"""Diagnostic functions for the equivariant compressor.

Pure functions, no CLI (driven by scripts/eval.py and the test suite);
each returns a flat dict of metrics. Coverage:

  equivariance_error   — relabel -> state permutes EXACTLY, essence invariant
  t1_binding_accuracy  — exact per-instruction wiring as T2 consumes it (T1)
  t2_exact_binding     — exact per-chunk input/output lists, by chunk shape (T2)
  gvn_collapse         — essence distance split by pair type: rename-twin
                         (free/structural) vs behavioral-non-rename
                         (commutativity — TRAINED) vs distinct
"""

import numpy as np
import torch
import torch.nn.functional as F

from datagen import (
    padding_mask, heldout_anchor_states, precompute_chunk,
    DEFAULT_DISTRIBUTION, build_opcode_table, random_instruction,
    AUX_CE_IGNORE, N_REGS,
)
from datagen.compare import _source_regs, _written_regs, instruction_reads
from datagen.generate import random_perm, relabel
from emulator import Instruction, R_TYPE
from tokenizer import encode_instruction, PAD

from .model import instruction_wiring, X0_SLOT
from .train import (
    _t1_wiring, _t1_predicted_wiring, _split_to_per_instruction,
    _t2_assemble, t2_chunk_forward,
)
from tokenizer import decode_instruction

_MAX_INSTR_TOKENS = 9   # matches datagen.batch.MAX_INSTR_TOKENS
_COMMUTATIVE = frozenset({'ADD', 'AND', 'OR', 'XOR'})


@torch.no_grad()
def _chunks_to_split(chunks):
    """Build the _split_to_per_instruction tuple directly from a list of
    chunks (each a list[Instruction]) — the controlled-eval analog of the
    batch splitter, with token-decoded per-instruction wiring (which only
    feeds T1; T2 routes off T1's prediction)."""
    rows, ci, si, in0, in1, out = [], [], [], [], [], []
    n_per_chunk = np.zeros(len(chunks), dtype=np.int32)
    for c, instrs in enumerate(chunks):
        n_per_chunk[c] = len(instrs)
        for j, instr in enumerate(instrs):
            rows.append(encode_instruction(instr))
            ci.append(c); si.append(j)
            a, b, o = instruction_wiring(instr)
            in0.append(a); in1.append(b); out.append(o)
    total = len(rows)
    instr_tokens = np.full((total, _MAX_INSTR_TOKENS), PAD, dtype=np.int8)
    instr_pad = np.ones((total, _MAX_INSTR_TOKENS), dtype=bool)
    for i, toks in enumerate(rows):
        instr_tokens[i, :len(toks)] = toks
        instr_pad[i, :len(toks)] = False
    to_i64 = lambda x: np.array(x, dtype=np.int64)
    in0, in1, out = to_i64(in0), to_i64(in1), to_i64(out)
    for a in (in0, in1, out):
        a[(a < 0) | (a >= N_REGS)] = 0
    return (instr_tokens, instr_pad, to_i64(ci), to_i64(si), n_per_chunk,
            in0, in1, out)


@torch.no_grad()
def encode_chunks_t2(t1, t2, chunks, device, *, route='binding'):
    """Essence (n_chunks, d_out) for a list of Instruction-list chunks, run
    through frozen T1 -> T2 with the given routing. Deterministic: the same
    chunks always give the same essences."""
    split = _chunks_to_split(chunks)
    n_chunks = len(chunks)
    max_ni = max((len(c) for c in chunks), default=1)
    return t2_chunk_forward(t1, t2, split, n_chunks, max_ni, device,
                            route=route)


# ===========================================================================
# Equivariance + tag-invariance
# ===========================================================================

@torch.no_grad()
def equivariance_error(encoder, batch, *, device, seed=0):
    """Exact-equivariance check on a trained T1: relabel registers (permute
    the wiring; content is register-invariant by construction) and confirm the
    emitted state permutes identically and the essence is invariant. Both ~0 by
    construction — this verifies it on the *trained* weights. Returns
    {'state_max_err', 'essence_max_err'}."""
    encoder.eval()
    R = encoder.n_regs
    tok = torch.from_numpy(batch.tokens).to(device).long()
    pad = torch.from_numpy(padding_mask(batch)).to(device)
    in0, in1, out = _t1_wiring(batch)
    B = tok.shape[0]
    rng = np.random.default_rng(seed)
    perm = np.arange(R)
    perm[1:] = rng.permutation(R - 1) + 1            # fix slot 0 (x0)
    def enc(i0, i1, o):
        return encoder.encode_state(
            tok, pad,
            torch.from_numpy(i0).to(device), torch.from_numpy(i1).to(device),
            torch.from_numpy(o).to(device))

    T, ess = enc(in0, in1, out)
    T_p, ess_p = enc(perm[in0], perm[in1], perm[out])
    perm_t = torch.from_numpy(perm).to(device)
    T_exp = torch.empty_like(T)
    T_exp[:, perm_t] = T
    return {
        'state_max_err': float((T_p - T_exp).abs().max()),
        'essence_max_err': float((ess_p - ess).abs().max()),
    }


# ===========================================================================
# Binding accuracy, exact, as the next tier routes on it
# ===========================================================================

def _decode_order(scores):
    """Per row, the predicted slot ordering = argsort of per-register scores
    descending, with x0 forced last (never a GP operand/output). (B, n_regs)
    long. argsort is duplicate-free by construction."""
    s = scores.clone()
    s[:, X0_SLOT] = torch.finfo(s.dtype).min
    return s.argsort(dim=-1, descending=True)


def _operand_form(instr):
    """The operand pattern a binding head has to read: which sources are x0 or
    alias each other, and whether the destination aliases a source."""
    rd, *rest = instr.args
    if instr.opcode in R_TYPE:
        rs1, rs2 = rest
        form = ('x0,x0' if rs1 == rs2 == 0 else 'x0,rs' if rs1 == 0
                else 'rs,x0' if rs2 == 0 else 'rs,rs' if rs1 == rs2
                else 'rs1,rs2')
        srcs = (rs1, rs2)
    else:
        form, srcs = 'imm', (rest[0],)
    return f'{instr.opcode} {form}' + (' (self-write)' if rd in srcs else '')


@torch.no_grad()
def t1_binding_accuracy(encoder, batches, *, device, max_batches=20):
    """Held-out T1 binding: the fraction of instructions whose EMITTED wiring
    (`_t1_predicted_wiring`, the (in0, in1, out) T2 routes on) equals the
    oracle's — the first two input slots in order and the first output slot,
    x0 where absent. The target is 1.0. Per-register averages over 32 slots
    saturate at 1.000 with one instruction in seventy mis-wired, so this scores
    whole instructions, and breaks the misses down by operand form."""
    encoder.eval()
    hits = total = 0
    forms = {}
    for n, batch in enumerate(batches):
        if n >= max_batches:
            break
        tok = torch.from_numpy(batch.tokens).to(device).long()
        pad = torch.from_numpy(padding_mask(batch)).to(device)
        in0, in1, out = (torch.from_numpy(a).to(device) for a in _t1_wiring(batch))
        T, _ = encoder.encode_state(tok, pad, in0, in1, out)
        emitted = torch.stack(_t1_predicted_wiring(encoder, T), 1).cpu().numpy()
        slots = np.where(batch.in_slot_regs[:, :2] == AUX_CE_IGNORE, 0,
                         batch.in_slot_regs[:, :2])
        dest = np.where(batch.out_slot_regs[:, 0] == AUX_CE_IGNORE, 0,
                        batch.out_slot_regs[:, 0])
        want = np.concatenate([slots, dest[:, None]], 1)
        ok = (emitted == want).all(1)
        for b in np.nonzero(batch.out_regs_valid)[0]:
            ins, _ = decode_instruction(batch.tokens[b].tolist(), 0)
            f = forms.setdefault(_operand_form(ins), [0, 0])
            f[0] += int(ok[b]); f[1] += 1
            hits += int(ok[b]); total += 1
    return {
        'exact': hits / max(total, 1), 'n_rows': total,
        'by_form': {k: (h / n, n) for k, (h, n) in forms.items()},
    }


def _emitted_binding(model, T):
    """The binding a tier emits, as the next tier would route on it: per row,
    the registers its live heads say are read (written), ordered by the score
    heads. Generalises `_t1_predicted_wiring` from (in0, in1, out) to every
    slot. Returns two lists of per-row register lists."""
    def decode(live_head, score_head):
        live = torch.sigmoid(live_head(T).squeeze(-1)) > 0.5
        live[:, X0_SLOT] = False
        n = live.sum(-1).cpu().numpy()
        order = _decode_order(score_head(T).squeeze(-1)).cpu().numpy()
        return [order[b, :n[b]].tolist() for b in range(order.shape[0])]
    return (decode(model.live_in_head, model.in_score_head),
            decode(model.live_out_head, model.out_score_head))


def _filled(slot_regs):
    return [[int(r) for r in row if r != AUX_CE_IGNORE] for row in slot_regs]


def _chunk_instrs(batch, b):
    toks, pos, out = batch.tokens[b].tolist(), 0, []
    for _ in range(int((batch.instr_lens[b] > 0).sum())):
        ins, pos = decode_instruction(toks, pos)
        out.append(ins)
    return out


def _chunk_shape(instrs):
    """The dataflow features a chunk's binding has to get right, as one label:
    an instruction reading another's result, two instructions reading one
    input, a write overwritten unread, a read the output ignores, x0 or aliased
    operands, a destination that is also a source."""
    tags, written, unread, readers = set(), set(), set(), {}
    for i, ins in enumerate(instrs):
        srcs, rel = _source_regs(ins), instruction_reads(ins)
        if 0 in srcs:
            tags.add('x0 operand')
        if len(srcs) == 2 and srcs[0] == srcs[1] != 0:
            tags.add('aliased')
        if any(r and r not in rel for r in srcs):
            tags.add('ignored read')
        for r in rel:
            if r in written:
                tags.add('composed')
            elif readers.setdefault(r, i) != i:
                tags.add('shared input')
            unread.discard(r)
        for r in _written_regs(ins):
            if r in unread:
                tags.add('dead write')
            if r in srcs:
                tags.add('self-write')
            unread.add(r)
            written.add(r)
    return ' + '.join(sorted(tags)) or 'plain'


@torch.no_grad()
def t2_exact_binding(t1, t2, batches, *, device, route='binding',
                     max_batches=20):
    """Held-out T2 binding: the fraction of chunks whose EMITTED binding — every
    input register in first-read order and every output register in write
    order — equals the oracle's. The target is 1.0. Broken down by chunk shape
    (`_chunk_shape`), and by opcode-and-operand form for the misses."""
    t1.eval(); t2.eval()
    hits = hits_in = hits_out = total = 0
    shapes, forms = {}, {}
    for n, batch in enumerate(batches):
        if n >= max_batches:
            break
        split = _split_to_per_instruction(batch)
        if split[0].shape[0] == 0:
            continue
        n_chunks, max_ni = batch.instr_lens.shape
        chunk_t1, c0, c1, co, cact = _t2_assemble(
            t1, split, n_chunks, max_ni, device, route=route)
        T, _ = t2.encode_state(chunk_t1, c0, c1, co, cact)
        ins_e, outs_e = _emitted_binding(t2, T)
        ins_w, outs_w = _filled(batch.in_slot_regs), _filled(batch.out_slot_regs)
        for b in np.nonzero(batch.out_regs_valid)[0]:
            ok_in, ok_out = ins_e[b] == ins_w[b], outs_e[b] == outs_w[b]
            ok = ok_in and ok_out
            instrs = _chunk_instrs(batch, b)
            for d, k in ((shapes, _chunk_shape(instrs)),
                         (forms, ' ; '.join(map(_operand_form, instrs)))):
                f = d.setdefault(k, [0, 0])
                f[0] += int(ok); f[1] += 1
            hits += ok; hits_in += ok_in; hits_out += ok_out; total += 1
    t = max(total, 1)
    return {
        'exact': hits / t, 'exact_in': hits_in / t, 'exact_out': hits_out / t,
        'n_rows': total,
        'by_shape': {k: (h / m, m) for k, (h, m) in shapes.items()},
        'by_form': {k: (h / m, m) for k, (h, m) in forms.items()},
    }


# ===========================================================================
# GVN collapse — split by pair type
# ===========================================================================

def _commute_variant(instrs, rng):
    """A behavioral-but-NOT-rename equivalent: swap the operands of one
    commutative R-type instruction (a+b == b+a). Returns a new chunk, or None
    if the chunk has no commutative op to swap."""
    idxs = [i for i, ins in enumerate(instrs)
            if ins.opcode in _COMMUTATIVE and ins.opcode in R_TYPE
            and ins.args[1] != ins.args[2]]
    if not idxs:
        return None
    i = idxs[rng.integers(len(idxs))]
    out = list(instrs)
    a = out[i].args
    out[i] = Instruction(out[i].opcode, a[0], a[2], a[1])   # swap rs1<->rs2
    return out


def _vp_target(p):
    """The values value-prediction is supervised on: the canonical output
    register file read at the WRITTEN slots, in write order. Comparing the whole
    32-register file instead would call two chunks different whenever they merely
    relocated their inputs to different canonical positions — true of most
    commutative-swap pairs, whose written outputs are identical."""
    slots = [int(r) for r in p.out_slot_regs if r != AUX_CE_IGNORE]
    return np.stack([p.out_regs[:, r] for r in slots]) if slots else None


def _behaviorally_equal(a_chunk, b_chunk, anchors):
    """True iff two chunks compute the same function of their canonical inputs,
    i.e. produce identical values at their written slots across anchors — the
    corpus's own equivalence label. None on undecodable / memory-op chunks."""
    try:
        ta = _vp_target(precompute_chunk(a_chunk, anchors))
        tb = _vp_target(precompute_chunk(b_chunk, anchors))
    except Exception:
        return None
    if ta is None or tb is None:
        return ta is None and tb is None
    return bool(ta.shape == tb.shape and np.array_equal(ta, tb))


@torch.no_grad()
def gvn_collapse(t1, t2, *, device, route='binding', n=200, chunk_len=4,
                 seed=0):
    """Essence-distance GVN collapse, SPLIT by pair type:

      rename     — chunk vs a random register-relabeling. Equivariance is
                   exact by construction, so this must read ~0 for ANY model:
                   it is an architecture assertion, not a quality signal.
      behavioral — chunk vs a commutative-operand-swap variant (a+b==b+a),
                   reported SPLIT by whether the swap actually preserved the
                   chunk's function. Collapsing the equivalent ones is TRAINED
                   value-numbering, not architecture; the non-equivalent ones
                   SHOULD be far apart. The mixed mean is not interpretable —
                   the ~2% non-equivalent pairs sit at the `distinct` scale and
                   dominate it.
      distinct   — two unrelated chunks confirmed behaviorally different.

    Returns mean (and the ratio) of normalized-essence L2 distance per type.
    A real value-numbering encoder collapses BOTH equivalent types far below
    `distinct`; if only `rename` collapses, the trained GVN is doing nothing."""
    t1.eval(); t2.eval()
    rng = np.random.default_rng(seed)
    anchors = heldout_anchor_states()
    table = build_opcode_table(DEFAULT_DISTRIBUTION)

    bases, twins, commutes = [], [], []
    while len(bases) < n:
        chunk = [random_instruction(rng, opcode_table=table)
                 for _ in range(chunk_len)]
        try:
            precompute_chunk(chunk, anchors)        # skip mem-op / undecodable
        except Exception:
            continue
        cv = _commute_variant(chunk, rng)
        if cv is None:
            continue
        perm = random_perm(rng)
        bases.append(chunk)
        twins.append(relabel(chunk, perm))
        commutes.append(cv)

    def enc(chunks):
        return F.normalize(
            encode_chunks_t2(t1, t2, chunks, device, route=route), dim=-1)

    eb, et, ec = enc(bases), enc(twins), enc(commutes)
    d_rename = (eb - et).norm(dim=-1)
    d_behav = (eb - ec).norm(dim=-1)
    # Swapping a commutative op's operands changes the chunk's first-read order.
    # If a later instruction reads one of the swapped registers, the canonical
    # input positions shift and the two chunks compute genuinely DIFFERENT
    # functions — ~2% of pairs, and they sit at the `distinct` scale, so they
    # dominate the mean. Separating them is correct behaviour, so the equivalent
    # subset is the row to judge collapse on.
    eq_flag = torch.tensor(
        [_behaviorally_equal(b, c, anchors) is True
         for b, c in zip(bases, commutes)], device=d_behav.device)
    d_behav_eq = d_behav[eq_flag]
    d_behav_ne = d_behav[~eq_flag]
    # distinct: pair base i with base j (i != j) only where behaviorally !=.
    perm = rng.permutation(len(bases))
    d_distinct = []
    for i, j in enumerate(perm):
        if i == j:
            continue
        eq = _behaviorally_equal(bases[i], bases[j], anchors)
        if eq is False:
            d_distinct.append((eb[i] - eb[j]).norm().item())
    d_distinct = torch.tensor(d_distinct) if d_distinct else torch.zeros(1)

    rn, bh, ds = (float(d_rename.mean()), float(d_behav.mean()),
                  float(d_distinct.mean()))
    return {
        'rename_mean': rn, 'behavioral_mean': bh, 'distinct_mean': ds,
        'behavioral_eq_mean': float(d_behav_eq.mean()) if d_behav_eq.numel() else None,
        'behavioral_ne_mean': float(d_behav_ne.mean()) if d_behav_ne.numel() else None,
        'n_behavioral_eq': int(eq_flag.sum()), 'n_behavioral_ne': int((~eq_flag).sum()),
        'rename_ratio': rn / ds if ds > 1e-9 else None,
        'behavioral_ratio': bh / ds if ds > 1e-9 else None,
        'n_pairs': len(bases), 'n_distinct': len(d_distinct) if isinstance(
            d_distinct, list) else int((d_distinct > 0).sum()),
    }
