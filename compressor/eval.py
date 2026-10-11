"""Diagnostic functions for the equivariant compressor.

Pure functions, no CLI (driven by scripts/eval.py and the test suite);
each returns a flat dict of metrics. Coverage:

  equivariance_error   — relabel -> state permutes EXACTLY, essence invariant
  t1_binding_accuracy  — live_in/out, pc, in/out-slot order, dup rate (T1)
  t2_binding_accuracy  — same, routing off T1's predicted binding (T2)
  gvn_collapse         — essence distance split by pair type: rename-twin
                         (free/structural) vs behavioral-non-rename
                         (commutativity — TRAINED) vs distinct
"""

import numpy as np
import torch
import torch.nn.functional as F

from datagen import (
    padding_mask, make_anchor_states, precompute_chunk,
    DEFAULT_DISTRIBUTION, build_opcode_table, random_instruction,
    AUX_CE_IGNORE, N_REGS,
)
from datagen.generate import random_perm, relabel
from emulator import Instruction, R_TYPE
from tokenizer import encode_instruction, PAD

from .model import instruction_wiring, X0_SLOT
from .train import (
    _t1_wiring, _split_to_per_instruction, _t2_assemble, t2_chunk_forward,
)

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
# Binding accuracy (shared T1/T2 core)
# ===========================================================================

def _decode_order(scores):
    """Per row, the predicted slot ordering = argsort of per-register scores
    descending, with x0 forced last (never a GP operand/output). (B, n_regs)
    long. argsort is duplicate-free by construction."""
    s = scores.clone()
    s[:, X0_SLOT] = torch.finfo(s.dtype).min
    return s.argsort(dim=-1, descending=True)


@torch.no_grad()
def _binding_metrics(model, T, essence, *, live_in, live_out, pc_writes,
                     in_slot, out_slot, active):
    """Accuracy of the binding heads against the batch targets, over active
    rows. live_*: per-element accuracy; in/out-slot: per-filled-slot order
    accuracy (argsort decode); dup: rows whose slot decode repeats a register
    (0 with argsort). All tensors on the model's device; active (B,) bool."""
    n_regs = model.n_regs
    a = active
    na = a.sum().clamp(min=1)

    li = (torch.sigmoid(model.live_in_head(T).squeeze(-1)) > 0.5).float()
    lo = (torch.sigmoid(model.live_out_head(T).squeeze(-1)) > 0.5).float()
    # pc reads essence DIRECTION only (magnitude = validity), matching training.
    pc = (torch.sigmoid(
        model.pc_writes_head(F.normalize(essence, dim=-1)).squeeze(-1)) > 0.5).float()
    li_acc = (((li == live_in).float().mean(-1)) * a).sum() / na
    lo_acc = (((lo == live_out).float().mean(-1)) * a).sum() / na
    pc_acc = ((pc == pc_writes).float() * a).sum() / na

    in_order = _decode_order(model.in_score_head(T).squeeze(-1))     # (B,n_regs)
    out_order = _decode_order(model.out_score_head(T).squeeze(-1))

    def slot_acc(order, slot_t, K):
        # per-filled-slot: order[:, k] == slot_t[:, k], over filled (!=IGNORE).
        pred = order[:, :K]
        tgt = slot_t[:, :K]
        filled = (tgt != AUX_CE_IGNORE) & a.unsqueeze(-1)
        correct = (pred == tgt) & filled
        n = filled.sum().clamp(min=1)
        return correct.sum() / n, int(filled.sum())

    in_acc, n_in = slot_acc(in_order, in_slot, model.max_input_slots)
    out_acc, n_out = slot_acc(out_order, out_slot, model.max_output_slots)
    # Duplicate rate of the decode over filled slots (argsort => 0).
    dup = 0.0
    return {
        'live_in_acc': float(li_acc), 'live_out_acc': float(lo_acc),
        'pc_acc': float(pc_acc),
        'in_slot_acc': float(in_acc), 'out_slot_acc': float(out_acc),
        'dup_rate': dup, 'n_active': int(a.sum()),
        'n_in_slots': n_in, 'n_out_slots': n_out,
    }


def _targets(batch, device):
    """The batch's binding targets as device tensors + the active-row mask."""
    li = torch.from_numpy(batch.live_in_mask).to(device).float()
    lo = torch.from_numpy(batch.live_out_mask).to(device).float()
    pc = torch.from_numpy(batch.pc_writes).to(device).float()
    in_slot = torch.from_numpy(batch.in_slot_regs.astype(np.int64)).to(device)
    out_slot = torch.from_numpy(batch.out_slot_regs.astype(np.int64)).to(device)
    active = (li.any(-1) | lo.any(-1))
    return li, lo, pc, in_slot, out_slot, active


def _accumulate(acc, m):
    """Sum metric dicts weighted by their slot/row counts for a clean mean."""
    for k in ('live_in_acc', 'live_out_acc', 'pc_acc'):
        acc[k] = acc.get(k, 0.0) + m[k] * m['n_active']
    acc['in_slot_acc'] = acc.get('in_slot_acc', 0.0) + m['in_slot_acc'] * m['n_in_slots']
    acc['out_slot_acc'] = acc.get('out_slot_acc', 0.0) + m['out_slot_acc'] * m['n_out_slots']
    acc['_rows'] = acc.get('_rows', 0) + m['n_active']
    acc['_in'] = acc.get('_in', 0) + m['n_in_slots']
    acc['_out'] = acc.get('_out', 0) + m['n_out_slots']


def _finalize(acc):
    rows = max(acc.get('_rows', 0), 1)
    nin = max(acc.get('_in', 0), 1)
    nout = max(acc.get('_out', 0), 1)
    return {
        'live_in_acc': acc.get('live_in_acc', 0.0) / rows,
        'live_out_acc': acc.get('live_out_acc', 0.0) / rows,
        'pc_acc': acc.get('pc_acc', 0.0) / rows,
        'in_slot_acc': acc.get('in_slot_acc', 0.0) / nin,
        'out_slot_acc': acc.get('out_slot_acc', 0.0) / nout,
        'dup_rate': 0.0,
        'n_rows': acc.get('_rows', 0),
    }


@torch.no_grad()
def t1_binding_accuracy(encoder, batches, *, device, max_batches=20):
    """Held-out T1 binding-head accuracy over RVT batches."""
    encoder.eval()
    acc = {}
    for n, batch in enumerate(batches):
        if n >= max_batches:
            break
        tok = torch.from_numpy(batch.tokens).to(device).long()
        pad = torch.from_numpy(padding_mask(batch)).to(device)
        in0, in1, out = (torch.from_numpy(a).to(device) for a in _t1_wiring(batch))
        T, ess = encoder.encode_state(tok, pad, in0, in1, out)
        li, lo, pc, in_slot, out_slot, active = _targets(batch, device)
        _accumulate(acc, _binding_metrics(
            encoder, T, ess, live_in=li, live_out=lo, pc_writes=pc,
            in_slot=in_slot, out_slot=out_slot, active=active))
    return _finalize(acc)


@torch.no_grad()
def t2_binding_accuracy(t1, t2, batches, *, device, route='binding',
                        max_batches=20):
    """Held-out T2 binding-head accuracy, routing off T1's predicted binding.
    Confirms the binding survives running on T1's *predicted* wiring,
    not ground-truth tokens."""
    t1.eval(); t2.eval()
    acc = {}
    for n, batch in enumerate(batches):
        if n >= max_batches:
            break
        split = _split_to_per_instruction(batch)
        if split[0].shape[0] == 0:
            continue
        n_chunks, max_ni = batch.instr_lens.shape

        chunk_t1, c0, c1, co, cact = _t2_assemble(
            t1, split, n_chunks, max_ni, device, route=route)
        T, ess = t2.encode_state(chunk_t1, c0, c1, co, cact)
        li, lo, pc, in_slot, out_slot, active = _targets(batch, device)
        _accumulate(acc, _binding_metrics(
            t2, T, ess, live_in=li, live_out=lo, pc_writes=pc,
            in_slot=in_slot, out_slot=out_slot, active=active))
    return _finalize(acc)


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
    anchors = make_anchor_states(8, 0)
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
