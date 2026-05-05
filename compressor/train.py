"""Training loops for the T1 encoder and decoder.

Two public training functions:
  train_encoder  — pair-MSE + magnitude-validity on RVT batches.
                   Step 1 of the piecemeal-assembly plan.
  (train_decoder is in scripts/train_decoder.py — it's a CLI shell
   over a small loop, not factored out here yet.)

Shared helpers (used by trainers, eval, and the acceptance suite):
  load_checkpoint           — load a torch.compile-aware state_dict.
  encode_instrs             — tokenize + pad + encode an Instruction list.
  prepare_decoder_targets   — build (input, target, padding) for CE.
  equivalence_loss          — auxiliary loss for MANIFEST collapse,
                              optional add-on to the encoder loop.
"""

import time
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import BOS, EOS, PAD, VOCAB_SIZE, encode_instruction
from tokenizer.tokenizer import decode_instruction
from emulator import (
    Instruction, R_TYPE, I_TYPE, B_TYPE, LOAD_TYPE, STORE_TYPE,
)
from datagen.batch import padding_mask
from datagen.compare import pairwise_distance_canonical
from datagen.generate import MANIFEST, sample_binding, materialize
from .model import T1Compressor, T2Compressor


# Register-set + commutativity used by the operand-extraction helper.
# A commutative R/B-type op has its two source registers sorted before
# being assigned to slot 0 vs slot 1 — so `ADD x5,x1,x2` and
# `ADD x5,x2,x1` produce the same aux targets and don't fight pair-MSE
# (which already gives them behavioral_distance = 0).
_COMMUTATIVE_INSTRS = frozenset({
    'ADD', 'XOR', 'OR', 'AND',  # R-type commutative ALU
    'BEQ', 'BNE',               # B-type symmetric comparators
})

# Cross-entropy ignore index. Invalid chunks have no decodable
# instruction, so their aux targets are -100 and they don't contribute
# to the CE losses.
_CE_IGNORE = -100


def _extract_operands(instr):
    """Return (dest_type, dest_reg, src_reg_0, src_reg_1) for one
    instruction. dest_type 0 = register write, 1 = memory write
    (STORE), branches use 0 with dest_reg=0. Sources are sorted for
    commutative ops so ADD x5,x1,x2 ≡ ADD x5,x2,x1 at the aux level."""
    op = instr.opcode
    if op in STORE_TYPE:
        # args: rs2, imm, rs1 — both rs1 (addr) and rs2 (value) are sources.
        return 1, 0, instr.args[2], instr.args[0]
    if op in B_TYPE:
        s0, s1 = instr.args[0], instr.args[1]
        if op in _COMMUTATIVE_INSTRS:
            s0, s1 = sorted((s0, s1))
        return 0, 0, s0, s1
    if op in LOAD_TYPE:
        # args: rd, imm, rs1
        return 0, instr.args[0], instr.args[2], 0
    if op in R_TYPE:
        s0, s1 = instr.args[1], instr.args[2]
        if op in _COMMUTATIVE_INSTRS:
            s0, s1 = sorted((s0, s1))
        return 0, instr.args[0], s0, s1
    if op in I_TYPE:
        return 0, instr.args[0], instr.args[1], 0
    if op == 'JALR':
        return 0, instr.args[0], instr.args[1], 0
    # LUI, AUIPC, JAL: rd, imm — no register sources.
    return 0, instr.args[0], 0, 0


def _extract_aux_targets(batch):
    """Per chunk: (dest_type, dest_reg, src_reg_0, src_reg_1) by
    decoding the chunk's first instruction. Invalid chunks (or any
    decode failure) get -100 (CE ignore_index). Single-instruction
    chunks (--rule single) populate one row; multi-instruction chunks
    use the first instruction only — aux supervision is currently
    only meaningful at the per-instruction level."""
    B = batch.tokens.shape[0]
    dt = np.full(B, _CE_IGNORE, dtype=np.int64)
    dr = np.full(B, _CE_IGNORE, dtype=np.int64)
    s0 = np.full(B, _CE_IGNORE, dtype=np.int64)
    s1 = np.full(B, _CE_IGNORE, dtype=np.int64)
    for c in range(B):
        if not batch.valid[c]:
            continue
        n = int(batch.instr_lens[c, 0])
        if n == 0:
            continue
        toks = batch.tokens[c, :n].tolist()
        try:
            instr, _ = decode_instruction(toks, 0)
        except Exception:
            continue
        a, b, c0, c1 = _extract_operands(instr)
        dt[c] = a
        dr[c] = b
        s0[c] = c0
        s1[c] = c1
    return dt, dr, s0, s1


# ===========================================================================
# Probe instructions — fixed pairs whose distances we log during training
# to watch the encoder's geometry develop in real time.
# ===========================================================================

_PROBE_INSTRS = [
    Instruction('ADD',  5, 7, 7),   # 0: ADD rs,rs (double canonical)
    Instruction('SLLI', 5, 7, 1),   # 1: SLLI rs,1 (shl1 canonical)
    Instruction('ADD',  5, 1, 2),   # 2: random ADD
    Instruction('SLLI', 5, 3, 2),   # 3: SLLI neighbor
    Instruction('ADD',  5, 2, 1),   # 4: commutative swap of #2
]


def _probe_tensor(device):
    encoded = [encode_instruction(i) for i in _PROBE_INSTRS]
    max_len = max(len(e) for e in encoded)
    tok = np.full((len(encoded), max_len), PAD, dtype=np.int64)
    pad = np.ones((len(encoded), max_len), dtype=np.bool_)
    for i, e in enumerate(encoded):
        tok[i, :len(e)] = e
        pad[i, :len(e)] = False
    return (torch.from_numpy(tok).to(device),
            torch.from_numpy(pad).to(device))


def _probe_distances(encoder, probe_tok, probe_pad):
    with torch.no_grad():
        v = encoder.encode(probe_tok, probe_pad)
        v = F.normalize(v, dim=-1)
        return {
            'aa':    float((v[0] - v[2]).norm()),
            'canon': float((v[0] - v[1]).norm()),
            'ss':    float((v[1] - v[3]).norm()),
            'comm':  float((v[2] - v[4]).norm()),
        }


# ===========================================================================
# Shared helpers
# ===========================================================================

def load_checkpoint(path, device):
    """Load a state dict, stripping the torch.compile prefix.

    `torch.compile` wraps modules so saved state dicts carry an
    `_orig_mod.` prefix on every key. This helper handles both
    compiled and uncompiled checkpoints transparently.
    """
    state = torch.load(path, map_location=device, weights_only=True)
    return {k.removeprefix('_orig_mod.'): v for k, v in state.items()}


def encode_instrs(model, instrs, device):
    """Tokenize, pad, and encode a list of Instructions."""
    encoded = [encode_instruction(i) for i in instrs]
    max_len = max(len(e) for e in encoded)
    tok = np.full((len(encoded), max_len), PAD, dtype=np.int64)
    pad = np.ones((len(encoded), max_len), dtype=np.bool_)
    for i, e in enumerate(encoded):
        tok[i, :len(e)] = e
        pad[i, :len(e)] = False
    return model.encode(torch.from_numpy(tok).to(device),
                        torch.from_numpy(pad).to(device))


def prepare_decoder_targets(token_lists, device):
    """Build decoder input/target tensors from lists of token IDs.

    Each token list gets wrapped: [BOS] + tokens + [EOS].
    Returns dec_input (shifted right), dec_target, dec_padding.
    Returns (None, None, None) if token_lists is empty.
    """
    if not token_lists:
        return None, None, None

    seqs = [[BOS] + toks + [EOS] for toks in token_lists]
    max_len = max(len(s) - 1 for s in seqs)
    N = len(seqs)
    dec_input = np.full((N, max_len), PAD, dtype=np.int64)
    dec_target = np.full((N, max_len), PAD, dtype=np.int64)
    dec_padding = np.ones((N, max_len), dtype=np.bool_)

    for j, seq in enumerate(seqs):
        L = len(seq) - 1
        dec_input[j, :L] = seq[:-1]
        dec_target[j, :L] = seq[1:]
        dec_padding[j, :L] = False

    return (torch.from_numpy(dec_input).to(device),
            torch.from_numpy(dec_target).to(device),
            torch.from_numpy(dec_padding).to(device))


def equivalence_loss(model, device, rng):
    """Per-class MSE on canonical *directional* pairwise distances, target=0.

    For every manifest class with ≥2 canonical templates: sample a
    fresh binding, materialize, encode, normalize to unit direction,
    measure mean squared within-class distance. Return the mean across
    classes. Suitable as an auxiliary loss to nudge equivalence
    geometry in addition to the per-pair distance signal.
    """
    all_instrs = []
    class_ranges = []
    start = 0
    for klass in MANIFEST:
        if len(klass.canonical) < 2:
            continue
        binding = sample_binding(klass, rng)
        instrs = [materialize(t, binding) for t in klass.canonical]
        all_instrs.extend(instrs)
        class_ranges.append((start, start + len(instrs)))
        start += len(instrs)

    if not class_ranges:
        return torch.tensor(0.0, device=device)

    vecs = encode_instrs(model, all_instrs, device)
    vecs = F.normalize(vecs, dim=-1)

    losses = []
    for s, e in class_ranges:
        cv = vecs[s:e]
        N = cv.shape[0]
        idx = torch.triu_indices(N, N, offset=1, device=device)
        d = torch.cdist(cv.unsqueeze(0), cv.unsqueeze(0)).squeeze(0)
        losses.append(d[idx[0], idx[1]].square().mean())
    return torch.stack(losses).mean()


# ===========================================================================
# train_encoder — pair-MSE + magnitude-validity on RVT batches
# ===========================================================================

def _resolve_device(spec):
    if spec == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return spec


def behavioral_loss(vecs, pair_indices, distances, *,
                    eps=1e-12, scale=10.0):
    """MSE of cosine distance against compressed behavioral_distance.

    Cosine distance (1 - cos θ) is the canonical direction-only
    comparison for vector embeddings: it measures only the angle
    between two vectors, with magnitude factored out by F.cosine_-
    similarity's internal normalization. This honors the design
    intent from Phase 9 of EXPERIMENT_LOG.md and WHAT_IS_A_THOUGHT.md:
    T1 lives in the unit ball B^128 with magnitude = validity/
    coherence and direction = semantics, two orthogonal axes. Magnitude
    is reserved for valid_loss; behavioral_loss touches only direction.

    The previous Euclidean formulation collapsed both axes into one
    signal and pulled magnitude well above 1 to fit large pair targets.
    An interim L2-chord-on-the-unit-sphere version fixed the magnitude
    leak but still required an eps-inside-sqrt trick for the sibling
    optimum's 0/0 gradient. Cosine distance has clean gradients
    everywhere by construction.

    Targets are compressed via tanh into [0, 2] — the natural range
    of cosine distance. behavioral_distance can reach ~55 on
    genuinely different chunks; cosine distance maxes at 2 (anti-
    parallel directions). Compression preserves ordering (siblings
    stay at 0, "very different" saturates near 2) while losing the
    long tail's fine-grained ordinal info, which we don't use for
    supervision anyway.
    """
    if pair_indices.shape[0] == 0:
        return torch.tensor(0.0, device=vecs.device)
    cos_sim = F.cosine_similarity(
        vecs[pair_indices[:, 0]],
        vecs[pair_indices[:, 1]],
        dim=-1, eps=eps,
    )
    d_pred = 1.0 - cos_sim
    d_target = 2.0 * torch.tanh(distances / scale)
    return ((d_pred - d_target) ** 2).mean()


def behavioral_loss_row_outputs(vecs, row_outputs, n_inputs, input_mags,
                                has_rd, rd_mag, pair_valid, *,
                                eps=1e-12, scale=10.0):
    """Cosine-distance MSE loss for the row-outputs T1 path.

    Forms the (B, B) target distance matrix on-device from the per-row
    canonical execution outputs via `pairwise_distance_canonical`, then
    compares against the encoder's pairwise cosine distances. Same
    direction-only formulation as `behavioral_loss`, just with implicit
    pair structure (every (i, j) pair where both rows are pair_valid
    contributes).

    All row_* args are torch tensors on the same device as `vecs`. The
    pair_valid mask drops pairs involving invalid windows, mem-op rows,
    or any row whose canonical outputs aren't meaningful.

    Returns a scalar loss; 0 when the mask leaves no valid pairs (e.g.,
    a batch that's all invalid windows).
    """
    if pair_valid.numel() == 0 or not pair_valid.any():
        return torch.tensor(0.0, device=vecs.device)

    target_d = pairwise_distance_canonical(
        row_outputs, n_inputs, input_mags, has_rd, rd_mag)  # (B, B)
    d_target = 2.0 * torch.tanh(target_d / scale)

    vecs_n = F.normalize(vecs, dim=-1, eps=eps)
    cos_sim = vecs_n @ vecs_n.T   # (B, B)
    d_pred = 1.0 - cos_sim

    pair_mask = pair_valid[:, None] & pair_valid[None, :]
    eye = torch.eye(
        pair_valid.shape[0], dtype=torch.bool, device=pair_valid.device)
    pair_mask = pair_mask & ~eye

    if not pair_mask.any():
        return torch.tensor(0.0, device=vecs.device)

    diff_sq = (d_pred - d_target) ** 2
    return diff_sq[pair_mask].mean()


def train_encoder(batch_iter, *,
                  d_model=128, n_heads=4, n_layers=2, d_out=128,
                  max_window=72,
                  lr=3e-4, n_steps=None, log_every=100, lr_min=1e-6,
                  behavioral_weight=1.0, behavioral_scale=10.0,
                  valid_weight=0.1, equiv_weight=0.0,
                  dest_type_weight=0.1, dest_reg_weight=0.1,
                  src_reg_weight=0.1,
                  equiv_seed=0, device='auto',
                  on_log=None):
    """Train a T1 encoder on RVT batches.

    behavioral_weight: weight on (||v_i - v_j|| - distance)**2 over pair_indices.
    valid_weight:      weight on (||v_c|| - (1 if valid else 0))**2.
    equiv_weight:      optional MANIFEST equivalence_loss. 0 disables.
    dest_type_weight:  CE on dest_type_head — does this instr write a
                       register or memory.
    dest_reg_weight:   CE on dest_reg_head — which register is dest.
    src_reg_weight:    CE on src_reg_head_0 + src_reg_head_1 (summed,
                       same weight for both slots) — which registers
                       are sources. Sources are sorted for commutative
                       ops so they don't fight pair-MSE.
    on_log:            optional callable(step, encoder, losses)
                       called after each log point.

    Returns (encoder, losses). losses[i] dict has keys: step, total,
    behavioral, valid, equiv, dest_type, dest_reg, src_reg, plus
    probe_* keys at each log point.
    """
    device = _resolve_device(device)

    encoder = T1Compressor(
        VOCAB_SIZE, d_model, n_heads, n_layers, d_out,
        max_window=max_window,
    ).to(device)
    opt = torch.optim.Adam(encoder.parameters(), lr=lr, fused=(device == 'cuda'))

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=n_steps, eta_min=lr_min)
        if n_steps else None)

    probe_tok, probe_pad = _probe_tensor(device)
    eq_rng = np.random.default_rng(equiv_seed)

    losses = []
    step = 0
    t0 = time.time()
    t_last = t0

    for batch in batch_iter:
        if n_steps is not None and step >= n_steps:
            break

        tok = torch.from_numpy(batch.tokens).to(device)
        pad = torch.from_numpy(padding_mask(batch)).to(device)
        valid = torch.from_numpy(batch.valid).to(device)

        encoder.train()
        vecs = encoder.encode(tok, pad)              # (B, d_out)

        # Pair-distance loss: dispatch on which payload mode the batch
        # carries. Row-outputs mode (T1 single-instruction) forms the
        # (B, B) target on-device from per-row canonical outputs;
        # pair-indices mode (multi-instruction) reads pre-computed
        # scalar distances.
        if batch.row_outputs.shape[0] > 0:
            ro = torch.from_numpy(batch.row_outputs).to(device)
            n_in = torch.from_numpy(batch.row_n_inputs).to(device).long()
            in_mags = torch.from_numpy(batch.row_input_mags).to(device)
            hrd = torch.from_numpy(batch.row_has_rd).to(device)
            rmag = torch.from_numpy(batch.row_rd_mag).to(device)
            pvalid = torch.from_numpy(batch.pair_valid).to(device)
            behavioral = behavioral_loss_row_outputs(
                vecs, ro, n_in, in_mags, hrd, rmag, pvalid,
                scale=behavioral_scale)
        else:
            ij = torch.from_numpy(
                batch.pair_indices.astype(np.int64)).to(device)
            d_target = torch.from_numpy(batch.distances).to(device)
            behavioral = behavioral_loss(vecs, ij, d_target,
                                          scale=behavioral_scale)

        # Validity magnitude.
        target_mag = valid.float()
        actual_mag = vecs.norm(dim=-1)
        valid_loss = ((actual_mag - target_mag) ** 2).mean()

        # Optional equivalence collapse.
        if equiv_weight > 0:
            eq_loss = equivalence_loss(encoder, device, eq_rng)
        else:
            eq_loss = torch.tensor(0.0, device=device)

        # Aux register-identity CE losses — read straight from the T1
        # vector via the four heads. Targets come from decoding the
        # chunk's first instruction; invalid chunks contribute -100
        # (CE ignore_index) so they don't fight the magnitude loss.
        if (dest_type_weight > 0 or dest_reg_weight > 0
                or src_reg_weight > 0):
            dt_t, dr_t, s0_t, s1_t = _extract_aux_targets(batch)
            dt_t = torch.from_numpy(dt_t).to(device)
            dr_t = torch.from_numpy(dr_t).to(device)
            s0_t = torch.from_numpy(s0_t).to(device)
            s1_t = torch.from_numpy(s1_t).to(device)
            # Aux heads project from direction only — passing raw vecs
            # would let the encoder use magnitude as a side-channel for
            # register identity, colliding with the validity-magnitude
            # loss. Magnitude must remain reserved for validity (unit
            # ball design, Phase 9).
            vecs_dir = F.normalize(vecs, dim=-1)
            dt_loss = F.cross_entropy(
                encoder.dest_type_head(vecs_dir), dt_t, ignore_index=_CE_IGNORE)
            dr_loss = F.cross_entropy(
                encoder.dest_reg_head(vecs_dir), dr_t, ignore_index=_CE_IGNORE)
            s0_loss = F.cross_entropy(
                encoder.src_reg_head_0(vecs_dir), s0_t, ignore_index=_CE_IGNORE)
            s1_loss = F.cross_entropy(
                encoder.src_reg_head_1(vecs_dir), s1_t, ignore_index=_CE_IGNORE)
            src_loss = s0_loss + s1_loss
        else:
            dt_loss = torch.tensor(0.0, device=device)
            dr_loss = torch.tensor(0.0, device=device)
            src_loss = torch.tensor(0.0, device=device)

        total = (behavioral_weight * behavioral
                 + valid_weight * valid_loss
                 + equiv_weight * eq_loss
                 + dest_type_weight * dt_loss
                 + dest_reg_weight * dr_loss
                 + src_reg_weight * src_loss)
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        opt.step()
        if scheduler:
            scheduler.step()

        step += 1
        record = {
            'step': step,
            'total': float(total),
            'behavioral': float(behavioral),
            'valid': float(valid_loss),
            'equiv': float(eq_loss),
            'dest_type': float(dt_loss),
            'dest_reg': float(dr_loss),
            'src_reg': float(src_loss),
        }
        if step % log_every == 0 or step == 1:
            probes = _probe_distances(encoder, probe_tok, probe_pad)
            record.update({f'probe_{k}': v for k, v in probes.items()})
            now = time.time()
            ms_per_step = (now - t_last) / max(1, log_every) * 1000
            t_last = now
            lr_now = scheduler.get_last_lr()[0] if scheduler else lr
            eta = ((n_steps - step) * ms_per_step / 1000) if n_steps else 0
            print(
                f'step {step:>5d}  loss {record["total"]:.4f} '
                f'(beh {record["behavioral"]:.4f} val {record["valid"]:.4f}'
                f' eq {record["equiv"]:.4f}'
                f' dt {record["dest_type"]:.3f} dr {record["dest_reg"]:.3f}'
                f' src {record["src_reg"]:.3f})  '
                f'lr {lr_now:.1e}  {ms_per_step:.0f}ms/step  '
                f'eta {timedelta(seconds=int(eta))}  '
                f'probes[aa={probes["aa"]:.3f} canon={probes["canon"]:.3f} '
                f'ss={probes["ss"]:.3f} comm={probes["comm"]:.3f}]')
            if on_log is not None:
                on_log(step, encoder, losses + [record])
        losses.append(record)

    return encoder, losses


# ===========================================================================
# T2: per-chunk encoder over T1 emission vectors
# ===========================================================================

# Per-instruction max token count. Match datagen.batch.MAX_INSTR_TOKENS;
# duplicated here only to avoid the cross-package import in a hot path.
_MAX_INSTR_TOKENS = 9


def _split_to_per_instruction(batch):
    """From a Batch with instr_lens, build flat per-instruction token
    arrays + a mapping back to (chunk, slot).

    Returns:
        instr_tokens: (N_instr_total, MAX_INSTR_TOKENS) int64 — padded
                      with PAD past each instruction's actual length.
        instr_pad:    (N_instr_total, MAX_INSTR_TOKENS) bool — True
                      where padded.
        chunk_idx:    (N_instr_total,) int64 — which row in the
                      original batch each instruction came from.
        slot_idx:     (N_instr_total,) int64 — which position-within-chunk.
        n_per_chunk:  (B,) int32 — instructions in each chunk
                      (0 for invalid rows).
    """
    B, max_n_instrs = batch.instr_lens.shape
    n_per_chunk = (batch.instr_lens > 0).sum(axis=1).astype(np.int32)
    total = int(n_per_chunk.sum())
    if total == 0:
        empty = np.zeros((0, _MAX_INSTR_TOKENS), dtype=np.int64)
        return (
            empty,
            np.ones((0, _MAX_INSTR_TOKENS), dtype=bool),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            n_per_chunk,
        )

    instr_tokens = np.full((total, _MAX_INSTR_TOKENS), PAD, dtype=np.int64)
    instr_pad = np.ones((total, _MAX_INSTR_TOKENS), dtype=bool)
    chunk_idx = np.empty(total, dtype=np.int64)
    slot_idx = np.empty(total, dtype=np.int64)

    out = 0
    for c in range(B):
        token_offset = 0
        for j in range(max_n_instrs):
            L = int(batch.instr_lens[c, j])
            if L == 0:
                continue
            instr_tokens[out, :L] = batch.tokens[c, token_offset:token_offset + L]
            instr_pad[out, :L] = False
            chunk_idx[out] = c
            slot_idx[out] = j
            out += 1
            token_offset += L
    return instr_tokens, instr_pad, chunk_idx, slot_idx, n_per_chunk


def train_t2_encoder(batch_iter, t1_encoder, *,
                     d_model=256, n_heads=4, n_layers=2, d_out=256,
                     max_chunk_len=32,
                     lr=3e-4, n_steps=None, log_every=100, lr_min=1e-6,
                     behavioral_weight=1.0, behavioral_scale=10.0,
                     valid_weight=0.0,
                     device='auto',
                     on_log=None):
    """Train a T2 encoder on top of a frozen T1 encoder.

    For each RVT batch:
      1. Split flat chunk tokens into per-instruction segments via
         batch.instr_lens.
      2. Run frozen T1.encode on every instruction in the batch
         (one big GPU call across all chunks).
      3. Reshape T1 outputs back into per-chunk sequences with padding.
      4. Run T2.encode over the sequences → one T2 vector per chunk.
      5. Pair-MSE on (||t2_a - t2_b|| - distance)^2 over batch.pair_indices.
      6. Optional magnitude-validity loss (defaults off — corpora
         generated without --inject-invalid have no negative class).

    on_log: optional callable(step, t2, losses) called after each log
            point. Use it to checkpoint mid-run.

    Returns (t2, losses).
    """
    device = _resolve_device(device)
    t1_encoder.eval()
    for p in t1_encoder.parameters():
        p.requires_grad = False
    t1_encoder = t1_encoder.to(device)

    t2 = T2Compressor(
        d_t1=t1_encoder.d_out, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_out=d_out, max_chunk_len=max_chunk_len,
    ).to(device)
    opt = torch.optim.Adam(t2.parameters(), lr=lr,
                           fused=(device == 'cuda'))

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=n_steps, eta_min=lr_min)
        if n_steps else None)

    losses = []
    step = 0
    t0 = time.time()
    t_last = t0

    for batch in batch_iter:
        if n_steps is not None and step >= n_steps:
            break

        # 1-3. Per-instruction T1 encode → per-chunk T1 sequences.
        instr_tokens, instr_pad, chunk_idx, slot_idx, n_per_chunk = \
            _split_to_per_instruction(batch)
        if instr_tokens.shape[0] == 0:
            continue

        instr_tok_t = torch.from_numpy(instr_tokens).to(device)
        instr_pad_t = torch.from_numpy(instr_pad).to(device)
        with torch.no_grad():
            t1_vecs = t1_encoder.encode(instr_tok_t, instr_pad_t)
            # (N_instr_total, d_t1)

        B, max_n_instrs = batch.instr_lens.shape
        chunk_t1 = torch.zeros(
            B, max_n_instrs, t1_encoder.d_out, device=device)
        chunk_idx_t = torch.from_numpy(chunk_idx).to(device)
        slot_idx_t = torch.from_numpy(slot_idx).to(device)
        chunk_t1[chunk_idx_t, slot_idx_t] = t1_vecs

        n_per_chunk_t = torch.from_numpy(n_per_chunk).to(device)
        slot_arange = torch.arange(max_n_instrs, device=device)
        chunk_pad = slot_arange[None, :] >= n_per_chunk_t[:, None]

        # Guard against all-padded rows (chunks with 0 valid instructions
        # — invalid windows). Attention with key_padding_mask=all-True
        # yields softmax-of-all-(-inf) = NaN, and even though those rows
        # aren't in the loss, the gradient through shared parameters
        # (pool_query) becomes 0 * NaN = NaN. Unmask position 0 so each
        # row has at least one (zero) key; the row's output is
        # deterministic but irrelevant since pair_indices doesn't
        # reference invalid rows.
        chunk_pad[:, 0] = False

        # 4. T2 forward.
        t2.train()
        t2_vecs = t2.encode(chunk_t1, chunk_pad)  # (B, d_out)

        # 5. Pair-MSE.
        ij = torch.from_numpy(
            batch.pair_indices.astype(np.int64)).to(device)
        d_target = torch.from_numpy(batch.distances).to(device)
        behavioral = behavioral_loss(t2_vecs, ij, d_target,
                                      scale=behavioral_scale)

        # 6. Optional magnitude-validity (0 by default for clean corpora).
        if valid_weight > 0:
            valid = torch.from_numpy(batch.valid).to(device).float()
            actual_mag = t2_vecs.norm(dim=-1)
            valid_loss = ((actual_mag - valid) ** 2).mean()
        else:
            valid_loss = torch.tensor(0.0, device=device)

        total = behavioral_weight * behavioral + valid_weight * valid_loss
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(t2.parameters(), 1.0)
        opt.step()
        if scheduler:
            scheduler.step()

        step += 1
        record = {
            'step': step,
            'total': float(total),
            'behavioral': float(behavioral),
            'valid': float(valid_loss),
            'n_chunks': B,
            'n_instrs': int(n_per_chunk.sum()),
        }
        if step % log_every == 0 or step == 1:
            now = time.time()
            ms_per_step = (now - t_last) / max(1, log_every) * 1000
            t_last = now
            lr_now = scheduler.get_last_lr()[0] if scheduler else lr
            eta = ((n_steps - step) * ms_per_step / 1000) if n_steps else 0
            print(
                f'step {step:>5d}  loss {record["total"]:.4f} '
                f'(behavioral {record["behavioral"]:.4f} valid {record["valid"]:.4f}) '
                f'B={record["n_chunks"]} I={record["n_instrs"]}  '
                f'lr {lr_now:.1e}  {ms_per_step:.0f}ms/step  '
                f'eta {timedelta(seconds=int(eta))}')
            if on_log is not None:
                on_log(step, t2, losses + [record])
        losses.append(record)

    return t2, losses
