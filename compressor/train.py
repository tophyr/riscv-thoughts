"""Training loops for the T1 encoder, T2 encoder, and decoder.

Three public training functions — train_encoder, train_t2_encoder, and
train_decoder — each a batch loop the CLI shells in scripts/ drive.

Shared helpers (used by trainers, eval, and the acceptance suite):
  load_checkpoint           — load a torch.compile-aware state_dict.
  encode_instrs             — tokenize + pad + encode an Instruction list.
  prepare_decoder_targets   — build (input, target, padding) for CE.
  binding_losses            — rename-equivariant register-binding losses,
                              shared by the T1 and T2 encoder loops.
"""

import itertools
import queue
import threading
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import BOS, EOS, PAD, VOCAB_SIZE, encode_instruction
from tokenizer.tokenizer import decode_instruction
from datagen import (
    padding_mask,
    make_anchor_states, precompute_chunk,
    N_REGS, MAX_INPUT_SLOTS, MAX_OUTPUT_SLOTS, AUX_CE_IGNORE,
)
from emulator import Instruction
from .model import T1Compressor, T2Compressor, Decoder


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
    """Probe geometry. Distances are measured on the unit-normalized
    full encoded vector."""
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


def decoder_targets_fixed(batch, max_dec_len):
    """Fixed-shape (B, max_dec_len) teacher-forcing tensors over ALL rows —
    the compile-able analog of prepare_decoder_targets (whose shape varies
    with the valid-row count and per-batch max length, which would force a
    CUDA-graph re-record every batch).

    Valid rows: input=[BOS]+toks, target=toks+[EOS]. Invalid rows: target
    all-PAD (CE ignore_index=PAD → they contribute nothing). Position 0 is
    always BOS and never masked, so no row is fully-padded (an all-padding
    row NaNs the decoder's attention softmax). max_dec_len must be >=
    max token_len + 1; the caller derives it from the corpus-constant
    batch.tokens width (max_tokens + 1)."""
    B, max_tokens = batch.tokens.shape
    valid = batch.valid
    tl = batch.token_lens
    dec_in = np.full((B, max_dec_len), PAD, dtype=np.int64)
    dec_tgt = np.full((B, max_dec_len), PAD, dtype=np.int64)
    dec_in[:, 0] = BOS
    # input = [BOS] + toks (valid rows only; invalid rows stay [BOS, PAD…],
    # fully masked below). target = toks + [EOS] for valid rows.
    dec_in[valid, 1:1 + max_tokens] = batch.tokens[valid]
    dec_tgt[valid, :max_tokens] = batch.tokens[valid]
    rows = np.nonzero(valid)[0]
    dec_tgt[rows, tl[rows]] = EOS              # EOS at each row's token_len
    # padding mask: valid rows unmask token_len+1 positions ([BOS]+toks);
    # invalid rows unmask only position 0 (BOS) so no row is fully padded.
    lengths = np.where(valid, tl + 1, 1)
    dec_pad = np.arange(max_dec_len)[None, :] >= lengths[:, None]
    return dec_in, dec_tgt, dec_pad


# ===========================================================================
# train_encoder — magnitude-validity + rename-equivariant register binding
# + value-prediction on RVT batches (the T2 loss set, single-instruction)
# ===========================================================================

def resolve_device(spec):
    if spec == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return spec


def _is_cuda(device):
    return device == 'cuda' or getattr(device, 'type', None) == 'cuda'


def _h2d(arr, device, dtype=None):
    """Host->device copy for a batch array. On CUDA, route through pinned
    memory + non_blocking so the 11 per-step copies don't each block the
    CPU (they queue on the default stream and the CPU races ahead to launch
    the compiled step) — this recovers most of the per-step H2D gap and
    lifts GPU util ~81%->~92% with NO CUDA-graph hazard (the copy is on the
    default stream, before/outside the captured region; unlike a side-stream
    prefetcher, which collides with graph capture). dtype casts on the host
    so the device sees the final dtype directly."""
    t = torch.from_numpy(arr)
    if dtype is not None:
        t = t.to(dtype)
    if _is_cuda(device):
        return t.pin_memory().to(device, non_blocking=True)
    return t.to(device)


def _value_compress(x):
    """Sign-preserving double-log1p compression. Maps int32 range
    [-2^31, 2^31] roughly to [-3.0, 3.0] while preserving sign and
    monotonicity. Matches the project's residual-compression
    convention used by the canonical-output comparison."""
    return torch.sign(x) * torch.log1p(torch.log1p(x.abs()))


def value_predict_loss(vp_head, vecs, anchor_states,
                       src_reg_0, src_reg_1, row_outputs,
                       pair_valid, has_rd):
    """MSE on the predicted rd-value-per-anchor.

    The hypothesis: if the vector encodes operator-essence (the I/O
    function of the instruction), then a head reading (vec, input
    register values) — and *crucially* not register IDs — can predict
    the output value. The head sees inputs as values only, so it has
    to read "what operation" from the vector; there's no path to
    identify which register the inputs came from.

    Args:
      vp_head:       nn.Module, (B*A, d+2) → (B*A, 1)
      vecs:          (B, d) float
      anchor_states: (n_anchors, 32) int32 — register-file snapshots
      src_reg_0,1:   (B,) int64 — source-register IDs per row. 0 means
                     "no source" (x0, which is hardwired to 0 — so the
                     lookup naturally yields zero contribution).
      row_outputs:   (B, n_anchors) float — target rd value per anchor.
      pair_valid:    (B,) bool
      has_rd:        (B,) bool

    Returns scalar MSE on (pair_valid & has_rd) rows.
    """
    device = vecs.device
    B, d = vecs.shape
    n_anchors = anchor_states.shape[0]

    # Row mask as a multiplicative weight (not a boolean index / early
    # return): keeps the op graph static-shape and host-sync-free, so the
    # whole training step can be torch.compile'd / CUDA-graph captured. The
    # masked-out rows still compute (their per-row loss is finite — pred and
    # target are always finite) but contribute zero. clamp(min=1) makes the
    # all-masked batch return 0.0, matching the old early-return.
    mask = (pair_valid & has_rd).float()

    # Lookup per-row input values: (n_anchors, B) → (B, n_anchors).
    v_src0 = anchor_states[:, src_reg_0].T.float()
    v_src1 = anchor_states[:, src_reg_1].T.float()
    v_src0_c = _value_compress(v_src0)
    v_src1_c = _value_compress(v_src1)

    # Feature: per (row, anchor), [vec || v_src0 || v_src1].
    vec_exp = vecs.unsqueeze(1).expand(-1, n_anchors, -1)
    feat = torch.cat([
        vec_exp,
        v_src0_c.unsqueeze(-1),
        v_src1_c.unsqueeze(-1),
    ], dim=-1)    # (B, n_anchors, d + 2)

    pred = vp_head(feat).squeeze(-1)    # (B, n_anchors)

    target_raw = row_outputs.float()    # (B, n_anchors)
    target_c = _value_compress(target_raw)

    diff_sq = (pred - target_c) ** 2
    per_row = diff_sq.mean(dim=-1)    # (B,)
    return (per_row * mask).sum() / mask.sum().clamp(min=1)


def build_optim_sched(params, lr, n_steps, *, warmup_steps=0,
                      lr_min=1e-6, device='cuda'):
    """Adam (fused on cuda) + an optional-warmup→cosine LR schedule, shared
    by all trainers. Returns (opt, scheduler); scheduler is None when
    n_steps is falsy (LR held constant, e.g. unbounded streaming)."""
    opt = torch.optim.Adam(params, lr=lr, fused=(device == 'cuda'))
    if not n_steps:
        return opt, None
    cosine_steps = max(1, n_steps - warmup_steps)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cosine_steps, eta_min=lr_min)
    if warmup_steps > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[warmup_steps])
    else:
        scheduler = cosine
    return opt, scheduler


def _current_lr(scheduler, lr):
    return scheduler.get_last_lr()[0] if scheduler else lr


class StepTimer:
    """Wall-clock pacing for log lines. tick() returns (ms_per_step, eta_s)
    for the current step given the total and the log cadence."""

    def __init__(self):
        self._t_last = time.time()

    def tick(self, step, n_steps, log_every):
        now = time.time()
        ms_per_step = (now - self._t_last) / max(1, log_every) * 1000
        self._t_last = now
        eta = ((n_steps - step) * ms_per_step / 1000) if n_steps else 0
        return ms_per_step, eta


def capture_train_state(step, opt, scheduler):
    """Snapshot optimizer moments + scheduler position + step + RNG for an
    as-close-as-possible resume (the data stream from gen_batches is random,
    so resume continues optimizer/schedule, not the exact byte stream).
    Shared by all trainers; fires at log-boundary cadence, so the device->host
    copies it triggers (state_dict / get_rng_state) are off the hot path."""
    return {
        'step': step,
        'opt': opt.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'torch_rng': torch.get_rng_state(),
        'cuda_rng': (torch.cuda.get_rng_state()
                     if torch.cuda.is_available() else None),
    }


def restore_train_state(state, opt, scheduler):
    """Inverse of capture_train_state: reload optimizer/scheduler/RNG and
    return the step to resume from. Shared by all trainers."""
    opt.load_state_dict(state['opt'])
    if scheduler is not None and state.get('scheduler'):
        scheduler.load_state_dict(state['scheduler'])
    torch_rng = state.get('torch_rng')
    if torch_rng is not None:
        torch.set_rng_state(torch_rng.cpu() if hasattr(torch_rng, 'cpu')
                            else torch_rng)
    cuda_rng = state.get('cuda_rng')
    if cuda_rng is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(cuda_rng.cpu() if hasattr(cuda_rng, 'cpu')
                                 else cuda_rng)
    return int(state.get('step', 0))


class TrainLog:
    """Owns the training-loop log boundary, shared by all trainers.

    The discipline it centralizes: keep loss values as GPU tensors every
    step (no device->host sync), and ONLY at log boundaries materialize them
    to floats, build + append the record, print, and invoke on_log. Each
    float(tensor) drains the async CUDA pipeline; doing it every step
    serializes CPU/GPU (a dominant cost on WSL2/GPU-PV) and would break any
    CUDA-graph capture of the step. Concentrating every sync here keeps the
    hot path async and gives one audited home for the invariant.

    The logger is optimizer-agnostic: it never touches opt/scheduler. Resume
    state is a separate concern (capture_train_state) — the loop hands this
    logger an opaque dict via state_fn and the logger only forwards it to
    on_log. Per-trainer differences are injected, not forked:
      formatter(record, ms_per_step, eta, lr) -> str   (the human log line)
      extra_fn() -> dict                                (probes / counts;
                    called only at boundaries so its own GPU work/syncs are
                    likewise off the hot path)
    on_log is always called (model, losses, train_state) — train_state is
    None for trainers that don't checkpoint resume state.
    """

    def __init__(self, *, n_steps, log_every, lr, scheduler, formatter,
                 on_log=None):
        self.n_steps = n_steps
        self.log_every = log_every
        self.lr = lr
        self.scheduler = scheduler
        self.formatter = formatter
        self.on_log = on_log
        self.timer = StepTimer()
        self.losses = []

    def should_log(self, step):
        return step % self.log_every == 0 or step == 1

    def log(self, step, loss_tensors, *, model=None, extra_fn=None,
            state_fn=None):
        """At a log boundary: materialize loss_tensors (dict of name->scalar
        tensor) to floats, merge extra_fn(), record, print, on_log. Off
        boundary: a no-op (NO sync). Returns the record dict or None."""
        if not self.should_log(step):
            return None
        record = {'step': step}
        record.update({k: float(v) for k, v in loss_tensors.items()})
        if extra_fn is not None:
            record.update(extra_fn())
        ms_per_step, eta = self.timer.tick(step, self.n_steps, self.log_every)
        lr_now = _current_lr(self.scheduler, self.lr)
        print(self.formatter(record, ms_per_step, eta, lr_now))
        self.losses.append(record)
        if self.on_log is not None:
            train_state = state_fn() if state_fn is not None else None
            self.on_log(model, self.losses, train_state)
        return record


def _peek(it):
    """Pull the first item so corpus-constant shapes/flags can be read, then
    return (first, chained_iter) with the item put back. (first, iter) is
    (None, it) for an empty stream. Lets trainers bake batch-derived constants
    into the compiled graph without consuming the first batch."""
    it = iter(it)
    try:
        first = next(it)
    except StopIteration:
        return None, it
    return first, itertools.chain([first], it)


def run_train_loop(batch_source, *, model, opt, scheduler, log, device,
                   prep_fn, fwd_loss_fn, extra_fn=None, clip=1.0,
                   compile_step=True, capture_state=False, start_step=0):
    """The shared compiled training loop for all trainers (T1/T2/decoder).

    Owns the universal mechanics: process/device setup (TF32 matmul + disabling
    the fused-attention fast path that the compiled step can't capture),
    torch.compile (CUDA-graph, reduce-overhead) of the forward+loss+backward,
    the per-step backward / grad-clip / opt / scheduler, and logging via `log`
    (TrainLog) — including the log-boundary sync discipline and optional
    resume-state capture. The two genuinely trainer-specific pieces are
    injected:

      prep_fn(item) -> tuple of GPU inputs for fwd_loss_fn, or None to skip
          the item (empty batch). Does the H2D copies (via _h2d) plus any
          eager, dynamic-shape precompute that must stay OUTSIDE the captured
          graph (e.g. T2's frozen-T1 per-instruction encode + chunk assembly,
          whose instruction count varies per batch). Every tensor it returns
          must be fixed-shape so the graph records once.
      fwd_loss_fn(*inputs) -> (total, metrics, aux)
          total:   scalar loss to backward.
          metrics: dict[str, scalar tensor] logged at each boundary.
          aux:     tensors a boundary-only extra_fn needs (or None).
          The torch.compile target — keep it fixed-shape + host-sync-free.
      extra_fn(item, aux) -> dict  boundary-only extra fields (probes, chunk
          counts, decoder accuracy), or None.

    `model` is the trainable module (its parameters are grad-clipped). Returns
    log.losses. start_step seeds the step counter (resume)."""
    if _is_cuda(device):
        # TF32 tensor cores for fp32 matmul (~2-4x, ~1 bit precision). Set
        # here so every trainer gets it uniformly (it had drifted to only
        # T2/decoder, leaving T1 on slow fp32 matmuls).
        torch.set_float32_matmul_precision('high')
    # Disable the fused attention fast path (_transformer_encoder_layer_fwd /
    # nested-tensor) — not torch.compile-able; it shatters the captured graph.
    # Global + numerically transparent; set once for all trainers.
    torch.backends.mha.set_fastpath_enabled(False)
    step_fn = (torch.compile(fwd_loss_fn, mode='reduce-overhead')
               if compile_step else fwd_loss_fn)
    step = start_step
    for item in batch_source:
        inputs = prep_fn(item)
        if inputs is None:
            continue
        total, metrics, aux = step_fn(*inputs)
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        if scheduler:
            scheduler.step()
        step += 1
        log.log(
            step, metrics, model=model,
            extra_fn=((lambda it=item, a=aux: extra_fn(it, a))
                      if extra_fn is not None else None),
            state_fn=((lambda s=step: capture_train_state(s, opt, scheduler))
                      if capture_state else None))
    return log.losses


def train_encoder(batch_iter, *,
                  d_model=128, n_heads=4, n_layers=2, d_out=128,
                  max_window=72,
                  lr=3e-4, n_steps=None, log_every=100, lr_min=1e-6,
                  valid_weight=0.1,
                  live_in_weight=0.1, live_out_weight=0.1,
                  pc_writes_weight=0.1,
                  in_slot_weight=0.1, out_slot_weight=0.1,
                  value_predict_weight=1.0,
                  anchor_seed=0, n_anchor_states=8,
                  device='auto', on_log=None, compile_step=True):
    """Train a T1 encoder on RVT batches.

    T1 is the single-instruction special case of T2 and uses the same
    rename-equivariant register-binding supervision (binding_losses):
    live_in/out + pc_writes BCE, in/out-slot ListMLE, plus value
    prediction. There is no equivalence-collapse loss — equivalence is
    an emergent property of the consistent per-instruction targets, not a
    separate objective that would fight register identity.

    valid_weight:        weight on (||v_c|| - (1 if valid else 0))**2.
    live_in/out_weight:  BCE on the behavioral input/output register masks.
    pc_writes_weight:    BCE on the explicit-PC-write flag.
    in/out_slot_weight:  ListMLE on the input/output slot orderings.
    value_predict_weight: MSE on per-anchor rd value prediction.
    on_log:              optional callable(model, losses, train_state=None)
                         called after each log point (uniform across
                         trainers; the encoder passes no train_state).
    compile_step:        torch.compile the forward+loss with CUDA graphs
                         (mode='reduce-overhead'). The whole forward+loss+
                         backward is captured as one graph (~9 launches vs
                         ~485 eager), saturating the GPU (~99% vs ~50%) and
                         ~2.5x faster. Requires the single-instruction T1
                         regime (row_outputs present); the loss path is
                         sync-free + static-shape, so it captures cleanly.
                         Numerics differ slightly from eager (fusion / TF32
                         reduction order) — set False for a bit-faithful
                         eager run.

    Returns (encoder, losses). losses[i] dict has keys: step, total,
    valid, live_in, live_out, pc_writes, in_slot, out_slot, value_pred,
    plus probe_* keys at each log point.
    """
    device = resolve_device(device)

    encoder = T1Compressor(
        VOCAB_SIZE, d_model, n_heads, n_layers, d_out,
        max_window=max_window,
    ).to(device)
    opt, scheduler = build_optim_sched(
        encoder.parameters(), lr, n_steps, lr_min=lr_min, device=device)

    probe_tok, probe_pad = _probe_tensor(device)

    # Anchor states for value-prediction loss. Reconstructed from
    # anchor_seed + n_anchor_states must match what gen_batches used —
    # otherwise row_outputs targets don't correspond to what executing
    # an instruction on these states would produce. Only used when
    # value_predict_weight > 0; built unconditionally because it's tiny.
    anchor_np = make_anchor_states(n_anchor_states, anchor_seed)
    anchor_states_t = torch.from_numpy(anchor_np).to(device)    # (A, 32) int32

    def _fmt(r, ms, eta, lr_now):
        return (
            f'step {r["step"]:>5d}  loss {r["total"]:.4f} '
            f'(val {r["valid"]:.4f}'
            f' li {r["live_in"]:.3f} lo {r["live_out"]:.3f}'
            f' pc {r["pc_writes"]:.3f}'
            f' in {r["in_slot"]:.3f} out {r["out_slot"]:.3f}'
            f' vp {r["value_pred"]:.4f})  '
            f'lr {lr_now:.1e}  {ms:.0f}ms/step  '
            f'eta {timedelta(seconds=int(eta))}  '
            f'probes[aa={r["probe_aa"]:.3f} canon={r["probe_canon"]:.3f} '
            f'ss={r["probe_ss"]:.3f} comm={r["probe_comm"]:.3f}]')

    log = TrainLog(n_steps=n_steps, log_every=log_every, lr=lr,
                   scheduler=scheduler, formatter=_fmt, on_log=on_log)

    # value-prediction needs the single-instruction row_outputs payload;
    # decide once from the first batch (the corpus rule is fixed for a run,
    # so row_outputs presence is constant) — a Python constant baked into the
    # compiled graph.
    first_batch, batch_iter = _peek(batch_iter)
    if first_batch is None:
        return encoder, log.losses
    vp_active = value_predict_weight > 0 and first_batch.row_outputs.shape[0] > 0

    encoder.train()
    _dummy = torch.zeros(1, device=device)

    # The fixed-shape forward+loss captured as one CUDA graph (the input
    # (B, max_tokens) is corpus-constant; in_k_eff/out_k_eff are Python ints
    # and value_predict masks with a multiply, so it's host-sync-free).
    def _fwd_loss(tok, pad, valid_f, li, lo, pc, ins, outs, ro, pv, hr,
                  ike, oke):
        vecs = encoder.encode(tok, pad)              # (B, d_out)
        valid_loss = ((vecs.norm(dim=-1) - valid_f) ** 2).mean()
        bl = binding_losses(
            encoder, vecs, live_in_t=li, live_out_t=lo, pc_writes_t=pc,
            in_slot_t=ins, out_slot_t=outs, in_k_eff=ike, out_k_eff=oke)
        if vp_active:
            vp = value_predict_loss(
                encoder.vp_head, vecs, anchor_states_t,
                ins[:, 0].clamp(min=0), ins[:, 1].clamp(min=0), ro, pv, hr)
        else:
            vp = vecs.new_zeros(())
        total = (valid_weight * valid_loss
                 + live_in_weight * bl['live_in']
                 + live_out_weight * bl['live_out']
                 + pc_writes_weight * bl['pc_writes']
                 + in_slot_weight * bl['in_slot']
                 + out_slot_weight * bl['out_slot']
                 + value_predict_weight * vp)
        metrics = {'total': total, 'valid': valid_loss,
                   'live_in': bl['live_in'], 'live_out': bl['live_out'],
                   'pc_writes': bl['pc_writes'], 'in_slot': bl['in_slot'],
                   'out_slot': bl['out_slot'], 'value_pred': vp}
        return total, metrics, None

    def _prep(batch):
        if vp_active:
            ro = _h2d(batch.row_outputs, device)
            pv = _h2d(batch.pair_valid, device)
            hr = _h2d(batch.row_has_rd, device)
        else:
            ro = pv = hr = _dummy
        return (_h2d(batch.tokens, device, dtype=torch.long),
                _h2d(padding_mask(batch), device),
                _h2d(batch.valid, device, dtype=torch.float32),
                _h2d(batch.live_in_mask, device, dtype=torch.float32),
                _h2d(batch.live_out_mask, device, dtype=torch.float32),
                _h2d(batch.pc_writes, device, dtype=torch.float32),
                _h2d(batch.in_slot_regs, device, dtype=torch.long),
                _h2d(batch.out_slot_regs, device, dtype=torch.long),
                ro, pv, hr,
                _slot_k_eff(batch.in_slot_regs), _slot_k_eff(batch.out_slot_regs))

    def _extra(batch, aux):
        return {f'probe_{k}': v
                for k, v in _probe_distances(encoder, probe_tok, probe_pad).items()}

    losses = run_train_loop(
        batch_iter, model=encoder, opt=opt, scheduler=scheduler, log=log,
        device=device,
        prep_fn=_prep, fwd_loss_fn=_fwd_loss, extra_fn=_extra,
        compile_step=compile_step)
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
        instr_tokens: (N_instr_total, MAX_INSTR_TOKENS) int8 — padded
                      with PAD past each instruction's actual length.
                      Caller must .long() before nn.Embedding.
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
        empty = np.zeros((0, _MAX_INSTR_TOKENS), dtype=np.int8)
        return (
            empty,
            np.ones((0, _MAX_INSTR_TOKENS), dtype=bool),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            n_per_chunk,
        )

    instr_tokens = np.full((total, _MAX_INSTR_TOKENS), PAD, dtype=np.int8)
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


def _decode_chunk_instructions(batch, row_idx):
    """Decode all instructions in chunk row_idx back to Instruction
    objects. Returns the list, or None if any decode fails or the
    chunk is empty/invalid. Caller masks via batch.valid."""
    if not batch.valid[row_idx]:
        return None
    instr_lens = batch.instr_lens[row_idx]
    tokens = batch.tokens[row_idx]
    instrs = []
    offset = 0
    for j in range(len(instr_lens)):
        L = int(instr_lens[j])
        if L == 0:
            continue
        toks = tokens[offset:offset + L].tolist()
        try:
            instr, _ = decode_instruction(toks, 0)
        except Exception:
            return None
        instrs.append(instr)
        offset += L
    return instrs if instrs else None


def _assemble_chunk_seq(t1_vecs, split_outputs, n_chunks, max_n_instrs, device):
    """Scatter per-instruction T1 vectors into padded per-chunk sequences
    and build the key-padding mask. Shared by t2_chunk_forward and the
    gradient-subspace diagnostic (which differ only in how they run the
    T1/T2 encodes around it)."""
    _, _, chunk_idx, slot_idx, n_per_chunk = split_outputs
    chunk_t1 = torch.zeros(
        n_chunks, max_n_instrs, t1_vecs.shape[-1], device=device)
    chunk_t1[torch.from_numpy(chunk_idx).to(device),
             torch.from_numpy(slot_idx).to(device)] = t1_vecs
    n_per_chunk_t = torch.from_numpy(n_per_chunk).to(device)
    chunk_pad = (torch.arange(max_n_instrs, device=device)[None, :]
                 >= n_per_chunk_t[:, None])
    # Unmask position 0: an all-padded row (invalid window) would NaN the
    # attention softmax and 0*NaN the shared pool_query gradient.
    chunk_pad[:, 0] = False
    return chunk_t1, chunk_pad


def t2_chunk_forward(t1, t2, split_outputs, n_chunks, max_n_instrs, device):
    """Per-instruction T1 encode → per-chunk T1 sequence → T2 encode — the
    forward shared by the T2 trainer and eval. T1 runs frozen (no_grad); T2
    runs in the caller's grad context (the caller sets t2.train()/eval() and
    wraps in torch.no_grad() as appropriate). split_outputs is the tuple from
    _split_to_per_instruction. Returns t2_out (n_chunks, d_out), or None for
    an empty batch."""
    instr_tokens, instr_pad, *_ = split_outputs
    if instr_tokens.shape[0] == 0:
        return None
    it = torch.from_numpy(instr_tokens).to(device).long()
    ip = torch.from_numpy(instr_pad).to(device)
    with torch.no_grad():
        t1_vecs = t1.encode(it, ip)
    chunk_t1, chunk_pad = _assemble_chunk_seq(
        t1_vecs, split_outputs, n_chunks, max_n_instrs, device)
    return t2.encode(chunk_t1, chunk_pad)


def t2_value_predict_loss(t2, vecs, anchor_states_t,
                          in_slot_regs_t, out_slot_regs_t,
                          out_regs_t, mask_t, return_count=False):
    """Per-anchor, per-out-slot value-prediction MSE for T2.

    The head reads (vec, anchor input-slot values) and predicts
    per-anchor per-slot output values. The supervision is "for slot k
    bound to register r in this chunk, the value at slot k should be
    register r's actual value after executing the chunk on this
    anchor."

    Args:
      t2:               T2Compressor (uses t2.vp_head)
      vecs:             (B, d) — T2 output vector
      anchor_states_t:  (A, 32) int32 — register file per anchor (pre-exec)
      in_slot_regs_t:   (B, MAX_INPUT_SLOTS) int64 — input slot register IDs
                        (AUX_CE_IGNORE for unused slots)
      out_slot_regs_t:  (B, MAX_OUTPUT_SLOTS) int64 — output slot register IDs
                        (AUX_CE_IGNORE for unused slots)
      out_regs_t:       (B, A, 32) int32 — register file per anchor AFTER
                        executing each chunk
      mask_t:           (B,) bool — rows to include (decoded successfully)

    Returns scalar MSE over (row, output-slot, anchor) triplets where
    out_slot_regs is not IGNORE.
    """
    n_anchors = anchor_states_t.shape[0]

    # Build per-anchor input values: clamp IGNORE→0 (x0=0 in all anchors)
    # for the lookup; x0's value is 0 by convention so unused slots
    # contribute zero, which matches "no input" naturally.
    in_lookup = in_slot_regs_t.clamp(min=0).long()    # (B, MAX_IN)
    # anchor_states[:, in_lookup] -> (A, B, MAX_IN); permute to (B, A, MAX_IN)
    in_vals = anchor_states_t[:, in_lookup].permute(1, 0, 2).float()

    in_vals_c = _value_compress(in_vals)

    # Feature: (B, A, d + MAX_IN)
    vec_exp = vecs.unsqueeze(1).expand(-1, n_anchors, -1)
    feat = torch.cat([vec_exp, in_vals_c], dim=-1)

    # Predict: (B, A, MAX_OUT).
    pred = t2.vp_head(feat)

    # Target: target[b, a, k] = out_regs[b, a, out_slot_regs[b, k]].
    # Clamp IGNORE→0 for lookup; mask out via out_slot_valid below.
    out_lookup = out_slot_regs_t.clamp(min=0).long()    # (B, MAX_OUT)
    out_lookup_expanded = out_lookup.unsqueeze(1).expand(-1, n_anchors, -1)
    target = out_regs_t.gather(dim=-1, index=out_lookup_expanded).float()
    target_c = _value_compress(target)

    out_valid = (out_slot_regs_t != AUX_CE_IGNORE)    # (B, MAX_OUT)
    pair_mask = out_valid & mask_t.unsqueeze(-1)        # (B, MAX_OUT)

    diff_sq = (pred - target_c) ** 2                    # (B, A, MAX_OUT)
    per_slot = diff_sq.mean(dim=1)                      # (B, MAX_OUT)
    # Mask-multiply (not boolean index / early return) so the op stays
    # static-shape and host-sync-free — capturable in the full-step graph.
    # clamp(min=1) returns 0.0 for an all-masked batch, matching the old
    # early-return.
    pm = pair_mask.float()
    mse = (per_slot * pm).sum() / pm.sum().clamp(min=1)
    return (mse, int(pair_mask.sum())) if return_count else mse


def _slot_k_eff(slot_regs_np):
    """Host-side effective slot width: last column (across the batch) that
    any row actually fills, +1. Computed from the numpy batch BEFORE the
    H2D copy, so it costs no GPU sync — the ListMLE loop bound becomes a
    plain Python int, keeping the training step graph-capturable (no
    device->host sync, no data-dependent GPU control flow). Returns 0 when
    no row fills any slot (all-IGNORE — invalid/mem-op batch)."""
    nz = np.nonzero((slot_regs_np != AUX_CE_IGNORE).any(axis=0))[0]
    return int(nz[-1]) + 1 if nz.size else 0


def _listmle_loss(scores, slot_regs, K_max, active_f, k_eff=None):
    """Plackett-Luce / ListMLE log-likelihood loss on per-register scores.

    scores:    (B, n_regs) — predicted per-register scalar scores.
    slot_regs: (B, K_max)  — GT slot ordering; AUX_CE_IGNORE for unfilled.
    K_max:     length of slot_regs second dim.
    active_f:  (B,) float — chunk validity mask.
    k_eff:     optional host int — iterate only the first k_eff slot columns
               (trailing all-IGNORE columns are a no-op on the loss, so this
               is exact). Training loops pass it host-computed via
               _slot_k_eff (sync-free). If None, it's derived on the GPU
               here — correct but forces a host sync; fine for tests /
               offline diagnostics, not the hot path.

    For each chunk b and each filled slot k:
        loss_k = -log softmax(scores_b \\ prev_chosen)[seq_b[k]]
    where "\\ prev_chosen" means previously-chosen regs are masked to -inf
    so they cannot be re-drawn. Per-chunk loss = sum_k loss_k / n_filled.
    Final loss = mean over active chunks with at least one filled slot.

    Properties:
      - Direct per-position supervision (each slot is its own categorical
        with the right answer at position k, identical structure to
        per-slot CE).
      - Distinctness baked in: the mask prevents the same reg from being
        drawn twice.
      - Info density ~K * log(n_regs) bits/chunk — same as per-slot CE,
        much higher than pairwise ranking's K^2 margin signal.
      - Decode (argsort) is unchanged.
    """
    B, N = scores.shape
    device = scores.device
    # Truncate to the last actually-filled slot column. A column where every
    # row is AUX_CE_IGNORE has `valid` all-False, so it adds nothing to
    # chunk_nll / n_filled / mask — iterating it is a no-op on the loss.
    # K_max is the padded width (32 in / 16 out), but a single-instruction
    # T1 chunk fills only ~2 in / 1 out, so iterating the full width launches
    # ~10 tiny kernels per empty column and dominates the train step.
    if k_eff is None:
        nz = (slot_regs != AUX_CE_IGNORE).any(dim=0).nonzero()
        k_eff = int(nz[-1].item()) + 1 if nz.numel() else 0
    mask = torch.zeros_like(scores, dtype=torch.bool)
    chunk_nll = torch.zeros(B, device=device)
    n_filled = torch.zeros(B, device=device)
    for k in range(k_eff):
        target = slot_regs[:, k]
        valid = (target != AUX_CE_IGNORE)
        masked_scores = scores.masked_fill(mask, float('-inf'))
        log_p = F.log_softmax(masked_scores, dim=-1)
        safe_target = target.clamp(min=0)
        step_nll = -log_p.gather(1, safe_target.unsqueeze(1)).squeeze(1)
        # Use torch.where, NOT step_nll * valid.float(): when k exceeds
        # the chunk's filled-slot count, safe_target collapses to 0 and
        # log_p[0] may be -inf (if reg 0 has been masked at an earlier
        # step), giving step_nll == +inf. Multiplying +inf * 0 yields
        # nan; torch.where avoids touching step_nll on invalid rows.
        chunk_nll = chunk_nll + torch.where(
            valid, step_nll, torch.zeros_like(step_nll))
        n_filled = n_filled + valid.float()
        update = torch.zeros_like(mask).scatter(
            1, safe_target.unsqueeze(1),
            torch.ones_like(safe_target.unsqueeze(1), dtype=torch.bool))
        mask = mask | (update & valid.unsqueeze(1))
    chunk_loss = chunk_nll / n_filled.clamp(min=1)
    chunk_has_slots = (n_filled > 0).float()
    weight = active_f * chunk_has_slots
    return (chunk_loss * weight).sum() / weight.sum().clamp(min=1)


def binding_losses(model, vec, *, live_in_t, live_out_t, pc_writes_t,
                   in_slot_t, out_slot_t, in_k_eff=None, out_k_eff=None):
    """Rename-equivariant register-binding losses, shared by T1 and T2.

    T1 is the n_instrs=1 special case of T2: both read these targets from
    `precompute_chunk` and supervise the same heads on the normalized
    binding direction. BCE on live_in/out masks + pc_writes; ListMLE
    (Plackett-Luce) on the in/out slot orderings via per-register score
    heads. Rows with all-zero masks (invalid / mem-op) are excluded.

    Every target derives from one dataflow analysis, so these losses are
    mutually consistent — no objective collapses what another must keep
    distinct (this is what the old equivalence_loss vs src-register CE
    could not jointly satisfy).

    `model` must expose live_in_head / live_out_head / pc_writes_head /
    in_score_head / out_score_head and n_regs / max_input_slots /
    max_output_slots. Returns a dict of the five scalar losses plus
    `active_f` (the per-row validity mask, for the caller's reuse).

    in_k_eff / out_k_eff: optional host-computed effective slot widths
    (see _slot_k_eff) forwarded to the ListMLE calls so the slot losses
    iterate only filled columns without a GPU sync. Omit them and the
    widths are derived on-device (a host sync — fine off the hot path).
    """
    any_in = live_in_t.any(dim=-1)
    any_out = live_out_t.any(dim=-1)
    active_f = (any_in | any_out).float()
    n_active = active_f.sum().clamp(min=1)
    n_regs = model.n_regs

    binding_dir = F.normalize(vec, dim=-1)
    li_logits = model.live_in_head(binding_dir)
    lo_logits = model.live_out_head(binding_dir)
    pc_logits = model.pc_writes_head(binding_dir).squeeze(-1)

    li_per = F.binary_cross_entropy_with_logits(
        li_logits, live_in_t, reduction='none')
    li_loss = (li_per * active_f.unsqueeze(-1)).sum() / (n_active * n_regs)
    lo_per = F.binary_cross_entropy_with_logits(
        lo_logits, live_out_t, reduction='none')
    lo_loss = (lo_per * active_f.unsqueeze(-1)).sum() / (n_active * n_regs)
    pc_per = F.binary_cross_entropy_with_logits(
        pc_logits, pc_writes_t, reduction='none')
    pc_loss = (pc_per * active_f).sum() / n_active

    in_scores = model.in_score_head(binding_dir)
    out_scores = model.out_score_head(binding_dir)
    in_slot_loss = _listmle_loss(
        in_scores, in_slot_t, model.max_input_slots, active_f, k_eff=in_k_eff)
    out_slot_loss = _listmle_loss(
        out_scores, out_slot_t, model.max_output_slots, active_f, k_eff=out_k_eff)

    return {
        'live_in': li_loss, 'live_out': lo_loss, 'pc_writes': pc_loss,
        'in_slot': in_slot_loss, 'out_slot': out_slot_loss,
        'active_f': active_f,
    }


def _compute_chunk_out_regs(batch, anchor_states_np):
    """Per row of the batch, run precompute_chunk to get the post-
    execution full register file per anchor. Returns:
      out_regs: (B, n_anchors, 32) int32
      mask:     (B,) bool — True for rows we got valid out_regs from
    Rows that fail (decode error, memory ops, etc.) get zero out_regs."""
    B = batch.tokens.shape[0]
    n_anchors = anchor_states_np.shape[0]
    out = np.zeros((B, n_anchors, 32), dtype=np.int32)
    mask = np.zeros(B, dtype=bool)
    for b in range(B):
        instrs = _decode_chunk_instructions(batch, b)
        if instrs is None:
            continue
        try:
            pre = precompute_chunk(instrs, anchor_states_np)
        except NotImplementedError:
            continue
        except Exception:
            continue
        out[b] = pre.out_regs
        mask[b] = True
    return out, mask


_GRAD_DIM_LOSSES = ['vp', 'in_slot', 'out_slot', 'live_in', 'live_out']


def _pr_from_cov(C):
    """Participation ratio (effective dimensionality) of a covariance/
    second-moment matrix: (trace)^2 / ||.||_F^2 = (sum eig)^2 / sum eig^2."""
    tr = torch.diagonal(C).sum()
    fro2 = (C * C).sum().clamp(min=1e-30)
    return float((tr * tr / fro2).item())


def _cov_cosine(A, B):
    """Matrix-cosine between two PSD matrices (0=orthogonal, 1=same)."""
    inner = (A * B).sum()
    na = torch.sqrt((A * A).sum())
    nb = torch.sqrt((B * B).sum())
    return float((inner / (na * nb).clamp(min=1e-30)).item())


def _pr_and_overlaps(grads, center=True):
    """grads: name -> (B, d) per-sample gradient on `pooled`.

    Reports BOTH:
      - uncentered second-moment C = g^T g: includes the mean (DC) gradient
        direction the loss pushes every sample along. Dominated by that
        common direction → small PR when gradients align.
      - centered covariance Cc = (g-mean)^T (g-mean): the dimensionality of
        how the gradient VARIES across samples (the common push removed),
        the analog of mean-centered PCA on representations.

    Returns dims = {name: {'unc': PR_uncentered, 'cen': PR_centered}} and
    overlaps (matrix-cosine) computed on the chosen basis (centered by
    default)."""
    C = {n: g.t() @ g for n, g in grads.items()}
    Cc = {n: (g - g.mean(0, keepdim=True)).t() @ (g - g.mean(0, keepdim=True))
          for n, g in grads.items()}
    dims = {n: {'unc': _pr_from_cov(C[n]), 'cen': _pr_from_cov(Cc[n])}
            for n in grads}
    basis = Cc if center else C
    overlaps = {}
    names = list(grads.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            overlaps[f'{a}|{b}'] = _cov_cosine(basis[a], basis[b])
    return dims, overlaps


def loss_grad_dims(t2, t1, batch, anchor_states_t, device,
                   out_regs_np=None, vp_mask_np=None):
    """Per-loss gradient-subspace dims + overlaps on the shared `pooled`
    vector (the score-heads + vp T2). Backprops each loss separately to
    pooled and summarizes via _pr_and_overlaps. Dirties param .grad —
    caller zeroes afterward. Pass precomputed out_regs to skip the anchor
    execution when measuring a fixed batch repeatedly."""
    split_outputs = _split_to_per_instruction(batch)
    instr_tokens, instr_pad, *_ = split_outputs
    if instr_tokens.shape[0] == 0:
        return None, None
    it = torch.from_numpy(instr_tokens).to(device).long()
    ip = torch.from_numpy(instr_pad).to(device)
    with torch.no_grad():
        t1_vecs = t1.encode(it, ip)
    B, max_n = batch.instr_lens.shape
    chunk_t1, pad = _assemble_chunk_seq(
        t1_vecs, split_outputs, B, max_n, device)
    # return_pooled exposes the pre-projection vector that each loss term is
    # backprop'd to below.
    t2_out, pooled = t2.encode(chunk_t1, pad, return_pooled=True)
    pooled.retain_grad()
    binding_dir = F.normalize(t2_out, dim=-1)

    in_slot_t = torch.from_numpy(
        batch.in_slot_regs.astype(np.int64)).to(device)
    out_slot_t = torch.from_numpy(
        batch.out_slot_regs.astype(np.int64)).to(device)
    live_in_t = torch.from_numpy(batch.live_in_mask).to(device).float()
    live_out_t = torch.from_numpy(batch.live_out_mask).to(device).float()
    active = (live_in_t.any(-1) | live_out_t.any(-1)).float()
    n_active = active.sum().clamp(min=1)
    nreg = t2.n_regs

    if out_regs_np is None:
        out_regs_np, vp_mask_np = _compute_chunk_out_regs(
            batch, anchor_states_t.cpu().numpy())
    out_regs_t = torch.from_numpy(out_regs_np).to(device)
    vp_mask_t = torch.from_numpy(vp_mask_np).to(device)

    losses = {
        'vp': t2_value_predict_loss(
            t2, t2_out, anchor_states_t, in_slot_t, out_slot_t,
            out_regs_t, vp_mask_t),
        'in_slot': _listmle_loss(
            t2.in_score_head(binding_dir), in_slot_t,
            t2.max_input_slots, active),
        'out_slot': _listmle_loss(
            t2.out_score_head(binding_dir), out_slot_t,
            t2.max_output_slots, active),
    }
    li = F.binary_cross_entropy_with_logits(
        t2.live_in_head(binding_dir), live_in_t, reduction='none')
    losses['live_in'] = (li * active.unsqueeze(-1)).sum() / (n_active * nreg)
    lo = F.binary_cross_entropy_with_logits(
        t2.live_out_head(binding_dir), live_out_t, reduction='none')
    losses['live_out'] = (lo * active.unsqueeze(-1)).sum() / (n_active * nreg)

    grads = {}
    for name in _GRAD_DIM_LOSSES:
        if pooled.grad is not None:
            pooled.grad = None
        losses[name].backward(retain_graph=True)
        grads[name] = pooled.grad.detach().clone()
    return _pr_and_overlaps(grads)


def _mp_worker_main(in_q, out_q, anchor_states_np, compute_out_regs):
    """Worker-process entry point. Re-imports needed modules (we're under
    spawn-context, so a fresh interpreter). Reads Batches from in_q,
    runs CPU-side prep (split + per-chunk analysis loop), sends results on
    out_q. EOF signaled by None.
    """
    # Imports must happen inside the worker (spawn context = fresh
    # interpreter, doesn't inherit parent's already-loaded modules).
    from compressor.train import (
        _split_to_per_instruction, _compute_chunk_out_regs,
    )
    try:
        while True:
            batch = in_q.get()
            if batch is None:
                break
            split = _split_to_per_instruction(batch)
            if compute_out_regs:
                out_regs_np, vp_mask_np = _compute_chunk_out_regs(
                    batch, anchor_states_np)
            else:
                out_regs_np = vp_mask_np = None
            out_q.put((batch, split, out_regs_np, vp_mask_np))
    finally:
        out_q.put(None)


class TrainBatchPrefetcher:
    """Multiprocessing prefetcher: spawn a worker *process* for CPU-side
    prep so it runs in true parallel with the main loop's GPU work (no
    GIL contention).

    Architecture:
      - Worker process (spawn context, fresh Python interpreter):
          pulls Batches from input queue, runs _split_to_per_instruction
          and _compute_chunk_out_regs, pushes results to output queue.
      - Reader thread (in main process, daemon):
          pulls Batches from batch_iter (the RVT stream), pushes to
          input queue. Decoupling main loop from stdin read.
      - Main loop: consumes prepped tuples from output queue.

    Why processes instead of threads: the per-chunk loop inside
    _compute_chunk_out_regs is pure-Python (decoding tokens, SSA
    analysis, exception handling) and entirely GIL-bound.
    Threading caused net regression (worker fought main for GIL).
    Process boundary eliminates GIL contention entirely; cost is
    pickling each batch across the queue (numpy arrays in dataclass,
    ~1-2ms per batch).

    Spawn context (not fork) avoids any CUDA-state contamination since
    the worker is a fresh interpreter.
    """

    _EOF = None  # sentinel

    def __init__(self, batch_iter, anchor_states_np, compute_out_regs,
                 maxsize=4):
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        self._in_q = ctx.Queue(maxsize=maxsize)
        self._out_q = ctx.Queue(maxsize=maxsize)
        self._worker = ctx.Process(
            target=_mp_worker_main,
            args=(self._in_q, self._out_q, anchor_states_np, compute_out_regs),
            daemon=True,
        )
        self._worker.start()

        # Reader thread feeds the worker's input queue from the batch
        # iterator. Runs in the main process; minimal Python work
        # (just reading from stdin and queue.put) so GIL contention
        # with the main GPU loop is small.
        self._reader_thread = threading.Thread(
            target=self._read_loop, args=(batch_iter,), daemon=True)
        self._reader_thread.start()

    def _read_loop(self, batch_iter):
        try:
            for batch in batch_iter:
                self._in_q.put(batch)
        finally:
            self._in_q.put(self._EOF)

    def __iter__(self):
        return self

    def __next__(self):
        item = self._out_q.get()
        if item is self._EOF:
            self._worker.join(timeout=10)
            raise StopIteration
        return item


def train_t2_encoder(batch_iter, t1_encoder, *,
                     d_model=256, n_heads=4, n_layers=2, d_out=256,
                     max_chunk_len=32,
                     lr=3e-4, n_steps=None, log_every=100, lr_min=1e-6,
                     warmup_steps=0,
                     valid_weight=0.0,
                     live_in_weight=0.1, live_out_weight=0.1,
                     pc_writes_weight=0.1,
                     in_slot_weight=0.1, out_slot_weight=0.1,
                     value_predict_weight=0.0,
                     value_predict_every=1,
                     anchor_seed=0, n_anchor_states=8,
                     t2_checkpoint=None,
                     device='auto',
                     on_log=None, compile_step=True):
    """Train a T2 encoder on top of a frozen T1 encoder.

    For each RVT batch:
      1. Split flat chunk tokens into per-instruction segments via
         batch.instr_lens.
      2. Run frozen T1.encode on every instruction in the batch
         (one big GPU call across all chunks).
      3. Reshape T1 outputs back into per-chunk sequences with padding.
      4. Run T2.encode over the sequences → one T2 vector per chunk.
      5. Optional magnitude-validity loss (defaults off — corpora
         generated without --inject-invalid have no negative class).
      6. Aux register-identity supervision (BCE on live_in/out masks
         and pc_writes; CE on slot-positional register predictions).

    Aux heads project from F.normalize(t2_vecs) so register-identity
    info is encoded into direction, not magnitude.

    on_log: optional callable(model, losses, train_state) called after
            each log point. Use it to checkpoint mid-run (train_state
            carries optimizer/scheduler/RNG for --resume).

    Returns (t2, losses).
    """
    device = resolve_device(device)
    # TF32 + fused-attention-fastpath disable are handled in run_train_loop
    # (shared across trainers).
    t1_encoder.eval()
    for p in t1_encoder.parameters():
        p.requires_grad = False
    t1_encoder = t1_encoder.to(device)

    t2 = T2Compressor(
        d_t1=t1_encoder.d_out, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_out=d_out, max_chunk_len=max_chunk_len,
    ).to(device)
    # Track resumed training state (optimizer/scheduler/step/rng). When
    # t2_checkpoint is a directory containing train_state.pt, we restore
    # everything for a continue-from-where-we-left-off resume. When it's
    # a .pt file, we only load model weights (legacy "polish" resume —
    # Adam moments reset, schedule restarts).
    resume_train_state = None
    if t2_checkpoint is not None:
        ckpt_path = Path(t2_checkpoint)
        if ckpt_path.is_dir():
            model_path = ckpt_path / 't2.pt'
            train_state_path = ckpt_path / 'train_state.pt'
            t2.load_state_dict(load_checkpoint(model_path, device),
                               strict=False)
            if train_state_path.exists():
                resume_train_state = torch.load(
                    train_state_path, map_location=device,
                    weights_only=False)
        else:
            t2.load_state_dict(load_checkpoint(ckpt_path, device),
                               strict=False)
    opt, scheduler = build_optim_sched(
        t2.parameters(), lr, n_steps, warmup_steps=warmup_steps,
        lr_min=lr_min, device=device)

    # Anchor states for value-prediction. Must match gen-time settings.
    anchor_np = make_anchor_states(n_anchor_states, anchor_seed)
    anchor_states_t = torch.from_numpy(anchor_np).to(device)

    def _fmt(r, ms, eta, lr_now):
        return (
            f'step {r["step"]:>5d}  loss {r["total"]:.4f} '
            f'(val {r["valid"]:.4f} '
            f'li {r["live_in"]:.3f} lo {r["live_out"]:.3f} '
            f'pc {r["pc_writes"]:.3f} '
            f'is {r["in_slot"]:.3f} os {r["out_slot"]:.3f} '
            f'vp {r["value_pred"]:.4f}) '
            f'B={r["n_chunks"]} I={r["n_instrs"]}  '
            f'lr {lr_now:.1e}  {ms:.0f}ms/step  '
            f'eta {timedelta(seconds=int(eta))}')

    log = TrainLog(n_steps=n_steps, log_every=log_every, lr=lr,
                   scheduler=scheduler, formatter=_fmt, on_log=on_log)
    step = 0

    # Restore optimizer/scheduler/step/RNG from a previous checkpoint dir
    # (set above in the t2_checkpoint handling block), so resume continues
    # where the prior run left off.
    if resume_train_state is not None:
        step = restore_train_state(resume_train_state, opt, scheduler)
        print(f'Resumed from step {step}')

    # Trains until stdin EOF — pipeline upstream (cat / mux_batches /
    # multinode_gen) controls total batch count. n_steps shapes the cosine
    # scheduler and ETA display; if the pipeline runs longer the LR stays at
    # lr_min, if shorter the schedule is truncated.
    #
    # Wrap batch_iter in a prefetcher so CPU-side prep (numpy splitting and
    # the per-chunk loop for value-predict targets) overlaps
    # with GPU work on the prior batch.
    batch_source = TrainBatchPrefetcher(
        batch_iter, anchor_np,
        compute_out_regs=(value_predict_weight > 0))

    # Peek the first item: batch shape (n_chunks, max_n_instrs) and vp/valid
    # activity are corpus-constant, so bake them as Python constants into the
    # compiled graph.
    first, batch_source = _peek(batch_source)
    if first is None:
        return t2, log.losses
    N_CHUNKS, MAX_NI = first[0].instr_lens.shape
    d_t1 = t1_encoder.d_out
    vp_active = value_predict_weight > 0
    valid_active = valid_weight > 0

    t2.train()
    _dummy = torch.zeros(1, device=device)
    _arange_ni = torch.arange(MAX_NI, device=device)[None, :]

    # Compiled T2 forward+loss — FIXED-shape inputs (chunk_t1 is (N_CHUNKS,
    # MAX_NI, d_t1), corpus-constant) so the graph records once. The frozen-T1
    # per-instruction encode + chunk assembly stay EAGER in _prep: their input
    # (total_instrs, ...) varies per batch, and feeding that to a CUDA graph
    # re-records per size (we observed 51 — catastrophic). T1 (d128) eager is
    # the cheap part; the T2 (d512) forward+loss+backward we capture is the bulk.
    def _fwd_loss(chunk_t1, chunk_pad, li, lo, pc, in_slot, out_slot,
                  out_regs, vp_mask, valid_f, ike, oke):
        t2_out = t2.encode(chunk_t1, chunk_pad)
        if valid_active:
            valid_loss = ((t2_out.norm(dim=-1) - valid_f) ** 2).mean()
        else:
            valid_loss = t2_out.new_zeros(())
        bl = binding_losses(
            t2, t2_out, live_in_t=li, live_out_t=lo, pc_writes_t=pc,
            in_slot_t=in_slot, out_slot_t=out_slot, in_k_eff=ike, out_k_eff=oke)
        if vp_active:
            vp = t2_value_predict_loss(
                t2, t2_out, anchor_states_t, in_slot, out_slot, out_regs, vp_mask)
        else:
            vp = t2_out.new_zeros(())
        total = (valid_weight * valid_loss
                 + live_in_weight * bl['live_in']
                 + live_out_weight * bl['live_out']
                 + pc_writes_weight * bl['pc_writes']
                 + in_slot_weight * bl['in_slot']
                 + out_slot_weight * bl['out_slot']
                 + value_predict_weight * vp)
        metrics = {'total': total, 'valid': valid_loss,
                   'live_in': bl['live_in'], 'live_out': bl['live_out'],
                   'pc_writes': bl['pc_writes'], 'in_slot': bl['in_slot'],
                   'out_slot': bl['out_slot'], 'value_pred': vp}
        return total, metrics, None

    def _prep(item):
        batch, split_outputs, out_regs_np, vp_mask_np = item
        instr_tokens, instr_pad, chunk_idx, slot_idx, n_per_chunk = split_outputs
        if instr_tokens.shape[0] == 0:        # all-invalid batch
            return None
        # Eager (dynamic-shape) frozen-T1 encode + assemble → fixed chunk_t1.
        it = _h2d(instr_tokens, device, dtype=torch.long)
        ip = _h2d(instr_pad, device)
        with torch.no_grad():
            t1_vecs = t1_encoder.encode(it, ip)
        ci = _h2d(chunk_idx, device, dtype=torch.long)
        si = _h2d(slot_idx, device, dtype=torch.long)
        npc = _h2d(n_per_chunk, device, dtype=torch.long)
        chunk_t1 = torch.zeros(N_CHUNKS, MAX_NI, d_t1,
                               device=device, dtype=t1_vecs.dtype)
        chunk_t1[ci, si] = t1_vecs
        chunk_pad = _arange_ni >= npc[:, None]
        chunk_pad[:, 0] = False
        valid_f = (_h2d(batch.valid, device, dtype=torch.float32)
                   if valid_active else _dummy)
        if vp_active:
            out_regs_t = _h2d(out_regs_np, device)
            vp_mask_t = _h2d(vp_mask_np, device)
        else:
            out_regs_t = vp_mask_t = _dummy
        return (chunk_t1, chunk_pad,
                _h2d(batch.live_in_mask, device, dtype=torch.float32),
                _h2d(batch.live_out_mask, device, dtype=torch.float32),
                _h2d(batch.pc_writes, device, dtype=torch.float32),
                _h2d(batch.in_slot_regs, device, dtype=torch.long),
                _h2d(batch.out_slot_regs, device, dtype=torch.long),
                out_regs_t, vp_mask_t, valid_f,
                _slot_k_eff(batch.in_slot_regs), _slot_k_eff(batch.out_slot_regs))

    def _extra(item, aux):
        n_per_chunk = item[1][4]
        return {'n_chunks': N_CHUNKS, 'n_instrs': int(n_per_chunk.sum())}

    losses = run_train_loop(
        batch_source, model=t2, opt=opt, scheduler=scheduler, log=log,
        device=device,
        prep_fn=_prep, fwd_loss_fn=_fwd_loss, extra_fn=_extra,
        compile_step=compile_step, capture_state=True, start_step=step)
    return t2, losses

    return t2, log.losses


def _decoder_accuracy(logits, dec_tgt, dec_pad):
    """Per-token and per-instruction (whole-row exact-match) accuracy."""
    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        non_pad = ~dec_pad
        tok_total = non_pad.sum().item()
        tok_acc = (((pred == dec_tgt) & non_pad).sum().item() / tok_total
                   if tok_total > 0 else 0.0)
        B = dec_tgt.shape[0]
        instr_correct = 0
        for b in range(B):
            n = non_pad[b].sum().item()
            if n > 0 and (pred[b, :n] == dec_tgt[b, :n]).all():
                instr_correct += 1
        instr_acc = instr_correct / B if B > 0 else 0.0
    return tok_acc, instr_acc


def train_decoder(batch_iter, encoder, *, d_model, n_heads, n_layers,
                  n_memory=1, lr=3e-4, n_steps=None, log_every=100,
                  lr_min=1e-6, device='auto', on_log=None, compile_step=True):
    """Train a decoder on a frozen encoder with teacher-forced CE.

    Reconstructs each valid chunk's tokens from the frozen encoder's vector.
    Processes ALL rows at a fixed shape (invalid rows get all-PAD targets,
    ignored by CE) so the step is CUDA-graph-capturable like the encoders.
    Returns (decoder, losses); losses[i] = {step, loss, tok_acc, instr_acc}.
    """
    device = resolve_device(device)
    # TF32 + fused-attention-fastpath disable are handled in run_train_loop
    # (shared across trainers). The decoder reconstructs straight from the
    # batch each step (light prep), so it has no separate prefetcher.
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    encoder = encoder.to(device)

    decoder = Decoder(
        VOCAB_SIZE, d_model, n_heads, n_layers,
        d_emb=encoder.d_out, n_memory_tokens=n_memory).to(device)
    decoder.train()
    opt, scheduler = build_optim_sched(
        decoder.parameters(), lr, n_steps, lr_min=lr_min, device=device)

    def _fmt(r, ms, eta, lr_now):
        return (f'step {r["step"]:>5d}  loss {r["loss"]:.4f}  '
                f'tok_acc {r["tok_acc"]:.1%}  instr_acc {r["instr_acc"]:.1%}  '
                f'lr {lr_now:.1e}  {ms:.0f}ms/step  '
                f'eta {timedelta(seconds=int(eta))}')

    log = TrainLog(n_steps=n_steps, log_every=log_every, lr=lr,
                   scheduler=scheduler, formatter=_fmt, on_log=on_log)

    # max_dec_len from the corpus-constant token width (max_tokens + 1).
    first, batch_iter = _peek(batch_iter)
    if first is None:
        return decoder, log.losses
    max_dec_len = first.tokens.shape[1] + 1
    autocast = _is_cuda(device)

    def _fwd_loss(vecs, dec_in, dec_pad, dec_tgt):
        if autocast:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = decoder(vecs, dec_in, dec_pad)
                loss = F.cross_entropy(
                    logits.reshape(-1, VOCAB_SIZE), dec_tgt.reshape(-1),
                    ignore_index=PAD)
        else:
            logits = decoder(vecs, dec_in, dec_pad)
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE), dec_tgt.reshape(-1),
                ignore_index=PAD)
        return loss, {'loss': loss}, (logits, dec_tgt, dec_pad)

    def _prep(batch):
        tok = _h2d(batch.tokens, device, dtype=torch.long)
        pad = _h2d(padding_mask(batch), device)
        with torch.no_grad():
            vecs = encoder.encode(tok, pad)
        di, dt, dp = decoder_targets_fixed(batch, max_dec_len)
        return (vecs, _h2d(di, device, dtype=torch.long),
                _h2d(dp, device), _h2d(dt, device, dtype=torch.long))

    def _extra(batch, aux):
        logits, dec_tgt, dec_pad = aux
        tok_acc, instr_acc = _decoder_accuracy(logits, dec_tgt, dec_pad)
        return {'tok_acc': tok_acc, 'instr_acc': instr_acc}

    losses = run_train_loop(
        batch_iter, model=decoder, opt=opt, scheduler=scheduler, log=log,
        device=device,
        prep_fn=_prep, fwd_loss_fn=_fwd_loss, extra_fn=_extra,
        compile_step=compile_step)
    return decoder, losses
