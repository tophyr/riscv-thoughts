"""Training loops for the T1 and T2 encoders.

Three public training functions — train_encoder, train_t2_encoder, and
each a batch loop the CLI shells in scripts/ drive.

Shared helpers (used by trainers, eval, and the acceptance suite):
  load_checkpoint           — load a torch.compile-aware state_dict.
  encode_instrs             — tokenize + pad + encode an Instruction list.
  binding_losses            — rename-equivariant register-binding losses,
                              shared by the T1 and T2 encoder loops.
"""

import ctypes
import itertools
import os
import queue
import signal
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
    precompute_chunk,
    N_REGS, MAX_INPUT_SLOTS, MAX_OUTPUT_SLOTS, AUX_CE_IGNORE,
)
from emulator import Instruction, I_TYPE
from .model import (
    Compressor, T1Compressor, T2Compressor, apply_head_touched,
    instruction_wiring, X0_SLOT)


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


# Token front-end submodule prefixes (live in TokenEmbedder); everything else
# in a T1 checkpoint (core + heads) lives in the Compressor.
_T1_EMBEDDER_KEYS = ('tok_emb.', 'win_pos_emb.', 'content_encoder.')


def _split_t1_state(state):
    """Map a T1 state_dict to the T1Compressor layout (`embedder.*` +
    `compressor.*`). Old flat checkpoints (pre-`TokenEmbedder` split) keyed the
    front-end + core + heads at the top level; route the embedder submodules to
    `embedder.` and the rest to `compressor.`. Already-split checkpoints (any
    key starts with `embedder.`/`compressor.`) pass through unchanged."""
    if any(k.startswith(('embedder.', 'compressor.')) for k in state):
        return state
    out = {}
    for k, v in state.items():
        pre = 'embedder.' if k.startswith(_T1_EMBEDDER_KEYS) else 'compressor.'
        out[pre + k] = v
    return out


def encode_instrs(model, instrs, device):
    """Tokenize, pad, and encode a list of Instructions to their essence
    vectors. Registers enter as wiring (instruction_wiring) only — the essence
    is a deterministic function of the instruction, so repeated calls on the
    same input return the same point."""
    encoded = [encode_instruction(i) for i in instrs]
    max_len = max(len(e) for e in encoded)
    n = len(encoded)
    tok = np.full((n, max_len), PAD, dtype=np.int64)
    pad = np.ones((n, max_len), dtype=np.bool_)
    for i, e in enumerate(encoded):
        tok[i, :len(e)] = e
        pad[i, :len(e)] = False
    wiring = np.array([instruction_wiring(i) for i in instrs], dtype=np.int64)
    return model.encode(
        torch.from_numpy(tok).to(device),
        torch.from_numpy(pad).to(device),
        torch.from_numpy(wiring[:, 0]).to(device),
        torch.from_numpy(wiring[:, 1]).to(device),
        torch.from_numpy(wiring[:, 2]).to(device))


def resolve_device(spec):
    if spec == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return spec


def _is_cuda(device):
    return device == 'cuda' or getattr(device, 'type', None) == 'cuda'


class _H2DStager:
    """Host->device copy of a step's packed batch arrays (`_pack_step_arrays`):
    one pinned buffer, one copy. Pinned + non_blocking lets the CPU race ahead
    to launch the compiled step, and the copy sits on the default stream
    outside the captured graph, so it cannot collide with capture. Pinned
    buffers are reused round-robin once their copy has completed."""

    def __init__(self, device, n_buffers=4):
        self.device = device
        self.bufs = [None] * n_buffers
        self.done = [None] * n_buffers
        self.k = 0
        self.dev = None

    def __call__(self, packed, specs):
        """packed: uint8 ndarray; specs: [(offset, dtype, shape)] -> tensors."""
        if _is_cuda(self.device):
            k, self.k = self.k, (self.k + 1) % len(self.bufs)
            if self.done[k] is not None:
                self.done[k].synchronize()
            if self.bufs[k] is None or self.bufs[k].numel() < packed.size:
                self.bufs[k] = torch.empty(packed.size, dtype=torch.uint8,
                                           pin_memory=True)
            buf = self.bufs[k][:packed.size]
            np.copyto(buf.numpy(), packed)
            # One device buffer reused every step, so the inputs sit at fixed
            # addresses a captured CUDA graph can read. The copy is ordered on
            # the stream after the previous step's reads of it.
            if self.dev is None or self.dev.numel() < packed.size:
                self.dev = torch.empty(packed.size, dtype=torch.uint8,
                                       device=self.device)
            dev = self.dev[:packed.size]
            dev.copy_(buf, non_blocking=True)
            self.done[k] = torch.cuda.Event()
            self.done[k].record()
        else:
            dev = torch.from_numpy(packed.copy())
        return [dev[o:o + int(np.prod(shape)) * dt.itemsize]
                .view(_TORCH_DTYPE[dt]).view(shape) for o, dt, shape in specs]


_TORCH_DTYPE = {np.dtype(d): t for d, t in (
    (np.int64, torch.long), (np.float32, torch.float32), (np.int32, torch.int32),
    (np.bool_, torch.bool), (np.int8, torch.int8))}
_PACK_ALIGN = 16


def _pack_step_arrays(batch, split, *, vp, valid, imm):
    """Everything one training step copies to the device, as one aligned uint8
    buffer: the dense instruction grid, the binding and vp targets, and the
    validity / immediate targets when those losses are on. Runs in the
    prefetch workers so the trainer's main thread only stages a single buffer.
    Returns (packed or None for an all-invalid batch, specs, n_instrs, shape),
    specs listing (offset, dtype, shape) per array in _prep's order."""
    shape = batch.instr_lens.shape
    n_instrs = int(split[4].sum())
    if split[0].shape[0] == 0:
        return None, None, n_instrs, shape
    n_chunks, max_ni = shape
    tok, pad, gi0, gi1, go, active = _dense_chunk_grid(split, n_chunks, max_ni)
    # The losses iterate input / output slots only up to these bounds; the rest
    # must be empty (a chunk of K instructions reads at most 2K and writes at
    # most K).
    in_k, out_k = min(2 * max_ni, MAX_INPUT_SLOTS), min(max_ni, MAX_OUTPUT_SLOTS)
    assert (batch.in_slot_regs[:, in_k:] == AUX_CE_IGNORE).all()
    assert (batch.out_slot_regs[:, out_k:] == AUX_CE_IGNORE).all()
    i64, f32 = np.int64, np.float32
    arrays = [(tok, i64), (pad, None), (gi0, i64), (gi1, i64), (go, i64),
              (active, f32), (batch.live_in_mask, f32),
              (batch.live_out_mask, f32), (batch.pc_writes, f32),
              (batch.in_slot_regs, i64), (batch.out_slot_regs, i64)]
    if vp:
        arrays += [(batch.out_regs, None), (batch.out_regs_valid, None),
                   (batch.anchors, None), (batch.out_inputs, i64)]
    if valid:
        arrays += [(batch.valid, f32)]
    if imm:
        imm_np, imm_mask_np = _t1_immediates(batch)
        arrays += [(_immediate_rotary_np(imm_np), f32), (imm_mask_np, f32)]
    arrays = [np.ascontiguousarray(a if dt is None else a.astype(dt))
              for a, dt in arrays]
    specs, n = [], 0
    for a in arrays:
        specs.append((n, a.dtype, a.shape))
        n += -(-a.nbytes // _PACK_ALIGN) * _PACK_ALIGN
    packed = np.zeros(n, dtype=np.uint8)
    for (o, _, _), a in zip(specs, arrays):
        packed[o:o + a.nbytes] = a.reshape(-1).view(np.uint8)
    return packed, specs, n_instrs, shape


_VP_VALUE_BITS = 32     # two's-complement width; must match model._VP_VALUE_BITS


def _value_bits(x):
    """Two's-complement 32-bit decomposition (LSB first) of an INTEGER tensor
    (RV32 register values). Returns (..., 32) float in {0,1}. Uniform
    resolution: the low bit — the +1 immediate distinguisher — is carry-free,
    so it gets a clean gradient that loglog compression destroys (immediate-gap
    finding). x MUST be an integer dtype — do NOT pre-cast to float32, which
    silently loses every bit above 2^24 and corrupts exactly the low bits we
    are here to supervise."""
    xi = x.to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(_VP_VALUE_BITS, device=x.device, dtype=torch.int64)
    return ((xi.unsqueeze(-1) >> shifts) & 1).to(torch.float32)


def _value_smooth(x):
    """Bounded sign-preserving smooth-magnitude encoding in (-1, 1). The vp
    ordering term — resolution comes from the bits; this only keeps
    numerically-close values close / ordered (the interpolation property)."""
    return torch.tanh((torch.sign(x) * torch.log1p(x.abs())) / 8.0)


def _value_feat(x_i):
    """One value -> the (..., 33) feature the vp head reads it as: 32
    two's-complement bits + one bounded smooth-magnitude scalar. `x_i` MUST be
    an integer tensor (see _value_bits)."""
    return torch.cat(
        [_value_bits(x_i), _value_smooth(x_i.float()).unsqueeze(-1)], dim=-1)


def vp_slot_loss(comp, T, anchors_t, out_slot_regs_t, out_inputs_t, out_regs_t,
                 mask_t, vp_smooth_weight=0.2, return_count=False, n_out=None):
    """Value prediction evaluated PER WRITTEN SLOT.

    For output slot k of a row, the head reads the written register's
    description — that slot's value channels in `T` — and, for each external
    input that output depends on (`out_inputs`, in the oracle's colour order),
    the input's value at its own register in the row's anchor state (32 bits +
    1 smooth scalar) and the input's structural colour from `T`, and predicts
    that register's value after the chunk (32 bit-logits + 1 smooth scalar).
    Nothing in the features is a register number or a position in the text:
    the description, the colours and the gathered values are all reached
    through one-hot gathers that a rename permutes, so a renamed twin presents
    the same description with the same input values and the same target.

    Args:
      comp:             the Compressor (uses comp.vp_head, d_out, d_event)
      T:                (B, n_regs, d_out+d_event+d_colour) the emitted state
      anchors_t:        (B, A, 32) int32 — the register file each row was
                        executed from
      out_slot_regs_t:  (B, MAX_OUT) int64 — output registers (AUX_CE_IGNORE
                        for unused slots)
      out_inputs_t:     (B, MAX_OUT, MAX_IN) int64 — each output's external
                        input registers in colour order (AUX_CE_IGNORE padded)
      out_regs_t:       (B, A, 32) int32 — the register file after the chunk
      mask_t:           (B,) bool — rows to include
      n_out:            bound on filled output slots; slots past it are
                        skipped (they must be AUX_CE_IGNORE)

    Returns a scalar loss (per-bit BCE + vp_smooth_weight · smooth-magnitude
    MSE) over (row, output slot, anchor) triples with a filled output slot.
    """
    B, A = anchors_t.shape[0], anchors_t.shape[1]
    d, dc = comp.d_out, comp.d_colour
    if n_out is not None:
        out_slot_regs_t = out_slot_regs_t[:, :n_out]
        out_inputs_t = out_inputs_t[:, :n_out]
    O, I = out_inputs_t.shape[1], out_inputs_t.shape[2]
    out_valid = out_slot_regs_t != AUX_CE_IGNORE                      # (B, O)
    out_idx = out_slot_regs_t.clamp(min=0).long()
    desc = T[..., :d].gather(1, out_idx.unsqueeze(-1).expand(-1, -1, d))   # (B, O, d)
    in_valid = (out_inputs_t != AUX_CE_IGNORE)                        # (B, O, I)
    in_idx = out_inputs_t.clamp(min=0).long()
    colours = T[..., d + comp.d_event:]                               # (B, R, dc)
    in_col = colours.gather(
        1, in_idx.reshape(B, -1, 1).expand(-1, -1, dc)).reshape(B, O, I, dc)
    in_col = in_col * in_valid.unsqueeze(-1)
    # Unfilled inputs feed value 0 and colour 0 = "no input". Kept INTEGER
    # until the bit decomposition: float32 loses every bit above 2^24.
    vals = anchors_t.to(torch.int64).gather(
        -1, in_idx.reshape(B, 1, -1).expand(-1, A, -1)).reshape(B, A, O, I)
    vals = vals * in_valid.unsqueeze(1)
    vfeat = torch.cat([_value_feat(vals),
                       in_col.unsqueeze(1).expand(-1, A, -1, -1, -1)], -1)
    x = torch.cat([desc.unsqueeze(1).expand(-1, A, -1, -1),
                   vfeat.reshape(B, A, O, I * (_VP_VALUE_BITS + 1 + dc))], -1)
    pred = comp.vp_head(x)                                            # (B, A, O, 33)
    target_i = out_regs_t.to(torch.int64).gather(
        -1, out_idx.unsqueeze(1).expand(-1, A, -1))                  # (B, A, O)
    nbits = _VP_VALUE_BITS
    bit_loss = F.binary_cross_entropy_with_logits(
        pred[..., :nbits], _value_bits(target_i), reduction='none').mean(-1)
    sm_loss = (pred[..., nbits] - _value_smooth(target_i.float())) ** 2
    per_slot = (bit_loss + vp_smooth_weight * sm_loss).mean(dim=1)   # (B, O)
    # Mask-multiply (not boolean index / early return) so the op stays
    # static-shape and host-sync-free — capturable in the full-step graph.
    pm = (out_valid & mask_t.unsqueeze(-1)).float()
    loss = (per_slot * pm).sum() / pm.sum().clamp(min=1)
    return (loss, int(pm.sum())) if return_count else loss


def vp_slot_predict(comp, T, anchors_t, out_slot_regs_t, out_inputs_t):
    """The head's prediction per (row, anchor, output slot), (B, A, O, 33),
    through the same features `vp_slot_loss` builds — the one definition every
    probe that scores the head reads."""
    B, A = anchors_t.shape[0], anchors_t.shape[1]
    d, dc = comp.d_out, comp.d_colour
    O, I = out_inputs_t.shape[1], out_inputs_t.shape[2]
    out_idx = out_slot_regs_t.clamp(min=0).long()
    desc = T[..., :d].gather(1, out_idx.unsqueeze(-1).expand(-1, -1, d))
    in_valid = (out_inputs_t != AUX_CE_IGNORE)
    in_idx = out_inputs_t.clamp(min=0).long()
    colours = T[..., d + comp.d_event:]
    in_col = colours.gather(
        1, in_idx.reshape(B, -1, 1).expand(-1, -1, dc)).reshape(B, O, I, dc)
    in_col = in_col * in_valid.unsqueeze(-1)
    vals = anchors_t.to(torch.int64).gather(
        -1, in_idx.reshape(B, 1, -1).expand(-1, A, -1)).reshape(B, A, O, I)
    vals = vals * in_valid.unsqueeze(1)
    vfeat = torch.cat([_value_feat(vals),
                       in_col.unsqueeze(1).expand(-1, A, -1, -1, -1)], -1)
    x = torch.cat([desc.unsqueeze(1).expand(-1, A, -1, -1),
                   vfeat.reshape(B, A, O, I * (_VP_VALUE_BITS + 1 + dc))], -1)
    return comp.vp_head(x)


# Rotary immediate encoding (T1). POWER-OF-2 periods 2,4,...,4096: each frequency
# is `imm mod 2^k` = the low k bits, which are directly readable from the hex
# tokens. This is bit-aligned and non-redundant across scales — unlike an
# arbitrary geometric ladder, which asks the encoder for `imm mod 7` / `imm mod
# 11` (modular reductions it can't compute, so it hedges those features to ~0 —
# the diagnosed source of the residual imm+1 collisions). angular freq = 2π/period.
_IMM_N_FREQ = 12        # must match model._IMM_N_FREQ (head output = 2*this)
_IMM_OMEGA = (2.0 * np.pi / (2.0 ** np.arange(1, _IMM_N_FREQ + 1))).astype(np.float32)


def _immediate_rotary_np(imm):
    """Rotary/Fourier encoding of integer immediates: imm (B,) int ->
    (B, 2*_IMM_N_FREQ) float [cos.. , sin..]. imm+δ rotates each frequency's
    (cos,sin) pair by δ·ω — a consistent, composable, opcode-independent
    transform (the geometry we want the essence's immediate axis to have),
    multi-scale so high ω resolves +1 and low ω carries magnitude."""
    ph = imm[:, None].astype(np.float32) * _IMM_OMEGA[None, :]     # (B, K)
    return np.concatenate([np.cos(ph), np.sin(ph)], axis=1).astype(np.float32)


def _t1_immediates(batch):
    """Per single-instruction row: the in-scope (I-type ALU) immediate value +
    a supervision mask. Only I_TYPE ops (ADDI / logical-imm / shifts) carry a
    ≤12-bit data immediate we supervise; every other row is masked. Trainer-side
    decode from the tokens (no RVT change). Returns (imm (B,) int64, mask (B,)
    float32)."""
    B = batch.tokens.shape[0]
    imm = np.zeros(B, dtype=np.int64)
    mask = np.zeros(B, dtype=np.float32)
    for b in range(B):
        if not batch.valid[b]:
            continue
        L = int(batch.token_lens[b])
        try:
            instr, _ = decode_instruction(batch.tokens[b, :L].tolist(), 0)
        except Exception:
            continue
        # I-type wiring is (rd, rs1, imm). Supervise the immediate ONLY where it
        # is behaviorally LIVE. An x0-write (rd==0) is a no-op — the result is
        # discarded — so its immediate is dead: `ADDI x0,rs,k` is the same no-op
        # for every k. The essence correctly COLLAPSES those equivalents (low
        # magnitude, immediate-independent); supervising the dead immediate would
        # fight that collapse with a gradient it can never satisfy (the ~3% imm
        # floor). So mask x0-writes out, like memory-ops and non-I-type already are.
        if instr.opcode in I_TYPE and instr.args[0] != 0:
            imm[b] = instr.args[2]
            mask[b] = 1.0
    return imm, mask


class _LRClock:
    """The float LR schedule driving an optimizer whose lr is a device tensor.
    A tensor lr is what lets clip + Adam compile into one graph without a
    recompile per step, but the schedulers do not compute the same sequence on
    a tensor lr (up to 10% off by mid-run). So the schedule runs unchanged on a
    stand-in optimizer and its value is copied into the tensor after each step.
    Quacks like the scheduler it wraps, for the loop, the log and resume."""

    def __init__(self, scheduler, clock, opt):
        self.scheduler, self.clock, self.opt = scheduler, clock, opt
        self.lr_t = opt.param_groups[0]['lr']

    def sync(self):
        for g in self.opt.param_groups:
            g['lr'] = self.lr_t
        self.lr_t.fill_(self.clock.param_groups[0]['lr'])

    def step(self):
        self.scheduler.step()
        self.sync()

    def get_last_lr(self):
        return self.scheduler.get_last_lr()

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, state):
        self.scheduler.load_state_dict(state)
        self.clock.param_groups[0]['lr'] = state['_last_lr'][0]
        self.sync()


def _lr_schedule(opt, n_steps, warmup_steps, lr_min):
    cosine_steps = max(1, n_steps - warmup_steps)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cosine_steps, eta_min=lr_min)
    if warmup_steps <= 0:
        return cosine
    warmup = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
    return torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup, cosine], milestones=[warmup_steps])


def build_optim_sched(params, lr, n_steps, *, warmup_steps=0,
                      lr_min=1e-6, device='cuda'):
    """Adam + an optional-warmup→cosine LR schedule, shared by all trainers.
    Returns (opt, scheduler); scheduler is None when n_steps is falsy (LR held
    constant, e.g. unbounded streaming). On CUDA the Adam is capturable with a
    device-tensor lr, so run_train_loop can compile it together with gradient
    clipping, and the scheduler is an _LRClock."""
    if not _is_cuda(device):
        opt = torch.optim.Adam(params, lr=lr)
        return opt, (_lr_schedule(opt, n_steps, warmup_steps, lr_min)
                     if n_steps else None)
    clock = torch.optim.SGD([torch.zeros(1)], lr=lr)
    opt = torch.optim.Adam(params, lr=torch.tensor(float(lr), device=device),
                           foreach=True, capturable=True)
    if not n_steps:
        return opt, None
    scheduler = _LRClock(_lr_schedule(clock, n_steps, warmup_steps, lr_min),
                         clock, opt)
    scheduler.sync()
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
    as-close-as-possible resume (the data stream from datagen is random,
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
    if isinstance(scheduler, _LRClock):
        # A checkpoint from a fused Adam carries its flags and a float lr.
        for g in opt.param_groups:
            g.update(fused=None, foreach=True, capturable=True)
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
    on_log(model, losses, train_state) — train_state is None for trainers
    that don't checkpoint resume state — fires at the first log boundary at
    least `save_every` steps after the previous one. It writes the checkpoint,
    which stalls the GPU for a fraction of a second — at every 100-step log
    boundary that cost a quarter of the step rate.
    """

    def __init__(self, *, n_steps, log_every, lr, scheduler, formatter,
                 on_log=None, save_every=10000):
        self.n_steps = n_steps
        self.log_every = log_every
        self.save_every = save_every
        self._next_save = save_every
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
        if self.on_log is not None and step >= self._next_save:
            self._next_save = step + self.save_every
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


def _graph_signature(inputs):
    return [(t.data_ptr(), t.shape, t.dtype) for t in inputs if torch.is_tensor(t)]


def run_train_loop(batch_source, *, model, opt, scheduler, log, device,
                   prep_fn, fwd_loss_fn, extra_fn=None, clip=1.0,
                   compile_step=True, capture_state=False, start_step=0):
    """The shared compiled training loop for both trainers (T1/T2).

    Owns the universal mechanics: process/device setup (TF32 matmul + disabling
    the fused-attention fast path that the compiled step can't capture),
    torch.compile (max-autotune) of the forward+loss, the whole step —
    forward, backward, grad-clip, Adam — captured as one CUDA graph on CUDA,
    the scheduler, and logging via `log`
    (TrainLog) — including the log-boundary sync discipline and optional
    resume-state capture. The two genuinely trainer-specific pieces are
    injected:

      prep_fn(item) -> tuple of GPU inputs for fwd_loss_fn, or None to skip
          the item (empty batch). Does the H2D copy (_H2DStager). Every
          tensor it returns must keep one shape AND one address across steps:
          the captured graph replays against the addresses it recorded.
      fwd_loss_fn(*inputs) -> (total, metrics, aux)
          total:   scalar loss to backward.
          metrics: dict[str, scalar tensor] logged at each boundary.
          aux:     tensors a boundary-only extra_fn needs (or None).
          The torch.compile target — keep it fixed-shape + host-sync-free.
      extra_fn(item, aux) -> dict  boundary-only extra fields (probes, chunk
          counts), or None.

    `model` is the trainable module (its parameters are grad-clipped). Returns
    log.losses. start_step seeds the step counter (resume)."""
    if _is_cuda(device):
        # TF32 tensor cores for fp32 matmul (~2-4x, ~1 bit precision). Set
        # here so every trainer gets it uniformly (it had drifted to only
        # T2, leaving T1 on slow fp32 matmuls).
        torch.set_float32_matmul_precision('high')
    # The host's per-step tensor work is small; more intra-op threads than this
    # spin and contend with the stream's receive path, slowing the step ~7%.
    torch.set_num_threads(min(4, torch.get_num_threads()))
    # Disable the fused attention fast path (_transformer_encoder_layer_fwd /
    # nested-tensor) — not torch.compile-able; it shatters the captured graph.
    # Global + numerically transparent; set once for all trainers.
    torch.backends.mha.set_fastpath_enabled(False)
    # On CUDA the whole step — forward, backward, gradient clip, Adam — is
    # captured as ONE CUDA graph and replayed, so the host issues one launch
    # per step instead of hundreds. The forward therefore compiles without
    # torch.compile's own CUDA graphs (those cannot nest in a capture);
    # max-autotune still benchmarks its GEMM kernels, which is 9% faster per
    # step than the default kernels and the same computation up to float
    # summation order. The search costs minutes on a cold inductor cache, once.
    graphed = compile_step and _is_cuda(device)
    step_fn = (torch.compile(fwd_loss_fn, mode=('max-autotune-no-cudagraphs'
                                               if graphed else 'max-autotune'))
               if compile_step else fwd_loss_fn)
    params = list(model.parameters())

    def _scale_and_step(norm):
        grads = [p.grad for p in params if p.grad is not None]
        torch._foreach_mul_(grads, torch.clamp(clip / (norm + 1e-6), max=1.0))
        opt.step()
    # Compiled, the clip scaling and Adam fuse into a few kernels: ~0.6 ms per
    # step less GPU time inside the graph than the eager foreach kernels.
    scale_and_step = (torch.compile(_scale_and_step) if graphed
                      else _scale_and_step)

    def train_step(*inputs):
        """forward + backward + clip_grad_norm_(params, clip) + Adam."""
        total, metrics, aux = step_fn(*inputs)
        total.backward()
        grads = [p.grad for p in params if p.grad is not None]
        scale_and_step(torch.linalg.vector_norm(
            torch.stack(torch._foreach_norm(grads))))
        return metrics, aux

    # Capture needs a few eager steps first (on a side stream) so the compiled
    # kernels, the backward and the optimizer state all exist. They are real
    # training steps on real batches, so nothing is trained twice.
    WARMUP = 3
    graph, captured, n_eager = None, None, 0
    side = torch.cuda.Stream() if graphed else None
    step = start_step
    for item in batch_source:
        inputs = prep_fn(item)
        if inputs is None:
            continue
        if not graphed:
            opt.zero_grad(set_to_none=True)
            metrics, aux = train_step(*inputs)
        elif graph is None and n_eager < WARMUP:
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                opt.zero_grad(set_to_none=True)
                metrics, aux = train_step(*inputs)
            torch.cuda.current_stream().wait_stream(side)
            n_eager += 1
        elif graph is None:
            # Gradients start as None, so the captured backward allocates them
            # once in the graph's pool and every replay rewrites them in place.
            opt.zero_grad(set_to_none=True)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                metrics, aux = train_step(*inputs)
            captured = _graph_signature(inputs)
            graph.replay()
        else:
            # A replay reads the captured addresses with the captured shapes,
            # so both must hold or it silently computes on the wrong layout.
            assert _graph_signature(inputs) == captured
            graph.replay()
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


def _t1_wiring(batch):
    """Per single-instruction row, the GP wiring (in0, in1, out) the state
    machine routes on, decoded from the row's tokens. Invalid / undecodable
    rows get (0, 0, 0) (x0, masked by the behavioral targets anyway). CPU-
    side (no RVT-format change); returns three (B,) int64 arrays."""
    B = batch.tokens.shape[0]
    in0 = np.zeros(B, dtype=np.int64)
    in1 = np.zeros(B, dtype=np.int64)
    out = np.zeros(B, dtype=np.int64)
    for b in range(B):
        if not batch.valid[b]:
            continue
        L = int(batch.token_lens[b])
        try:
            instr, _ = decode_instruction(batch.tokens[b, :L].tolist(), 0)
        except Exception:
            continue
        in0[b], in1[b], out[b] = instruction_wiring(instr)
    return in0, in1, out


def train_compressor(batch_iter, *, ingest, lower_encoder=None,
                     d_out=128, d_event=16, op_cell_depth=1, vp_head_depth=3,
                     n_pool_heads=1,
                     d_model=128, n_heads=4, n_layers=2, max_window=72,
                     route='binding',
                     lr=3e-4, n_steps=None, log_every=100, lr_min=1e-6,
                     warmup_steps=0,
                     valid_weight=0.1, live_in_weight=0.1, live_out_weight=0.1,
                     pc_writes_weight=0.1, in_slot_weight=0.1,
                     out_slot_weight=0.1, value_predict_weight=1.0,
                     vp_smooth_weight=0.2, imm_weight=1.0,
                     resume=None, device='auto', on_log=None, save_every=10000,
                     compile_step=True):
    """The ONE trainer for the equivariant Compressor — T1 and T2 run the SAME
    execution path. The only tier-specific piece is the content
    provider (`_content` below): ingest='tokens' (T1) embeds opcode+immediate
    tokens with the model itself (gradients flow into the front-end);
    ingest='vectors' (T2) runs a FROZEN lower-tier Compressor per instruction
    and routes T2's wiring off its PREDICTED binding (route='binding')
    rather than decoded tokens. Everything else is shared: the dense-grid prep
    (`_dense_chunk_grid`), the binding + canonical value-prediction + magnitude
    losses, the single compiled CUDA-graph step (frozen-T1 encode + routing +
    core + losses folded in), the prefetcher, and the logging.

    Returns (model, losses). loss records: step, total, valid, live_in,
    live_out, pc_writes, in_slot, out_slot, value_pred, n_chunks,
    n_instrs.

    """
    device = resolve_device(device)
    # `model` is the trained+saved+grad-clipped entity; `comp` is the Compressor
    # whose heads/core the losses + routing read. For T2 they're the same; for
    # T1, model = TokenEmbedder + Compressor and comp = model.compressor.
    if ingest == 'vectors':
        lower_encoder.eval()
        for p in lower_encoder.parameters():
            p.requires_grad = False
        lower_encoder = lower_encoder.to(device)
        comp = Compressor(
            d_content=lower_encoder.d_out, d_out=d_out, d_event=d_event,
            op_cell_depth=op_cell_depth, vp_head_depth=vp_head_depth,
            n_pool_heads=n_pool_heads)
        model = comp.to(device)
        model_file = 't2.pt'
    else:
        model = T1Compressor(
            vocab_size=VOCAB_SIZE, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, max_window=max_window, d_out=d_out,
            d_event=d_event, op_cell_depth=op_cell_depth,
            vp_head_depth=vp_head_depth, n_pool_heads=n_pool_heads).to(device)
        comp = model.compressor
        model_file = 'encoder.pt'
    route_binding = (route == 'binding')

    # Optional resume: a run-dir (model + train_state.pt) continues
    # optimizer/schedule/step/RNG; a bare .pt loads weights only. T1 state is
    # split-remapped (old flat -> embedder./compressor.) if needed.
    resume_train_state = None
    if resume is not None:
        ckpt_path = Path(resume)
        weights_path = ckpt_path / model_file if ckpt_path.is_dir() else ckpt_path
        state = load_checkpoint(weights_path, device)
        if ingest == 'tokens':
            state = _split_t1_state(state)
        model.load_state_dict(state, strict=False)
        if ckpt_path.is_dir():
            ts = ckpt_path / 'train_state.pt'
            if ts.exists():
                resume_train_state = torch.load(
                    ts, map_location=device, weights_only=False)

    opt, scheduler = build_optim_sched(
        model.parameters(), lr, n_steps, warmup_steps=warmup_steps,
        lr_min=lr_min, device=device)

    def _fmt(r, ms, eta, lr_now):
        return (
            f'step {r["step"]:>5d}  loss {r["total"]:.4f} '
            f'(val {r["valid"]:.4f} li {r["live_in"]:.3f} lo {r["live_out"]:.3f} '
            f'pc {r["pc_writes"]:.3f} is {r["in_slot"]:.3f} os {r["out_slot"]:.3f} '
            f'vp {r["value_pred"]:.4f} im {r["imm"]:.4f}) '
            f'B={r["n_chunks"]} I={r["n_instrs"]}  '
            f'lr {lr_now:.1e}  {ms:.0f}ms/step  '
            f'eta {timedelta(seconds=int(eta))}')

    log = TrainLog(n_steps=n_steps, log_every=log_every, lr=lr,
                   scheduler=scheduler, formatter=_fmt, on_log=on_log,
                   save_every=save_every)
    step = 0
    if resume_train_state is not None:
        step = restore_train_state(resume_train_state, opt, scheduler)
        print(f'Resumed from step {step}')

    vp_active = value_predict_weight > 0
    # immediate head is T1-only: one instruction = one unambiguous immediate.
    # (T2's chunk essence has no single immediate; its imm_head stays unused.)
    imm_active = imm_weight > 0 and ingest == 'tokens'
    valid_active = valid_weight > 0
    # Prefetcher overlaps CPU-side prep with GPU work — shared by both tiers.
    batch_source = TrainBatchPrefetcher(
        batch_iter, pack=dict(vp=vp_active, valid=valid_active, imm=imm_active))
    first, batch_source = _peek(batch_source)
    if first is None:
        return model, log.losses
    N_CHUNKS, MAX_NI = first[3]
    # CONSTANT ListMLE loop bounds (corpus-constant, so the compiled step never
    # recompiles on a varying slot width). A K-instr chunk reads <=2K regs,
    # writes <=K; iterating extra all-IGNORE columns is a no-op on the loss.
    IN_K = min(2 * MAX_NI, MAX_INPUT_SLOTS)
    OUT_K = min(MAX_NI, MAX_OUTPUT_SLOTS)
    model.train()
    _dummy = torch.zeros(1, device=device)

    def _content(tok, pad, w0, w1, wo, active):
        """Tier front-end (the one tier-specific path): a (B, MNI) dense
        token grid -> (content_seq (B,MNI,d_content), in0,in1,out (B,MNI)).
        tokens: embed with the trained model (grad flows). vectors: frozen
        lower-tier encode + predicted-binding routing (no grad)."""
        B, MNI = active.shape
        flat = B * MNI
        Ltok = tok.shape[-1]
        if ingest == 'tokens':
            content = model.embedder(
                tok.reshape(flat, Ltok), pad.reshape(flat, Ltok)).reshape(B, MNI, -1)
            return content, w0, w1, wo
        with torch.no_grad():
            T1s, ess = lower_encoder.encode_state(
                tok.reshape(flat, Ltok), pad.reshape(flat, Ltok),
                w0.reshape(flat), w1.reshape(flat), wo.reshape(flat))
            if route_binding:
                r0, r1, ro = _t1_predicted_wiring(lower_encoder, T1s)
            else:
                r0, r1, ro = w0.reshape(flat), w1.reshape(flat), wo.reshape(flat)
        return (ess.reshape(B, MNI, -1),
                r0.reshape(B, MNI), r1.reshape(B, MNI), ro.reshape(B, MNI))

    # ONE compiled CUDA-graph step (the one-graph fold): front-end + core + losses, all
    # at a fixed-padded instruction count (B*MNI). Host-sync-free, static-shape.
    def _fwd_loss(tok, pad, w0, w1, wo, active,
                  li, lo, pc, in_slot, out_slot, out_regs, vp_mask, anchors,
                  out_inputs, valid_f,
                  ike, oke, imm_target, imm_mask):
        content_seq, c0, c1, co = _content(tok, pad, w0, w1, wo, active)
        T, essence = comp.encode_state(content_seq, c0, c1, co, active)
        valid_loss = (((essence.norm(dim=-1) - valid_f) ** 2).mean()
                      if valid_active else essence.new_zeros(()))
        bl = binding_losses(
            comp, T, essence, live_in_t=li, live_out_t=lo, pc_writes_t=pc,
            in_slot_t=in_slot, out_slot_t=out_slot, in_k_eff=ike, out_k_eff=oke)
        vp = (vp_slot_loss(
                  comp, T, anchors, out_slot, out_inputs, out_regs, vp_mask,
                  vp_smooth_weight=vp_smooth_weight, n_out=oke)
              if vp_active else essence.new_zeros(()))
        # Immediate head (T1 only): linear readout of the essence must match the
        # rotary encoding of the instruction's immediate, masked to I-type rows.
        if imm_active:
            # direction only (magnitude = validity).
            imm_pred = comp.imm_head(F.normalize(essence, dim=-1))   # (B, 2*_IMM_N_FREQ)
            imm_per = ((imm_pred - imm_target) ** 2).mean(-1)   # (B,)
            imm = (imm_per * imm_mask).sum() / imm_mask.sum().clamp(min=1)
        else:
            imm = essence.new_zeros(())
        total = (valid_weight * valid_loss
                 + live_in_weight * bl['live_in']
                 + live_out_weight * bl['live_out']
                 + pc_writes_weight * bl['pc_writes']
                 + in_slot_weight * bl['in_slot']
                 + out_slot_weight * bl['out_slot']
                 + value_predict_weight * vp
                 + imm_weight * imm
                 )
        metrics = {'total': total, 'valid': valid_loss,
                   'live_in': bl['live_in'], 'live_out': bl['live_out'],
                   'pc_writes': bl['pc_writes'], 'in_slot': bl['in_slot'],
                   'out_slot': bl['out_slot'], 'value_pred': vp,
                   'imm': imm}
        return total, metrics, None

    stage = _H2DStager(device)

    def _prep(item):
        packed, specs = item[0], item[1]
        if packed is None:                        # all-invalid batch
            return None
        staged = iter(stage(packed, specs))
        fixed = [next(staged) for _ in range(11)]
        out_regs_t, vp_mask_t, anchors_t, out_inputs_t = (
            (next(staged), next(staged), next(staged), next(staged)) if vp_active
            else (_dummy, _dummy, _dummy, _dummy))
        valid_f = next(staged) if valid_active else _dummy
        imm_target_t, imm_mask_t = ((next(staged), next(staged)) if imm_active
                                    else (_dummy, _dummy))
        return (*fixed, out_regs_t, vp_mask_t, anchors_t, out_inputs_t, valid_f,
                IN_K, OUT_K, imm_target_t, imm_mask_t)

    def _extra(item, aux):
        return {'n_chunks': N_CHUNKS, 'n_instrs': item[2]}

    losses = run_train_loop(
        batch_source, model=model, opt=opt, scheduler=scheduler, log=log,
        device=device, prep_fn=_prep, fwd_loss_fn=_fwd_loss, extra_fn=_extra,
        compile_step=compile_step, capture_state=True, start_step=step)
    return model, losses


def train_encoder(batch_iter, *,
                  d_model=128, n_heads=4, n_layers=2, d_out=128,
                  max_window=72,
                  d_event=16, op_cell_depth=1, vp_head_depth=3,
                  n_pool_heads=1,
                  lr=3e-4, n_steps=None, log_every=100, lr_min=1e-6,
                  valid_weight=0.1,
                  live_in_weight=0.1, live_out_weight=0.1,
                  pc_writes_weight=0.1,
                  in_slot_weight=0.1, out_slot_weight=0.1,
                  value_predict_weight=1.0, vp_smooth_weight=0.2, imm_weight=1.0,
                  device='auto', on_log=None, save_every=10000,
                  compile_step=True):
    """Train a T1 encoder on RVT batches.

    T1 is the single-instruction special case of T2 and uses the same
    rename-equivariant register-binding supervision (binding_losses):
    live_in/out + pc_writes BCE, in/out-slot ListMLE, plus value
    prediction. There is no equivalence-collapse loss — equivalence is
    an emergent property of the consistent per-instruction targets, not a
    separate objective that would fight register identity.

    d_event..n_pool_heads: the equivariant core's own dimensions, forwarded
                         unchanged to train_compressor — T1 and T2 share one
                         core, so the tiers must be able to run the same
                         configuration of it. See RegisterStateMachine.

    valid_weight:        weight on (||v_c|| - (1 if valid else 0))**2.
    live_in/out_weight:  BCE on the behavioral input/output register masks.
    pc_writes_weight:    BCE on the explicit-PC-write flag.
    in/out_slot_weight:  ListMLE on the input/output slot orderings.
    value_predict_weight: MSE on per-anchor rd value prediction.
    on_log:              optional callable(model, losses, train_state=None)
                         called at the first log point at or after every
                         `save_every` steps (uniform across trainers).
    compile_step:        torch.compile the forward+loss with CUDA graphs
                         (mode='max-autotune'). The whole forward+loss+
                         backward is captured as one graph (~9 launches vs
                         ~485 eager), saturating the GPU (~99% vs ~50%) and
                         ~2.5x faster. The loss path is sync-free +
                         static-shape, so it captures cleanly.
                         Numerics differ slightly from eager (fusion / TF32
                         reduction order) — set False for a bit-faithful
                         eager run.

    Returns (encoder, losses). losses[i] dict has keys: step, total,
    valid, live_in, live_out, pc_writes, in_slot, out_slot, value_pred,
    plus probe_* keys at each log point.
    """
    return train_compressor(
        batch_iter, ingest='tokens', d_out=d_out, d_model=d_model,
        n_heads=n_heads, n_layers=n_layers, max_window=max_window,
        d_event=d_event, op_cell_depth=op_cell_depth,
        vp_head_depth=vp_head_depth, n_pool_heads=n_pool_heads,
        lr=lr, n_steps=n_steps, log_every=log_every, lr_min=lr_min,
        valid_weight=valid_weight, live_in_weight=live_in_weight,
        live_out_weight=live_out_weight, pc_writes_weight=pc_writes_weight,
        in_slot_weight=in_slot_weight, out_slot_weight=out_slot_weight,
        value_predict_weight=value_predict_weight,
        vp_smooth_weight=vp_smooth_weight, imm_weight=imm_weight,
        device=device, on_log=on_log, save_every=save_every,
        compile_step=compile_step)


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
        instr_in0/in1/out: (N_instr_total,) int64 — per-instruction GP
                      wiring for the state machine (x0=0). Decoded here so
                      it rides through the prefetcher with the token split.
    """
    B, max_n_instrs = batch.instr_lens.shape
    n_per_chunk = (batch.instr_lens > 0).sum(axis=1).astype(np.int32)
    total = int(n_per_chunk.sum())
    if total == 0:
        empty = np.zeros((0, _MAX_INSTR_TOKENS), dtype=np.int8)
        z = np.zeros(0, dtype=np.int64)
        return (empty, np.ones((0, _MAX_INSTR_TOKENS), dtype=bool),
                z, z, n_per_chunk, z.copy(), z.copy(), z.copy())

    instr_tokens = np.full((total, _MAX_INSTR_TOKENS), PAD, dtype=np.int8)
    instr_pad = np.ones((total, _MAX_INSTR_TOKENS), dtype=bool)
    chunk_idx = np.empty(total, dtype=np.int64)
    slot_idx = np.empty(total, dtype=np.int64)
    instr_in0 = np.zeros(total, dtype=np.int64)
    instr_in1 = np.zeros(total, dtype=np.int64)
    instr_out = np.zeros(total, dtype=np.int64)

    out = 0
    for c in range(B):
        token_offset = 0
        for j in range(max_n_instrs):
            L = int(batch.instr_lens[c, j])
            if L == 0:
                continue
            toks = batch.tokens[c, token_offset:token_offset + L]
            instr_tokens[out, :L] = toks
            instr_pad[out, :L] = False
            chunk_idx[out] = c
            slot_idx[out] = j
            try:
                instr, _ = decode_instruction(toks.tolist(), 0)
                instr_in0[out], instr_in1[out], instr_out[out] = \
                    instruction_wiring(instr)
            except Exception:
                pass    # leave (0,0,0); undecodable rows route to x0
            out += 1
            token_offset += L
    # sanitize out-of-range reg ids (garbage windows) -> x0.
    for a in (instr_in0, instr_in1, instr_out):
        a[(a < 0) | (a >= N_REGS)] = 0
    return (instr_tokens, instr_pad, chunk_idx, slot_idx, n_per_chunk,
            instr_in0, instr_in1, instr_out)


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


@torch.no_grad()
def _t1_predicted_wiring(t1, T):
    """Per instruction, derive the GP wiring (in0, in1, out) for T2 from the
    FROZEN T1's own binding PREDICTIONS — never decoded tokens. This is the
    tier-recursion routing: T2 consumes only T1's emitted output
    (essence + binding), so the experiment actually tests "is T1's emission
    sufficient for the next tier?". With token wiring (the old diagnostic path) T2
    would score identically given a useless T1, so it tested nothing.

      out      = argmax out_score over predicted-written regs (gated by
                 predicted live_out count; absent => x0).
      in0, in1 = top-2 in_score over predicted-read regs in read order
                 (gated by predicted live_in count; absent => x0).

    T1's binding heads are behavioral (reads labelled at fixed witness states),
    so this routes the behaviorally-relevant operands — exactly what T2's own
    binding targets supervise, and what value-numbering needs (a non-behavioral
    read can't change the output, so dropping it is correct).

    Equivariance is preserved EXACTLY: in_score/out_score/live heads are
    equivariant per-slot readouts of T, so an argmax/topk over the register
    axis permutes with a rename and the counts are rename-invariant — the
    derived wiring permutes with π, keeping T2 exactly equivariant.

    T: (N, n_regs, d_slot) frozen-T1 per-instruction state. Returns three
    (N,) long slot-index tensors. x0 (=0) stands for an absent operand."""
    n_regs = t1.n_regs
    device = T.device
    in_scores = t1.in_score_head(T).squeeze(-1)             # (N, n_regs)
    out_scores = t1.out_score_head(T).squeeze(-1)
    live_in = torch.sigmoid(t1.live_in_head(T).squeeze(-1))
    live_out = torch.sigmoid(t1.live_out_head(T).squeeze(-1))
    # x0 is never a GP operand/output, and a register the instruction never
    # touched (an all-zero slot) is never one either — every such slot has the
    # same score, and a tie among them would be broken by index, which a
    # rename does not permute. Only touched, non-x0 slots are candidates.
    not_x0 = torch.ones(n_regs, device=device)
    not_x0[X0_SLOT] = 0.0
    candidate = (T != 0).any(-1) & (not_x0 > 0)             # (N, n_regs)
    neg = torch.finfo(in_scores.dtype).min
    in_s = in_scores.masked_fill(~candidate, neg)
    out_s = out_scores.masked_fill(~candidate, neg)
    n_in = ((live_in > 0.5) * candidate).sum(-1)            # (N,)
    n_out = ((live_out > 0.5) * candidate).sum(-1)
    top2 = in_s.topk(2, dim=-1)                             # in0 then in1
    out_top = out_s.max(dim=-1)
    z = torch.zeros_like(out_top.indices)
    in0 = torch.where((n_in >= 1) & (top2.values[:, 0] > neg), top2.indices[:, 0], z)
    in1 = torch.where((n_in >= 2) & (top2.values[:, 1] > neg), top2.indices[:, 1], z)
    out = torch.where((n_out >= 1) & (out_top.values > neg), out_top.indices, z)
    return in0, in1, out


def _dense_chunk_grid(split_outputs, n_chunks, max_n_instrs):
    """Scatter the ragged per-instruction split into a FIXED-shape
    (n_chunks, max_n_instrs, ...) dense grid, so the frozen-T1 encode +
    predicted-binding routing can run INSIDE the single compiled CUDA-graph step
    instead of as eager, variable-shape, per-step CPU dispatch (the throughput fold).

    Padding slots (a chunk's missing instructions) are PAD-filled with
    active=0. T1 still encodes them, but they scatter to inactive T2 slots
    where the core's write/event are gated to zero — so every ACTIVE slot is
    bit-identical to the ragged path; only discarded padding work is added.

    Returns numpy: tokens (B, MNI, 9) int64, pad (B, MNI, 9) bool,
    in0/in1/out (B, MNI) int64 (token-decoded wiring; feeds T1 ingestion),
    active (B, MNI) float32."""
    (instr_tokens, instr_pad, chunk_idx, slot_idx, n_per_chunk,
     in0, in1, out) = split_outputs
    B, MNI = n_chunks, max_n_instrs
    tok = np.full((B, MNI, _MAX_INSTR_TOKENS), PAD, dtype=np.int64)
    pad = np.ones((B, MNI, _MAX_INSTR_TOKENS), dtype=bool)
    g0 = np.zeros((B, MNI), dtype=np.int64)
    g1 = np.zeros((B, MNI), dtype=np.int64)
    go = np.zeros((B, MNI), dtype=np.int64)
    ci, si = chunk_idx, slot_idx
    tok[ci, si] = instr_tokens.astype(np.int64)
    pad[ci, si] = instr_pad
    g0[ci, si] = in0
    g1[ci, si] = in1
    go[ci, si] = out
    active = (np.arange(MNI)[None, :] < n_per_chunk[:, None]).astype(np.float32)
    return tok, pad, g0, g1, go, active


def _t2_assemble(t1, split_outputs, n_chunks, max_n_instrs, device,
                 t1_encode=None, route='binding'):
    """Eager (dynamic-shape) frozen-T1 encode + scatter into fixed per-chunk
    tensors for the T2 state machine. Returns:
      chunk_t1   (n_chunks, max_n_instrs, d_t1)  — per-instr T1 essence (content)
      chunk_in0/in1/out (n_chunks, max_n_instrs) long — per-instr wiring
      chunk_active (n_chunks, max_n_instrs) float — 1 where an instr sits
    Frozen-T1 essence is tag-invariant, so any in-distribution tag gives the
    same content — we use a fresh random tag here purely to stay in
    distribution. Padded slots route to x0 with active=0 (no-op in the core).

    route: 'binding' (default) derives T2's per-instruction wiring from
    T1's PREDICTED binding (_t1_predicted_wiring) — the tier-recursion path.
    'tokens' uses the token-decoded wiring carried in split_outputs (the old
    diagnostic path) — kept ONLY as a diagnostic A/B, not a valid recursion result.
    Either way the wiring fed to T1 itself is token-derived (T1 ingesting its
    own instruction); only what flows to T2 changes.

    t1_encode: optional compiled `t1.encode_state` (the register-state machine
    has many small ops; eager per-step launch of them over ~all-chunk
    instructions was the T2 trainer's CPU bottleneck — the trainer passes a
    compiled, dynamic-shape version so they fuse). Must return (T, essence)."""
    (instr_tokens, instr_pad, chunk_idx, slot_idx, n_per_chunk,
     in0, in1, out) = split_outputs
    enc = t1_encode if t1_encode is not None else t1.encode_state
    it = torch.from_numpy(instr_tokens).to(device).long()
    ip = torch.from_numpy(instr_pad).to(device)
    t1_in0 = torch.from_numpy(in0).to(device)
    t1_in1 = torch.from_numpy(in1).to(device)
    t1_out = torch.from_numpy(out).to(device)
    with torch.no_grad():
        # T1 runs on its OWN token-derived wiring (ingestion); it emits state
        # T and essence.
        T1_state, t1_ess = enc(it, ip, t1_in0, t1_in1, t1_out)
    if route == 'binding':
        r_in0, r_in1, r_out = _t1_predicted_wiring(t1, T1_state)
    else:                                    # 'tokens' diagnostic
        r_in0, r_in1, r_out = t1_in0, t1_in1, t1_out
    ci = torch.from_numpy(chunk_idx).to(device)
    si = torch.from_numpy(slot_idx).to(device)
    chunk_t1 = torch.zeros(n_chunks, max_n_instrs, t1_ess.shape[-1], device=device)
    chunk_t1[ci, si] = t1_ess
    chunk_in0 = torch.zeros(n_chunks, max_n_instrs, dtype=torch.long, device=device)
    chunk_in1 = torch.zeros(n_chunks, max_n_instrs, dtype=torch.long, device=device)
    chunk_out = torch.zeros(n_chunks, max_n_instrs, dtype=torch.long, device=device)
    chunk_in0[ci, si] = r_in0
    chunk_in1[ci, si] = r_in1
    chunk_out[ci, si] = r_out
    npc = torch.from_numpy(n_per_chunk).to(device)
    chunk_active = (torch.arange(max_n_instrs, device=device)[None, :]
                    < npc[:, None]).float()
    return chunk_t1, chunk_in0, chunk_in1, chunk_out, chunk_active


def t2_chunk_forward(t1, t2, split_outputs, n_chunks, max_n_instrs, device,
                     route='binding'):
    """Frozen-T1 per-instruction essence → T2 register-state machine over the
    chunk → essence. The whole forward, for eval / non-compiled callers (the
    trainer splits the eager assemble from the compiled core). Returns the
    chunk essence (n_chunks, d_out), or None for an empty batch."""
    if split_outputs[0].shape[0] == 0:
        return None
    chunk_t1, in0, in1, out, active = _t2_assemble(
        t1, split_outputs, n_chunks, max_n_instrs, device, route=route)
    return t2.encode(chunk_t1, in0, in1, out, active)


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


def binding_losses(model, T, essence, *, live_in_t, live_out_t, pc_writes_t,
                   in_slot_t, out_slot_t, in_k_eff=None, out_k_eff=None):
    """Rename-equivariant register-binding losses, shared by T1 and T2.

    T1 is the n_instrs=1 special case of T2: both emit one register-indexed
    state `T` (B, n_regs, d_slot) plus a rename-invariant `essence`. The
    binding heads are per-slot readouts of `T` — `head(T) -> (B, n_regs, 1)`
    — so a register rename permutes the scores identically (equivariant by
    construction, duplicate-free). `pc_writes` reads `essence` (PC effects
    are opcode-determined, rename-invariant). BCE on live_in/out + pc_writes;
    ListMLE (Plackett-Luce) on the in/out slot orderings. Rows with all-zero
    masks (invalid / mem-op) are excluded.

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

    li_logits = model.live_in_head(T).squeeze(-1)        # (B, n_regs)
    lo_logits = model.live_out_head(T).squeeze(-1)
    # pc reads the essence DIRECTION only (magnitude = validity); binding heads
    # above read per-slot T, which is off the essence-magnitude axis entirely.
    pc_logits = model.pc_writes_head(F.normalize(essence, dim=-1)).squeeze(-1)

    li_per = F.binary_cross_entropy_with_logits(
        li_logits, live_in_t, reduction='none')
    li_loss = (li_per * active_f.unsqueeze(-1)).sum() / (n_active * n_regs)
    lo_per = F.binary_cross_entropy_with_logits(
        lo_logits, live_out_t, reduction='none')
    lo_loss = (lo_per * active_f.unsqueeze(-1)).sum() / (n_active * n_regs)
    pc_per = F.binary_cross_entropy_with_logits(
        pc_logits, pc_writes_t, reduction='none')
    pc_loss = (pc_per * active_f).sum() / n_active

    # An instruction reads two slots and writes one, so a chunk touches at most
    # in_k_eff + out_k_eff of them; the in-score MLP need not see the rest.
    in_scores = (apply_head_touched(model.in_score_head, T, in_k_eff + out_k_eff)
                 if in_k_eff is not None and out_k_eff is not None
                 else model.in_score_head(T)).squeeze(-1)   # (B, n_regs)
    out_scores = model.out_score_head(T).squeeze(-1)
    in_slot_loss = _listmle_loss(
        in_scores, in_slot_t, model.max_input_slots, active_f, k_eff=in_k_eff)
    out_slot_loss = _listmle_loss(
        out_scores, out_slot_t, model.max_output_slots, active_f, k_eff=out_k_eff)

    return {
        'live_in': li_loss, 'live_out': lo_loss, 'pc_writes': pc_loss,
        'in_slot': in_slot_loss, 'out_slot': out_slot_loss,
        'active_f': active_f,
    }


def _compute_chunk_out_regs(batch):
    """Per row of the batch, execute the chunk on the emulator from the
    register files the row ships as its anchors. Returns:
      out_regs: (B, n_anchors, 32) int32
      mask:     (B,) bool — True for rows we got valid out_regs from
    Rows that fail (decode error, memory ops, etc.) get zero out_regs. The
    shipped `out_regs` must equal this: the anchors shipped are the register
    files the oracle executed from, with no further placement. The oracle's
    SSA evaluator is used rather than the emulator because the two differ on
    intra-chunk control flow (`_eval_ssa_numpy`)."""
    from datagen.compare import to_ssa, live_nodes, _eval_ssa_numpy
    B = batch.tokens.shape[0]
    n_anchors = batch.anchors.shape[1]
    out = np.zeros((B, n_anchors, 32), dtype=np.int32)
    mask = np.zeros(B, dtype=bool)
    for b in range(B):
        instrs = _decode_chunk_instructions(batch, b)
        if instrs is None or not batch.out_regs_valid[b]:
            continue
        try:
            ssa = to_ssa(instrs)
            out[b], _ = _eval_ssa_numpy(ssa, len(instrs), batch.anchors[b],
                                        live_nodes(ssa))
        except Exception:
            continue
        mask[b] = True
    return out, mask


_PR_SET_PDEATHSIG = 1


def _mp_worker_main(in_q, out_q, parent_pid, pack, n_slots):
    """Worker-process entry point. Re-imports needed modules (we're under
    spawn-context, so a fresh interpreter). Reads Batches from in_q,
    runs CPU-side prep (_split_to_per_instruction), sends results on
    out_q. EOF signaled by None.
    """
    # `daemon=True` only reaps the worker on a CLEAN parent exit, so a killed
    # trainer left it running forever. Have the kernel SIGKILL it with the
    # parent instead, and bail if the parent is already gone.
    try:
        ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_PDEATHSIG, signal.SIGKILL)
    except Exception:
        pass
    if os.getppid() != parent_pid:
        os._exit(0)
    # Imports must happen inside the worker (spawn context = fresh
    # interpreter, doesn't inherit parent's already-loaded modules).
    from compressor.train import _pack_step_arrays, _split_to_per_instruction
    # Packed steps go through a ring of slots in a /dev/shm file rather than
    # through the queue: unpickling ~0.7 MB per step cost the trainer's main
    # thread more than packing saved. The queue carries only the slot index.
    ring, k = None, 0
    try:
        while True:
            batch = in_q.get()
            if batch is None:
                break
            packed, specs, n_instrs, shape = _pack_step_arrays(
                batch, _split_to_per_instruction(batch), **pack)
            if packed is None:
                out_q.put((None, None, specs, n_instrs, shape))
                continue
            if ring is None:
                path = f'/dev/shm/rvt-prefetch-{os.getpid()}'
                ring = np.memmap(path, dtype=np.uint8, mode='w+',
                                 shape=(n_slots, packed.size))
            ring[k] = packed
            out_q.put(((path, ring.shape), k, specs, n_instrs, shape))
            k = (k + 1) % n_slots
    finally:
        out_q.put(None)


def _remove_orphan_rings():
    """Delete prefetch ring files whose worker is gone. The consumer unlinks a
    ring as soon as it maps it, so one survives only if the trainer died first."""
    import glob
    for path in glob.glob('/dev/shm/rvt-prefetch-*'):
        try:
            os.kill(int(path.rsplit('-', 1)[1]), 0)
        except ProcessLookupError:
            os.unlink(path)
        except (ValueError, PermissionError, FileNotFoundError):
            pass


class TrainBatchPrefetcher:
    """Multiprocessing prefetcher: CPU-side prep — the per-instruction split and
    packing a step's arrays for the device (`_pack_step_arrays`, configured by
    `pack`) — runs in `n_workers` spawned worker *processes*, in true parallel
    with the main loop's GPU work.

    Architecture:
      - Reader thread (main process, daemon): pulls Batches from batch_iter
        (the RVT stream) and deals batch i to worker i % n_workers.
      - Worker processes (spawn context, fresh interpreters): split and pack
        each Batch and push the packed step to their own output queue.
      - Main loop: reads the output queues in the same round-robin order, so
        it sees exactly the stream's batch order whatever the worker count.

    Processes, not threads: the split is GIL-bound Python, and a thread fought
    the main loop for the GIL. One worker delivered ~240 batches/s, about what
    the GPU step consumes; four deliver about twice that. Spawn (not fork)
    keeps CUDA state out of the workers.
    """

    _EOF = None  # sentinel

    # Deep enough to ride out the multinode stream's lulls: at 4 batches, a gap
    # of tens of ms in delivery idled the GPU; 128 does not.
    def __init__(self, batch_iter, pack, maxsize=128, n_workers=4):
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        depth = max(1, maxsize // n_workers)
        # A worker can run at most `depth` queued items plus the two the
        # consumer holds (a peeked first item and the one being prepped) ahead
        # of the consumer, so a slot is never rewritten before it is copied.
        n_slots = depth + 4
        _remove_orphan_rings()
        self._in_qs = [ctx.Queue(maxsize=depth) for _ in range(n_workers)]
        self._out_qs = [ctx.Queue(maxsize=depth) for _ in range(n_workers)]
        self._rings = [None] * n_workers
        self._workers = [
            ctx.Process(target=_mp_worker_main,
                        args=(iq, oq, os.getpid(), pack, n_slots), daemon=True)
            for iq, oq in zip(self._in_qs, self._out_qs)]
        for w in self._workers:
            w.start()
        self._next = 0
        self._reader_thread = threading.Thread(
            target=self._read_loop, args=(batch_iter,), daemon=True)
        self._reader_thread.start()

    def _read_loop(self, batch_iter):
        n = len(self._in_qs)
        try:
            for i, batch in enumerate(batch_iter):
                self._in_qs[i % n].put(batch)
        finally:
            for q in self._in_qs:
                q.put(self._EOF)

    def __iter__(self):
        return self

    def __next__(self):
        """(packed step, specs, n_instrs, shape). The packed step is a view of
        a worker's ring slot: copy it before fetching the next item."""
        w = self._next
        item = self._out_qs[w].get()
        if item is self._EOF:
            for p in self._workers:
                p.join(timeout=10)
            raise StopIteration
        self._next = (w + 1) % len(self._out_qs)
        ring, slot, specs, n_instrs, shape = item
        if ring is None:
            return None, specs, n_instrs, shape
        if self._rings[w] is None:
            path, ring_shape = ring
            self._rings[w] = np.memmap(path, dtype=np.uint8, mode='r',
                                       shape=ring_shape)
            os.unlink(path)            # both ends are mapped; no file can leak
        return self._rings[w][slot], specs, n_instrs, shape


def train_t2_encoder(batch_iter, t1_encoder, *,
                     d_out=256, d_event=16, op_cell_depth=1, vp_head_depth=12,
                     n_pool_heads=1,
                     lr=3e-4, n_steps=None, log_every=100, lr_min=1e-6,
                     warmup_steps=0,
                     valid_weight=0.0,
                     live_in_weight=0.1, live_out_weight=0.1,
                     pc_writes_weight=0.1,
                     in_slot_weight=0.1, out_slot_weight=0.1,
                     value_predict_weight=0.0, vp_smooth_weight=0.2,
                     t2_checkpoint=None, t2_route='binding',
                     device='auto',
                     on_log=None, save_every=10000, compile_step=True):
    """T2 preset over `train_compressor` — frozen-T1 vector front-end, wiring
    routed off T1's predicted binding (t2_route). Same execution path
    as T1; only the content provider + hparams differ."""
    return train_compressor(
        batch_iter, ingest='vectors', lower_encoder=t1_encoder,
        d_out=d_out, d_event=d_event, op_cell_depth=op_cell_depth,
        vp_head_depth=vp_head_depth,
        n_pool_heads=n_pool_heads, route=t2_route,
        lr=lr, n_steps=n_steps, log_every=log_every, lr_min=lr_min,
        warmup_steps=warmup_steps,
        valid_weight=valid_weight, live_in_weight=live_in_weight,
        live_out_weight=live_out_weight, pc_writes_weight=pc_writes_weight,
        in_slot_weight=in_slot_weight, out_slot_weight=out_slot_weight,
        value_predict_weight=value_predict_weight,
        vp_smooth_weight=vp_smooth_weight, resume=t2_checkpoint,
        device=device, on_log=on_log, save_every=save_every,
        compile_step=compile_step)

