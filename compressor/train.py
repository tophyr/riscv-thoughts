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
from .model import (
    Compressor, T1Compressor, T2Compressor, Decoder, instruction_wiring,
    X0_SLOT)


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


def _wiring_from_token_lists(token_lists):
    """Per token sequence, the GP wiring (in0, in1, out) for the state
    machine. Undecodable windows (partial / bogus) get (0, 0, 0). Three
    (N,) int64 arrays."""
    n = len(token_lists)
    in0 = np.zeros(n, dtype=np.int64)
    in1 = np.zeros(n, dtype=np.int64)
    out = np.zeros(n, dtype=np.int64)
    for i, toks in enumerate(token_lists):
        try:
            instr, _ = decode_instruction(list(toks), 0)
        except Exception:
            continue
        in0[i], in1[i], out[i] = instruction_wiring(instr)
    # decode_instruction doesn't validate token classes, so a malformed
    # window can yield out-of-range register ids — clamp those to x0 (no-op).
    for a in (in0, in1, out):
        a[(a < 0) | (a >= N_REGS)] = 0
    return in0, in1, out


def encode_instrs(model, instrs, device, tags=None):
    """Tokenize, pad, and encode a list of Instructions to their essence
    vectors. Registers enter as wiring (instruction_wiring); a SHARED per-
    register tag (broadcast across the instrs) is used by default so the
    essence comparison isolates the operator signal (equivalence / probe
    geometry), matching how a trained model becomes tag-invariant."""
    encoded = [encode_instruction(i) for i in instrs]
    max_len = max(len(e) for e in encoded)
    n = len(encoded)
    tok = np.full((n, max_len), PAD, dtype=np.int64)
    pad = np.ones((n, max_len), dtype=np.bool_)
    for i, e in enumerate(encoded):
        tok[i, :len(e)] = e
        pad[i, :len(e)] = False
    wiring = np.array([instruction_wiring(i) for i in instrs], dtype=np.int64)
    if tags is None:
        shared = np.random.standard_normal(model.n_regs).astype(np.float32)
        shared[0] = 0.0
        tags = np.broadcast_to(shared, (n, model.n_regs)).copy()
    return model.encode(
        torch.from_numpy(tok).to(device),
        torch.from_numpy(pad).to(device),
        torch.from_numpy(wiring[:, 0]).to(device),
        torch.from_numpy(wiring[:, 1]).to(device),
        torch.from_numpy(wiring[:, 2]).to(device),
        torch.as_tensor(tags, dtype=torch.float32, device=device))


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
                     d_out=128, d_event=16,
                     d_model=128, n_heads=4, n_layers=2, max_window=72,
                     route='binding',
                     lr=3e-4, n_steps=None, log_every=100, lr_min=1e-6,
                     warmup_steps=0,
                     valid_weight=0.1, live_in_weight=0.1, live_out_weight=0.1,
                     pc_writes_weight=0.1, in_slot_weight=0.1,
                     out_slot_weight=0.1, value_predict_weight=1.0,
                     anchor_seed=0, n_anchor_states=8,
                     resume=None, device='auto', on_log=None,
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
    live_out, pc_writes, in_slot, out_slot, value_pred, pair, n_chunks,
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
            d_content=lower_encoder.d_out, d_out=d_out, d_event=d_event)
        model = comp.to(device)
        model_file = 't2.pt'
    else:
        model = T1Compressor(
            vocab_size=VOCAB_SIZE, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, max_window=max_window, d_out=d_out,
            d_event=d_event).to(device)
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
    anchor_states_t = torch.from_numpy(
        make_anchor_states(n_anchor_states, anchor_seed)).to(device)

    def _fmt(r, ms, eta, lr_now):
        return (
            f'step {r["step"]:>5d}  loss {r["total"]:.4f} '
            f'(val {r["valid"]:.4f} li {r["live_in"]:.3f} lo {r["live_out"]:.3f} '
            f'pc {r["pc_writes"]:.3f} is {r["in_slot"]:.3f} os {r["out_slot"]:.3f} '
            f'vp {r["value_pred"]:.4f}) '
            f'B={r["n_chunks"]} I={r["n_instrs"]}  '
            f'lr {lr_now:.1e}  {ms:.0f}ms/step  '
            f'eta {timedelta(seconds=int(eta))}')

    log = TrainLog(n_steps=n_steps, log_every=log_every, lr=lr,
                   scheduler=scheduler, formatter=_fmt, on_log=on_log)
    step = 0
    if resume_train_state is not None:
        step = restore_train_state(resume_train_state, opt, scheduler)
        print(f'Resumed from step {step}')

    # Prefetcher overlaps CPU-side split with GPU work — shared by both tiers.
    batch_source = TrainBatchPrefetcher(batch_iter)
    first, batch_source = _peek(batch_source)
    if first is None:
        return model, log.losses
    N_CHUNKS, MAX_NI = first[0].instr_lens.shape
    # CONSTANT ListMLE loop bounds (corpus-constant, so the compiled step never
    # recompiles on a varying slot width). A K-instr chunk reads <=2K regs,
    # writes <=K; iterating extra all-IGNORE columns is a no-op on the loss.
    IN_K = min(2 * MAX_NI, MAX_INPUT_SLOTS)
    OUT_K = min(MAX_NI, MAX_OUTPUT_SLOTS)
    vp_active = value_predict_weight > 0
    out_regs_active = vp_active
    valid_active = valid_weight > 0
    model.train()
    _dummy = torch.zeros(1, device=device)

    def _content(tok, pad, w0, w1, wo, active, t1_tags):
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
                w0.reshape(flat), w1.reshape(flat), wo.reshape(flat), t1_tags)
            if route_binding:
                r0, r1, ro = _t1_predicted_wiring(lower_encoder, T1s)
            else:
                r0, r1, ro = w0.reshape(flat), w1.reshape(flat), wo.reshape(flat)
        return (ess.reshape(B, MNI, -1),
                r0.reshape(B, MNI), r1.reshape(B, MNI), ro.reshape(B, MNI))

    # ONE compiled CUDA-graph step (the one-graph fold): front-end + core + losses, all
    # at a fixed-padded instruction count (B*MNI). Host-sync-free, static-shape.
    def _fwd_loss(tok, pad, w0, w1, wo, active, t1_tags, t2_tags,
                  li, lo, pc, in_slot, out_slot, out_regs, vp_mask, valid_f,
                  ike, oke):
        content_seq, c0, c1, co = _content(tok, pad, w0, w1, wo, active, t1_tags)
        T, essence = comp.encode_state(content_seq, c0, c1, co, active, t2_tags)
        valid_loss = (((essence.norm(dim=-1) - valid_f) ** 2).mean()
                      if valid_active else essence.new_zeros(()))
        bl = binding_losses(
            comp, T, essence, live_in_t=li, live_out_t=lo, pc_writes_t=pc,
            in_slot_t=in_slot, out_slot_t=out_slot, in_k_eff=ike, out_k_eff=oke)
        vp = (t2_value_predict_loss(
                  comp, essence, anchor_states_t, in_slot, out_slot,
                  out_regs, vp_mask)
              if vp_active else essence.new_zeros(()))
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
        batch, split_outputs = item
        if split_outputs[0].shape[0] == 0:        # all-invalid batch
            return None
        tok, pad, gi0, gi1, go, active = _dense_chunk_grid(
            split_outputs, N_CHUNKS, MAX_NI)
        t1_tags = torch.randn(N_CHUNKS * MAX_NI, model.n_regs, device=device)
        t2_tags = torch.randn(N_CHUNKS, model.n_regs, device=device)
        valid_f = (_h2d(batch.valid, device, dtype=torch.float32)
                   if valid_active else _dummy)
        if out_regs_active:
            out_regs_t = _h2d(batch.out_regs, device)
            vp_mask_t = _h2d(batch.out_regs_valid, device)
        else:
            out_regs_t = vp_mask_t = _dummy
        return (_h2d(tok, device, dtype=torch.long), _h2d(pad, device),
                _h2d(gi0, device, dtype=torch.long),
                _h2d(gi1, device, dtype=torch.long),
                _h2d(go, device, dtype=torch.long),
                _h2d(active, device, dtype=torch.float32),
                t1_tags, t2_tags,
                _h2d(batch.live_in_mask, device, dtype=torch.float32),
                _h2d(batch.live_out_mask, device, dtype=torch.float32),
                _h2d(batch.pc_writes, device, dtype=torch.float32),
                _h2d(batch.in_slot_regs, device, dtype=torch.long),
                _h2d(batch.out_slot_regs, device, dtype=torch.long),
                out_regs_t, vp_mask_t, valid_f, IN_K, OUT_K)

    def _extra(item, aux):
        n_per_chunk = item[1][4]
        return {'n_chunks': N_CHUNKS, 'n_instrs': int(n_per_chunk.sum())}

    losses = run_train_loop(
        batch_source, model=model, opt=opt, scheduler=scheduler, log=log,
        device=device, prep_fn=_prep, fwd_loss_fn=_fwd_loss, extra_fn=_extra,
        compile_step=compile_step, capture_state=True, start_step=step)
    return model, losses


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
        lr=lr, n_steps=n_steps, log_every=log_every, lr_min=lr_min,
        valid_weight=valid_weight, live_in_weight=live_in_weight,
        live_out_weight=live_out_weight, pc_writes_weight=pc_writes_weight,
        in_slot_weight=in_slot_weight, out_slot_weight=out_slot_weight,
        value_predict_weight=value_predict_weight,
        anchor_seed=anchor_seed, n_anchor_states=n_anchor_states,
        device=device, on_log=on_log, compile_step=compile_step)


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

    T1's binding heads are behavioral (magnitude-filtered live sets), so this
    routes the behaviorally-relevant operands — exactly what T2's own binding
    targets supervise, and what value-numbering needs (a non-behavioral read
    can't change the output, so dropping it is correct).

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
    # x0 is never a GP operand/output — take it out of contention.
    not_x0 = torch.ones(n_regs, device=device)
    not_x0[X0_SLOT] = 0.0
    neg = torch.finfo(in_scores.dtype).min
    in_s = in_scores.masked_fill(not_x0 == 0, neg)
    out_s = out_scores.masked_fill(not_x0 == 0, neg)
    n_in = ((live_in > 0.5) * not_x0).sum(-1)               # (N,)
    n_out = ((live_out > 0.5) * not_x0).sum(-1)
    top2 = in_s.topk(2, dim=-1).indices                     # (N, 2): in0 then in1
    out_top = out_s.argmax(dim=-1)                          # (N,)
    z = torch.zeros_like(out_top)
    in0 = torch.where(n_in >= 1, top2[:, 0], z)
    in1 = torch.where(n_in >= 2, top2[:, 1], z)
    out = torch.where(n_out >= 1, out_top, z)
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
    t1_tags = torch.randn(it.shape[0], t1.n_regs, device=device)
    with torch.no_grad():
        # T1 runs on its OWN token-derived wiring (ingestion); it emits state
        # T and essence.
        T1_state, t1_ess = enc(it, ip, t1_in0, t1_in1, t1_out, t1_tags)
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
                     t2_tags=None, route='binding'):
    """Frozen-T1 per-instruction essence → T2 register-state machine over the
    chunk → essence. The whole forward, for eval / non-compiled callers (the
    trainer splits the eager assemble from the compiled core). Returns the
    chunk essence (n_chunks, d_out), or None for an empty batch."""
    if split_outputs[0].shape[0] == 0:
        return None
    chunk_t1, in0, in1, out, active = _t2_assemble(
        t1, split_outputs, n_chunks, max_n_instrs, device, route=route)
    if t2_tags is None:
        t2_tags = torch.randn(n_chunks, t2.n_regs, device=device)
    return t2.encode(chunk_t1, in0, in1, out, active, t2_tags)


def t2_value_predict_loss(t2, vecs, anchor_states_t,
                          in_slot_regs_t, out_slot_regs_t,
                          out_regs_t, mask_t, return_count=False):
    """Per-anchor, per-out-slot value-prediction MSE (canonical).

    The head reads (vec, CANONICAL input-slot values) and predicts per-anchor
    per-slot output values. Both the inputs fed and the target read are in the
    canonical frame: input slot k is fed anchor[:, k+1] (the canonical position
    precompute_chunk sourced the k-th input from) and the target is out_regs
    (the canonical post-execution register file). So the task is "given the
    canonical input values, predict the canonical output values" — a
    self-consistent, register-name-free I/O function the essence must encode.

    Args:
      t2:               T2Compressor (uses t2.vp_head)
      vecs:             (B, d) — T2 output vector
      anchor_states_t:  (A, 32) int32 — register file per anchor (pre-exec)
      in_slot_regs_t:   (B, MAX_INPUT_SLOTS) int64 — input slot register IDs;
                        used ONLY as a fill mask (which slots carry an input).
                        The values fed are canonical (anchor[:, k+1]), not the
                        actual registers' values.
      out_slot_regs_t:  (B, MAX_OUTPUT_SLOTS) int64 — output slot register IDs
                        (AUX_CE_IGNORE for unused slots)
      out_regs_t:       (B, A, 32) int32 — register file per anchor AFTER
                        executing each chunk
      mask_t:           (B,) bool — rows to include (decoded successfully)

    Returns scalar MSE over (row, output-slot, anchor) triplets where
    out_slot_regs is not IGNORE.
    """
    n_anchors = anchor_states_t.shape[0]
    max_in = in_slot_regs_t.shape[1]

    # CANONICAL input values. `out_regs` is the canonical baseline:
    # precompute_chunk relocates the chunk's i-th first-read input to canonical
    # position i+1 (_canonical_state) and executes there, so out_regs depends
    # only on operand STRUCTURE, not register names. The vp head must therefore
    # see the canonical-position values anchor[:, k+1] for input slot k — NOT
    # the raw value at the slot's actual register (the old bug, which fed
    # anchor[:, in_slot_regs] and asked the head to predict f(canon inputs) from
    # raw inputs = uncorrelated numbers). Canonical also leaks no register
    # identity into vp, which is the right choice for an equivariant encoder.
    # Unfilled slots (AUX_CE_IGNORE) feed 0 = "no input".
    # (Caveat: canonical positions are assigned to all live syntactic inputs,
    # while in_slot_regs lists only the behavioral subset; the two coincide for
    # T1 and ~all T2 chunks, diverging only for degenerate ops with a live-but-
    # value-irrelevant input, e.g. ANDI _,_,0 — rare, accepted.)
    device = anchor_states_t.device
    canon_pos = torch.arange(1, max_in + 1, device=device).clamp(max=N_REGS - 1)
    in_filled = (in_slot_regs_t != AUX_CE_IGNORE).float()       # (B, max_in)
    # (A, max_in) canonical values, broadcast over rows, zeroed at unfilled
    # slots -> (B, A, max_in).
    in_vals = (anchor_states_t[:, canon_pos].float().unsqueeze(0)
               * in_filled.unsqueeze(1))

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
    pc_logits = model.pc_writes_head(essence).squeeze(-1)

    li_per = F.binary_cross_entropy_with_logits(
        li_logits, live_in_t, reduction='none')
    li_loss = (li_per * active_f.unsqueeze(-1)).sum() / (n_active * n_regs)
    lo_per = F.binary_cross_entropy_with_logits(
        lo_logits, live_out_t, reduction='none')
    lo_loss = (lo_per * active_f.unsqueeze(-1)).sum() / (n_active * n_regs)
    pc_per = F.binary_cross_entropy_with_logits(
        pc_logits, pc_writes_t, reduction='none')
    pc_loss = (pc_per * active_f).sum() / n_active

    in_scores = model.in_score_head(T).squeeze(-1)       # (B, n_regs)
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


def _mp_worker_main(in_q, out_q):
    """Worker-process entry point. Re-imports needed modules (we're under
    spawn-context, so a fresh interpreter). Reads Batches from in_q,
    runs CPU-side prep (_split_to_per_instruction), sends results on
    out_q. EOF signaled by None.
    """
    # Imports must happen inside the worker (spawn context = fresh
    # interpreter, doesn't inherit parent's already-loaded modules).
    from compressor.train import _split_to_per_instruction
    try:
        while True:
            batch = in_q.get()
            if batch is None:
                break
            out_q.put((batch, _split_to_per_instruction(batch)))
    finally:
        out_q.put(None)


class TrainBatchPrefetcher:
    """Multiprocessing prefetcher: spawn a worker *process* for CPU-side
    prep so it runs in true parallel with the main loop's GPU work (no
    GIL contention).

    Architecture:
      - Worker process (spawn context, fresh Python interpreter):
          pulls Batches from input queue, runs _split_to_per_instruction,
          pushes results to the output queue.
      - Reader thread (in main process, daemon):
          pulls Batches from batch_iter (the RVT stream), pushes to
          input queue. Decoupling main loop from stdin read.
      - Main loop: consumes prepped tuples from output queue.

    Why processes instead of threads: _split_to_per_instruction is
    pure-Python numpy reshaping and entirely GIL-bound. Threading caused
    net regression (worker fought main for GIL). Process boundary
    eliminates GIL contention entirely; cost is pickling each batch across
    the queue (numpy arrays in dataclass, ~1-2ms per batch).

    Spawn context (not fork) avoids any CUDA-state contamination since
    the worker is a fresh interpreter.
    """

    _EOF = None  # sentinel

    def __init__(self, batch_iter, maxsize=4):
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        self._in_q = ctx.Queue(maxsize=maxsize)
        self._out_q = ctx.Queue(maxsize=maxsize)
        self._worker = ctx.Process(
            target=_mp_worker_main,
            args=(self._in_q, self._out_q),
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
                     d_out=256, d_event=16,
                     lr=3e-4, n_steps=None, log_every=100, lr_min=1e-6,
                     warmup_steps=0,
                     valid_weight=0.0,
                     live_in_weight=0.1, live_out_weight=0.1,
                     pc_writes_weight=0.1,
                     in_slot_weight=0.1, out_slot_weight=0.1,
                     value_predict_weight=0.0,
                     anchor_seed=0, n_anchor_states=8,
                     t2_checkpoint=None, t2_route='binding',
                     device='auto',
                     on_log=None, compile_step=True):
    """T2 preset over `train_compressor` — frozen-T1 vector front-end, wiring
    routed off T1's predicted binding (t2_route). Same execution path
    as T1; only the content provider + hparams differ."""
    return train_compressor(
        batch_iter, ingest='vectors', lower_encoder=t1_encoder,
        d_out=d_out, d_event=d_event, route=t2_route,
        lr=lr, n_steps=n_steps, log_every=log_every, lr_min=lr_min,
        warmup_steps=warmup_steps,
        valid_weight=valid_weight, live_in_weight=live_in_weight,
        live_out_weight=live_out_weight, pc_writes_weight=pc_writes_weight,
        in_slot_weight=in_slot_weight, out_slot_weight=out_slot_weight,
        value_predict_weight=value_predict_weight,
        anchor_seed=anchor_seed,
        n_anchor_states=n_anchor_states, resume=t2_checkpoint,
        device=device, on_log=on_log, compile_step=compile_step)


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
