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
from emulator import Instruction
from datagen.batch import padding_mask
from datagen.generate import MANIFEST, sample_binding, materialize
from .model import T1Compressor


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


def train_encoder(batch_iter, *,
                  d_model=128, n_heads=4, n_layers=2, d_out=128,
                  max_window=72,
                  lr=3e-4, n_steps=None, log_every=100, lr_min=1e-6,
                  pair_weight=1.0, valid_weight=0.1, equiv_weight=0.0,
                  equiv_seed=0, device='auto'):
    """Train a T1 encoder on RVT batches.

    pair_weight:  weight on (||v_i - v_j|| - distance)**2 over pair_indices.
    valid_weight: weight on (||v_c|| - (1 if valid else 0))**2 over rows.
    equiv_weight: optional add-on equivalence_loss (direction-only collapse
                  on MANIFEST canonical templates). 0 disables.

    Returns (encoder, losses).
      losses: list[dict] with keys total/pair/valid/equiv plus probe_*.
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

        # Pair-MSE.
        if batch.pair_indices.shape[0] > 0:
            ij = torch.from_numpy(
                batch.pair_indices.astype(np.int64)).to(device)
            d_target = torch.from_numpy(batch.distances).to(device)
            d_pred = (vecs[ij[:, 0]] - vecs[ij[:, 1]]).norm(dim=-1)
            pair_loss = ((d_pred - d_target) ** 2).mean()
        else:
            pair_loss = torch.tensor(0.0, device=device)

        # Validity magnitude.
        target_mag = valid.float()
        actual_mag = vecs.norm(dim=-1)
        valid_loss = ((actual_mag - target_mag) ** 2).mean()

        # Optional equivalence collapse.
        if equiv_weight > 0:
            eq_loss = equivalence_loss(encoder, device, eq_rng)
        else:
            eq_loss = torch.tensor(0.0, device=device)

        total = (pair_weight * pair_loss
                 + valid_weight * valid_loss
                 + equiv_weight * eq_loss)
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
            'pair': float(pair_loss),
            'valid': float(valid_loss),
            'equiv': float(eq_loss),
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
                f'(pair {record["pair"]:.4f} valid {record["valid"]:.4f}'
                f' eq {record["equiv"]:.4f})  '
                f'lr {lr_now:.1e}  {ms_per_step:.0f}ms/step  '
                f'eta {timedelta(seconds=int(eta))}  '
                f'probes[aa={probes["aa"]:.3f} canon={probes["canon"]:.3f} '
                f'ss={probes["ss"]:.3f} comm={probes["comm"]:.3f}]')
        losses.append(record)

    return encoder, losses
