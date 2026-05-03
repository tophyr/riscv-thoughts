"""Diagnostic functions for the encoder + decoder.

Pure functions, no CLI. Importable from training loops (for periodic
mid-training evals) or from a test suite (for post-training validation
against acceptance thresholds). Each function returns a flat dict of
metrics; the caller decides what to log, print, or assert.

Coverage by plan step:
  Step 1 (encoder pair-MSE):
    - pair_distance_correlation  → Pearson > 0.85
    - equivalence_collapse       → intra/inter < 0.3 per class
    - validity_separation        → ‖T1‖ separates valid from invalid
  Step 2 (decoder reconstruction):
    - decoder_accuracy           → instr_acc > 0.95
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

from datagen.batch import padding_mask
from datagen.generate import (
    DEFAULT_DISTRIBUTION, _build_opcode_table,
    MANIFEST, sample_binding, materialize, random_instruction,
)
from datagen.invalidity import gen_partial, gen_spanning, gen_multi, gen_bogus
from emulator import batch_is_complete_instruction
from tokenizer import PAD, encode_instruction

from .train import encode_instrs, prepare_decoder_targets


# ===========================================================================
# Step 1 diagnostics
# ===========================================================================

@torch.no_grad()
def pair_distance_correlation(encoder, batches, *, device, max_batches=100):
    """Pearson/Spearman correlation between encoder pairwise distance
    and target distance over each batch's pre-computed pair structure.

    batches: iterable of datagen.batch.Batch (RVT). Pairs with target
             distance 0 (cluster siblings) and pairs with computed
             chunk_distance both contribute.

    Returns: {'pearson': r, 'spearman': r, 'n_pairs': int}.
    """
    encoder.eval()
    model_dists = []
    target_dists = []

    for i, batch in enumerate(batches):
        if i >= max_batches:
            break
        if batch.pair_indices.shape[0] == 0:
            continue
        tok = torch.from_numpy(batch.tokens).to(device)
        pad = torch.from_numpy(padding_mask(batch)).to(device)
        vecs = encoder.encode(tok, pad)
        ij = torch.from_numpy(batch.pair_indices.astype(np.int64)).to(device)
        d_pred = (vecs[ij[:, 0]] - vecs[ij[:, 1]]).norm(dim=-1)
        model_dists.append(d_pred.cpu().numpy())
        target_dists.append(batch.distances.copy())

    if not model_dists:
        return {'pearson': float('nan'), 'spearman': float('nan'),
                'n_pairs': 0}

    md = np.concatenate(model_dists)
    td = np.concatenate(target_dists)
    if md.std() < 1e-9 or td.std() < 1e-9:
        # Degenerate (all-zero distances) — correlation undefined.
        return {'pearson': float('nan'), 'spearman': float('nan'),
                'n_pairs': int(md.size)}
    pr, _ = stats.pearsonr(md, td)
    sr, _ = stats.spearmanr(md, td)
    return {'pearson': float(pr), 'spearman': float(sr),
            'n_pairs': int(md.size)}


@torch.no_grad()
def equivalence_collapse(encoder, *, device, n_samples=50, seed=42):
    """Per-MANIFEST-class intra-class vs canonical-vs-contrast distance
    on direction-normalized T1 vectors.

    Step 1's geometry success criterion: every class has intra/inter
    well below 1 (~0.3 = clear pass).

    Returns: {
        'classes':  {name: {'intra': float|None,
                            'inter': float|None,
                            'ratio': float|None}},
        'mean_ratio': float | None,   # over classes where ratio is defined
    }
    """
    encoder.eval()
    rng = np.random.default_rng(seed)
    out = {}
    ratios = []

    for klass in MANIFEST:
        intras, inters = [], []
        for _ in range(n_samples):
            binding = sample_binding(klass, rng)
            canon = [materialize(t, binding) for t in klass.canonical]
            contrast = [materialize(t, binding) for t in klass.contrast]
            instrs = canon + contrast
            vecs = F.normalize(encode_instrs(encoder, instrs, device), dim=-1)
            cv = vecs[:len(canon)]
            xv = vecs[len(canon):]
            if cv.shape[0] >= 2:
                d = torch.cdist(cv.unsqueeze(0), cv.unsqueeze(0)).squeeze(0)
                idx = torch.triu_indices(cv.shape[0], cv.shape[0],
                                         offset=1, device=device)
                intras.append(d[idx[0], idx[1]].mean().item())
            if xv.shape[0] > 0:
                d = torch.cdist(cv.unsqueeze(0), xv.unsqueeze(0)).squeeze(0)
                inters.append(d.mean().item())

        intra = float(np.mean(intras)) if intras else None
        inter = float(np.mean(inters)) if inters else None
        ratio = (intra / inter) if (intra is not None
                                    and inter is not None
                                    and inter > 1e-6) else None
        out[klass.name] = {'intra': intra, 'inter': inter, 'ratio': ratio}
        if ratio is not None:
            ratios.append(ratio)

    return {
        'classes': out,
        'mean_ratio': float(np.mean(ratios)) if ratios else None,
    }


@torch.no_grad()
def validity_separation(encoder, *, device, n_per_class=2000,
                        max_window=32, seed=0,
                        magnitude_threshold=0.5, batch_size=512):
    """‖T1‖ separation across {valid, partial, spanning, multi, bogus}.

    Returns:
      'class_stats':  per-class ‖T1‖ (mean, std, min, max) and
                      ground-truth is_complete rate.
      'magnitude_acc': accuracy of (‖T1‖ > threshold) as is_complete predictor.
      'majority_baseline': 0/1 baseline accuracy.
    """
    encoder.eval()
    rng = np.random.default_rng(seed)
    opcode_table = _build_opcode_table(DEFAULT_DISTRIBUTION)

    gens = [
        ('valid',    lambda: encode_instruction(
            random_instruction(rng, opcode_table=opcode_table))),
        ('partial',  lambda: gen_partial(rng, opcode_table, max_window)),
        ('spanning', lambda: gen_spanning(rng, opcode_table, max_window)),
        ('multi',    lambda: gen_multi(rng, opcode_table, max_window)),
        ('bogus',    lambda: gen_bogus(rng, max_window)),
    ]

    tokens_list, class_ids = [], []
    for cid, (_, fn) in enumerate(gens):
        for _ in range(n_per_class):
            toks = fn()
            if len(toks) > max_window:
                toks = toks[:max_window]
            tokens_list.append(toks)
            class_ids.append(cid)

    def _pad(seqs):
        N = len(seqs)
        tok = np.full((N, max_window), PAD, dtype=np.int64)
        pad = np.ones((N, max_window), dtype=bool)
        lens = np.zeros(N, dtype=np.int64)
        for i, s in enumerate(seqs):
            L = len(s)
            tok[i, :L] = s
            pad[i, :L] = False
            lens[i] = L
        return tok, pad, lens

    norms = []
    is_complete = []
    for i in range(0, len(tokens_list), batch_size):
        chunk = tokens_list[i:i + batch_size]
        tok, pad, lens = _pad(chunk)
        v = encoder.encode(torch.from_numpy(tok).to(device),
                           torch.from_numpy(pad).to(device))
        norms.append(v.norm(dim=-1).cpu())
        c = batch_is_complete_instruction(
            torch.from_numpy(tok).to(device),
            torch.from_numpy(lens).to(device),
            device)
        is_complete.append(c.cpu())
    norms = torch.cat(norms)
    is_complete = torch.cat(is_complete).float()
    cls = torch.tensor(class_ids)

    class_stats = {}
    for cid, (name, _) in enumerate(gens):
        mask = cls == cid
        if not mask.any():
            continue
        n = norms[mask]
        class_stats[name] = {
            'n': int(mask.sum()),
            'norm_mean': float(n.mean()),
            'norm_std': float(n.std()),
            'norm_min': float(n.min()),
            'norm_max': float(n.max()),
            'is_complete_rate': float(is_complete[mask].mean()),
        }

    pred = (norms > magnitude_threshold).float()
    mag_acc = float((pred == is_complete).float().mean())
    pos = float(is_complete.mean())
    baseline = max(pos, 1 - pos)

    return {
        'class_stats': class_stats,
        'magnitude_acc': mag_acc,
        'majority_baseline': baseline,
        'magnitude_threshold': magnitude_threshold,
    }


# ===========================================================================
# Step 2 diagnostics
# ===========================================================================

@torch.no_grad()
def decoder_accuracy(encoder, decoder, batches, *,
                     device, max_batches=20):
    """Teacher-forced reconstruction accuracy of the decoder on RVT
    batches. Only valid rows contribute (invalid rows have no target).

    Step 2 success criterion: instr_acc > 0.95.

    Returns: {'tok_acc': float, 'instr_acc': float,
              'n_tokens': int, 'n_instrs': int, 'n_batches': int}.
    """
    encoder.eval()
    decoder.eval()
    tok_correct = tok_total = 0
    instr_correct = instr_total = 0
    n_batches = 0

    for batch in batches:
        if n_batches >= max_batches:
            break
        n_batches += 1
        valid = batch.valid
        if not valid.any():
            continue
        tok = torch.from_numpy(batch.tokens).to(device)
        pad = torch.from_numpy(padding_mask(batch)).to(device)
        vecs_all = encoder.encode(tok, pad)
        valid_idx = np.flatnonzero(valid)
        vecs = vecs_all[valid_idx]
        token_lists = []
        for i in valid_idx:
            n = int(batch.token_lens[i])
            token_lists.append(batch.tokens[i, :n].tolist())
        dec_in, dec_tgt, dec_pad = prepare_decoder_targets(token_lists, device)
        if dec_in is None:
            continue
        logits = decoder(vecs, dec_in, dec_pad)
        pred = logits.argmax(dim=-1)
        non_pad = ~dec_pad
        tok_correct += int(((pred == dec_tgt) & non_pad).sum())
        tok_total += int(non_pad.sum())
        for b in range(pred.shape[0]):
            n = int(non_pad[b].sum())
            instr_total += 1
            if n > 0 and bool((pred[b, :n] == dec_tgt[b, :n]).all()):
                instr_correct += 1

    return {
        'tok_acc': tok_correct / tok_total if tok_total else 0.0,
        'instr_acc': instr_correct / instr_total if instr_total else 0.0,
        'n_tokens': tok_total,
        'n_instrs': instr_total,
        'n_batches': n_batches,
    }
