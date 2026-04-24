"""Training loops for the compressor.

- train_batches(): N×N pairwise MSE on single-instruction batches (step 1)
- train_sequences(): fixed-window training on sequence batches (context experiments)
- streaming_train(): shift-reduce compressor with REINFORCE gate training
"""

import time
from datetime import timedelta
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import BOS, EOS, PAD, VOCAB_SIZE, encode_instruction
from emulator import (
    Instruction,
    batch_execute, batch_parse_tokens, random_regs_gpu,
)
from datagen.equivalences import MANIFEST, sample_binding, materialize
from .model import T1Compressor, Decoder


# Fixed probe instructions for watching geometry during training.
# Distances reported in each log line:
#   aa:    ADD x5,x7,x7 vs ADD x5,x1,x2    (different-operand ADDs)
#   canon: ADD x5,x7,x7 vs SLLI x5,x7,1    (double_is_shl1 canonical)
#   ss:    SLLI x5,x7,1 vs SLLI x5,x3,2    (SLLI neighbor)
#   comm:  ADD x5,x1,x2 vs ADD x5,x2,x1    (commutative swap — should collapse)
_PROBE_INSTRS = [
    Instruction('ADD',  5, 7, 7),   # 0: ADD rs,rs (double canonical)
    Instruction('SLLI', 5, 7, 1),   # 1: SLLI rs,1 (shl1 canonical)
    Instruction('ADD',  5, 1, 2),   # 2: random ADD
    Instruction('SLLI', 5, 3, 2),   # 3: SLLI neighbor
    Instruction('ADD',  5, 2, 1),   # 4: commutative swap of #2
]


def _probe_tensor(device):
    """Tokenize + pad the fixed probe instructions for periodic eval."""
    encoded = [encode_instruction(i) for i in _PROBE_INSTRS]
    max_len = max(len(e) for e in encoded)
    tok = np.full((len(encoded), max_len), PAD, dtype=np.int64)
    pad = np.ones((len(encoded), max_len), dtype=np.bool_)
    for i, e in enumerate(encoded):
        tok[i, :len(e)] = e
        pad[i, :len(e)] = False
    return (torch.from_numpy(tok).to(device),
            torch.from_numpy(pad).to(device))


def _encode_instrs(model, instrs, device):
    """Tokenize + pad a list of Instructions and encode them."""
    encoded = [encode_instruction(i) for i in instrs]
    max_len = max(len(e) for e in encoded)
    tok = np.full((len(encoded), max_len), PAD, dtype=np.int64)
    pad = np.ones((len(encoded), max_len), dtype=np.bool_)
    for i, e in enumerate(encoded):
        tok[i, :len(e)] = e
        pad[i, :len(e)] = False
    tok_t = torch.from_numpy(tok).to(device)
    pad_t = torch.from_numpy(pad).to(device)
    return model.encode(tok_t, pad_t)


def equivalence_loss(model, device, rng):
    """Per-class MSE on canonical pairwise distances, target=0.

    Each call: for every manifest class with >=2 canonical templates,
    sample a fresh binding, materialize, collect all instructions
    across classes into a single padded batch, encode once, then
    slice per-class to compute mean squared within-class distance.
    Returns the mean across classes.
    """
    all_instrs = []
    class_ranges = []  # list of (start, end) slices into the batch
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

    vecs = _encode_instrs(model, all_instrs, device)  # (N_total, d_out)

    losses = []
    for s, e in class_ranges:
        class_vecs = vecs[s:e]
        N = class_vecs.shape[0]
        idx = torch.triu_indices(N, N, offset=1, device=device)
        dists = torch.cdist(class_vecs.unsqueeze(0),
                            class_vecs.unsqueeze(0)).squeeze(0)
        losses.append(dists[idx[0], idx[1]].square().mean())
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Execution distance metrics
# ---------------------------------------------------------------------------

def exec_distance_deltas(deltas, device):
    """Pairwise execution distance from per-register deltas.
    Used by sequence-based training (RVS).
    """
    d = deltas.to(device=device, dtype=torch.float32)
    diff = (d.unsqueeze(1) - d.unsqueeze(0)).abs()
    return diff.log1p_().mean(dim=(-1, -2))


def _exec_distance_scalar_impl(dv, pv):
    """Pairwise distance from scalar data_vals and pc_vals.

    mean_s(log1p(log1p(|data_diff_s| + |pc_diff_s|)))

    Nested log: first log1p compresses the int32 range (0..4e9) to
    (0..22); second log1p compresses that to (0..3.14). The second
    compression spreads the low end more uniformly, so near-equivalence
    pairs (e.g., ADDI imm=0 vs imm=1, raw diff=1) get meaningfully
    larger targets and thus more gradient pressure during training.
    """
    B, S = dv.shape
    acc = torch.zeros(B, B, dtype=torch.float32, device=dv.device)
    for s in range(S):
        d = (dv[:, s].unsqueeze(1) - dv[:, s].unsqueeze(0)).abs()
        p = (pv[:, s].unsqueeze(1) - pv[:, s].unsqueeze(0)).abs()
        acc += torch.log1p(torch.log1p(d + p))
    return acc / S


_exec_distance_scalar_compiled = torch.compile(_exec_distance_scalar_impl)


def exec_distance_scalar(data_vals, pc_vals, device):
    """Pairwise execution distance from computed output values.
    Used by single-instruction batch training (RVB).

    data_vals: (B, n_inputs) int64 numpy array
    pc_vals: (B, n_inputs) int64 numpy array
    Returns: (B, B) float32 tensor
    """
    dv = torch.tensor(data_vals, dtype=torch.bfloat16, device=device)
    pv = torch.tensor(pc_vals, dtype=torch.bfloat16, device=device)
    return _exec_distance_scalar_compiled(dv, pv)


# ---------------------------------------------------------------------------
# Execution distance scaling
# ---------------------------------------------------------------------------

_EXEC_DIST_SCALE = 2.0 / 3.14  # nested-log range [0, ~3.14] → [0, ~2]


# ---------------------------------------------------------------------------
# N×N pairwise MSE training on single-instruction batches (step 1)
# ---------------------------------------------------------------------------

def train_batches(batch_iter, d_model=128, n_heads=4, n_layers=2,
                  d_out=128, lr=3e-4, device='auto', n_steps=None,
                  log_every=100, lr_min=1e-6, equiv_weight=0.0,
                  equiv_seed=0, recon_weight=0.0,
                  dec_d_model=128, dec_n_heads=4, dec_n_layers=2,
                  k_samples=10, n_reward_inputs=4):
    """Train the T1 encoder on single-instruction batches (RVB).

    N×N pairwise MSE between T1 vector distances and execution
    distances computed from scalar data_vals and pc_vals.

    Returns (model, losses).
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.set_float32_matmul_precision('high')

    model = T1Compressor(VOCAB_SIZE, d_model, n_heads, n_layers, d_out)
    model = model.to(device)

    decoder = None
    if recon_weight > 0:
        decoder = Decoder(VOCAB_SIZE, dec_d_model, dec_n_heads,
                          dec_n_layers, d_emb=d_out)
        decoder = decoder.to(device)
        all_params = list(model.parameters()) + list(decoder.parameters())
    else:
        all_params = list(model.parameters())

    opt = torch.optim.Adam(all_params, lr=lr)
    scheduler = None
    if n_steps:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=n_steps, eta_min=lr_min)

    losses = []
    step = 0
    t0 = time.time()
    probe_tok, probe_pad = _probe_tensor(device)
    equiv_rng = np.random.default_rng(equiv_seed)
    recon_baseline = 0.0

    for batch in batch_iter:
        tok = torch.from_numpy(batch.token_ids).to(device)
        pad = torch.from_numpy(batch.padding_mask).to(device)
        dt_targets = torch.from_numpy(batch.dest_types).to(device)
        dr_targets = torch.from_numpy(batch.dest_regs).to(device)

        vecs = model.compiled_encode(tok, pad)  # (B, d_out)
        dt_logits = model.dest_type_head(vecs)
        dr_logits = model.dest_reg_head(vecs)

        # N×N pairwise distances.
        model_dists = torch.cdist(vecs.unsqueeze(0), vecs.unsqueeze(0),
                                  p=2).squeeze(0)
        exec_dists = exec_distance_scalar(
            batch.data_vals, batch.pc_vals, device)

        # MSE on upper triangle with fixed scaling.
        N = vecs.shape[0]
        tri = torch.triu_indices(N, N, offset=1, device=device)
        target = exec_dists[tri[0], tri[1]] * _EXEC_DIST_SCALE
        mse_loss = (model_dists[tri[0], tri[1]] - target).square().mean()

        # Destination classification: dest_type for all instructions,
        # dest_reg only for register-writing instructions (dt=0).
        dt_loss = F.cross_entropy(dt_logits, dt_targets)
        reg_mask = (dt_targets == 0)
        if reg_mask.any():
            dr_loss = F.cross_entropy(
                dr_logits[reg_mask], dr_targets[reg_mask])
        else:
            dr_loss = torch.tensor(0.0, device=device)

        if equiv_weight > 0:
            eq_loss = equivalence_loss(model, device, equiv_rng)
        else:
            eq_loss = torch.tensor(0.0, device=device)

        if recon_weight > 0 and decoder is not None:
            B_dec = tok.shape[0]
            token_lists = []
            for b in range(B_dec):
                mask = ~batch.padding_mask[b]
                token_lists.append(batch.token_ids[b][mask].tolist())
            dec_in, dec_tgt, dec_pad = _prepare_decoder_targets(
                token_lists, device)

            # Decoder forward — vecs NOT detached so REINFORCE
            # gradient flows through decoder to encoder.
            dec_logits = decoder(vecs, dec_in, dec_pad)
            non_pad_dec = ~dec_pad
            orig_lengths = non_pad_dec.sum(dim=1)

            dist = torch.distributions.Categorical(logits=dec_logits)

            # K-sample REINFORCE with shaped execution reward.
            reinforce_sum = torch.tensor(0.0, device=device)
            total_reward = 0.0
            total_equiv = 0

            for _ in range(k_samples):
                sampled = dist.sample()
                lp = (dist.log_prob(sampled)
                      * non_pad_dec.float()).sum(dim=1)

                with torch.no_grad():
                    o_op, o_rd, o_rs1, o_rs2, o_imm, o_v = \
                        batch_parse_tokens(tok, orig_lengths, device)
                    d_op, d_rd, d_rs1, d_rs2, d_imm, d_v = \
                        batch_parse_tokens(sampled, orig_lengths, device)
                    both_v = o_v & d_v

                    op_m = (o_op == d_op).float()
                    rd_m = (o_rd == d_rd).float()
                    dv_m = torch.zeros(B_dec, device=device)
                    pc_m = torch.zeros(B_dec, device=device)

                    for _ in range(n_reward_inputs):
                        rgs = random_regs_gpu(B_dec, device=device)
                        pcr = torch.randint(
                            0, 256, (B_dec,),
                            dtype=torch.int32, device=device) * 4
                        o_dv, o_pc = batch_execute(
                            o_op, o_rd, o_rs1, o_rs2, o_imm, rgs, pcr)
                        d_dv, d_pc = batch_execute(
                            d_op, d_rd, d_rs1, d_rs2, d_imm, rgs, pcr)
                        dv_m += (o_dv == d_dv).float()
                        pc_m += (o_pc == d_pc).float()

                    rewards = torch.where(
                        both_v,
                        0.25 * op_m + 0.25 * rd_m
                        + 0.25 * dv_m / n_reward_inputs
                        + 0.25 * pc_m / n_reward_inputs,
                        torch.zeros(B_dec, device=device))

                    equiv = both_v & (dv_m == n_reward_inputs) \
                                   & (pc_m == n_reward_inputs)

                advantage = rewards - recon_baseline
                adv_std = advantage.std() + 1e-8
                advantage = (advantage - advantage.mean()) / adv_std
                reinforce_sum = reinforce_sum \
                    - (lp * advantage).mean()
                total_reward += rewards.mean().item()
                total_equiv += equiv.sum().item()

            recon_loss = reinforce_sum / k_samples
            recon_baseline = (0.99 * recon_baseline
                              + 0.01 * total_reward / k_samples)
            recon_acc = total_equiv / (B_dec * k_samples)
        else:
            recon_loss = torch.tensor(0.0, device=device)
            dec_logits = None
            recon_acc = 0.0

        loss = mse_loss + dt_loss + dr_loss + equiv_weight * eq_loss \
            + recon_weight * recon_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if scheduler:
            scheduler.step()

        loss_val = loss.item()
        losses.append(loss_val)
        step += 1

        if step % log_every == 0:
            elapsed = time.time() - t0
            ms_per = elapsed / step * 1000
            current_lr = scheduler.get_last_lr()[0] if scheduler else lr
            eta_str = ''
            if n_steps:
                eta_sec = int(max(0, n_steps - step) * ms_per / 1000)
                eta_str = f'  eta {timedelta(seconds=eta_sec)}'
            with torch.no_grad():
                pv = model.encode(probe_tok, probe_pad)
            d_add_add = (pv[0] - pv[2]).norm().item()
            d_canon = (pv[0] - pv[1]).norm().item()
            d_sll_sll = (pv[1] - pv[3]).norm().item()
            d_comm = (pv[2] - pv[4]).norm().item()
            recon_str = ''
            if dec_logits is not None:
                recon_str = (f' rl {recon_loss.item():.3f}'
                             f' eq {recon_acc:.0%}')
            print(f'step {step:>6d}  '
                  f'loss {loss_val:.4f} '
                  f'(mse {mse_loss.item():.4f} '
                  f'dt {dt_loss.item():.3f} '
                  f'dr {dr_loss.item():.3f} '
                  f'eq {eq_loss.item():.4f}'
                  f'{recon_str})  '
                  f'lr {current_lr:.1e}  {ms_per:.0f}ms/step  '
                  f'N={N}{eta_str}  '
                  f'probe[aa={d_add_add:.3f} '
                  f'canon={d_canon:.3f} '
                  f'ss={d_sll_sll:.3f} '
                  f'comm={d_comm:.3f}]')

        if n_steps and step >= n_steps:
            break

    return model, decoder, losses


# ---------------------------------------------------------------------------
# Fixed-window training (context experiment)
# ---------------------------------------------------------------------------

def extract_windows(batch, window_size=1):
    """Extract per-instruction training examples from a sequence batch."""
    B = batch.token_ids.shape[0]
    all_tokens = []
    all_deltas = []

    for b in range(B):
        n = int(batch.n_instructions[b])
        tok = batch.token_ids[b]
        idx = batch.token_instr_idx[b]
        mask = ~batch.padding_mask[b]

        for i in range(window_size - 1, n):
            start_instr = i - window_size + 1
            sel = mask & (idx >= start_instr) & (idx <= i)
            instr_tokens = tok[sel].tolist()
            all_tokens.append([BOS] + instr_tokens + [EOS])
            delta = (batch.per_instr_regs[b, i + 1, :, :]
                     - batch.per_instr_regs[b, i, :, :])
            all_deltas.append(delta)

    if not all_tokens:
        return None, None, None

    max_len = max(len(t) for t in all_tokens)
    N = len(all_tokens)
    token_ids = np.full((N, max_len), PAD, dtype=np.int64)
    padding_mask = np.ones((N, max_len), dtype=np.bool_)
    for j, t in enumerate(all_tokens):
        token_ids[j, :len(t)] = t
        padding_mask[j, :len(t)] = False

    deltas = np.stack(all_deltas, axis=0)
    return token_ids, padding_mask, deltas


def train_sequences(batch_iter, **kwargs):
    """Placeholder for sequence-based training.

    The original implementation (context experiments, Exp 19) used
    fixed windows with per-register delta execution distance. It will
    be replaced with streaming_train when T2 fine-tuning is built.
    """
    raise NotImplementedError(
        'train_sequences is not implemented. Use train_batches for '
        'encoder training (step 1) or streaming_train for gate training.')


# ---------------------------------------------------------------------------
# Decoder target preparation
# ---------------------------------------------------------------------------

def _prepare_decoder_targets(token_lists, device):
    """Build decoder input/target tensors from lists of token IDs.

    Each token list gets wrapped: [BOS] + tokens + [EOS].
    Returns dec_input (shifted right), dec_target, dec_padding.
    Returns None tuple if token_lists is empty.
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


@torch.no_grad()
def autoregressive_sample(decoder, vecs, max_len, device):
    """Sample token sequences autoregressively (no gradient).

    The decoder sees its OWN previous tokens at each position.
    Returns sampled tokens only (BOS stripped) — log_probs are
    computed separately in a single parallel forward pass to save
    memory.
    """
    B = vecs.shape[0]
    generated = torch.full((B, 1), BOS, dtype=torch.long, device=device)
    for _ in range(max_len):
        pad_mask = torch.zeros(
            B, generated.shape[1], dtype=torch.bool, device=device)
        logits = decoder(vecs, generated, pad_mask)
        next_token = torch.distributions.Categorical(
            logits=logits[:, -1, :]).sample()
        generated = torch.cat(
            [generated, next_token.unsqueeze(1)], dim=1)
    return generated[:, 1:]


def compute_log_probs(decoder, vecs, sampled, orig_lengths, device):
    """Compute log_probs for a sampled sequence in one parallel forward.

    Uses the sampled tokens as decoder input (shifted right with BOS).
    The log_probs are conditioned on the sampled prefix, not ground
    truth — teacher-forcing on the decoder's own output.
    """
    B, T = sampled.shape
    bos_col = torch.full((B, 1), BOS, dtype=torch.long, device=device)
    dec_input = torch.cat([bos_col, sampled[:, :-1]], dim=1)
    pad_mask = torch.zeros(B, T, dtype=torch.bool, device=device)

    logits = decoder(vecs, dec_input, pad_mask)
    dist = torch.distributions.Categorical(logits=logits)
    log_probs = dist.log_prob(sampled)

    pos_mask = (torch.arange(T, device=device).unsqueeze(0)
                < orig_lengths.unsqueeze(1))
    return (log_probs * pos_mask.float()).sum(dim=1)


def gpu_reward(orig_tokens, sampled_tokens, orig_lengths,
               sampled_lengths, device, n_inputs=4):
    """Fully-GPU shaped execution-equivalence reward.

    Decomposes the reward into per-field signals so the decoder
    gets credit for partially-correct decodings:
      0.25 × opcode match
      0.25 × dest_reg match
      0.25 × data_val match (fraction across n_inputs)
      0.25 × pc match (fraction across n_inputs)

    Returns (rewards, valid_mask, equiv_mask) as (B,) tensors.
    """
    o_op, o_rd, o_rs1, o_rs2, o_imm, o_valid = batch_parse_tokens(
        orig_tokens, orig_lengths, device)
    d_op, d_rd, d_rs1, d_rs2, d_imm, d_valid = batch_parse_tokens(
        sampled_tokens, sampled_lengths, device)

    both_valid = o_valid & d_valid
    B = orig_tokens.shape[0]

    op_match = (o_op == d_op).float()
    rd_match = (o_rd == d_rd).float()

    dv_match_count = torch.zeros(B, dtype=torch.float32, device=device)
    pc_match_count = torch.zeros(B, dtype=torch.float32, device=device)

    for _ in range(n_inputs):
        regs = random_regs_gpu(B, device=device)
        pc = torch.randint(0, 256, (B,), dtype=torch.int32,
                           device=device) * 4
        o_dv, o_pc = batch_execute(
            o_op, o_rd, o_rs1, o_rs2, o_imm, regs, pc)
        d_dv, d_pc = batch_execute(
            d_op, d_rd, d_rs1, d_rs2, d_imm, regs, pc)
        dv_match_count += (o_dv == d_dv).float()
        pc_match_count += (o_pc == d_pc).float()

    dv_match = dv_match_count / n_inputs
    pc_match = pc_match_count / n_inputs

    shaped_reward = (0.25 * op_match + 0.25 * rd_match
                     + 0.25 * dv_match + 0.25 * pc_match)

    rewards = torch.where(
        both_valid, shaped_reward,
        torch.zeros(B, dtype=torch.float32, device=device))

    equiv_mask = (both_valid
                  & (dv_match_count == n_inputs)
                  & (pc_match_count == n_inputs))

    return rewards, d_valid, equiv_mask


def load_checkpoint(path, device):
    """Load a state dict, stripping the torch.compile prefix.

    `torch.compile` wraps modules so saved state dicts carry an
    `_orig_mod.` prefix on every key. This helper transparently
    handles both compiled and uncompiled checkpoints so the result
    loads into an uncompiled module.
    """
    state = torch.load(path, map_location=device, weights_only=True)
    return {k.removeprefix('_orig_mod.'): v for k, v in state.items()}


def _compute_emission_deltas(batch, emission_info, device):
    """Compute cumulative register deltas for complete emissions."""
    deltas = []
    for info in emission_info:
        b = info['batch_idx']
        i_start = info['instr_start']
        i_end = info['instr_end']
        delta = (batch.per_instr_regs[b, i_end + 1, :, :]
                 - batch.per_instr_regs[b, i_start, :, :])
        deltas.append(torch.from_numpy(delta.astype(np.float32)))

    if not deltas:
        return torch.zeros(0, 1, 32, device=device)
    return torch.stack(deltas).to(device)


# ---------------------------------------------------------------------------
# Streaming compressor training with REINFORCE gate training
# ---------------------------------------------------------------------------

def streaming_train(batch_iter, d_model=128, n_heads=4, n_layers=2,
                    d_out=128, dec_d_model=128, dec_n_heads=4,
                    dec_n_layers=2, lr=3e-4, device='auto',
                    n_steps=None, log_every=100, lr_min=1e-6,
                    pairwise_weight=1.0, reinforce_lr=1e-3,
                    baseline_decay=0.99):
    """Train shift-reduce compressor + decoder.

    The encoder (window encoder + GRU + projector) and decoder are
    trained via normal backprop through reconstruction loss.

    The gates are trained via REINFORCE: at every iteration, the
    decoder evaluates the T1 candidate against the current window
    contents. The per-iteration reconstruction loss serves as the
    reward signal (negated: lower loss = higher reward).

    Returns (encoder, decoder, losses, gate_stats).
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    encoder = T1Compressor(
        VOCAB_SIZE, d_model, n_heads, n_layers, d_out)
    encoder = encoder.to(device)

    decoder = Decoder(
        VOCAB_SIZE, dec_d_model, dec_n_heads, dec_n_layers, d_emb=d_out)
    decoder = decoder.to(device)

    # Separate optimizers: normal backprop for encoder+decoder,
    # REINFORCE for gates.
    encoder_params = (
        list(encoder.tok_emb.parameters())
        + list(encoder.win_pos_emb.parameters())
        + list(encoder.win_encoder.parameters())
        + list(encoder.proj.parameters())
        + list(encoder.gru.parameters()))
    gate_params = (
        list(encoder.accept_head.parameters())
        + list(encoder.emit_head.parameters())
        + list(encoder.evict_head.parameters()))

    opt_main = torch.optim.Adam(
        encoder_params + list(decoder.parameters()), lr=lr)
    opt_gates = torch.optim.Adam(gate_params, lr=reinforce_lr)

    scheduler = None
    if n_steps:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_main, T_max=n_steps, eta_min=lr_min)

    # REINFORCE baseline (moving average of per-iteration loss).
    baseline = 0.0

    losses = []
    gate_stats = []
    step = 0
    t0 = time.time()

    for batch in batch_iter:
        tok = torch.from_numpy(batch.token_ids).to(device)
        pad = torch.from_numpy(batch.padding_mask).to(device)
        idx = torch.from_numpy(batch.token_instr_idx).to(device)
        n_instr = torch.from_numpy(batch.n_instructions).to(device)

        B = tok.shape[0]

        # --- Forward pass (shift-reduce loop) ---
        (iter_t1s, iter_window_tokens,
         iter_gate_log_probs, iter_gate_decisions,
         iter_gate_logits,
         iter_window_buf_t, iter_window_lens_t,
         emission_info, emit_counts) = encoder(tok, pad, idx, n_instr)

        n_iters = len(iter_t1s)
        if n_iters == 0:
            continue

        # --- Per-iteration decoder evaluation (chunked + length-sorted) ---
        # Flatten all (iter, batch_item) pairs with non-empty windows,
        # sort by window length descending, process in chunks. Sorting
        # keeps each chunk tightly padded; chunking keeps the decoder
        # forward at a size the GPU can handle efficiently. A single
        # unchunked forward with global-max padding turned out to be
        # much slower than many small forwards.
        flat_iter_idx = []
        flat_batch_idx = []
        flat_tokens = []
        flat_win_sizes = []
        for it in range(n_iters):
            wt = iter_window_tokens[it]
            for b in range(B):
                if wt[b]:
                    flat_iter_idx.append(it)
                    flat_batch_idx.append(b)
                    flat_tokens.append(wt[b])
                    flat_win_sizes.append(len(wt[b]))

        if not flat_tokens:
            iter_recon_losses = [
                torch.zeros(B, device=device) for _ in range(n_iters)]
        else:
            DECODER_CHUNK_SIZE = 1024
            order = sorted(
                range(len(flat_tokens)),
                key=lambda i: -flat_win_sizes[i])

            iter_idx_t = torch.tensor(
                flat_iter_idx, dtype=torch.long, device=device)
            batch_idx_t = torch.tensor(
                flat_batch_idx, dtype=torch.long, device=device)
            stacked_t1 = torch.stack(iter_t1s)  # (n_iters, B, d_out)

            N = len(flat_tokens)
            per_item_loss = torch.zeros(N, device=device)
            for start in range(0, N, DECODER_CHUNK_SIZE):
                chunk_flat_idx = order[start:start + DECODER_CHUNK_SIZE]
                chunk_tokens = [flat_tokens[i] for i in chunk_flat_idx]
                chunk_iter = torch.tensor(
                    [flat_iter_idx[i] for i in chunk_flat_idx],
                    dtype=torch.long, device=device)
                chunk_batch = torch.tensor(
                    [flat_batch_idx[i] for i in chunk_flat_idx],
                    dtype=torch.long, device=device)
                chunk_t1 = stacked_t1[chunk_iter, chunk_batch]

                dec_in, dec_tgt, dec_pad = _prepare_decoder_targets(
                    chunk_tokens, device)
                logits = decoder(chunk_t1, dec_in, dec_pad)

                pt = F.cross_entropy(
                    logits.reshape(-1, VOCAB_SIZE),
                    dec_tgt.reshape(-1),
                    ignore_index=PAD, reduction='none')
                pt = pt.reshape(len(chunk_tokens), -1)
                non_pad_counts = (~dec_pad).sum(dim=1).clamp(min=1).float()
                chunk_loss = pt.sum(dim=1) / non_pad_counts
                ws = torch.tensor(
                    [flat_win_sizes[i] for i in chunk_flat_idx],
                    dtype=torch.float32, device=device)
                chunk_loss = chunk_loss * ws

                chunk_idx_tensor = torch.tensor(
                    chunk_flat_idx, dtype=torch.long, device=device)
                per_item_loss = per_item_loss.scatter(
                    0, chunk_idx_tensor, chunk_loss)

            flat_flat_idx = iter_idx_t * B + batch_idx_t
            losses_flat = torch.zeros(
                n_iters * B, device=device).scatter(
                    0, flat_flat_idx, per_item_loss)
            losses_matrix = losses_flat.view(n_iters, B)
            iter_recon_losses = [losses_matrix[i] for i in range(n_iters)]

        # --- Reconstruction loss for encoder+decoder (normal backprop) ---
        # Average across all iterations and sequences.
        all_recon = torch.stack(iter_recon_losses)  # (n_iters, B)
        # Mask out done sequences (zero loss).
        recon_mask = all_recon > 0
        if recon_mask.any():
            recon_loss = all_recon[recon_mask].mean()
        else:
            recon_loss = torch.tensor(0.0, device=device)

        # --- Pairwise MSE loss (on actual emissions with complete instrs) ---
        complete_info = [info for info in emission_info
                         if info['has_complete']]

        if len(complete_info) >= 2:
            # Gather emission T1 vectors.
            emission_vecs = []
            for info in complete_info:
                it = info['iteration']
                b = info['batch_idx']
                emission_vecs.append(iter_t1s[it][b])
            emission_vecs = torch.stack(emission_vecs)

            emission_deltas = _compute_emission_deltas(
                batch, complete_info, device)
            Nc = emission_vecs.shape[0]
            model_dists = torch.cdist(
                emission_vecs.unsqueeze(0),
                emission_vecs.unsqueeze(0),
                p=2).squeeze(0)
            exec_dists = exec_distance_deltas(emission_deltas, device)
            scale = 2.0 / max(exec_dists.max().item(), 1e-6)
            exec_dists = exec_dists * min(scale, 4.0)
            tri = torch.triu_indices(Nc, Nc, offset=1, device=device)
            pairwise_loss = F.mse_loss(
                model_dists[tri[0], tri[1]],
                exec_dists[tri[0], tri[1]])
        else:
            pairwise_loss = torch.tensor(0.0, device=device)

        # --- REINFORCE for gates ---
        # Compute REINFORCE loss BEFORE main backward (shares graph).
        with torch.no_grad():
            iter_rewards = torch.stack(iter_recon_losses)
            mean_reward = iter_rewards[iter_rewards > 0].mean().item() \
                if (iter_rewards > 0).any() else 0.0
            baseline = baseline_decay * baseline + (1 - baseline_decay) * mean_reward

        reinforce_loss = torch.tensor(0.0, device=device)
        n_terms = 0
        for it in range(n_iters):
            log_probs = iter_gate_log_probs[it]  # (B, 3)
            rewards = iter_recon_losses[it].detach()  # (B,)
            advantages = rewards - baseline

            total_lp = log_probs.sum(dim=1)
            active = rewards > 0
            if active.any():
                reinforce_loss = reinforce_loss + \
                    (total_lp[active] * advantages[active]).mean()
                n_terms += 1

        if n_terms > 0:
            reinforce_loss = reinforce_loss / n_terms

        # --- Combined backward ---
        main_loss = recon_loss + pairwise_weight * pairwise_loss
        total_loss = main_loss + reinforce_loss
        opt_main.zero_grad()
        opt_gates.zero_grad()
        total_loss.backward()
        opt_main.step()
        opt_gates.step()
        if scheduler:
            scheduler.step()

        # --- Logging ---
        recon_val = recon_loss.item()
        pair_val = pairwise_loss.item()
        loss_val = main_loss.item()
        reinforce_val = reinforce_loss.item() if n_terms > 0 else 0.0

        with torch.no_grad():
            # Gate decision rates.
            total_a = total_e = total_v = total_steps = 0
            total_window = 0
            for it in range(n_iters):
                decs = iter_gate_decisions[it]  # (B, 3)
                for b in range(B):
                    wt = iter_window_tokens[it][b]
                    if wt or (it == 0):  # active sequence
                        total_a += decs[b, 0].item() > 0.5
                        total_e += decs[b, 1].item() > 0.5
                        total_v += decs[b, 2].item() > 0.5
                        total_steps += 1
                        total_window += len(wt)

            accept_rate = total_a / max(total_steps, 1)
            emit_rate = total_e / max(total_steps, 1)
            evict_rate = total_v / max(total_steps, 1)
            mean_emits = emit_counts.float().mean().item()
            mean_instrs = n_instr.float().mean().item()
            mean_window = total_window / max(total_steps, 1)
            complete_frac = len(complete_info) / max(len(emission_info), 1)

            # Reconstruction accuracy on actual emissions.
            if emission_info:
                all_tgt_toks = [info['target_tokens'] for info in emission_info]
                dec_in, dec_tgt, dec_pad = _prepare_decoder_targets(
                    all_tgt_toks, device)
                if dec_in is not None:
                    # Find emission T1 vectors.
                    em_vecs = []
                    for info in emission_info:
                        em_vecs.append(
                            iter_t1s[info['iteration']][info['batch_idx']])
                    em_vecs = torch.stack(em_vecs)
                    em_logits = decoder(em_vecs, dec_in, dec_pad)
                    pred = em_logits.argmax(dim=-1)
                    non_pad_dec = ~dec_pad
                    correct = (pred == dec_tgt) & non_pad_dec
                    recon_acc = correct.sum().item() / non_pad_dec.sum().item()
                else:
                    recon_acc = 0.0
            else:
                recon_acc = 0.0

        losses.append({
            'total': loss_val,
            'recon': recon_val,
            'pairwise': pair_val,
            'reinforce': reinforce_val,
            'mean_window': mean_window,
        })
        gs = {
            'accept_rate': accept_rate,
            'emit_rate': emit_rate,
            'evict_rate': evict_rate,
            'emits_per_seq': mean_emits,
            'instrs_per_seq': mean_instrs,
            'n_emissions': len(emission_info),
            'complete_frac': complete_frac,
            'recon_acc': recon_acc,
            'iters_per_seq': n_iters,
        }
        gate_stats.append(gs)

        step += 1

        if step % log_every == 0:
            elapsed = time.time() - t0
            ms_per = elapsed / step * 1000
            current_lr = scheduler.get_last_lr()[0] if scheduler else lr
            eta_str = ''
            if n_steps:
                eta_sec = int(max(0, n_steps - step) * ms_per / 1000)
                eta_str = f'  eta {timedelta(seconds=eta_sec)}'
            print(f'step {step:>6d}  '
                  f'loss {loss_val:.4f} '
                  f'(recon {recon_val:.3f} pair {pair_val:.4f} '
                  f'rl {reinforce_val:.3f})  '
                  f'acc {recon_acc:.1%}  '
                  f'lr {current_lr:.1e}  {ms_per:.0f}ms/step  '
                  f'a/e/v={accept_rate:.0%}/{emit_rate:.0%}/'
                  f'{evict_rate:.0%}  '
                  f'{gs["emits_per_seq"]:.1f}/'
                  f'{gs["instrs_per_seq"]:.1f}  '
                  f'win {mean_window:.1f}  '
                  f'ok {complete_frac:.0%}  '
                  f'iters {n_iters}{eta_str}')

        if n_steps and step >= n_steps:
            break

    return encoder, decoder, losses, gate_stats

