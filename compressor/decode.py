"""Gradient-based decoder for T1 vectors.

Given a target T1 vector (a point on S^127), finds an instruction
whose tokens compress to (near) that point by optimizing continuous
embeddings through the frozen compressor and snapping to discrete
tokens.
"""

import torch
import torch.nn.functional as F
import numpy as np

from emulator import Instruction, run as run_instruction, random_regs, make_ctx
from tokenizer import encode_instruction, VOCAB, VOCAB_SIZE
from datagen import extract_data_val


def gradient_decode(model, target_t1, n_restarts=10, n_steps=500,
                    lr=0.1):
    """Find an instruction whose T1 vector matches the target.

    Optimizes continuous embeddings in the model's embedding space,
    then snaps to nearest discrete tokens. Tries multiple sequence
    lengths (4-8) to cover all RV32I instruction formats.

    Args:
        model: frozen T1Compressor
        target_t1: (d_out,) target vector on the unit sphere
        n_restarts: number of random initializations to try per length
        n_steps: gradient steps per restart
        lr: learning rate for embedding optimization

    Returns:
        list of (distance, token_ids, instruction_or_None) tuples,
        sorted by distance. instruction_or_None is the parsed
        Instruction if the token sequence is structurally valid.
    """
    device = next(model.parameters()).device
    target = target_t1.to(device).detach()

    # Get the full token embedding table for snapping.
    tok_emb = model.tok_embedding.weight.detach()  # (vocab, d_model)

    results = []

    for seq_len in range(4, 9):
        for _ in range(n_restarts):
            # Initialize continuous embeddings randomly.
            emb_mean = tok_emb.mean(dim=0, keepdim=True)
            emb_std = tok_emb.std()
            cont_emb = (emb_mean + emb_std * torch.randn(
                1, seq_len, model.d_model, device=device))
            cont_emb.requires_grad_(True)

            optimizer = torch.optim.Adam([cont_emb], lr=lr)

            for step in range(n_steps):
                optimizer.zero_grad()

                # Forward through the compressor (bypass tok_embedding).
                positions = torch.arange(seq_len, device=device).unsqueeze(0)
                x = cont_emb + model.pos_embedding(positions)
                x = model.encoder(x)
                x = x.mean(dim=1)  # mean pool (no padding)
                t1 = F.normalize(model.output_proj(x), dim=-1)

                loss = (t1 - target).square().sum()
                loss.backward()
                optimizer.step()

            # Snap each position to nearest token embedding.
            with torch.no_grad():
                final_emb = cont_emb.squeeze(0)  # (seq_len, d_model)
                sim = F.cosine_similarity(
                    final_emb.unsqueeze(1),
                    tok_emb.unsqueeze(0),
                    dim=-1)
                token_ids = sim.argmax(dim=-1)

                # Recompress the discrete tokens to get actual distance.
                ids = token_ids.unsqueeze(0)
                t1_actual, _, _ = model(ids)
                dist = (t1_actual - target).square().sum().sqrt().item()

            results.append((dist, token_ids.cpu().tolist()))

    # Sort by distance, try to parse each as an instruction.
    results.sort(key=lambda x: x[0])
    parsed = []
    for dist, ids in results:
        instr = _try_parse(ids)
        parsed.append((dist, ids, instr))

    return parsed


def _try_parse(token_ids):
    """Try to parse a token ID sequence as a valid RV32I instruction."""
    # Build reverse lookup.
    id_to_tok = {i: tok for i, tok in enumerate(VOCAB)}
    tokens = [id_to_tok.get(tid, '?') for tid in token_ids]

    # Strip padding tokens.
    tokens = [t for t in tokens if t != 'PAD']
    if not tokens:
        return None

    # First token should be an opcode.
    opcode = tokens[0]

    from emulator import R_TYPE, I_TYPE, B_TYPE, LOAD_TYPE, STORE_TYPE

    try:
        if opcode in R_TYPE:
            # R-type: opcode rd rs1 rs2
            rd = _parse_reg(tokens[1])
            rs1 = _parse_reg(tokens[2])
            rs2 = _parse_reg(tokens[3])
            return Instruction(opcode, rd, rs1, rs2)
        elif opcode in I_TYPE:
            # I-type: opcode rd rs1 imm
            rd = _parse_reg(tokens[1])
            rs1 = _parse_reg(tokens[2])
            imm = _parse_imm(tokens[3:])
            return Instruction(opcode, rd, rs1, imm)
        elif opcode in LOAD_TYPE:
            # Load: opcode rd imm rs1
            rd = _parse_reg(tokens[1])
            imm = _parse_imm(tokens[2:-1])
            rs1 = _parse_reg(tokens[-1])
            return Instruction(opcode, rd, imm, rs1)
        elif opcode in STORE_TYPE:
            # Store: opcode rs2 imm rs1
            rs2 = _parse_reg(tokens[1])
            imm = _parse_imm(tokens[2:-1])
            rs1 = _parse_reg(tokens[-1])
            return Instruction(opcode, rs2, imm, rs1)
        elif opcode in B_TYPE:
            # Branch: opcode rs1 rs2 imm
            rs1 = _parse_reg(tokens[1])
            rs2 = _parse_reg(tokens[2])
            imm = _parse_imm(tokens[3:])
            return Instruction(opcode, rs1, rs2, imm)
        elif opcode in ('LUI', 'AUIPC'):
            rd = _parse_reg(tokens[1])
            imm = _parse_imm(tokens[2:])
            return Instruction(opcode, rd, imm)
        elif opcode == 'JAL':
            rd = _parse_reg(tokens[1])
            imm = _parse_imm(tokens[2:])
            return Instruction(opcode, rd, imm)
        elif opcode == 'JALR':
            rd = _parse_reg(tokens[1])
            rs1 = _parse_reg(tokens[2])
            imm = _parse_imm(tokens[3:])
            return Instruction(opcode, rd, rs1, imm)
    except (IndexError, ValueError):
        return None
    return None


def _parse_reg(tok):
    """Parse 'X5' → 5."""
    if tok.startswith('X'):
        return int(tok[1:])
    raise ValueError(f'Not a register: {tok}')


def _parse_imm(tokens):
    """Parse hex digit tokens (H0-HF, optional <NEG>) into an immediate."""
    if not tokens:
        raise ValueError('No immediate tokens')
    neg = False
    digits = []
    for t in tokens:
        if t == '<NEG>':
            neg = True
        elif t.startswith('H') and len(t) == 2:
            digits.append(t[1])
        elif t == '<PAD>':
            continue
        else:
            raise ValueError(f'Unexpected token in immediate: {t}')
    if not digits:
        raise ValueError('No hex digits')
    val = int(''.join(digits), 16)
    return -val if neg else val


def verify_equivalence(model, original, decoded, n_inputs=64):
    """Check if two instructions are execution-equivalent.

    Returns (equiv, mean_dist) where equiv is True if mean_dist < 1e-6.
    """
    if original is None or decoded is None:
        return False, float('inf')

    rng = np.random.default_rng(42)
    ctx = make_ctx()
    diffs = []

    for _ in range(n_inputs):
        regs = random_regs(rng)
        pc = int(rng.integers(0, 1024)) * 4

        s1, pc1, m1 = run_instruction([original], regs=regs, pc=pc,
                                       rng=rng, _ctx=ctx)
        s2, pc2, m2 = run_instruction([decoded], regs=regs, pc=pc,
                                       rng=rng, _ctx=ctx)

        dv1 = extract_data_val(original, s1.regs, m1)
        dv2 = extract_data_val(decoded, s2.regs, m2)

        diff = abs(dv1 - dv2) + abs(pc1 - pc2)
        diffs.append(diff)

    mean_dist = np.mean(diffs)
    return mean_dist < 1e-6, mean_dist
