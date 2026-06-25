"""The equivariant Compressor, its token front-end, and the Decoder.

ONE `Compressor` class serves every tier: a
rename-equivariant register-indexed encoder built on the shared
RegisterStateMachine. It ALWAYS ingests a sequence of per-instruction content
vectors and, per instruction, runs one step of the machine:

    read operand slots -> op-cell(content, in0, in1) -> write dest slot

The tier-specific piece is only what produces those content vectors:
  - T1: a standalone `TokenEmbedder` turns one instruction's opcode+immediate
    tokens into a content vector (registers stripped to wiring); `T1Compressor`
    is the embedder feeding a Compressor at seq_len=1.
  - T2+: the content vectors ARE a frozen lower tier's per-instruction essence
    (the tier contract — never a re-embedded opcode); `T2Compressor` is a bare
    Compressor, run over seq_len=K.

The emitted object is one register-indexed state T (B, n_regs, d). A register
rename permutes the slots exactly (no per-register parameters — the only
register-indexed input is an anonymous per-chunk tag), so:

  - essence = a permutation-invariant attention pool over the register axis
    (the operator axis; value_predict / pc_writes read it),
  - binding = the per-slot remainder (live_in/out + in/out slot order; the
    binding heads read it per-slot).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from emulator import R_TYPE, I_TYPE, B_TYPE, LOAD_TYPE, STORE_TYPE
from tokenizer import REG_TOKEN_LO, REG_TOKEN_HI
from datagen import N_REGS, MAX_INPUT_SLOTS, MAX_OUTPUT_SLOTS


# x0 is the hardwired-zero, non-renameable slot. Register renames permute
# slots 1..n_regs-1 only; slot 0 seeds to zero, reads as zero, never written.
X0_SLOT = 0


def instruction_wiring(instr):
    """Map an Instruction to its GP-register wiring for the state machine:
    (in0, in1, out) slot indices, x0 (=0) standing for an absent operand.
    in0/in1 are the source registers read; out is the written GP register
    (0 when none is written — branches/stores, or an x0 write).

    PC effects are not a GP slot here (they are recovered from the rename-
    invariant essence, which encodes the opcode). Memory ops fall through
    with their GP base/value reads but carry no behavioral supervision.

    Argument orders follow tokenizer.decode_instruction:
      R: (rd,rs1,rs2)  I: (rd,rs1,imm)  B: (rs1,rs2,imm)
      LOAD: (rd,imm,rs1)  STORE: (rs2,imm,rs1)  JALR: (rd,rs1,imm)
      LUI/AUIPC/JAL: (rd,imm)
    """
    op, a = instr.opcode, instr.args
    if op in R_TYPE:
        return a[1], a[2], a[0]
    if op in I_TYPE:
        return a[1], 0, a[0]
    if op in B_TYPE:
        return a[0], a[1], 0
    if op in LOAD_TYPE:
        return a[2], 0, a[0]
    if op in STORE_TYPE:
        return a[2], a[0], 0
    if op == 'JALR':
        return a[1], 0, a[0]
    if op in ('LUI', 'AUIPC', 'JAL'):
        return 0, 0, a[0]
    return 0, 0, 0


class RegisterStateMachine(nn.Module):
    """Shared equivariant core of T1 (seq_len=1) and T2 (seq_len=K).

    Per register slot the state is [value | event]:
      value (d_value) — set on write, PRESERVED on read; the dataflow the
        op-cell threads (multi-hop value-numbering at T2).
      event (d_event) — accumulates read/write events with a read-order
        ramp, so live_in/in_slot are recoverable from the emitted object.

    Each step, per instruction:
      r0, r1 = value[in0], value[in1]          (one-hot reads)
      res    = op_cell(content, r0, r1)        (role-aware combine)
      value[out] = res                         (one-hot write; x0 = nop)
      event[in0],[in1] += read_pulse(content, r)   (content-dependent)
      event[out]       += write_pulse

    No per-register parameters: slots are seeded only with an anonymous
    per-chunk tag. Reads/writes are pure one-hot tensor ops, so a register
    rename permutes the state exactly (equivariant by construction) and any
    pool over the register axis is invariant. Vectorized (no per-slot Python
    loop) so the step captures into the compiled CUDA-graph train step.
    """

    def __init__(self, d_value, d_event, d_content, n_regs=32):
        super().__init__()
        self.d_value = d_value
        self.d_event = d_event
        self.n_regs = n_regs
        self.tag_proj = nn.Linear(1, d_value)
        self.op_cell = nn.Sequential(
            nn.Linear(d_content + 2 * d_value, 4 * d_value), nn.ReLU(),
            nn.Linear(4 * d_value, d_value))
        # content-dependent read pulse: a behaviorally-irrelevant read
        # (AND rs1,x0) can be marked differently from a relevant one, so the
        # magnitude-filtered live_in target is recoverable.
        self.read_pulse = nn.Linear(d_content + d_value, d_event)
        # Per-operand-role bias on the read event. in_slot is read-ORDER;
        # within one instruction both operands are read in the same step, so
        # the cross-step read-order ramp can't separate them — this role bias
        # (operand position 0 vs 1, register-independent so equivariance is
        # preserved) is what lets in_score recover which operand is slot 0.
        self.read_role = nn.Parameter(torch.randn(2, d_event) * 0.1)
        self.write_pulse = nn.Parameter(torch.randn(d_event) * 0.1)
        # essence pool: single learned query, no positional -> the softmax
        # over the register axis is exactly permutation-invariant.
        self.pool_q = nn.Parameter(torch.randn(d_value) * 0.02)
        self.pool_k = nn.Linear(d_value, d_value)
        self.pool_v = nn.Linear(d_value, d_value)

    def forward(self, content, in0, in1, out, active, tags):
        """content (B,K,d_content); in0/in1/out (B,K) long; active (B,K)
        float; tags (B,n_regs) float. Returns (T, essence):
          T (B, n_regs, d_value+d_event), essence (B, d_value)."""
        device = content.device
        K = content.shape[1]
        not_x0 = torch.ones(self.n_regs, device=device)
        not_x0[X0_SLOT] = 0.0
        not_x0_row = not_x0.view(1, -1)
        value = self.tag_proj(tags.unsqueeze(-1)) * not_x0.view(1, -1, 1)
        event = torch.zeros(
            content.shape[0], self.n_regs, self.d_event, device=device)
        # Per-slot ORDER timestamps (the cross-instruction analog of read_role):
        #   read_time  = step of FIRST read  -> decodes in_slot (first-read order)
        #   write_time = step of LAST write  -> decodes out_slot (SSA write order)
        # Register-INDEPENDENT step values placed by the (permuted) wiring, so a
        # rename permutes them identically — equivariance preserved.
        read_time = torch.zeros(content.shape[0], self.n_regs, device=device)
        write_time = torch.zeros(content.shape[0], self.n_regs, device=device)

        for k in range(K):
            a = active[:, k:k + 1]                          # (B,1)
            step_sig = float(k + 1) / float(K)              # normalized step ramp
            oh0 = F.one_hot(in0[:, k], self.n_regs).to(value.dtype)
            oh1 = F.one_hot(in1[:, k], self.n_regs).to(value.dtype)
            r0 = torch.bmm(oh0.unsqueeze(1), value).squeeze(1)   # (B,d_value)
            r1 = torch.bmm(oh1.unsqueeze(1), value).squeeze(1)
            c = content[:, k]
            res = self.op_cell(torch.cat([c, r0, r1], dim=-1))   # (B,d_value)
            # write value at dest (x0 write = nop), active rows only.
            oh_out = F.one_hot(out[:, k], self.n_regs).to(value.dtype) \
                * not_x0_row
            w = (oh_out * a).unsqueeze(-1)                       # (B,n_regs,1)
            value = value * (1.0 - w) + w * res.unsqueeze(1)
            # event channels: content-dependent read pulse (x0 reads leave no
            # trace) + write pulse.
            oh0e = oh0 * not_x0_row
            oh1e = oh1 * not_x0_row
            p0 = self.read_pulse(torch.cat([c, r0], dim=-1)) + self.read_role[0]
            p1 = self.read_pulse(torch.cat([c, r1], dim=-1)) + self.read_role[1]
            event = event + (oh0e * a).unsqueeze(-1) * p0.unsqueeze(1)
            event = event + (oh1e * a).unsqueeze(-1) * p1.unsqueeze(1)
            event = event + (oh_out * a).unsqueeze(-1) \
                * self.write_pulse.view(1, 1, -1)
            # order timestamps: first read sticks (operand 0 BEFORE operand 1
            # within a step, so same-instruction inputs aren't tied — the
            # within-instruction order read_role also marks); last write
            # overwrites (SSA write order).
            fr0 = (oh0e * a) * (read_time == 0).to(value.dtype)
            read_time = read_time + fr0 * (2.0 * (k + 1))
            fr1 = (oh1e * a) * (read_time == 0).to(value.dtype)
            read_time = read_time + fr1 * (2.0 * (k + 1) + 1.0)
            wr = oh_out * a
            write_time = write_time * (1.0 - wr) + wr * step_sig
        # fold the order timestamps into event channels 0 (read) and 1 (write),
        # normalized to ~(0,1].
        order = torch.stack([read_time / (2.0 * K), write_time], dim=-1)
        event = event + F.pad(order, (0, self.d_event - 2))
        T = torch.cat([value, event], dim=-1)

        kk = self.pool_k(value)
        att = torch.softmax((kk * self.pool_q.view(1, 1, -1)).sum(-1), dim=1)
        essence = (att.unsqueeze(-1) * self.pool_v(value)).sum(1)
        return T, essence

    def apply_slot_head(self, head, T):
        """Per-slot readout: Linear(d_slot,1) over every register slot ->
        (B, n_regs). Equivariant (a rename permutes the slots, permuting the
        scores identically)."""
        return head(T).squeeze(-1)


class TokenEmbedder(nn.Module):
    """Tier-0 front-end: embed one instruction's opcode+immediate tokens into a
    rename-invariant content vector — the input a T1 Compressor's core threads.
    Register-token positions are masked out of attention and the mean pool, so
    the content carries no register identity (registers enter the core as wiring
    + tags, never as content). T2+ feed their Compressor a lower tier's essence
    instead of token embeddings, so they need no embedder.
    """

    def __init__(self, *, vocab_size, d_model, n_heads, n_layers, max_window):
        super().__init__()
        self.d_model = d_model
        self.max_window = max_window
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.win_pos_emb = nn.Embedding(max_window, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 4,
            dropout=0.0, batch_first=True)
        # enable_nested_tensor=False: the nested-tensor fast path is not
        # torch.compile-able and shatters the full-step CUDA graph. Dense path
        # is numerically identical (padding-skip optimization only).
        self.content_encoder = nn.TransformerEncoder(
            layer, n_layers, enable_nested_tensor=False)

    def forward(self, tokens, pad):
        """(N, L) tokens + pad -> (N, d_model) content."""
        L = tokens.shape[1]
        device = tokens.device
        x = self.tok_emb(tokens) + self.win_pos_emb(
            torch.arange(L, device=device))
        is_reg = (tokens >= REG_TOKEN_LO) & (tokens < REG_TOKEN_HI)
        content_pad = pad | is_reg
        # Guard a fully-masked row (an invalid window with no content token,
        # e.g. all-register / no-opcode) so the attention softmax doesn't NaN.
        # Only position 0 of such a row is unmasked, and the pool still excludes
        # register tokens — valid instructions (opcode at 0) are unaffected and
        # content stays rename-invariant. Col-0 slice assign is compile-safe.
        attn_pad = content_pad.clone()
        attn_pad[:, 0] = content_pad[:, 0] & ~content_pad.all(dim=1)
        x = self.content_encoder(x, src_key_padding_mask=attn_pad)
        keep = (~content_pad).to(x.dtype).unsqueeze(-1)         # (N,L,1)
        return (x * keep).sum(1) / keep.sum(1).clamp(min=1.0)


# Head hidden capacities. Never tuned, so they live here as named constants
# rather than constructor params. Note vp_head is large and fixed (doesn't
# scale with d_out) — value_predict is a diagnostic probe (GVN collapse, not
# vp, is the headline metric), and its error is at an intrinsic floor, so the
# capacity is "don't be the bottleneck" rather than a fitted size.
_IN_SCORE_HIDDEN = 256
_VP_HIDDEN = 1024
_VP_DEPTH = 3          # number of hidden _VP_HIDDEN -> _VP_HIDDEN blocks


class Compressor(nn.Module):
    """The equivariant register-state-machine encoder (one per tier).

    Ingests a sequence of per-instruction CONTENT vectors (B, K, d_content) —
    token embeddings for T1 (via TokenEmbedder), a frozen lower tier's essence
    for T2+ (the tier contract) — threads them through the chunk's register
    state by each instruction's wiring, and emits one register-indexed state
    T (B, n_regs, d_out+d_event). A register rename permutes the slots exactly
    (no per-register parameters); essence = the permutation-invariant pool
    (operator axis), the per-slot remainder = binding.

    n_regs and the slot widths are fixed constants (RV32I has 32 registers; the
    slot widths are the supervision schema's). The state-geometry dims
    d_content, d_out, d_event are required — each tier specifies them.
    """

    def __init__(self, *, d_content, d_out, d_event):
        super().__init__()
        self.d_content = d_content
        self.d_out = d_out
        self.d_event = d_event
        self.n_regs = N_REGS
        self.max_input_slots = MAX_INPUT_SLOTS
        self.max_output_slots = MAX_OUTPUT_SLOTS

        self.core = RegisterStateMachine(
            d_value=d_out, d_event=d_event, d_content=d_content, n_regs=N_REGS)

        # Binding heads read the per-slot state T (d_out+d_event), one score per
        # register slot (equivariant); pc_writes + value_predict read the
        # rename-invariant essence. in_score is an MLP (relevance × order
        # product a Linear can't do); the rest are Linear.
        d_slot = d_out + d_event
        self.live_in_head = nn.Linear(d_slot, 1)
        self.live_out_head = nn.Linear(d_slot, 1)
        self.in_score_head = nn.Sequential(
            nn.Linear(d_slot, _IN_SCORE_HIDDEN), nn.ReLU(),
            nn.Linear(_IN_SCORE_HIDDEN, 1))
        self.out_score_head = nn.Linear(d_slot, 1)
        self.pc_writes_head = nn.Linear(d_out, 1)
        # value-prediction (canonical): predict per-anchor per-out-slot
        # output values from (essence, canonical input-slot values).
        vp_layers = [nn.Linear(d_out + MAX_INPUT_SLOTS, _VP_HIDDEN), nn.ReLU()]
        for _ in range(_VP_DEPTH):
            vp_layers += [nn.Linear(_VP_HIDDEN, _VP_HIDDEN), nn.ReLU()]
        vp_layers.append(nn.Linear(_VP_HIDDEN, MAX_OUTPUT_SLOTS))
        self.vp_head = nn.Sequential(*vp_layers)

    def encode_state(self, content_seq, in0, in1, out, active, tags):
        """content_seq (B, K, d_content), wiring (B, K) long, active (B, K)
        float, tags (B, n_regs) -> (T, essence)."""
        return self.core(content_seq, in0, in1, out, active, tags)

    def encode(self, content_seq, in0, in1, out, active, tags):
        """The rename-invariant essence (B, d_out)."""
        return self.encode_state(content_seq, in0, in1, out, active, tags)[1]


# Compressor head/loss interface that T1Compressor delegates to its compressor.
_COMPRESSOR_HEADS = ('live_in_head', 'live_out_head', 'in_score_head',
                     'out_score_head', 'pc_writes_head', 'vp_head')


class T1Compressor(nn.Module):
    """T1 = a TokenEmbedder feeding a Compressor: the embedder turns one
    instruction's tokens into the content vector the compressor threads
    (seq_len=1). Both are trained jointly. Delegates the compressor's head /
    dim interface (so it is interchangeable with a bare Compressor for
    binding_losses, value-prediction, and eval) and adds a tokens-in encode
    path. (Checkpoint compat: an old flat T1 state_dict splits cleanly into
    `embedder.*` + `compressor.*` — see compressor.train._split_t1_state.)
    """

    def __init__(self, *, vocab_size, d_model, n_heads, n_layers, max_window,
                 d_out, d_event):
        super().__init__()
        self.embedder = TokenEmbedder(
            vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, max_window=max_window)
        self.compressor = Compressor(
            d_content=d_model, d_out=d_out, d_event=d_event)
        # dim interface (plain ints — delegated by value, not submodules)
        self.n_regs = self.compressor.n_regs
        self.d_out = self.compressor.d_out
        self.d_event = self.compressor.d_event
        self.max_input_slots = self.compressor.max_input_slots
        self.max_output_slots = self.compressor.max_output_slots

    def __getattr__(self, name):
        # Delegate the binding/vp heads to the compressor WITHOUT re-registering
        # them as our own submodules (which would duplicate them in state_dict).
        # nn.Module.__getattr__ handles real submodules/params; we only reach
        # here for names it didn't find.
        if name in _COMPRESSOR_HEADS:
            return getattr(self.compressor, name)
        return super().__getattr__(name)

    def encode_state(self, tokens, pad, in0, in1, out, tags):
        content = self.embedder(tokens, pad).unsqueeze(1)           # (B,1,d)
        active = torch.ones(content.shape[0], 1, device=content.device)
        return self.compressor.encode_state(
            content, in0.unsqueeze(1), in1.unsqueeze(1), out.unsqueeze(1),
            active, tags)

    def encode(self, tokens, pad, in0, in1, out, tags):
        """The T1 'vector' = the rename-invariant essence (B, d_out)."""
        return self.encode_state(tokens, pad, in0, in1, out, tags)[1]


def T2Compressor(*, d_t1, d_out, d_event):
    """T2 preset: a Compressor fed a lower tier's essence (d_content=d_t1)."""
    return Compressor(d_content=d_t1, d_out=d_out, d_event=d_event)


class Decoder(nn.Module):
    """Autoregressive decoder conditioned on an emission vector."""

    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_emb,
                 max_seq_len=64, dropout=0.0, n_memory_tokens=1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_memory_tokens = n_memory_tokens
        self.emission_proj = nn.Linear(d_emb, n_memory_tokens * d_model)
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, n_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, emission_vecs, target_tokens, target_padding_mask):
        N, T = target_tokens.shape
        device = target_tokens.device
        memory = self.emission_proj(emission_vecs).view(
            N, self.n_memory_tokens, self.d_model)
        pos = torch.arange(T, device=device)
        tgt = self.tok_emb(target_tokens) + self.pos_emb(pos)
        causal_mask = torch.ones(
            T, T, dtype=torch.bool, device=device).triu(diagonal=1)
        out = self.decoder(
            tgt, memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=target_padding_mask,
            tgt_is_causal=False)  # explicit (not None) → skips
                                  # _detect_is_causal_mask's tensor->bool check
                                  # (a graph break) while still applying the
                                  # explicit causal mask. (True invokes a
                                  # fast path that doesn't capture under compile.)
        return self.head(out)
