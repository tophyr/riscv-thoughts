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


# What the essence pool reads per slot, beyond the `value` channels. The pool is
# a symmetric sum over slots, so it is blind to anything not fed to it — and
# every ORDER channel withheld so far has turned out to cost real accuracy:
#   'write_time' — which step wrote the slot. Without it the essence sees an
#       unordered bag and cannot tell output 1 from output 2 (measured: pooled
#       bit accuracy 80.0 -> 98.6 at len=2).
#   'order'      — write_time + read_time. read_time IS the canonical input
#       position (operand r at step k is stamped 2(k+1)+r, so sorting recovers
#       first-read order). Without it the essence cannot place an operand in the
#       canonical frame when two instructions share a source register — measured
#       82.96% vs 99.999% on overlapping- vs disjoint-source pairs.
#   'event'      — the whole event block, i.e. pool_src == T. RISCV_DESIGN's
#       position: the essence/binding split is emergent, and the vector must NOT
#       be hand-partitioned into "these dims = operator, those = binding".
# 'none' reproduces the pre-ordering-fix model bit-for-bit, and is still the
# constructor/CLI default; 'write_time' is the adopted setting for new runs.
POOL_EXTRA = ('none', 'write_time', 'order', 'event')


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

    def __init__(self, d_value, d_event, d_content, n_regs=32, op_cell_depth=1,
                 n_pool_heads=1, pool_mode='pma', pool_extra='none',
                 read_time_mode='stamp', op_cell_rank=False):
        super().__init__()
        self.d_value = d_value
        self.d_event = d_event
        self.n_regs = n_regs
        self.n_pool_heads = n_pool_heads
        self.pool_mode = pool_mode
        assert pool_extra in POOL_EXTRA, f'bad pool_extra: {pool_extra}'
        self.pool_extra = pool_extra
        assert read_time_mode in ('stamp', 'rank'), read_time_mode
        self.read_time_mode = read_time_mode
        # Give the op-cell each operand's canonical read position. The pool
        # cannot use read-order (it never sees content, so it must apply it
        # unconditionally, which breaks commutative collapse); the op-cell can,
        # because it sees the opcode and both operands together and can
        # symmetrize per opcode. Requires read_time_mode='rank' to be the
        # canonical position. Widens op_cell's input, so old checkpoints only
        # load with this off.
        self.op_cell_rank = op_cell_rank
        assert not (op_cell_rank and read_time_mode != 'rank'), \
            'op_cell_rank needs read_time_mode=rank (stamp is not a position)'
        self.tag_proj = nn.Linear(1, d_value)
        self.op_cell = nn.Sequential(
            nn.Linear(d_content + 2 * d_value + (2 if op_cell_rank else 0),
                      4 * d_value), nn.ReLU(),
            nn.Linear(4 * d_value, d_value))
        # Extra RESIDUAL FFN blocks on op_cell's output (op_cell_depth > 1). The
        # op-cell is reused RECURRENTLY once per instruction, so a naive deep
        # stack becomes a depth×K gradient path that won't train — the skip
        # connection (res = res + blk(res)) is the highway that keeps it
        # trainable. depth=1 -> empty list -> op_cell is bit-identical to the
        # shallow baseline and its state_dict is UNCHANGED, so depth-1
        # checkpoints (incl. the frozen T1) still load. Static ModuleList loop
        # -> unrolls at trace time, no graph break under torch.compile.
        self.op_cell_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_value, 4 * d_value), nn.ReLU(),
                nn.Linear(4 * d_value, d_value))
            for _ in range(op_cell_depth - 1)])
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
        # essence pool: Pooling-by-Multihead-Attention (Set Transformer). k=
        # n_pool_heads learned seed QUERIES, each doing a permutation-invariant
        # attention pool over the register slots (softmax over slots, no
        # positional -> exactly rename-invariant). A single query (k=1) SUMS all
        # slots -> superimposes multiple output values irrecoverably; k>1 lets
        # different heads content-address different slots, keeping outputs
        # separable. Heads concatenate to an essence of the SAME width d_value
        # (each head d_value/k), so no downstream interface changes. pool_q is
        # stored FLAT (k*d_key,), so k=1 -> shape (d_value,) = the old
        # single-query param exactly -> existing checkpoints load unchanged.
        assert d_value % n_pool_heads == 0, 'd_value must be divisible by n_pool_heads'
        d_pool_in = d_value + {'none': 0, 'write_time': 1,
                               'order': 2, 'event': d_event}[pool_extra]
        self.pool_q = nn.Parameter(torch.randn(n_pool_heads * d_value) * 0.02)
        self.pool_k = nn.Linear(d_pool_in, d_value)
        self.pool_v = nn.Linear(d_pool_in, d_value // n_pool_heads)

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
        n_read = torch.zeros(content.shape[0], 1, device=device)
        write_time = torch.zeros(content.shape[0], self.n_regs, device=device)

        for k in range(K):
            a = active[:, k:k + 1]                          # (B,1)
            step_sig = float(k + 1) / float(K)              # normalized step ramp
            oh0 = F.one_hot(in0[:, k], self.n_regs).to(value.dtype)
            oh1 = F.one_hot(in1[:, k], self.n_regs).to(value.dtype)
            r0 = torch.bmm(oh0.unsqueeze(1), value).squeeze(1)   # (B,d_value)
            r1 = torch.bmm(oh1.unsqueeze(1), value).squeeze(1)
            c = content[:, k]
            oh0e = oh0 * not_x0_row
            oh1e = oh1 * not_x0_row
            # Read-order is stamped BEFORE the op-cell so this step's operands
            # carry their own canonical position. Nothing between the two points
            # reads read_time, so with op_cell_rank off the step is unchanged.
            read_time, n_read = self._stamp_reads(
                read_time, n_read, write_time, oh0e, oh1e, a, k)
            cell_in = [c, r0, r1]
            if self.op_cell_rank:
                # the position of the register each operand reads, normalized;
                # 0 for an x0 operand (not an input). Per-chunk scalars placed
                # by the wiring, so no register index enters — equivariance holds.
                scale = float(self.n_regs)
                cell_in += [((oh0e * read_time).sum(-1, keepdim=True) / scale),
                            ((oh1e * read_time).sum(-1, keepdim=True) / scale)]
            res = self.op_cell(torch.cat(cell_in, dim=-1))       # (B,d_value)
            for blk in self.op_cell_blocks:                      # residual depth
                res = res + blk(res)
            # write value at dest (x0 write = nop), active rows only.
            oh_out = F.one_hot(out[:, k], self.n_regs).to(value.dtype) \
                * not_x0_row
            w = (oh_out * a).unsqueeze(-1)                       # (B,n_regs,1)
            value = value * (1.0 - w) + w * res.unsqueeze(1)
            # event channels: content-dependent read pulse (x0 reads leave no
            # trace) + write pulse.
            p0 = self.read_pulse(torch.cat([c, r0], dim=-1)) + self.read_role[0]
            p1 = self.read_pulse(torch.cat([c, r1], dim=-1)) + self.read_role[1]
            event = event + (oh0e * a).unsqueeze(-1) * p0.unsqueeze(1)
            event = event + (oh1e * a).unsqueeze(-1) * p1.unsqueeze(1)
            event = event + (oh_out * a).unsqueeze(-1) \
                * self.write_pulse.view(1, 1, -1)
            wr = oh_out * a
            write_time = write_time * (1.0 - wr) + wr * step_sig
        # fold the order timestamps into event channels 0 (read) and 1 (write),
        # normalized to ~(0,1].
        order = torch.stack([read_time / (2.0 * K), write_time], dim=-1)
        event = event + F.pad(order, (0, self.d_event - 2))
        T = torch.cat([value, event], dim=-1)

        # Essence pool. k seed queries; heads concatenate to width d_value.
        # pool_src carries the per-slot write ORDER when enabled (see __init__):
        # register-indexed, so it permutes with a rename and the pool stays
        # exactly invariant.
        # Everything appended here is register-indexed, so a rename permutes it
        # with the slots and the pool stays exactly invariant. 'event' feeds the
        # whole slot state (pool_src == T); note its channels 0,1 carry the
        # timestamps already mixed with the read/write pulses, whereas 'order'
        # feeds the clean ramps.
        if self.pool_extra == 'write_time':
            pool_src = torch.cat([value, write_time.unsqueeze(-1)], dim=-1)
        elif self.pool_extra == 'order':
            pool_src = torch.cat([value, order], dim=-1)
        elif self.pool_extra == 'event':
            pool_src = T
        else:
            pool_src = value
        q = self.pool_q.view(self.n_pool_heads, -1)          # (k, d_value)
        kk = self.pool_k(pool_src)                           # (B, n_regs, d_value)
        logits = kk @ q.t()                                  # (B, n_regs, k)
        if self.pool_mode == 'compete':
            # Slot-Attention competition: each register slot softmax-ASSIGNS
            # across the k heads (dim=2), so slots divide among heads (no more
            # than one head can 'own' a slot) -> heads are forced to COVER
            # different slots instead of piling on the salient few. Then each
            # head takes a weighted mean of ITS slots (per-head normalize over
            # slots). The final per-head sum over slots is symmetric -> still
            # rename-invariant. (k=1 would be a uniform mean, so this mode is
            # for k>1 only; k=1 stays 'pma'.)
            att = torch.softmax(logits, dim=2)
            att = att / (att.sum(dim=1, keepdim=True) + 1e-8)
        else:  # 'pma': each head independently softmaxes over slots (no competition)
            att = torch.softmax(logits, dim=1)               # (B, n_regs, k) over slots
        vv = self.pool_v(pool_src)                           # (B, n_regs, d_value//k)
        heads = att.transpose(1, 2) @ vv                     # (B, k, d_value//k)
        essence = heads.reshape(value.shape[0], -1)          # (B, d_value)
        return T, essence

    def _stamp_reads(self, read_time, n_read, write_time, oh0e, oh1e, a, k):
        """Stamp this step's operands with their read ORDER; first read sticks.
        Operand 0 is stamped before operand 1, so same-instruction inputs are
        not tied (read_role marks the within-instruction order separately).

        'stamp' (legacy) writes 2(k+1)+role, which equals the canonical position
        minus one only while every read is fresh: a repeat read is correctly not
        re-stamped, leaving a hole, so every later operand overshoots — wrong
        exactly on the shared-source subset. 'rank' counts distinct registers
        read BEFORE written, which IS the canonical position, so no consumer
        needs a conversion. (It still diverges on a DEAD input: the canonical
        frame covers live inputs only, and liveness is not knowable here.)
        All values are per-chunk scalars placed through the wiring one-hots, so
        equivariance is unaffected.
        """
        rank = self.read_time_mode == 'rank'
        for oh, role in ((oh0e, 0.0), (oh1e, 1.0)):
            fresh = (read_time == 0).to(read_time.dtype)
            if rank:
                fresh = fresh * (write_time == 0).to(read_time.dtype)
            fr = (oh * a) * fresh
            if rank:
                n_read = n_read + fr.sum(-1, keepdim=True)
                read_time = read_time + fr * n_read
            else:
                read_time = read_time + fr * (2.0 * (k + 1) + role)
        return read_time, n_read

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
        # A learnable aggregator (CLS) token, read out in place of a pool: a
        # mean would average the place-valued immediate digits uniformly and
        # destroy the additive magnitude, so the transformer's layers learn the
        # combination through this token instead.
        self.cls = nn.Parameter(torch.randn(d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 4,
            dropout=0.0, batch_first=True)
        # enable_nested_tensor=False: the nested-tensor fast path is not
        # torch.compile-able and shatters the full-step CUDA graph. Dense path
        # is numerically identical (padding-skip optimization only).
        self.content_encoder = nn.TransformerEncoder(
            layer, n_layers, enable_nested_tensor=False)

    def forward(self, tokens, pad):
        """(N, L) tokens + pad -> (N, d_model) content.

        Prepend the never-masked aggregator token; read out its final state. It
        attends only to opcode+immediate content (registers masked), so content
        stays rename-invariant. A fully-masked row (invalid window) leaves the
        token attending to itself only — finite, no NaN guard needed. The
        content tokens keep their win_pos_emb place (added before the prepend),
        so the immediate's place value is preserved."""
        N, L = tokens.shape
        device = tokens.device
        x = self.tok_emb(tokens) + self.win_pos_emb(
            torch.arange(L, device=device))
        is_reg = (tokens >= REG_TOKEN_LO) & (tokens < REG_TOKEN_HI)
        content_pad = pad | is_reg
        cls = self.cls.view(1, 1, -1).expand(N, 1, -1)
        x = torch.cat([cls, x], dim=1)                          # (N, 1+L, d)
        cls_pad = torch.zeros(N, 1, dtype=content_pad.dtype, device=device)
        attn_pad = torch.cat([cls_pad, content_pad], dim=1)
        x = self.content_encoder(x, src_key_padding_mask=attn_pad)
        return x[:, 0]


# Head hidden capacities. Never tuned, so they live here as named constants
# rather than constructor params. vp_head is large and fixed (does not scale
# with d_out): the capacity is "don't be the bottleneck" rather than a fitted
# size — vp is the training objective, not a diagnostic, so the head must not
# be what limits it.
def _build_vp_head(vp_in, vp_out, depth):
    """vp value-prediction head with `depth` hidden layers of _VP_HIDDEN.
    depth=0 -> a single linear readout. depth=4 reproduces the historical
    head exactly (same nn.Sequential layout -> same state_dict keys, so
    existing checkpoints load unchanged)."""
    if depth <= 0:
        return nn.Linear(vp_in, vp_out)
    layers = [nn.Linear(vp_in, _VP_HIDDEN), nn.ReLU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(_VP_HIDDEN, _VP_HIDDEN), nn.ReLU()]
    layers.append(nn.Linear(_VP_HIDDEN, vp_out))
    return nn.Sequential(*layers)


_IN_SCORE_HIDDEN = 256
_VP_HIDDEN = 1024
_VP_HEAD_DEPTH = 4     # vp_head hidden layers (default). 0 = linear readout —
                       # weakening the head forces the essence to expose value
                       # structure geometrically instead of the head untangling
                       # it (the essence cone-collapse / dim-underuse fix).
_VP_VALUE_BITS = 32    # two's-complement bit width of the vp value target (RV32)
_IMM_N_FREQ = 12       # rotary immediate: power-of-2 periods 2..4096 (2*this = target dim)


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

    def __init__(self, *, d_content, d_out, d_event, op_cell_depth=1,
                 vp_head_depth=_VP_HEAD_DEPTH, n_pool_heads=1, pool_mode='pma',
                 pool_extra='none', read_time_mode='stamp',
                 op_cell_rank=False):
        super().__init__()
        self.d_content = d_content
        self.d_out = d_out
        self.d_event = d_event
        self.n_regs = N_REGS
        self.max_input_slots = MAX_INPUT_SLOTS
        self.max_output_slots = MAX_OUTPUT_SLOTS

        self.core = RegisterStateMachine(
            d_value=d_out, d_event=d_event, d_content=d_content, n_regs=N_REGS,
            op_cell_depth=op_cell_depth, n_pool_heads=n_pool_heads,
            pool_mode=pool_mode, pool_extra=pool_extra,
            read_time_mode=read_time_mode, op_cell_rank=op_cell_rank)

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
        # value-prediction (canonical): predict per-anchor per-out-slot output
        # values from (essence, canonical input-slot values). Per slot: 32
        # two's-complement bit-logits + 1 smooth-magnitude scalar (inputs fed
        # likewise). Per-bit gives uniform resolution — the low bit (the +1
        # immediate distinguisher) is equally weighted and carry-free — so the
        # essence is pressured to retain immediates, which the older single
        # loglog-scalar target crushed away.
        vp_in = d_out + MAX_INPUT_SLOTS * (_VP_VALUE_BITS + 1)
        vp_out = MAX_OUTPUT_SLOTS * (_VP_VALUE_BITS + 1)
        self.vp_head = _build_vp_head(vp_in, vp_out, vp_head_depth)
        # immediate head (T1 only): a LINEAR readout of a rotary/Fourier encoding
        # of the instruction's immediate. Linear so the rotary structure is
        # forced to live in the essence (imm+δ -> a consistent rotation of the
        # essence's immediate subspace, opcode-independent), not hidden in a
        # nonlinear head. Present but unused at T2 (chunk essence has no single
        # immediate). See compressor.train._immediate_rotary.
        self.imm_head = nn.Linear(d_out, 2 * _IMM_N_FREQ)

    def encode_state(self, content_seq, in0, in1, out, active, tags):
        """content_seq (B, K, d_content), wiring (B, K) long, active (B, K)
        float, tags (B, n_regs) -> (T, essence)."""
        return self.core(content_seq, in0, in1, out, active, tags)

    def encode(self, content_seq, in0, in1, out, active, tags):
        """The rename-invariant essence (B, d_out)."""
        return self.encode_state(content_seq, in0, in1, out, active, tags)[1]


# Compressor head/loss interface that T1Compressor delegates to its compressor.
_COMPRESSOR_HEADS = ('live_in_head', 'live_out_head', 'in_score_head',
                     'out_score_head', 'pc_writes_head', 'vp_head', 'imm_head')


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
                 d_out, d_event, op_cell_depth=1, vp_head_depth=_VP_HEAD_DEPTH,
                 n_pool_heads=1, pool_mode='pma', pool_extra='none',
                 read_time_mode='stamp', op_cell_rank=False):
        super().__init__()
        self.embedder = TokenEmbedder(
            vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, max_window=max_window)
        self.compressor = Compressor(
            d_content=d_model, d_out=d_out, d_event=d_event,
            op_cell_depth=op_cell_depth, vp_head_depth=vp_head_depth,
            n_pool_heads=n_pool_heads, pool_mode=pool_mode,
            pool_extra=pool_extra,
            read_time_mode=read_time_mode, op_cell_rank=op_cell_rank)
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


def T2Compressor(*, d_t1, d_out, d_event, op_cell_depth=1,
                 vp_head_depth=_VP_HEAD_DEPTH, n_pool_heads=1, pool_mode='pma',
                 pool_extra='none', read_time_mode='stamp',
                 op_cell_rank=False):
    """T2 preset: a Compressor fed a lower tier's essence (d_content=d_t1)."""
    return Compressor(d_content=d_t1, d_out=d_out, d_event=d_event,
                      op_cell_depth=op_cell_depth, vp_head_depth=vp_head_depth,
                      n_pool_heads=n_pool_heads, pool_mode=pool_mode,
                      pool_extra=pool_extra,
                      read_time_mode=read_time_mode,
                      op_cell_rank=op_cell_rank)


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
