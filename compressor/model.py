"""The equivariant Compressor and its token front-end.

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
register-indexed input is each input's canonical read position), so:

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
      event[in0],[in1] += read_pulse(op_cell hidden)[role]   (whole operand context)
      event[out]       += write_pulse

    No per-register parameters: a slot is seeded only with the canonical read
    position of the input it holds. Reads/writes are pure one-hot tensor ops,
    so a register rename permutes the state exactly (equivariant by
    construction) and any pool over the register axis is invariant.
    Vectorized (no per-slot Python loop) so the step captures into the
    compiled CUDA-graph train step.
    """

    def __init__(self, d_value, d_event, d_content, n_regs=32, op_cell_depth=1,
                 n_pool_heads=1):
        super().__init__()
        self.d_value = d_value
        self.d_event = d_event
        self.n_regs = n_regs
        self.n_pool_heads = n_pool_heads
        # An input register's leaf identity IS its canonical read position,
        # written to its slot on first read. That is what distinguishes two
        # inputs from each other without naming either — a per-register
        # embedding would have to be constant to stay equivariant, which
        # destroys distinctness. Position is assigned by read ORDER, so a
        # rename permutes it with the slots and equivariance is exact; it is
        # deterministic, so the essence is a deterministic function of the
        # instruction; and it is the frame the value-prediction target is
        # already expressed in.
        self.rank_emb = nn.Embedding(n_regs, d_value)
        self.op_cell = nn.Sequential(
            nn.Linear(d_content + 2 * d_value, 4 * d_value), nn.ReLU(),
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
        # Read pulses, one per operand POSITION, from the op-cell's hidden layer:
        # whether a read matters depends on the opcode AND on the other operand
        # (`AND rs,x0` vs `AND rs,rt`; x0 arrives as the zero vector, aliasing
        # as r0 == r1), which no function of one operand can see.
        self.read_pulse = nn.Linear(4 * d_value, 2 * d_event)
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
        # pool_v is d_value wide and each head reads its OWN slice: one SHARED
        # d_value/k projection would cap what the essence can carry about any
        # single slot at d_value/k, however wide d_value is (it cost the 12-bit
        # I-type immediate). Per-head slices also let head count and per-slot
        # fidelity scale independently.
        # The pool is a symmetric sum over slots, so it is structurally blind to
        # which register was written FIRST — and the essence has to encode an
        # ORDERED function ("output 1 is this, output 2 is that"). Each slot's
        # write STEP is therefore appended to what the pool reads: one extra
        # channel, register-indexed, so a rename permutes it with the slots and
        # invariance stays exact. A commutative operand swap changes neither
        # which slot was written nor when, so equivalence collapse is not put at
        # risk — unlike read-order, which the pool must not see.
        assert d_value % n_pool_heads == 0, 'd_value must be divisible by n_pool_heads'
        d_pool_in = d_value + 1
        self.pool_q = nn.Parameter(torch.randn(n_pool_heads * d_value) * 0.02)
        self.pool_k = nn.Linear(d_pool_in, d_value)
        self.pool_v = nn.Linear(d_pool_in, d_value)

    def forward(self, content, in0, in1, out, active):
        """content (B,K,d_content); in0/in1/out (B,K) long; active (B,K)
        float. Returns (T, essence):
          T (B, n_regs, d_value+d_event), essence (B, d_value)."""
        device = content.device
        K = content.shape[1]
        not_x0 = torch.ones(self.n_regs, device=device)
        not_x0[X0_SLOT] = 0.0
        not_x0_row = not_x0.view(1, -1)
        # Slots start at exactly zero; a slot acquires its leaf identity only
        # when first READ (see rank_emb below), so a register the chunk never
        # touches contributes nothing to the pool.
        value = torch.zeros(content.shape[0], self.n_regs, self.d_value,
                            device=device)
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
        # Slots any instruction read or wrote. Every other slot is exactly zero
        # throughout, which lets the pool skip projecting it.
        touched = torch.zeros(content.shape[0], self.n_regs, device=device)

        for k in range(K):
            a = active[:, k:k + 1]                          # (B,1)
            step_sig = float(k + 1) / float(K)              # normalized step ramp
            oh0 = F.one_hot(in0[:, k], self.n_regs).to(value.dtype)
            oh1 = F.one_hot(in1[:, k], self.n_regs).to(value.dtype)
            c = content[:, k]
            oh0e = oh0 * not_x0_row
            oh1e = oh1 * not_x0_row
            # Read-order is stamped BEFORE the operands are read, so this
            # step's own inputs can be seeded from their positions.
            read_time, n_read, firsts, ranks = self._stamp_reads(
                read_time, n_read, write_time, oh0e, oh1e, a)
            # An input's leaf identity IS its canonical position, written on
            # first read. Indexed by read ORDER, not by register, so a rename
            # permutes it with the slots and equivariance stays exact.
            for fr, rk in zip(firsts, ranks):
                e = self.rank_emb(
                    rk.clamp(max=self.n_regs - 1).long()).squeeze(1)
                fr = fr.unsqueeze(-1)
                value = value * (1.0 - fr) + fr * e.unsqueeze(1)
            r0 = torch.bmm(oh0.unsqueeze(1), value).squeeze(1)   # (B,d_value)
            r1 = torch.bmm(oh1.unsqueeze(1), value).squeeze(1)
            hidden = F.relu(self.op_cell[0](torch.cat([c, r0, r1], dim=-1)))
            res = self.op_cell[2](hidden)                        # (B,d_value)
            for blk in self.op_cell_blocks:                      # residual depth
                res = res + blk(res)
            # write value at dest (x0 write = nop), active rows only.
            oh_out = F.one_hot(out[:, k], self.n_regs).to(value.dtype) \
                * not_x0_row
            w = (oh_out * a).unsqueeze(-1)                       # (B,n_regs,1)
            value = value * (1.0 - w) + w * res.unsqueeze(1)
            # event channels: per-position read pulses (x0 reads leave no trace
            # on the x0 slot) + write pulse.
            p0, p1 = self.read_pulse(hidden).view(-1, 2, self.d_event).unbind(1)
            event = event + (oh0e * a).unsqueeze(-1) * p0.unsqueeze(1)
            event = event + (oh1e * a).unsqueeze(-1) * p1.unsqueeze(1)
            event = event + (oh_out * a).unsqueeze(-1) \
                * self.write_pulse.view(1, 1, -1)
            wr = oh_out * a
            write_time = write_time * (1.0 - wr) + wr * step_sig
            touched = torch.clamp(touched + (oh0e + oh1e + oh_out) * a, max=1.0)
        # fold the order timestamps into event channels 0 (read) and 1 (write),
        # normalized to ~(0,1].
        order = torch.stack([read_time / (2.0 * K), write_time], dim=-1)
        event = event + F.pad(order, (0, self.d_event - 2))
        T = torch.cat([value, event], dim=-1)

        # Essence pool. k seed queries; heads concatenate to width d_value. The
        # pool reads each slot's value channels plus its write step — both
        # register-indexed, so a rename permutes them with the slots and the
        # pool stays exactly invariant. Read-order must NOT be added here: the
        # pool never sees content, so it would apply read-order unconditionally
        # and stop being blind to a commutative operand swap.
        pool_src = torch.cat([value, write_time.unsqueeze(-1)], dim=-1)
        q = self.pool_q.view(self.n_pool_heads, -1)          # (k, d_value)
        kk, vv = self._project_touched(pool_src, touched, 3 * K)
        logits = kk @ q.t()                                  # (B, n_regs, k)
        # Each head independently softmaxes over the slots; the per-head sum
        # over slots is symmetric, hence exactly rename-invariant.
        att = torch.softmax(logits, dim=1)                   # (B, n_regs, k)
        vv = vv.view(                                        # per-head slices
            value.shape[0], self.n_regs, self.n_pool_heads, -1)
        heads = torch.einsum('bsk,bskd->bkd', att, vv)       # (B, k, d_value//k)
        essence = heads.reshape(value.shape[0], -1)          # (B, d_value)
        return T, essence

    def _project_touched(self, pool_src, touched, max_touched):
        """(pool_k(pool_src), pool_v(pool_src)), projecting only the slots the
        chunk touched. An untouched slot's pool_src row is exactly zero, so its
        projection is exactly the bias; at most `max_touched` slots (two reads
        and a write per instruction) carry anything else. Which untouched slots
        fill a short row's spare picks does not matter — they project to bias."""
        R = pool_src.shape[1]
        n = min(max_touched, R)
        # One-hot selection rather than gather/scatter, like the core's own
        # reads and writes: same values, and a deterministic backward.
        sel = F.one_hot(touched.topk(n, dim=1).indices, R).to(pool_src.dtype)
        src = sel @ pool_src                                 # (B, n, d_in)
        uncovered = (1.0 - sel.sum(1)).unsqueeze(-1)         # (B, R, 1)
        return [sel.transpose(1, 2) @ proj(src) + uncovered * proj.bias
                for proj in (self.pool_k, self.pool_v)]

    def _stamp_reads(self, read_time, n_read, write_time, oh0e, oh1e, a):
        """Stamp this step's operands with their canonical read POSITION: the
        running count of distinct registers read BEFORE they are written. First
        read sticks; operand 0 is stamped before operand 1, so two inputs of one
        instruction get different positions (the per-position read pulses mark which
        operand slot each filled).

        A register read twice receives one position twice, which is correct —
        it is one input. The target's canonical frame counts a dead
        instruction's reads too, so given T1's wiring the count is the frame's
        exactly. All values are per-chunk scalars placed through the wiring
        one-hots, so equivariance is unaffected.
        """
        firsts, ranks = [], []
        for oh in (oh0e, oh1e):
            fresh = ((read_time == 0).to(read_time.dtype)
                     * (write_time == 0).to(read_time.dtype))
            fr = (oh * a) * fresh
            n_read = n_read + fr.sum(-1, keepdim=True)
            read_time = read_time + fr * n_read
            firsts.append(fr)
            ranks.append(n_read)
        return read_time, n_read, firsts, ranks

    def apply_slot_head(self, head, T):
        """Per-slot readout: Linear(d_slot,1) over every register slot ->
        (B, n_regs). Equivariant (a rename permutes the slots, permuting the
        scores identically)."""
        return head(T).squeeze(-1)


def apply_head_touched(head, T, max_touched):
    """head over T's register axis (B, R, d) -> (B, R, out), evaluated only on
    the slots that can be nonzero. A slot the chunk never touched is an exact
    zero row, so its output is head(0); at most `max_touched` slots differ."""
    R = T.shape[1]
    if max_touched >= R:
        return head(T)
    nonzero = (T != 0).any(-1).to(T.dtype)
    sel = F.one_hot(nonzero.topk(max_touched, dim=1).indices, R).to(T.dtype)
    uncovered = (1.0 - sel.sum(1)).unsqueeze(-1)             # (B, R, 1)
    # head(0) is evaluated once per row, alongside the touched slots, so its
    # gradient is an ordinary weight-gradient matmul and stays deterministic.
    y = head(torch.cat([sel @ T, T.new_zeros(T.shape[0], 1, T.shape[-1])], 1))
    return sel.transpose(1, 2) @ y[:, :-1] + uncovered @ y[:, -1:]


class TokenEmbedder(nn.Module):
    """Tier-0 front-end: embed one instruction's opcode+immediate tokens into a
    rename-invariant content vector — the input a T1 Compressor's core threads.
    Register-token positions are masked out of attention and the mean pool, so
    the content carries no register identity (registers enter the core as
    wiring only, never as content). T2+ feed their Compressor a lower tier's essence
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
# rather than constructor params. vp_head's DEPTH is not an implementation
# detail: it is the only consumer a T2 essence has, so its capacity is an
# implicit claim about what the next tier will be able to read, and `vp = 0` at
# a deeper head is a strictly weaker statement than at a shallow one. Quote the
# depth with the number. T1's is the smallest that reaches correct results; T2's
# is free in shape until T2 reaches vp = 0 (METRICS.md).
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
_VP_HEAD_DEPTH = 4     # vp_head hidden layers. Entry scripts pass their own
                       # (T1: 3, the smallest that reaches exact vp at held-out
                       # anchors; EXPERIMENT_LOG finding 33). Capacity here is
                       # evaluation the encoder is spared.
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
                 vp_head_depth=_VP_HEAD_DEPTH, n_pool_heads=1):
        super().__init__()
        self.d_content = d_content
        self.d_out = d_out
        self.d_event = d_event
        self.n_regs = N_REGS
        self.max_input_slots = MAX_INPUT_SLOTS
        self.max_output_slots = MAX_OUTPUT_SLOTS

        self.core = RegisterStateMachine(
            d_value=d_out, d_event=d_event, d_content=d_content, n_regs=N_REGS,
            op_cell_depth=op_cell_depth, n_pool_heads=n_pool_heads)

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

    def encode_state(self, content_seq, in0, in1, out, active):
        """content_seq (B, K, d_content), wiring (B, K) long, active (B, K)
        float -> (T, essence)."""
        return self.core(content_seq, in0, in1, out, active)

    def encode(self, content_seq, in0, in1, out, active):
        """The rename-invariant essence (B, d_out)."""
        return self.encode_state(content_seq, in0, in1, out, active)[1]


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
                 n_pool_heads=1):
        super().__init__()
        self.embedder = TokenEmbedder(
            vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, max_window=max_window)
        self.compressor = Compressor(
            d_content=d_model, d_out=d_out, d_event=d_event,
            op_cell_depth=op_cell_depth, vp_head_depth=vp_head_depth,
            n_pool_heads=n_pool_heads)
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

    def encode_state(self, tokens, pad, in0, in1, out):
        content = self.embedder(tokens, pad).unsqueeze(1)           # (B,1,d)
        active = torch.ones(content.shape[0], 1, device=content.device)
        return self.compressor.encode_state(
            content, in0.unsqueeze(1), in1.unsqueeze(1), out.unsqueeze(1),
            active)

    def encode(self, tokens, pad, in0, in1, out):
        """The T1 'vector' = the rename-invariant essence (B, d_out)."""
        return self.encode_state(tokens, pad, in0, in1, out)[1]


def T2Compressor(*, d_t1, d_out, d_event, op_cell_depth=1,
                 vp_head_depth=_VP_HEAD_DEPTH, n_pool_heads=1):
    """T2 preset: a Compressor fed a lower tier's essence (d_content=d_t1)."""
    return Compressor(d_content=d_t1, d_out=d_out, d_event=d_event,
                      op_cell_depth=op_cell_depth, vp_head_depth=vp_head_depth,
                      n_pool_heads=n_pool_heads)

