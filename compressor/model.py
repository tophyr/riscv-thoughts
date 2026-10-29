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
register-indexed input is each leaf's structural colour, itself a function of
the wiring alone), so:

  - essence = a permutation-invariant attention pool over the WRITTEN slots
    (the chunk's multiset of value-numbers; pc_writes reads it),
  - binding = the per-slot remainder (live_in/out + in/out slot order; the
    binding heads read it per-slot), including each written slot's pointer
    into the essence,
  - a written slot's value channels are the description of that output's
    function; the essence pools them, and value prediction evaluates each
    output from the essence through that output's pointer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from emulator import R_TYPE, I_TYPE, B_TYPE, LOAD_TYPE, STORE_TYPE
from tokenizer import REG_TOKEN_LO, REG_TOKEN_HI
from datagen import N_REGS, MAX_INPUT_SLOTS, MAX_OUTPUT_SLOTS
from datagen.signature import WL_ROUNDS


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


# Width of a leaf's structural colour. The colouring is a fixed random-feature
# message pass (below): wide enough that distinct colours collide only with
# vanishing probability, narrow enough to ride in every slot's state.
_COLOUR_DIM = 64

# Width of an output's pointer into the essence: a one-hot of the output's
# position in the content-hash order of the chunk's distinct output data paths.
# A chunk of K instructions has at most K outputs, so only the first K
# positions are ever used.
_PTR_DIM = MAX_OUTPUT_SLOTS

# Two output paths whose content-aware colours lie closer than this are one
# path. Over the frozen T1, identical paths differ by at most 3e-6 (float
# noise in its essence) and the closest distinct paths by 9e-2.
_PATH_TIE_EPS = 1e-4


class RegisterStateMachine(nn.Module):
    """Shared equivariant core of T1 (seq_len=1) and T2 (seq_len=K).

    Per register slot the state is [value | event | colour | pointer]:
      value (d_value) — set on write, PRESERVED on read; the dataflow the
        op-cell threads (multi-hop value-numbering at T2). A written slot's
        value is the description of that output's function.
      event (d_event) — accumulates read/write events with a read-order
        ramp, so live_in/in_slot are recoverable from the emitted object.
      colour (_COLOUR_DIM) — the leaf's structural colour (zero on slots that
        are not live leaves), so an evaluator can tell which input is which.
      pointer (_PTR_DIM) — on a written slot, which of the chunk's distinct
        output data paths it holds: where in the essence its function is
        filed (zero elsewhere).

    Before the steps run, a pre-pass over the wiring computes, from the wiring
    alone: which instructions are LIVE (reach a final write), and each leaf's
    STRUCTURAL COLOUR — a role-aware Weisfeiler–Lehman refinement over the live
    dataflow graph, realised as `WL_ROUNDS` rounds of message passing with
    FIXED random matrices. Every operation's two read roles are ordered: the
    encoder is told nothing about which operations are commutative, so a
    commutative swap is the op-cell's to learn. It is the colouring
    `datagen.signature` computes with strings (with every role ordered), in
    vector form: identical structure gives bit-identical colours, different
    structure different ones (with probability one over the random features).
    These rules are scaffolding — the learned form is the same pass with the
    matrices trained (RISCV_DESIGN, "What is built in for good, and what is
    built in for now").

    Each step, per instruction:
      r0, r1 = value[in0], value[in1]          (one-hot reads)
      res    = op_cell(content, r0, r1)        (role-aware combine)
      value[out] = res                         (one-hot write; x0 = nop)
      event[in0],[in1] += read_pulse(op_cell hidden)[role]   (whole operand context)
      event[out]       += write_pulse

    No per-register parameters: a leaf slot is seeded, on first read, from its
    colour. Reads/writes are pure one-hot tensor ops, so a register rename
    permutes the state exactly (equivariant by construction) and any pool over
    the register axis is invariant. Vectorized (no per-slot Python loop) so
    the step captures into the compiled CUDA-graph train step.
    """

    def __init__(self, d_value, d_event, d_content, n_regs=32, op_cell_depth=1,
                 n_pool_heads=1):
        super().__init__()
        self.d_value = d_value
        self.d_event = d_event
        self.d_colour = _COLOUR_DIM
        self.d_ptr = _PTR_DIM
        self.n_regs = n_regs
        self.n_pool_heads = n_pool_heads
        # A leaf's identity in the value channels is a learned projection of
        # its structural colour: deterministic, free of register identity and
        # of instruction order, and distinct exactly where the register
        # function distinguishes the leaves.
        self.seed_proj = nn.Linear(_COLOUR_DIM, d_value)
        self.op_cell = nn.Sequential(
            nn.Linear(d_content + 2 * d_value, 4 * d_value), nn.ReLU(),
            nn.Linear(4 * d_value, d_value))
        # Extra RESIDUAL FFN blocks on op_cell's output (op_cell_depth > 1). The
        # op-cell is reused RECURRENTLY once per instruction, so a naive deep
        # stack becomes a depth×K gradient path that won't train — the skip
        # connection (res = res + blk(res)) is the highway that keeps it
        # trainable. Static ModuleList loop -> unrolls at trace time, no graph
        # break under torch.compile.
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
        # Fixed random features of the colouring pass: not parameters. Seeded
        # so every model computes one colouring; saved as buffers so a
        # checkpoint carries its own.
        g = torch.Generator().manual_seed(0x5EED)
        dc = _COLOUR_DIM
        rand = lambda *s: torch.randn(*s, generator=g) / (dc ** 0.5)
        self.register_buffer('c_leaf', rand(dc))
        self.register_buffer('c_op', rand(dc))
        self.register_buffer('A', rand(dc, dc))
        self.register_buffer('M_role', rand(2, dc, dc))      # what an op reads, by role
        self.register_buffer('U_role', rand(2, dc, dc))      # who reads a node, by role
        # The content-aware colouring that names each output's data path: the
        # same pass with every operation seeded from its content as well, and
        # the fixed hash direction that sets the content-hash order of one
        # chunk's distinct paths.
        self.register_buffer('P_content', rand(d_content, dc))
        self.register_buffer('u_order', rand(dc))
        # essence pool: Pooling-by-Multihead-Attention (Set Transformer). k=
        # n_pool_heads learned seed QUERIES, each doing a permutation-invariant
        # attention pool over the WRITTEN register slots (softmax over slots, no
        # positional -> exactly rename-invariant). A single query (k=1) SUMS all
        # slots -> superimposes multiple output values irrecoverably; k>1 lets
        # different heads content-address different slots, keeping outputs
        # separable. Heads concatenate to an essence of the SAME width d_value
        # (each head d_value/k). pool_v is d_value wide and each head reads its
        # OWN slice: one SHARED d_value/k projection would cap what the essence
        # can carry about any single slot at d_value/k, however wide d_value is.
        # The pool reads written slots only: a read-only leaf's slot is
        # interface (whether an input still sits in a register of its own), and
        # the inputs reach the essence inside the results built from them.
        # Neither write order nor read order reaches the pool: both are the
        # linearization, not the function.
        assert d_value % n_pool_heads == 0, 'd_value must be divisible by n_pool_heads'
        self.pool_q = nn.Parameter(torch.randn(n_pool_heads * d_value) * 0.02)
        self.pool_k = nn.Linear(d_value, d_value)
        self.pool_v = nn.Linear(d_value, d_value)
        # Each written slot's pointer enters its pool key, so the pool can file
        # an output's description where the pointer will look for it.
        self.pool_ptr = nn.Linear(_PTR_DIM, d_value, bias=False)

    def _prepass(self, oh0e, oh1e, oh_oute, active, content):
        """Liveness and leaf colours from the wiring alone, and each written
        register's content-aware path colour.

        oh0e, oh1e: (B,K,R) one-hot reads with x0 masked; oh_oute: (B,K,R)
        one-hot writes with x0 and padding masked; active (B,K); content
        (B,K,d_content).
        Returns (leaf_colours (B,R,dc), live (B,K), written (B,R),
        path_colours (B,R,dc)). Leaf colours see only the wiring, so they say
        nothing about any operation; a path colour also sees every operation's
        content, so it tells apart outputs whose wiring is symmetric."""
        B, K, R = oh_oute.shape
        dc, dev = self.d_colour, oh_oute.device
        # Liveness, backwards from the final writers: a write is live when its
        # register's value is still needed — as an output, or by a live reader
        # before the next write. A dead instruction's reads add nothing.
        written = oh_oute.sum(1).clamp(max=1.0)
        needed = written
        live = []
        for k in reversed(range(K)):
            hit = (oh_oute[:, k] * needed).sum(-1)                   # (B,)
            live.append(hit)
            reads = ((oh0e[:, k] + oh1e[:, k]).clamp(max=1.0)
                     * (hit * active[:, k]).unsqueeze(-1))
            needed = (needed - oh_oute[:, k] * hit.unsqueeze(-1) + reads).clamp(0.0, 1.0)
        live = torch.stack(live[::-1], 1)                             # (B,K)
        # Each read's child: the instruction that last wrote the register
        # (a producer), else the register itself (a leaf).
        eye = torch.eye(K, device=dev, dtype=oh_oute.dtype)
        last_writer = torch.zeros(B, R, K, device=dev, dtype=oh_oute.dtype)
        child_leaf, child_op = ([], []), ([], [])
        for k in range(K):
            for r, oh in enumerate((oh0e[:, k], oh1e[:, k])):
                prod = (oh.unsqueeze(-1) * last_writer).sum(1)        # (B,K)
                child_leaf[r].append(oh * (1.0 - prod.sum(-1, keepdim=True)))
                child_op[r].append(prod)
            w = oh_oute[:, k].unsqueeze(-1)
            last_writer = last_writer * (1.0 - w) + w * eye[k].view(1, 1, K)
        cl = [torch.stack(c, 1) for c in child_leaf]                  # (B,K,R)
        co = [torch.stack(c, 1) for c in child_op]                    # (B,K,K)
        lv = live.unsqueeze(-1)                                       # (B,K,1)
        leaf_mask = ((cl[0] + cl[1]) * lv).sum(1).clamp(max=1.0)      # (B,R)
        CL = leaf_mask.unsqueeze(-1) * self.c_leaf
        leaves, _ = self._refine(CL, lv * self.c_op, cl, co, lv, leaf_mask)
        op_seed = self.c_op + F.normalize(content, dim=-1) @ self.P_content
        _, ops = self._refine(CL, lv * op_seed, cl, co, lv, leaf_mask)
        # A written register's path is its final writer's operation.
        paths = torch.einsum('brk,bkd->brd', last_writer, ops)
        return leaves, live, written, paths * written.unsqueeze(-1)

    def _refine(self, CL, CO, cl, co, lv, leaf_mask):
        """WL_ROUNDS of role-aware refinement over the live dataflow graph,
        from leaf colours CL (B,R,dc) and operation colours CO (B,K,dc)."""
        B, K, dc = CO.shape[0], CO.shape[1], CO.shape[2]
        R, dev = CL.shape[1], CO.device
        for _ in range(WL_ROUNDS):
            down = torch.zeros(B, K, dc, device=dev, dtype=CL.dtype)
            upL = torch.zeros(B, R, dc, device=dev, dtype=CL.dtype)
            upO = torch.zeros(B, K, dc, device=dev, dtype=CL.dtype)
            for r in range(2):
                child = (torch.einsum('bkr,brd->bkd', cl[r], CL)
                         + torch.einsum('bkj,bjd->bkd', co[r], CO))
                down = down + lv * (child @ self.M_role[r].t())
                um = lv * (CO @ self.U_role[r].t())
                upL = upL + torch.einsum('bkr,bkd->brd', cl[r], um)
                upO = upO + torch.einsum('bkj,bkd->bjd', co[r], um)
            CL = torch.tanh(CL @ self.A.t() + upL) * leaf_mask.unsqueeze(-1)
            CO = torch.tanh(CO @ self.A.t() + down + upO) * lv
        return CL, CO

    def _pointer(self, paths, written):
        """Each written register's pointer (B,R,_PTR_DIM): a one-hot of its
        path's position in the CONTENT-HASH ORDER of the chunk's distinct
        paths — sorted by a fixed hash of what each path computes, never by
        where its instructions are written. Identical paths share a position;
        no register number or instruction position enters it, so a rename
        permutes it with the slots and a reordering or a dead instruction
        leaves it unchanged. The hash reads syntax, so two equivalent paths
        spelled differently can sort differently."""
        s = paths @ self.u_order                                      # (B,R) the hash
        dist = (paths.unsqueeze(2) - paths.unsqueeze(1)).norm(dim=-1)  # (B,R,R)
        pair = written.unsqueeze(2) * written.unsqueeze(1)
        same = (dist <= _PATH_TIE_EPS).to(paths.dtype) * pair
        n_same = same.sum(-1).clamp(min=1.0)                          # (B,R)
        below = ((s.unsqueeze(1) < s.unsqueeze(2)).to(paths.dtype)
                 * (1.0 - same) * pair)                                # [r,q]: q below r
        order = (below / n_same.unsqueeze(1)).sum(-1).round().long()
        order = order.clamp(max=_PTR_DIM - 1)
        return F.one_hot(order, _PTR_DIM).to(paths.dtype) * written.unsqueeze(-1)

    def forward(self, content, in0, in1, out, active):
        """content (B,K,d_content); in0/in1/out (B,K) long; active (B,K)
        float. Returns (T, essence):
          T (B, n_regs, d_value+d_event+d_colour+d_ptr), essence (B, d_value)."""
        device = content.device
        B, K = active.shape
        not_x0 = torch.ones(self.n_regs, device=device)
        not_x0[X0_SLOT] = 0.0
        not_x0_row = not_x0.view(1, -1)
        oh0 = F.one_hot(in0, self.n_regs).to(content.dtype)          # (B,K,R)
        oh1 = F.one_hot(in1, self.n_regs).to(content.dtype)
        oh_out = F.one_hot(out, self.n_regs).to(content.dtype) * not_x0_row
        oh0e = oh0 * not_x0_row
        oh1e = oh1 * not_x0_row
        a3 = active.unsqueeze(-1)
        colours, live, written, paths = self._prepass(
            oh0e * a3, oh1e * a3, oh_out * a3, active, content)
        ptr = self._pointer(paths, written)                           # (B,R,P)
        seed = self.seed_proj(colours)                                # (B,R,d_value)
        # Slots start at exactly zero; a slot acquires its leaf identity only
        # when first READ, so a register the chunk never touches contributes
        # nothing to any consumer.
        value = torch.zeros(B, self.n_regs, self.d_value, device=device)
        event = torch.zeros(B, self.n_regs, self.d_event, device=device)
        # Per-slot ORDER timestamps, for the binding heads only (the event
        # channels never reach the pool):
        #   read_time  = canonical read rank  -> decodes in_slot (first-read order)
        #   write_time = step of LAST write   -> decodes out_slot (write order)
        read_time = torch.zeros(B, self.n_regs, device=device)
        n_read = torch.zeros(B, 1, device=device)
        write_time = torch.zeros(B, self.n_regs, device=device)

        for k in range(K):
            a = active[:, k:k + 1]                          # (B,1)
            step_sig = float(k + 1) / float(K)              # normalized step ramp
            c = content[:, k]
            read_time, n_read, firsts, _ = self._stamp_reads(
                read_time, n_read, write_time, oh0e[:, k], oh1e[:, k], a)
            # A leaf's identity IS its colour, written on first read.
            for fr in firsts:
                fr = fr.unsqueeze(-1)
                value = value * (1.0 - fr) + fr * seed
            r0 = torch.bmm(oh0[:, k].unsqueeze(1), value).squeeze(1)   # (B,d_value)
            r1 = torch.bmm(oh1[:, k].unsqueeze(1), value).squeeze(1)
            hidden = F.relu(self.op_cell[0](torch.cat([c, r0, r1], dim=-1)))
            res = self.op_cell[2](hidden)                        # (B,d_value)
            for blk in self.op_cell_blocks:                      # residual depth
                res = res + blk(res)
            # write value at dest (x0 write = nop), active rows only.
            w = (oh_out[:, k] * a).unsqueeze(-1)                 # (B,n_regs,1)
            value = value * (1.0 - w) + w * res.unsqueeze(1)
            # event channels: per-position read pulses (x0 reads leave no trace
            # on the x0 slot) + write pulse.
            p0, p1 = self.read_pulse(hidden).view(-1, 2, self.d_event).unbind(1)
            event = event + (oh0e[:, k] * a).unsqueeze(-1) * p0.unsqueeze(1)
            event = event + (oh1e[:, k] * a).unsqueeze(-1) * p1.unsqueeze(1)
            event = event + (oh_out[:, k] * a).unsqueeze(-1) \
                * self.write_pulse.view(1, 1, -1)
            wr = oh_out[:, k] * a
            write_time = write_time * (1.0 - wr) + wr * step_sig
        # fold the order timestamps into event channels 0 (read) and 1 (write),
        # normalized to ~(0,1].
        order = torch.stack([read_time / (2.0 * K), write_time], dim=-1)
        event = event + F.pad(order, (0, self.d_event - 2))
        T = torch.cat([value, event, colours, ptr], dim=-1)

        # Essence pool over the written slots: k seed queries; heads concatenate
        # to width d_value. Each head independently softmaxes over the slots;
        # the per-head sum over slots is symmetric, hence exactly rename-
        # invariant.
        q = self.pool_q.view(self.n_pool_heads, -1)          # (k, d_value)
        kk, vv = self._project_touched(value, written, K)
        kk = kk + self.pool_ptr(ptr)
        logits = kk @ q.t()                                  # (B, n_regs, k)
        att = torch.softmax(logits, dim=1)                   # (B, n_regs, k)
        vv = vv.view(B, self.n_regs, self.n_pool_heads, -1)  # per-head slices
        heads = torch.einsum('bsk,bskd->bkd', att, vv)       # (B, k, d_value//k)
        essence = heads.reshape(B, -1)                       # (B, d_value)
        return T, essence

    def _project_touched(self, pool_src, written, max_touched):
        """(pool_k, pool_v) of the state with every unwritten slot zeroed,
        projecting only the slots in `written`: an unwritten slot's projection
        is exactly the bias, and at most `max_touched` slots (one write per
        instruction) carry anything else. A read-only leaf's row is NOT zero
        (it holds the leaf's seed), so the rows are masked before selection:
        a short row's spare picks fall on the lowest-index unwritten slots,
        x1 among them, and would otherwise project that leaf into the pool."""
        R = pool_src.shape[1]
        n = min(max_touched, R)
        pool_src = pool_src * written.unsqueeze(-1)
        # One-hot selection rather than gather/scatter, like the core's own
        # reads and writes: same values, and a deterministic backward.
        sel = F.one_hot(written.topk(n, dim=1).indices, R).to(pool_src.dtype)
        src = sel @ pool_src                                 # (B, n, d_in)
        uncovered = (1.0 - sel.sum(1)).unsqueeze(-1)         # (B, R, 1)
        return [sel.transpose(1, 2) @ proj(src) + uncovered * proj.bias
                for proj in (self.pool_k, self.pool_v)]

    def _stamp_reads(self, read_time, n_read, write_time, oh0e, oh1e, a):
        """Stamp this step's operands with their canonical read POSITION: the
        running count of distinct registers read BEFORE they are written. First
        read sticks; operand 0 is stamped before operand 1. The positions feed
        the event channels (the binding heads' first-read order), never the
        value channels or the pool.

        All values are per-chunk scalars placed through the wiring one-hots, so
        equivariance is unaffected.
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
# depth with the number. T1's is the smallest that reaches correct results
# (METRICS.md).
class VPHead(nn.Module):
    """The vp value-prediction head, `depth` hidden layers of _VP_HIDDEN as a
    pre-activation residual MLP: h = W0·x, then h += W·relu(LN(h)) per block,
    then out = W·relu(LN(h)). A plain ReLU stack cannot evaluate a composed
    two-instruction chunk exactly even from a perfect description, and a plain
    stack this deep goes dead in training (EXPERIMENT_LOG, finding 34)."""

    def __init__(self, vp_in, vp_out, depth):
        super().__init__()
        assert depth >= 1, 'the vp head needs at least one hidden layer'
        self.inp = nn.Linear(vp_in, _VP_HIDDEN)
        self.norms = nn.ModuleList(nn.LayerNorm(_VP_HIDDEN) for _ in range(depth))
        self.lins = nn.ModuleList(nn.Linear(_VP_HIDDEN, _VP_HIDDEN)
                                  for _ in range(depth - 1))
        self.out = nn.Linear(_VP_HIDDEN, vp_out)
        for lin in (self.inp, *self.lins):
            nn.init.kaiming_normal_(lin.weight, nonlinearity='relu')
            nn.init.zeros_(lin.bias)

    def hidden(self, h):
        """From the input layer's output to the readout layer's input."""
        for norm, lin in zip(self.norms, self.lins):
            h = h + lin(F.relu(norm(h)))
        return F.relu(self.norms[-1](h))

    def forward(self, x):
        return self.out(self.hidden(self.inp(x)))


_IN_SCORE_HIDDEN = 256
_VP_HIDDEN = 1024
_VP_HEAD_DEPTH = 3     # vp_head hidden layers: T1's depth, the smallest that
                       # reaches exact vp at held-out anchors (finding 33). T2
                       # passes 12 (finding 35). Capacity here is evaluation
                       # the encoder is spared.
_VP_VALUE_BITS = 32    # two's-complement bit width of the vp value target (RV32)
_IMM_N_FREQ = 12       # rotary immediate: power-of-2 periods 2..4096 (2*this = target dim)


class Compressor(nn.Module):
    """The equivariant register-state-machine encoder (one per tier).

    Ingests a sequence of per-instruction CONTENT vectors (B, K, d_content) —
    token embeddings for T1 (via TokenEmbedder), a frozen lower tier's essence
    for T2+ (the tier contract) — threads them through the chunk's register
    state by each instruction's wiring, and emits one register-indexed state
    T (B, n_regs, d_out+d_event+d_colour+d_ptr). A register rename permutes the slots
    exactly (no per-register parameters); essence = the permutation-invariant
    pool over the written slots, the per-slot remainder = binding.

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
        self.d_colour = _COLOUR_DIM
        self.d_ptr = _PTR_DIM
        self.n_regs = N_REGS
        self.max_input_slots = MAX_INPUT_SLOTS
        self.max_output_slots = MAX_OUTPUT_SLOTS

        self.core = RegisterStateMachine(
            d_value=d_out, d_event=d_event, d_content=d_content, n_regs=N_REGS,
            op_cell_depth=op_cell_depth, n_pool_heads=n_pool_heads)

        # Binding heads read the per-slot state T, one score per register slot
        # (equivariant); pc_writes reads the rename-invariant essence.
        # in_score is an MLP (relevance × order product a Linear can't do); the
        # rest are Linear.
        d_slot = d_out + d_event + _COLOUR_DIM + _PTR_DIM
        self.live_in_head = nn.Linear(d_slot, 1)
        self.live_out_head = nn.Linear(d_slot, 1)
        self.in_score_head = nn.Sequential(
            nn.Linear(d_slot, _IN_SCORE_HIDDEN), nn.ReLU(),
            nn.Linear(_IN_SCORE_HIDDEN, 1))
        self.out_score_head = nn.Linear(d_slot, 1)
        self.pc_writes_head = nn.Linear(d_out, 1)
        # Value prediction, per WRITTEN output, read from the ESSENCE: from (the
        # unit-normalized essence, the output's pointer, and for each of the
        # output's inputs its value as 32 two's-complement bits + 1
        # smooth-magnitude scalar and its colour) predict that register's
        # value (32 bit-logits + 1 smooth scalar). The output's own
        # description is never handed over, so the essence has to carry every
        # output's function, filed where its pointer finds it. Per-bit gives
        # uniform resolution, so immediates are pressured to survive.
        vp_inputs = MAX_INPUT_SLOTS * (_VP_VALUE_BITS + 1 + _COLOUR_DIM)
        self.vp_head = VPHead(d_out + _PTR_DIM + vp_inputs, _VP_VALUE_BITS + 1,
                              vp_head_depth)
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
        self.d_colour = self.compressor.d_colour
        self.d_ptr = self.compressor.d_ptr
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
