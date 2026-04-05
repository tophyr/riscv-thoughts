# Next Steps: Generalized Instruction Compressor

## Context

Phase 1 (ALU-only T0→T1 compression) is complete. The compressor learns
execution-proportional geometry for ALU instructions with Pearson r=0.995.
The next step is generalizing to all RV32I instructions (loads, stores,
branches, jumps, LUI, AUIPC) and redesigning the training around a more
principled multi-component loss.

## The Insight: Instructions as Graph Nodes

Every instruction is a graph node with input edges and output edges:

```
ADD x5, x3, x7:   in: [x3, x7]     out: [x5 ← x3+x7,      PC ← PC+4]
SW x5, 4(x3):     in: [x5, x3]     out: [mem[x3+4] ← x5,   PC ← PC+4]
BEQ x1, x2, 16:   in: [x1, x2]     out: [x0 ← 0,           PC ← cond ? PC+16 : PC+4]
JAL x1, 100:       in: [PC]         out: [x1 ← PC+4,         PC ← PC+100]
JALR x1, x5, 0:   in: [PC, x5]     out: [x1 ← PC+4,         PC ← (x5+0)&~1]
LW x5, 4(x3):     in: [x3, mem]    out: [x5 ← mem[x3+4],    PC ← PC+4]
LUI x5, 0x12345:  in: [imm]        out: [x5 ← 0x12345<<12,  PC ← PC+4]
AUIPC x5, 0x100:  in: [PC, imm]    out: [x5 ← PC+(0x100<<12), PC ← PC+4]
```

Key properties:
- Every instruction writes to PC. Non-branches all write PC+4 (they
  cluster together on the PC dimension). Branches/jumps write different
  values, naturally separating them.
- Every instruction writes to at most one other location (a register or
  a memory address). Pure branches write to x0 (no data effect).
- PC is just another location in a flat address space — not special.
- Memory addressing is always base+offset (rs1 + imm12). One scheme,
  no complexity.

## Two-Component Execution Comparison

Instead of comparing full register states or a single computed value,
extract TWO values per instruction execution:

1. **Data value**: what was written to the data destination
   - Register-writing instructions: output.regs[rd]
   - Stores: the value written to memory (rs2, width-dependent)
   - Branches: 0 (writing to x0)

2. **PC value**: what was written to PC
   - Non-branches: initial_PC + 4
   - Branches: initial_PC + offset (if taken) or initial_PC + 4 (if not)
   - JAL: initial_PC + offset
   - JALR: (rs1 + imm) & ~1

The execution distance between two instructions becomes:

```
data_dist = mean_over_inputs(log(1 + |data_val_A - data_val_B|))
pc_dist = mean_over_inputs(log(1 + |pc_val_A - pc_val_B|))
dist = data_dist + pc_dist
```

No C=16 penalty. Destination identity is handled by classification heads.

## Destination Prediction Heads

The model gets two classification heads reading from the shared T1 vector:

1. **dest_type**: register (0) vs memory (1) — classifies whether the
   data output goes to a register or memory
2. **dest_reg**: which register (0-31) — meaningful when dest_type is
   register; branches predict x0

These heads provide structural gradient signal. The execution correlation
provides semantic gradient signal. Both read from the full T1 vector
(shared space — the model decides how to organize dimensions).

## Loss Function

Three components:

1. **Execution correlation**: 1 - Pearson(T1_pairwise_dists,
   exec_pairwise_dists). Same as Phase 1 but using the two-component
   exec distance. This is the semantic grounding.

2. **dest_type classification**: CrossEntropyLoss on register-vs-memory
   prediction. Lightweight structural signal.

3. **dest_reg classification**: CrossEntropyLoss on register number
   prediction (for register-dest instructions). Structural signal.

Weights between these components are TBD — start with equal weighting,
monitor gradient norms per HATA lesson 3 (gradient dominance in
multi-objective losses).

## Implementation Plan

### 1. Emulator Changes (emulator/emulator.py)

Add `run_detailed()` to Executor that returns full state:
- Final register file (32 × int32)
- Final PC (uint32)
- Final memory (byte array, or relevant slice)

This is needed because the current `run()` only returns registers. We
need PC and memory to extract both output values.

### 2. Instruction Generator (compressor/train.py or new module)

Expand `random_instruction()` to generate all RV32I types:

- ALU R-type: same as now (10 opcodes)
- ALU I-type: same as now (9 opcodes)
- LUI: random rd, random 20-bit immediate
- AUIPC: random rd, random 20-bit immediate
- Loads (LB, LBU, LH, LHU, LW): random rd, random rs1, random imm12
- Stores (SB, SH, SW): random rs2, random rs1, random imm12
- Branches (BEQ, BNE, BLT, BGE, BLTU, BGEU): random rs1, rs2, random
  imm13 (multiple of 2)
- JAL: random rd, random imm21 (multiple of 2)
- JALR: random rd, random rs1, random imm12

Probabilities of each type should be configurable. Start with uniform
across types (not across individual opcodes — otherwise the 10 R-type
ALU ops would dominate).

### 3. Memory Handling for Loads/Stores

For memory instructions, the address is rs1 + imm. With random register
values, addresses span the full 32-bit space. Cannot allocate 4GB.

Approach: for each (instruction, input_state) pair, if the instruction
accesses memory:
1. Compute the concrete address from the input register state
2. Constrain the base register to a valid memory range (e.g., set rs1
   to a value in [4096, mem_size - 4096] so ±2048 offset stays in bounds)
3. Initialize memory at the computed address with random bytes
4. Execute

This constrains one register per memory instruction per input state. The
rest of the register state stays fully random.

Alternative: use a sparse memory dict instead of TinyFive's byte array.
Simpler but requires bypassing TinyFive's memory system.

### 4. Random PC

Each input state gets a random initial PC (not always 0). This is needed
so AUIPC and LUI produce different results (AUIPC reads PC, LUI doesn't).

Implementation: one extra random uint32 per input state, set on the
Executor before execution, read back after execution for the PC output
value.

Important: branch offsets and JAL offsets are relative to PC. A branch
with offset +8 at random PC produces PC+8, not 8. The PC output value
is the full final PC, not just the offset. This means non-branches
(PC+4) will all agree on the PC delta (+4) but disagree on the absolute
PC value. Need to think about whether to compare absolute PC or PC delta.

Recommendation: compare absolute PC values. Non-branches all produce
initial_PC + 4, so they all agree. Branches produce different values.
Since initial_PC is the same for both instructions in a pair (same input
state), the comparison is valid.

### 5. Extraction Function

Per-instruction function that takes execution results and returns
(data_val, pc_val):

```python
def extract_outputs(instr, final_regs, final_pc, final_mem, input_regs, initial_pc):
    # PC output: always the final PC
    pc_val = final_pc

    # Data output: depends on instruction type
    if instr.opcode in STORE_TYPE:
        # Value written to memory. Read back from output memory.
        addr = compute_store_addr(instr, input_regs)
        data_val = read_stored_value(final_mem, addr, instr.opcode)
    elif instr.opcode in B_TYPE:
        # Branches write to x0 (no data effect)
        data_val = 0
    else:
        # Everything else writes to rd (args[0] for R/I/U-type, JAL, JALR, loads)
        rd = instr.args[0]
        data_val = final_regs[rd]

    return data_val, pc_val
```

### 6. Model Changes (compressor/model.py)

Add two classification heads to T1Compressor:

```python
self.dest_type_head = nn.Linear(d_out, 2)    # register vs memory
self.dest_reg_head = nn.Linear(d_out, 32)    # which register
```

Forward returns (t1_vec, dest_type_logits, dest_reg_logits).

### 7. Producer Changes

The producer process now:
1. Generates random instructions (all types)
2. For each input state:
   a. Random registers (x1-x15, x0=0)
   b. Random PC
   c. For memory instructions: constrain base register, init memory
   d. Execute via Executor.run_detailed()
   e. Extract (data_val, pc_val) via extraction function
3. Sends through queue:
   - token_ids_np (B, max_len)
   - padding_mask_np (B, max_len)
   - data_vals (B, n_inputs) int64
   - pc_vals (B, n_inputs) int64
   - dest_types (B,) int — 0=register, 1=memory
   - dest_regs (B,) int — register number (0-31)

### 8. GPU Distance Computation

```python
def exec_distance_from_two_vals(data_vals, pc_vals, device):
    # data_vals, pc_vals: (B, n_inputs) int64
    # Returns: (B, B) float32 distance matrix
    data_t = torch.tensor(data_vals, dtype=torch.float64, device=device)
    pc_t = torch.tensor(pc_vals, dtype=torch.float64, device=device)

    dist = torch.zeros(B, B, device=device)
    for s in range(n_inputs):
        dv = data_t[:, s]
        data_diff = (dv.unsqueeze(1) - dv.unsqueeze(0)).abs()
        dist += torch.log1p(data_diff)

        pv = pc_t[:, s]
        pc_diff = (pv.unsqueeze(1) - pv.unsqueeze(0)).abs()
        dist += torch.log1p(pc_diff)
    dist /= n_inputs

    return dist.float()
```

### 9. Combined Loss

```python
def combined_loss(t1_vecs, exec_dist_matrix,
                  dest_type_logits, dest_type_targets,
                  dest_reg_logits, dest_reg_targets):
    corr = correlation_loss(t1_vecs, exec_dist_matrix)
    type_loss = F.cross_entropy(dest_type_logits, dest_type_targets)
    reg_loss = F.cross_entropy(dest_reg_logits, dest_reg_targets)
    return corr + type_loss + reg_loss  # start with equal weights
```

### 10. Updated Defaults

- batch_size: 4096 (proven to work well)
- n_steps: 100000
- d_out: 128
- lr: 3e-4 with cosine decay to 1e-6

## Potential Issues

### Store value extraction
For SB/SH, the stored value is truncated. Need to read back from memory
after execution to get the correct width. For SW, it's the full 32-bit
register value.

### Branch condition evaluation
Branches write different PC values depending on whether the condition is
true. With random register inputs, BEQ (equal) is almost never taken
(probability ~2^-32), while BLT (less than) is taken ~50% of the time.
This means different branch types have very different PC output
distributions. This is correct — they ARE semantically different.

### AUIPC vs LUI
With random PC, AUIPC produces rd = random_PC + (imm << 12) while LUI
produces rd = imm << 12. They're distinct instructions with different
behaviors, correctly separated. At PC=0 they'd be equivalent — which
is why randomizing PC matters.

### Memory address validity
Need to ensure that constrained base registers still produce diverse
enough addresses for meaningful load/store behavior. If all memory
accesses hit the same small region, loads might not be well-differentiated.

### Gradient balance
The correlation loss and classification losses have different scales.
Monitor gradient norms early and adjust weights if one dominates.
Per HATA lesson 3.

## What Is NOT Changing

- Tokenizer (already supports all instruction types)
- Emulator wrapper API (adding run_detailed, not changing run)
- Model architecture fundamentals (transformer encoder + mean pooling)
- Correlation loss formulation (Pearson on pairwise distances)
- Pipeline architecture (producer processes + GPU consumer)
- Existing tests (all should still pass)
