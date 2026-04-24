"""RV32I emulators.

Two independent emulator implementations for different jobs:

- `cpu_emulator` (sequential, Python, numpy): instruction-by-instruction
  execution with full state, control flow, and memory. Used for ground-
  truth execution, data generation, and correctness testing.

- `gpu_emulator` (batched, PyTorch, torch.where): all 37 opcodes in
  parallel via tensor ops, no per-item control flow, no real memory.
  Used inside training loops for REINFORCE execution-equivalence
  rewards where speed matters more than fidelity.

These are NOT interchangeable — they have different interfaces because
they solve different problems. Import the one you need:

    from emulator import run, Instruction, ...       # CPU
    from emulator import batch_execute, ...          # GPU (lazy-loaded)

GPU names are lazy-loaded via module-level __getattr__ to avoid a
circular import: gpu_emulator imports from tokenizer, and tokenizer
imports from emulator. The lazy load defers gpu_emulator's resolution
until after tokenizer has finished loading.
"""

from .cpu_emulator import (
    R_TYPE, I_TYPE, B_TYPE, LOAD_TYPE, STORE_TYPE, ALL_OPCODES,
    MEM_WIDTH,
    SparseMemory, RV32IState, Instruction, make_ctx, run, random_regs,
)

_GPU_NAMES = frozenset({
    'batch_execute', 'batch_parse_tokens',
    'batch_is_complete_instruction',
    'instructions_to_tensors', 'random_regs_gpu',
})


def __getattr__(name):
    if name in _GPU_NAMES:
        from . import gpu_emulator
        return getattr(gpu_emulator, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
