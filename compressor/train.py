"""Training utilities for the T0→T1 compressor.

Execution distance computation, ranking loss, and training loop.
"""

import numpy as np
import torch

from emulator import Instruction, Executor, random_regs
from datagen import generate_sequence
from tokenizer import encode_instruction, PAD, VOCAB_SIZE

from .model import T1Compressor


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

def tokenize_batch(
    instructions: list[Instruction],
    device: torch.device = torch.device('cpu'),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize a batch of single instructions into padded tensors.

    Returns:
        (token_ids, padding_mask) where token_ids is (B, max_len) and
        padding_mask is (B, max_len) with True at padding positions.
    """
    encoded = [encode_instruction(instr) for instr in instructions]
    max_len = max(len(e) for e in encoded)

    token_ids = torch.full((len(encoded), max_len), PAD, dtype=torch.long,
                           device=device)
    padding_mask = torch.ones(len(encoded), max_len, dtype=torch.bool,
                              device=device)
    for i, enc in enumerate(encoded):
        token_ids[i, :len(enc)] = torch.tensor(enc, dtype=torch.long)
        padding_mask[i, :len(enc)] = False

    return token_ids, padding_mask


# ---------------------------------------------------------------------------
# Execution distance
# ---------------------------------------------------------------------------

def _exec_one_input_state(args):
    """Worker: execute all instructions on one input state. Returns (B, 32)."""
    instructions, input_regs = args
    exe = Executor(mem_size=64)  # one per worker process, reused across instructions
    results = np.zeros((len(instructions), 32), dtype=np.int64)
    for i, instr in enumerate(instructions):
        state = exe.run([instr], initial_regs=input_regs)
        results[i, :] = state.regs.astype(np.int64)
    return results


def make_pool(n_workers: int) -> 'multiprocessing.pool.Pool':
    """Create a reusable process pool with spawn context."""
    import multiprocessing
    ctx = multiprocessing.get_context('spawn')
    return ctx.Pool(processes=n_workers)


# Destination register penalty for the computed-value distance metric.
# C=16 ≈ log(1 + 8.9M), meaning a dest register mismatch is as significant
# as two instructions whose computed values differ by ~8.9 million.
_DEST_MISMATCH_PENALTY = 16.0


def compute_exec_distance_matrix(
    instructions: list[Instruction],
    n_inputs: int = 32,
    rng: np.random.Generator | None = None,
    pool: 'multiprocessing.pool.Pool | None' = None,
) -> np.ndarray:
    """Compute pairwise execution distances for a batch of instructions.

    Compares the *computed value* (output[rd]) rather than the full
    register state. Distance is:
        mean_over_inputs(log(1 + |val_A - val_B|)) + C * (rd_A != rd_B)

    This balances computational similarity against destination register
    identity, avoiding the ~10^6x scale gap that raw L1 register distance
    produces.

    Args:
        pool: Reusable process pool. If None, runs sequentially.

    Returns:
        (B, B) float64 array of distances.
    """
    if rng is None:
        rng = np.random.default_rng()

    B = len(instructions)

    # Extract destination register for each instruction.
    # For all ALU instructions (R-type and I-type), rd is args[0].
    rds = np.array([instr.args[0] for instr in instructions])

    # Execute all instructions on all input states.
    input_states = [random_regs(rng) for _ in range(n_inputs)]
    work = [(instructions, regs) for regs in input_states]

    if pool is not None:
        results = pool.map(_exec_one_input_state, work)
    else:
        results = [_exec_one_input_state(w) for w in work]

    # results: list of n_inputs × (B, 32) arrays of full register states.
    output_states = np.stack(results, axis=1)  # (B, n_inputs, 32)

    # Extract the computed value: output[rd] for each instruction.
    # computed_vals[i, s] = output_states[i, s, rd_i]
    computed_vals = output_states[np.arange(B), :, rds]  # (B, n_inputs)

    # Pairwise computed-value distance: mean of log(1 + |diff|) over inputs.
    dist_matrix = np.zeros((B, B), dtype=np.float64)
    for s in range(n_inputs):
        vals_s = computed_vals[:, s].astype(np.float64)  # (B,)
        diff = np.abs(vals_s[:, None] - vals_s[None, :])  # (B, B)
        dist_matrix += np.log1p(diff)
    dist_matrix /= n_inputs

    # Add destination register mismatch penalty.
    dest_mismatch = (rds[:, None] != rds[None, :]).astype(np.float64)
    dist_matrix += _DEST_MISMATCH_PENALTY * dest_mismatch

    return dist_matrix


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def correlation_loss(
    t1_vecs: torch.Tensor,
    exec_dist_matrix: torch.Tensor,
) -> torch.Tensor:
    """Pearson correlation loss over pairwise distances.

    Measures 1 - correlation(T1_distances, exec_distances) over all
    unique pairs in the batch. Reaches 0 when T1 distances are any
    positive linear function of exec distances — meaning the T1 space
    preserves proportional distance relationships, not just ordering.

    Args:
        t1_vecs: (B, d_out) T1 vectors from the compressor.
        exec_dist_matrix: (B, B) pairwise execution distances.

    Returns:
        Scalar loss in [0, 2]. 0 = perfect positive correlation,
        1 = no correlation, 2 = perfect negative correlation.
    """
    # Pairwise T1 distances: (B, B)
    t1_dists = torch.cdist(t1_vecs, t1_vecs)

    # Extract upper triangle (unique pairs, excluding self-distances).
    B = t1_vecs.shape[0]
    idx = torch.triu_indices(B, B, offset=1, device=t1_vecs.device)
    t1_flat = t1_dists[idx[0], idx[1]]
    exec_flat = exec_dist_matrix[idx[0], idx[1]]

    # Pearson correlation.
    t1_centered = t1_flat - t1_flat.mean()
    exec_centered = exec_flat - exec_flat.mean()

    num = (t1_centered * exec_centered).sum()
    denom = (t1_centered.norm() * exec_centered.norm()).clamp(min=1e-8)

    return 1.0 - num / denom


# ---------------------------------------------------------------------------
# Random instruction generation (single instructions, not sequences)
# ---------------------------------------------------------------------------

def random_instruction(rng: np.random.Generator) -> Instruction:
    """Generate a single random straight-line RV32I instruction."""
    return generate_sequence(1, rng, r_type_prob=0.5)[0]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _batch_producer(out_queue, n_batches, batch_size, n_inputs, seed):
    """Producer process: generates instructions, runs emulator, builds arrays.

    Sends raw computed values and rd indices through the queue. The main
    process computes the pairwise distance matrix on GPU, avoiding the B²
    numpy computation that dominates producer time at large batch sizes.
    """
    rng = np.random.default_rng(seed)
    exe = Executor(mem_size=64)

    for _ in range(n_batches):
        instructions = [random_instruction(rng) for _ in range(batch_size)]
        B = len(instructions)
        rds = np.array([instr.args[0] for instr in instructions], dtype=np.int64)

        # Execute all instructions on all input states.
        input_states = [random_regs(rng) for _ in range(n_inputs)]
        computed_vals = np.zeros((B, n_inputs), dtype=np.int64)
        for s, input_regs in enumerate(input_states):
            for i, instr in enumerate(instructions):
                state = exe.run([instr], initial_regs=input_regs)
                computed_vals[i, s] = state.regs[rds[i]]

        # Tokenize.
        encoded = [encode_instruction(instr) for instr in instructions]
        max_len = max(len(e) for e in encoded)
        token_ids = np.full((B, max_len), PAD, dtype=np.int64)
        padding_mask = np.ones((B, max_len), dtype=np.bool_)
        for i, enc in enumerate(encoded):
            token_ids[i, :len(enc)] = enc
            padding_mask[i, :len(enc)] = False

        out_queue.put((token_ids, padding_mask, computed_vals, rds))
    out_queue.put(None)  # sentinel


def _exec_distance_from_vals(computed_vals, rds, device):
    """Compute pairwise distance matrix on GPU from computed values and rd indices.

    Args:
        computed_vals: (B, n_inputs) int64 tensor of computed output values.
        rds: (B,) int64 tensor of destination register indices.
        device: torch device.

    Returns:
        (B, B) float32 tensor of pairwise distances.
    """
    vals = torch.tensor(computed_vals, dtype=torch.float64, device=device)
    B, n_inputs = vals.shape

    # Pairwise distance: mean of log(1 + |diff|) over input states.
    # vals[:, s] is (B,). Pairwise diff is (B, B) per input state.
    dist = torch.zeros(B, B, dtype=torch.float64, device=device)
    for s in range(n_inputs):
        v = vals[:, s]
        diff = (v.unsqueeze(1) - v.unsqueeze(0)).abs()
        dist += torch.log1p(diff)
    dist /= n_inputs

    # Destination register mismatch penalty.
    rds_t = torch.tensor(rds, dtype=torch.long, device=device)
    dest_mismatch = (rds_t.unsqueeze(1) != rds_t.unsqueeze(0)).float()
    dist += _DEST_MISMATCH_PENALTY * dest_mismatch

    return dist.float()


def train(
    batch_size: int,
    n_steps: int,
    n_inputs: int,
    n_producers: int,
    prefetch: int,
    torch_threads: int,
    lr: float,
    lr_min: float,
    d_model: int,
    n_heads: int,
    n_layers: int,
    d_out: int,
    device: str,
    log_every: int,
    seed: int,
):
    """Train the T0→T1 compressor.

    Uses a pipelined architecture: producer processes generate and prepare
    batches (instruction generation, emulator execution, tokenization)
    while the main process runs forward/backward/step. Each producer does
    its own emulation sequentially — parallelism comes from running
    multiple producers, not from parallelizing within one.

    Args:
        batch_size: Number of instructions per batch.
        n_steps: Total training steps.
        n_inputs: Random input states per instruction for exec distance.
        n_producers: Producer processes preparing batches in parallel.
        prefetch: Max batches queued ahead.
        torch_threads: Threads for torch BLAS operations.
        lr: Learning rate.
        d_model: Transformer hidden dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
        d_out: T1 vector dimension.
        device: 'cpu', 'cuda', or 'auto'.
        log_every: Print loss every N steps.
        seed: Random seed.

    Returns:
        Trained model and loss history.
    """
    import multiprocessing
    ctx = multiprocessing.get_context('spawn')

    torch.set_num_threads(torch_threads)

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    rng = np.random.default_rng(seed)

    model = T1Compressor(
        vocab_size=VOCAB_SIZE,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_out=d_out,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_steps, eta_min=lr_min)
    losses = []

    # Launch producer processes, each with its own RNG seed.
    batch_queue = ctx.Queue(maxsize=prefetch)
    producers = []
    steps_remaining = n_steps
    for p in range(n_producers):
        n = steps_remaining // (n_producers - p)
        steps_remaining -= n
        producer_seed = int(rng.integers(0, 2**63))
        proc = ctx.Process(
            target=_batch_producer,
            args=(batch_queue, n, batch_size, n_inputs, producer_seed),
            daemon=True,
        )
        proc.start()
        producers.append(proc)

    # Main training loop: consume prepared batches.
    sentinels_seen = 0
    step = 0
    while sentinels_seen < n_producers:
        item = batch_queue.get()
        if item is None:
            sentinels_seen += 1
            continue

        token_ids_np, padding_mask_np, computed_vals_np, rds_np = item
        token_ids = torch.from_numpy(token_ids_np).to(device)
        padding_mask = torch.from_numpy(padding_mask_np).to(device)
        exec_dists = _exec_distance_from_vals(computed_vals_np, rds_np, device)

        t1_vecs = model(token_ids, padding_mask)
        loss = correlation_loss(t1_vecs, exec_dists)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        losses.append(loss_val)

        if step % log_every == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'step {step:5d}  loss={loss_val:.4f}  lr={current_lr:.2e}')
        step += 1

    for proc in producers:
        proc.join()

    return model, losses


def save_run(model, losses, hparams=None, out_dir='runs'):
    """Save model checkpoint, loss history, and hyperparameters."""
    import json
    from pathlib import Path
    from datetime import datetime

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = out / stamp
    run_dir.mkdir()

    torch.save(model.state_dict(), run_dir / 'model.pt')
    with open(run_dir / 'losses.json', 'w') as f:
        json.dump(losses, f)
    if hparams is not None:
        with open(run_dir / 'hparams.json', 'w') as f:
            json.dump(hparams, f, indent=2)

    print(f'Saved to {run_dir}')
    return run_dir
