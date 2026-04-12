"""Shared utilities for batch pipeline scripts."""

import os
import sys

from datagen.seqgen import _BATCH_HEADER, _batch_body_size

# Re-export for scripts that need to scan batch headers.
BATCH_HEADER = _BATCH_HEADER
batch_body_size = _batch_body_size

# Sanity bounds for batch header validation.
_MAX_B = 1_000_000
_MAX_TOKENS = 10_000
_MAX_INSTRS = 1_000
_MAX_INPUTS = 10_000


def validate_batch_header(data):
    """Validate a batch header. Returns (B, max_tokens, max_instrs, n_inputs)."""
    vals = BATCH_HEADER.unpack(data[:BATCH_HEADER.size])
    B, max_tokens, max_instrs, n_inputs = vals
    if not (0 < B <= _MAX_B and 0 < max_tokens <= _MAX_TOKENS
            and 0 < max_instrs <= _MAX_INSTRS
            and 0 < n_inputs <= _MAX_INPUTS):
        raise ValueError(
            f'Invalid header: B={B}, max_tokens={max_tokens}, '
            f'max_instrs={max_instrs}, n_inputs={n_inputs}')
    return vals


def binary_stdout():
    """Return a binary fd for stdout, redirecting Python stdout to stderr.

    Prevents stray prints from corrupting binary output. Call once
    at script startup before any output.
    """
    out = os.fdopen(os.dup(sys.stdout.fileno()), 'wb')
    sys.stdout = sys.stderr
    return out
