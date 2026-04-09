"""Shared utilities for batch pipeline scripts."""

import os
import sys

from datagen.datagen import _BATCH_HEADER

# Sanity bounds for batch header validation.
MAX_B = 1_000_000
MAX_LEN = 1000
MAX_INPUTS = 10000


def validate_batch_header(data):
    """Validate a raw batch's header values. Returns (B, max_len, n_inputs)
    or raises ValueError."""
    B, max_len, n_inputs = _BATCH_HEADER.unpack(data[:_BATCH_HEADER.size])
    if not (0 < B <= MAX_B and 0 < max_len <= MAX_LEN
            and 0 < n_inputs <= MAX_INPUTS):
        raise ValueError(
            f'Invalid header: B={B}, max_len={max_len}, n_inputs={n_inputs}')
    return B, max_len, n_inputs


def binary_stdout():
    """Return a binary fd for stdout, redirecting Python stdout to stderr.

    Prevents stray prints from corrupting binary output. Call once
    at script startup before any output.
    """
    out = os.fdopen(os.dup(sys.stdout.fileno()), 'wb')
    sys.stdout = sys.stderr
    return out
