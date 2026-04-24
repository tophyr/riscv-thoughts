"""Shared utilities for batch pipeline scripts.

Supports RVS (sequence) and RVB (single-instruction) stream formats
via auto-detection from the 4-byte magic.
"""

import os
import struct
import sys

from datagen.seqgen import (
    _BATCH_HEADER as _RVS_BATCH_HEADER,
    _batch_body_size as _rvs_body_size,
    _DTYPE_CHARS as _RVS_DTYPE_CHARS,
    read_stream_header as _rvs_read_header,
    write_stream_header as _rvs_write_header,
    read_batch_bytes as _rvs_read_bytes,
)
from datagen.batchgen import (
    _BATCH_HEADER as _RVB_BATCH_HEADER,
    _batch_body_size as _rvb_body_size,
    _DTYPE_CHARS as _RVB_DTYPE_CHARS,
    read_stream_header as _rvb_read_header,
    write_stream_header as _rvb_write_header,
    read_batch_bytes as _rvb_read_bytes,
)

# Stream headers vary per format: magic + version + N dtype chars,
# where N is format-specific. Read in two stages.
_MAGIC_AND_VERSION = struct.Struct('<4sB')

_MAX_B = 1_000_000
_MAX_TOKENS = 10_000
_MAX_INPUTS = 10_000


class StreamFmt:
    """Format descriptor for batch stream I/O."""
    def __init__(self, name, version, dtype_chars, read_header, write_header,
                 read_bytes, batch_header, body_size, validate):
        self.name = name
        self.version = version
        self.dtype_chars = dtype_chars
        self.n_dtype_chars = len(dtype_chars)
        self.read_header = read_header
        self.write_header = write_header
        self.read_bytes = read_bytes
        self.batch_header = batch_header
        self.body_size = body_size
        self.validate = validate


def _validate_rvs(data):
    vals = _RVS_BATCH_HEADER.unpack(data[:_RVS_BATCH_HEADER.size])
    B, max_tokens, max_instrs, n_inputs = vals
    if not (0 < B <= _MAX_B and 0 < max_tokens <= _MAX_TOKENS
            and 0 < n_inputs <= _MAX_INPUTS):
        raise ValueError(f'Invalid RVS header: {vals}')
    return vals


def _validate_rvb(data):
    vals = _RVB_BATCH_HEADER.unpack(data[:_RVB_BATCH_HEADER.size])
    B, max_len, n_inputs = vals
    if not (0 < B <= _MAX_B and 0 < max_len <= _MAX_TOKENS
            and 0 < n_inputs <= _MAX_INPUTS):
        raise ValueError(f'Invalid RVB header: {vals}')
    return vals


RVS = StreamFmt('rvs', 1, _RVS_DTYPE_CHARS,
                _rvs_read_header, _rvs_write_header,
                _rvs_read_bytes, _RVS_BATCH_HEADER, _rvs_body_size,
                _validate_rvs)

RVB = StreamFmt('rvb', 2, _RVB_DTYPE_CHARS,
                _rvb_read_header, _rvb_write_header,
                _rvb_read_bytes, _RVB_BATCH_HEADER, _rvb_body_size,
                _validate_rvb)

_FORMATS = {b'RVS\x00': RVS, b'RVB\x00': RVB}


def detect_format(f):
    """Read stream header, auto-detect format, validate.

    Returns the StreamFmt. Consumes the header bytes. Stream
    headers are variable-length: 4-byte magic + 1-byte version +
    N dtype chars, where N depends on the format.
    """
    mv = f.read(_MAGIC_AND_VERSION.size)
    if len(mv) < _MAGIC_AND_VERSION.size:
        raise ValueError('Missing or truncated stream header')
    magic, version = _MAGIC_AND_VERSION.unpack(mv)

    fmt = _FORMATS.get(magic)
    if fmt is None:
        raise ValueError(f'Unknown format: {magic!r}')
    if version != fmt.version:
        raise ValueError(
            f'Unsupported {fmt.name} version: {version} '
            f'(expected {fmt.version}); regenerate.')

    dtype_chars = f.read(fmt.n_dtype_chars)
    if len(dtype_chars) < fmt.n_dtype_chars:
        raise ValueError(f'Truncated stream header for {fmt.name}')
    if dtype_chars != fmt.dtype_chars:
        raise ValueError(f'Dtype mismatch for {fmt.name}: {dtype_chars!r}')
    return fmt


def peek_format(f):
    """Peek at magic bytes without consuming the header.

    Uses BufferedReader.peek so it works on non-seekable streams
    (pipes, FIFOs from process substitution).
    """
    buf = b''
    while len(buf) < 4:
        prev = len(buf)
        buf = f.peek(4)
        if len(buf) <= prev:
            raise ValueError(f'Truncated stream: {len(buf)} bytes before magic')
    magic = bytes(buf[:4])
    fmt = _FORMATS.get(magic)
    if fmt is None:
        raise ValueError(f'Unknown format: {magic!r}')
    return fmt


def binary_stdout():
    """Return a binary fd for stdout, redirecting Python stdout to stderr."""
    out = os.fdopen(os.dup(sys.stdout.fileno()), 'wb')
    sys.stdout = sys.stderr
    return out
