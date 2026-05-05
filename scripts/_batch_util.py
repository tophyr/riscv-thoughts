"""Shared utilities for tools (scripts/).

One stream format: RVT (unified training batch). The pipe tools and
mux dispatch on the format magic anyway; the registry approach is
preserved so adding a new format later is one entry.
"""

import os
import struct
import sys

from datagen.batch import RVT_FORMAT

_MAGIC_AND_VERSION = struct.Struct('<4sB')

_MAX_B = 1_000_000
_MAX_TOKENS = 100_000
_MAX_PAIRS = 10_000_000


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


def _validate_rvt(data):
    vals = RVT_FORMAT.batch_header.unpack(
        data[:RVT_FORMAT.batch_header.size])
    # Header fields: B, max_tokens, max_n_instrs, P, RB, K, n_anchors,
    # n_channels, max_in. RB / K / n_anchors / n_channels / max_in are
    # 0 in pair-mode batches and equal to B / 2 / n_states / 2 / 2 in
    # row-outputs mode.
    B, max_tokens, max_n_instrs, P, RB, K, n_anchors, n_channels, max_in = vals
    if not (0 < B <= _MAX_B
            and 0 < max_tokens <= _MAX_TOKENS
            and 0 < max_n_instrs <= _MAX_TOKENS
            and 0 <= P <= _MAX_PAIRS
            and 0 <= RB <= _MAX_B
            and 0 <= K <= 16
            and 0 <= n_anchors <= 256
            and 0 <= n_channels <= 16
            and 0 <= max_in <= 16):
        raise ValueError(f'Invalid RVT header: {vals}')
    return vals


RVT = StreamFmt('rvt', RVT_FORMAT.version, RVT_FORMAT.dtype_chars,
                RVT_FORMAT.read_stream_header, RVT_FORMAT.write_stream_header,
                RVT_FORMAT.read_batch_bytes, RVT_FORMAT.batch_header,
                RVT_FORMAT.body_size, _validate_rvt)

_FORMATS = {b'RVT\x00': RVT}


def detect_format(f):
    """Read stream header, auto-detect format, validate.

    Returns the StreamFmt. Consumes the header bytes.
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

    Uses BufferedReader.peek so it works on non-seekable streams.
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
    """Return a binary fd for stdout, redirecting Python stdout to stderr.

    Force blocking mode on the fd: GNU `tee` (and other stream-pipe
    consumers) flip pipes non-blocking on their side as a performance
    optimization, and the flag is shared across the kernel's open file
    description with the writing peer. Without this guard, Python's
    BufferedWriter.write() raises BlockingIOError mid-batch when the
    downstream consumer is briefly slow, leaving the reader with a
    truncated body.
    """
    fd = os.dup(sys.stdout.fileno())
    os.set_blocking(fd, True)
    out = os.fdopen(fd, 'wb')
    sys.stdout = sys.stderr
    return out


def read_batch_or_error(fmt, f, *, lenient=False):
    """Read one full batch with structured error handling.

    Returns (data, error) where exactly one is None:
      (bytes, None)  → batch read OK
      (None, None)   → clean EOF (or lenient EOFError absorbed)
      (None, str)    → validation failure (always reported), or hard
                       EOFError when lenient=False
    """
    try:
        data = fmt.read_bytes(f)
    except EOFError as e:
        if lenient:
            return None, None
        return None, str(e)
    if data is None:
        return None, None
    try:
        fmt.validate(data)
    except ValueError as e:
        return None, str(e)
    return data, None
