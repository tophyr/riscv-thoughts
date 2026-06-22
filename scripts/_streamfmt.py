"""RVT stream-format IO for the binary batch pipe tools.

The wire layer shared by the UNIX-filter CLIs (gen_batches, mux_batches,
batch_slice, bench_throughput): a format registry keyed by
the 4-byte magic, header/byte readers, and stdout helpers. One format
exists today (RVT, the unified training batch); the registry keeps adding
another a one-line change. No torch, no model code — pure stream plumbing.
"""

import os
import struct
import sys

from datagen import RVT_FORMAT

_MAGIC_AND_VERSION = struct.Struct('<4sB')

_MAX_B = 1_000_000
_MAX_TOKENS = 100_000


class StreamFmt:
    """Format descriptor for batch stream I/O. Batches are self-describing
    (each carries magic+version+dtype) — there is no stream header."""
    def __init__(self, name, version, dtype_chars, read_bytes,
                 batch_header, prefix_size, body_size, validate):
        self.name = name
        self.version = version
        self.dtype_chars = dtype_chars
        self.n_dtype_chars = len(dtype_chars)
        self.read_bytes = read_bytes
        self.batch_header = batch_header
        self.prefix_size = prefix_size
        self.body_size = body_size
        self.validate = validate


def _validate_rvt(data):
    off = RVT_FORMAT.batch_prefix_size  # skip magic+version+dtype prefix
    vals = RVT_FORMAT.batch_header.unpack(
        data[off:off + RVT_FORMAT.batch_header.size])
    # Header fields: B, max_tokens, max_n_instrs, RB, n_anchors. RB is 0
    # for batches without a row-outputs payload and equal to B (with
    # n_anchors = n_states) in the row-outputs T1 mode.
    B, max_tokens, max_n_instrs, RB, n_anchors = vals
    if not (0 < B <= _MAX_B
            and 0 < max_tokens <= _MAX_TOKENS
            and 0 < max_n_instrs <= _MAX_TOKENS
            and 0 <= RB <= _MAX_B
            and 0 <= n_anchors <= 256):
        raise ValueError(f'Invalid RVT header: {vals}')
    return vals


RVT = StreamFmt('rvt', RVT_FORMAT.version, RVT_FORMAT.dtype_chars,
                RVT_FORMAT.read_batch_bytes, RVT_FORMAT.batch_header,
                RVT_FORMAT.batch_prefix_size,
                RVT_FORMAT.body_size, _validate_rvt)

_FORMATS = {b'RVT\x00': RVT}


def detect_format(f):
    """Identify the format from the first batch's self-describing prefix.

    Peeks magic + version (does NOT consume — batches are read whole
    afterward). Returns the StreamFmt. Requires a buffered/peekable stream.
    """
    need = _MAGIC_AND_VERSION.size
    buf = b''
    while len(buf) < need:
        prev = len(buf)
        buf = f.peek(need)
        if len(buf) <= prev:
            raise ValueError('Empty or truncated stream')
    magic, version = _MAGIC_AND_VERSION.unpack(buf[:need])
    fmt = _FORMATS.get(magic)
    if fmt is None:
        raise ValueError(f'Unknown format: {magic!r}')
    if version != fmt.version:
        raise ValueError(
            f'Unsupported {fmt.name} version: {version} '
            f'(expected {fmt.version}); regenerate.')
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
