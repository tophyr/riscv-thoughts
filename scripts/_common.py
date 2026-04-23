"""Shared utilities for training and evaluation scripts."""

import torch


def resolve_device(spec):
    """Map an 'auto'/'cuda'/'cpu' spec to a concrete device string.

    'auto' picks cuda if available, else cpu. Anything else is
    returned as-is.
    """
    if spec == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return spec


def format_eta(secs):
    """Format a duration in seconds as a compact human string."""
    secs = int(secs)
    if secs < 60:
        return f'{secs}s'
    if secs < 3600:
        return f'{secs // 60}m'
    if secs < 86400:
        return f'{secs // 3600}h{(secs % 3600) // 60}m'
    return f'{secs // 86400}d{(secs % 86400) // 3600}h'
