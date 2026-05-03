"""Pytest plugin for the model-acceptance suite.

Adds three CLI options:
  --encoder PATH   path to encoder.pt (parent dir must contain hparams.json)
  --decoder PATH   path to decoder.pt (PATH.hparams.json must exist alongside)
  --device SPEC    'auto' (default), 'cuda', or 'cpu'

Without --encoder, every test marked @pytest.mark.acceptance is skipped.
A test that also needs the decoder requires both --encoder and --decoder.

After the run, prints a metrics table that always shows the actual
numbers regardless of pass/fail. The default pytest noise (header,
tracebacks, per-outcome summary) is suppressed: you get the dots,
the count line, and the metrics table.
"""

import json
from pathlib import Path

import pytest
import torch

from compressor.model import T1Compressor, Decoder
from compressor.train import load_checkpoint
from tokenizer import VOCAB_SIZE


# ---------------------------------------------------------------------------
# CLI options
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    g = parser.getgroup('acceptance')
    g.addoption('--encoder', default=None, metavar='PATH',
                help='Encoder checkpoint. Triggers @acceptance tests.')
    g.addoption('--decoder', default=None, metavar='PATH',
                help='Decoder checkpoint. Required for decoder tests.')
    g.addoption('--device', default='auto',
                choices=('auto', 'cuda', 'cpu'),
                help='Device for acceptance fixtures.')


# ---------------------------------------------------------------------------
# Auto-skip @acceptance tests when checkpoints aren't given
# ---------------------------------------------------------------------------

def pytest_collection_modifyitems(config, items):
    enc = config.getoption('--encoder')
    dec = config.getoption('--decoder')
    skip_no_enc = pytest.mark.skip(reason='requires --encoder')
    skip_no_dec = pytest.mark.skip(reason='requires --decoder')
    for item in items:
        kw = item.keywords
        if 'acceptance' in kw and enc is None:
            item.add_marker(skip_no_enc)
        if 'needs_decoder' in kw and dec is None:
            item.add_marker(skip_no_dec)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def device(request):
    spec = request.config.getoption('--device')
    if spec == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return spec


@pytest.fixture(scope='session')
def encoder(request, device):
    path = request.config.getoption('--encoder')
    if path is None:
        pytest.skip('requires --encoder')
    hp_path = Path(path).parent / 'hparams.json'
    if not hp_path.exists():
        pytest.fail(f'missing companion hparams.json at {hp_path}')
    hp = json.loads(hp_path.read_text())
    enc = T1Compressor(
        VOCAB_SIZE,
        d_model=hp['d_model'], n_heads=hp['n_heads'],
        n_layers=hp['n_layers'], d_out=hp['d_out'],
        max_window=hp.get('max_window', 32),
    ).to(device)
    enc.load_state_dict(load_checkpoint(path, device), strict=False)
    enc.eval()
    return enc


@pytest.fixture(scope='session')
def decoder(request, device, encoder):
    path = request.config.getoption('--decoder')
    if path is None:
        pytest.skip('requires --decoder')
    hp_path = Path(path).with_suffix('.hparams.json')
    if not hp_path.exists():
        pytest.fail(f'missing companion hparams at {hp_path}')
    hp = json.loads(hp_path.read_text())
    dec = Decoder(
        VOCAB_SIZE,
        d_model=hp['dec_d_model'], n_heads=hp['dec_n_heads'],
        n_layers=hp['dec_n_layers'], d_emb=hp['d_out'],
        n_memory_tokens=hp.get('dec_n_memory', 1),
    ).to(device)
    dec.load_state_dict(load_checkpoint(path, device))
    dec.eval()
    return dec


# ---------------------------------------------------------------------------
# Marker registration
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line(
        'markers',
        'acceptance: trained-model evaluation; requires --encoder')
    config.addinivalue_line(
        'markers',
        'needs_decoder: also requires --decoder')
    # Silence pytorch's prototype-stage nested-tensor warning that fires
    # inside TransformerEncoder forward — drowns out the metrics table.
    config.addinivalue_line(
        'filterwarnings',
        'ignore:.*nested tensors.*prototype stage.*:UserWarning')


# ---------------------------------------------------------------------------
# Metrics table — always shown, regardless of pass/fail
# ---------------------------------------------------------------------------

# Tests record metrics here via the `metrics` fixture below. Keyed by
# nodeid (test path::test_name) for stable ordering with pytest output.
_METRICS: dict[str, dict] = {}
_OUTCOMES: dict[str, str] = {}


@pytest.fixture
def metrics(request):
    """Test fixture: record key/value metrics that should appear in
    the post-run table even if the test fails."""
    bag = {}
    _METRICS[request.node.nodeid] = bag
    return bag


def pytest_runtest_logreport(report):
    if report.when == 'call':
        _OUTCOMES[report.nodeid] = report.outcome


def _short_name(nodeid):
    """test_acceptance.py::test_pair_distance → pair_distance"""
    name = nodeid.rsplit('::', 1)[-1]
    return name.removeprefix('test_')


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not _METRICS:
        return
    tw = terminalreporter._tw
    tw.sep('=', 'model evaluation')
    rows = []
    for nodeid, m in _METRICS.items():
        outcome = _OUTCOMES.get(nodeid, 'skipped')
        status = {'passed': 'PASS', 'failed': 'FAIL',
                  'skipped': 'SKIP'}.get(outcome, outcome.upper())
        metrics_str = '  '.join(
            f'{k}={_fmt(v)}' for k, v in m.items())
        rows.append((_short_name(nodeid), status, metrics_str))
    if not rows:
        return
    name_w = max(len(r[0]) for r in rows)
    for name, status, m in rows:
        color = {'PASS': 'green', 'FAIL': 'red', 'SKIP': 'yellow'}.get(
            status, 'white')
        tw.write(f'  {name:<{name_w}}  ', bold=True)
        tw.write(f'{status:<6}', **{color: True})
        tw.write(f'  {m}\n')


def _fmt(v):
    if isinstance(v, float):
        return f'{v:.3f}' if abs(v) < 1000 else f'{v:.0f}'
    if isinstance(v, dict):
        return '{' + ', '.join(f'{k}:{_fmt(x)}' for k, x in v.items()) + '}'
    if isinstance(v, (list, tuple)):
        return '[' + ', '.join(_fmt(x) for x in v) + ']'
    return str(v)
