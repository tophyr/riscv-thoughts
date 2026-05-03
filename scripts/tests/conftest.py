"""Conftest for scripts/tests/.

Adds --run-grading. The grading suite (test_grading.py) is slow
(~3-5 minutes on GPU; longer on CPU) and is skipped by default.
The fast pipeline integration tests (test_pipeline.py) always run.
"""

import pytest


def pytest_addoption(parser):
    parser.addoption('--run-grading', action='store_true',
                     help='Run the slow end-to-end grading suite.')


def pytest_collection_modifyitems(config, items):
    if config.getoption('--run-grading'):
        return
    skip = pytest.mark.skip(reason='requires --run-grading')
    for item in items:
        if 'grading' in item.keywords:
            item.add_marker(skip)


def pytest_configure(config):
    config.addinivalue_line(
        'markers',
        'grading: end-to-end from-scratch grading; requires --run-grading')
