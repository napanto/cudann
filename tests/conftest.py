"""cudann test configuration: the shared fnn-testkit suite runs against this backend.

    pip install "fnn-testkit @ git+https://github.com/napanto/fnn-bench#subdirectory=testkit"
    pytest tests/                      # = pytest --pyargs fnn_testkit --backend cudann
    pytest tests/ --device gpu --dtype float --blas cublas
"""
import os
import sys

pytest_plugins = ["fnn_testkit.plugin"]


def pytest_load_initial_conftests(early_config, parser, args):
    # default --backend cudann unless the user passes one explicitly
    if not any(a.startswith("--backend") for a in args):
        args.insert(0, "--backend=cudann")
