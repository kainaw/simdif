"""
simdif: A unified similarity and difference library.
Supports sequence alignment, set logic, and vector-space metrics.
"""

import sys
from importlib.metadata import version as _version, PackageNotFoundError as _PackageNotFoundError

from .simdif import *
from .metrics import *

try:
    __version__ = _version("simdif")
except _PackageNotFoundError:
    # Not installed -- e.g. running straight from a source checkout with no
    # `pip install -e .`. pyproject.toml remains the single source of truth;
    # nothing here needs updating when the version changes.
    __version__ = "0.0.0+unknown"
__author__ = "C. Shaun Wagner"

def available_metrics():
    return sorted(METRICS.keys())

# This maps each metric to the default for that metric.
# For example, jaccard is mapped to sim_jaccard
def _initialize_convenience_names():
    current_module = sys.modules[__name__]
    for name, metadata in METRICS.items():
        default_key = metadata.get('default', 'sim')
        default_func = metadata.get(default_key)
        if default_func:
            setattr(current_module, name, default_func)

_initialize_convenience_names()