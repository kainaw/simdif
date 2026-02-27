"""
simdif: A unified similarity and difference library.
Supports sequence alignment, set logic, and vector-space metrics.
"""

from .simdif import *

__version__ = "0.1.0"
__author__ = "C. Shaun Wagner"

def available_metrics():
    return sorted(METRICS.keys())
