"""
simdif: A unified similarity and difference library.
Supports sequence alignment, set logic, and vector-space metrics.
"""

from .simdif import sim, dist, score, matrix, trace

__version__ = "1.0.0"
__author__ = "C. Shaun Wagner"

__all__ = [
    "sim",
    "dist",
    "score",
    "matrix",
    "trace"
]

__version__ = "0.1.0"

def available_metrics():
    return sorted(METRICS.keys())