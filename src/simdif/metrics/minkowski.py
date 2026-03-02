import math
import sys
from ..simdif import Metric, METRICS, to_list_numeric, _align_vectors


def info_minkowski() -> str:
    return """
Minkowski Distance (Lp Norm)
----------------------------
A generalized distance metric between two points in a normed vector space. 
By changing the 'p' parameter, it transforms into other distances.

Formula:
    D(A, B) = ( sum(|Ai - Bi|^p) )^(1/p)

Common values for p:
    p=1: Manhattan Distance
    p=2: Euclidean Distance
    p=inf: Chebyshev Distance
    """.strip()


def explain_minkowski(a, b, **kwargs) -> str:
    p = kwargs.get('p', 2)
    a, b = to_list_numeric(a, **kwargs), to_list_numeric(b, **kwargs)
    if len(a) != len(b):
        a, b = _align_vectors(a, b, **kwargs)
        a, b = to_list_numeric(a, **kwargs), to_list_numeric(b, **kwargs)
    if len(a) != len(b):
        return "Error: Vector length mismatch"
    terms = [f"|{x} - {y}|^{p}" for x, y in zip(a, b)]
    values = [abs(x - y)**p for x, y in zip(a, b)]
    sum_powers = sum(values)
    result = sum_powers ** (1/p)
    return f"""
A: {a}
B: {b}
Parameter p: {p}

Step 1: Calculate sum of absolute differences to the power of p:
  Σ(|Ai - Bi|^{p}
  = {' + '.join([f"{v:.4f}" for v in values])}
  = {sum_powers:.4f}

Step 2: Take the p-th root of the sum:
  ({sum_powers:.4f})^(1/{p})
  = {result:.4f}

Minkowski Distance: {result:.4f}
    """.strip()


@Metric
def dist_minkowski(a, b, **kwargs) -> float:
    p = kwargs.get('p', 2)
    a, b = to_list_numeric(a, **kwargs), to_list_numeric(b, **kwargs)
    if len(a) != len(b):
        a, b = _align_vectors(a, b, **kwargs)
        a, b = to_list_numeric(a, **kwargs), to_list_numeric(b, **kwargs)
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
    if 'scipy' in sys.modules:
        from scipy.spatial import distance
        return float(distance.minkowski(a, b, p))
    return sum(abs(x - y) ** p for x, y in zip(a, b)) ** (1/p)


@Metric
def sim_minkowski(a, b, **kwargs) -> float:
    return 1.0 / (1.0 + dist_minkowski(a, b, **kwargs))


METRICS['minkowski'] = {
    'class': 'vector',
    'default': 'dist',
    'dist': dist_minkowski,
    'sim': sim_minkowski,
    'info': info_minkowski,
	'explain': explain_minkowski,
}
