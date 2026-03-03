import math
from .minkowski import dist_minkowski
from ..simdif import Metric, METRICS, to_list_numeric_aligned


def info_euclidean() -> str:
    return """
Euclidean Distance (L2 Norm)
----------------------------
The "straight-line" distance between two points in Euclidean space. 

Formula:
    D(A, B) = sqrt( sum((Ai - Bi)^2) )
    (Minkowski Distance where p=2)
    """.strip()


def explain_euclidean(a, b, **kwargs) -> str:
    a, b = to_list_numeric_aligned(a, b, **kwargs)
    if len(a) != len(b):
        raise ValueException("Error: Vector length mismatch")
    steps = []
    sum_sq = 0.0
    for i, (x, y) in enumerate(zip(a, b)):
        diff = x - y
        diff_sq = diff ** 2
        sum_sq += diff_sq
        steps.append(f"  idx {i}: ({x} - {y})^2 = {diff}^2 = {diff_sq:.4f}")
    dist = math.sqrt(sum_sq)
    return f"""
A: {a}
B: {b}
Step-by-step Squared Differences:
{chr(10).join(steps)}
Sum of Squares: {sum_sq:.4f}
Square Root of Sum: sqrt({sum_sq:.4f}) = {dist:.4f}
Similarity (1 / (1 + d)): {1.0 / (1.0 + dist):.4f}
    """.strip()


@Metric
def dist_euclidean(a, b, **kwargs) -> float:
    kwargs.pop('p', None)
    return dist_minkowski(a, b, p=2, **kwargs)


@Metric
def sim_euclidean(a, b, **kwargs) -> float:
    return 1.0 / (1.0 + dist_euclidean(a, b, **kwargs))


METRICS['euclidean'] = {
    'class': 'vector',
    'default': 'dist',
    'dist': dist_euclidean,
    'sim': sim_euclidean,
    'info': info_euclidean,
	'explain': explain_euclidean,
}
