import math
from .minkowski import dist_minkowski
from ..simdif import Metric, METRICS


def info_euclidean() -> str:
    return """
Euclidean Distance (L2 Norm)
----------------------------
The "straight-line" distance between two points in Euclidean space. 

Formula:
    D(A, B) = sqrt( sum((Ai - Bi)^2) )
    (Minkowski Distance where p=2)
    """.strip()


def explain_euclidean(a, b, **_) -> str:
    v1, v2 = to_list_numeric(a), to_list_numeric(b)
    if len(v1) != len(v2):
        raise ValueException("Error: Vector length mismatch")
    steps = []
    sum_sq = 0.0
    for i, (x, y) in enumerate(zip(v1, v2)):
        diff = x - y
        diff_sq = diff ** 2
        sum_sq += diff_sq
        steps.append(f"  idx {i}: ({x} - {y})^2 = {diff}^2 = {diff_sq:.4f}")
    dist = math.sqrt(sum_sq)
    return f"""
A: {v1}
B: {v2}
Step-by-step Squared Differences:
{chr(10).join(steps)}
Sum of Squares: {sum_sq:.4f}
Square Root of Sum: sqrt({sum_sq:.4f}) = {dist:.4f}
Similarity (1 / (1 + d)): {1.0 / (1.0 + dist):.4f}
    """.strip()


@Metric
def dist_euclidean(a, b, **_) -> float:
    return dist_minkowski(a, b, p=2)


@Metric
def sim_euclidean(a, b, **_) -> float:
    return 1.0 / (1.0 + dist_euclidean(a, b))


METRICS['euclidean'] = {
    'class': 'vector',
    'default': 'dist',
    'dist': dist_euclidean,
    'sim': sim_euclidean,
    'info': info_euclidean,
	'explain': explain_euclidean,
}
