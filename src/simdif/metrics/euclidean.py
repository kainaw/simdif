import math
from .minkowski import dist_minkowski
from ..simdif import Metric, METRICS, to_list_numeric_aligned
from ._helpers import _sim_from_dist, _dif_from_dist, _max_line


def info_euclidean() -> str:
    return """
Euclidean Distance (L2 Norm)
----------------------------
The "straight-line" distance between two points in Euclidean space.

Formula:
    D(A, B) = sqrt( sum((Ai - Bi)^2) )
    (Minkowski Distance where p=2)

Roles:
    dist: sqrt(sum((Ai - Bi)^2)) (>= 0, unbounded)
    sim:  1 / (1 + D), or 1 - dif when d_max is supplied
    dif:  1 - sim,     or D / d_max when d_max is supplied

Note: no maximum exists on R^n, so sim defaults to the 1/(1+D) squash and
never reaches 0. Supply d_max to rescale linearly. If your coordinates are
range-normalized to [0,1], the bound is exact:

    d_max = sqrt(n)   -- the diagonal of the unit cube, attained by opposite
                         corners, e.g. sqrt(3) ~ 1.7321 for 3 features

WARNING -- d_max must be a real bound. Distances above it clamp to dif=1.0,
so every pair beyond d_max scores identically. explain_ reports the clamp.
    """.strip()


def explain_euclidean(a, b, **kwargs) -> str:
    a, b = to_list_numeric_aligned(a, b, **kwargs)
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
{_max_line(dist, kwargs.get('d_max'),
           unbounded_note=f"unbounded on R^n (on [0,1]^n it would be sqrt({len(a)}) = {math.sqrt(len(a)):.4f})")}
    """.strip()


@Metric
def dist_euclidean(a, b, **kwargs) -> float:
    kwargs.pop('p', None)
    return dist_minkowski(a, b, p=2, **kwargs)


@Metric
def dif_euclidean(a, b, **kwargs) -> float:
    return _dif_from_dist(dist_euclidean(a, b, **kwargs), kwargs.get('d_max'))


@Metric
def sim_euclidean(a, b, **kwargs) -> float:
    return _sim_from_dist(dist_euclidean(a, b, **kwargs), kwargs.get('d_max'))


METRICS['euclidean'] = {
    'class': 'vector',
    'default': 'dist',
    'dist': dist_euclidean,
    'dif': dif_euclidean,
    'sim': sim_euclidean,
    'info': info_euclidean,
	'explain': explain_euclidean,
}
