from .minkowski import dist_minkowski
from ..simdif import Metric, METRICS, to_list_numeric_aligned
from ._helpers import _sim_from_dist, _dif_from_dist, _max_line


def info_manhattan() -> str:
    return """
Manhattan Distance (L1 Norm / Taxicab)
--------------------------------------
The distance between two points measured along axes at right angles.
Imagine a taxi driving through a grid-based city.

Formula:
    D(A, B) = sum(|Ai - Bi|)
    (Minkowski Distance where p=1)

Roles:
    dist: sum(|Ai - Bi|) (>= 0, unbounded)
    sim:  1 / (1 + D), or 1 - dif when d_max is supplied
    dif:  1 - sim,     or D / d_max when d_max is supplied

Note: no maximum exists on R^n, so sim defaults to the 1/(1+D) squash. If
your coordinates are range-normalized to [0,1], the bound is exact:

    d_max = n   -- every coordinate maximally opposed

Dividing by n like this is Gower's general coefficient (1971): the mean
absolute difference across range-normalized features. It is the published,
scale-free way to turn Manhattan into a difference, which is why it is worth
range-normalizing first rather than guessing a bound.

WARNING -- d_max must be a real bound. Distances above it clamp to dif=1.0,
so every pair beyond d_max scores identically. explain_ reports the clamp.
    """.strip()
info_taxicab = info_manhattan
info_cityblock = info_manhattan


def explain_manhattan(a, b, **kwargs) -> str:
    v1, v2 = to_list_numeric_aligned(a, b, **kwargs)
    steps = []
    total_dist = 0.0
    for i, (x, y) in enumerate(zip(v1, v2)):
        diff = abs(x - y)
        total_dist += diff
        steps.append(f"  idx {i}: |{x} - {y}| = {diff:.4f}")
    return f"""
A: {v1}
B: {v2}
Step-by-step Absolute Differences:
{chr(10).join(steps)}
Total Manhattan Distance (Sum): {total_dist:.4f}
{_max_line(total_dist, kwargs.get('d_max'),
           unbounded_note=f"unbounded on R^n (on [0,1]^n it would be n = {len(v1)})")}
    """.strip()
explain_taxicab = explain_manhattan
explain_cityblock = explain_manhattan


@Metric
def dist_manhattan(a, b, **kwargs) -> float:
    # Force p=1 but forward the rest (e.g. pad_value) to Minkowski.
    return dist_minkowski(a, b, **{**kwargs, 'p': 1})
dist_taxicab = dist_manhattan
dist_cityblock = dist_manhattan


@Metric
def dif_manhattan(a, b, **kwargs) -> float:
    return _dif_from_dist(dist_manhattan(a, b, **kwargs), kwargs.get('d_max'))
dif_taxicab = dif_manhattan
dif_cityblock = dif_manhattan


@Metric
def sim_manhattan(a, b, **kwargs) -> float:
    return _sim_from_dist(dist_manhattan(a, b, **kwargs), kwargs.get('d_max'))
sim_taxicab = sim_manhattan
sim_cityblock = sim_manhattan


METRICS['manhattan'] = {
    'class': 'vector',
    'default': 'dist',
    'dist': dist_manhattan,
    'dif': dif_manhattan,
    'sim': sim_manhattan,
    'info': info_manhattan,
	'explain': explain_manhattan,
}
METRICS['taxicab'] = METRICS['manhattan']
METRICS['cityblock'] = METRICS['manhattan']
