import math
import sys
from .chebyshev import dist_chebyshev, explain_chebyshev
from ..simdif import Metric, METRICS, to_list_numeric_aligned
from ._helpers import _sim_from_dist, _dif_from_dist, _max_line


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

Raising p weights the larger coordinate gaps more heavily. In the limit only
the largest gap survives, which is why p=inf is the maximum rather than a sum.

Roles:
    dist: the Lp norm above (>= 0, unbounded)
    sim:  1 / (1 + D), or 1 - dif when d_max is supplied
    dif:  1 - sim,     or D / d_max when d_max is supplied

Note: no maximum exists on R^n, so sim defaults to the 1/(1+D) squash. If
your coordinates are range-normalized to [0,1], the bound is exact and
depends on both n and p:

    d_max = n^(1/p)      p=1: n      p=2: sqrt(n)      p=inf: 1

That is the whole reason to range-normalize first: it converts "no bound
exists" into a bound you can compute. Dividing Manhattan by n is Gower's
coefficient; note that p=inf needs no n at all, because the largest single
coordinate gap on [0,1]^n is 1 regardless of dimension.

WARNING -- d_max must be a real bound. Anything above it clamps to dif=1.0,
collapsing every distant pair onto the same score. explain_ says when a
clamp happened.

Note: p=inf is handled as a limit, not by substituting infinity into the
formula -- |Ai - Bi|^inf is inf above 1 and 0 below it, and inf^(1/inf) is
1.0, so evaluating it literally returns 1.0 for almost any input.
dist_minkowski(a, b, p=inf) delegates to dist_chebyshev instead.
    """.strip()


def explain_minkowski(a, b, **kwargs) -> str:
    p = kwargs.get('p', 2)
    if p == math.inf:
        return ("p=inf is the Chebyshev distance, reached as a limit rather than\n"
                "by evaluating the formula (see info_minkowski). Showing that instead:\n\n"
                + explain_chebyshev(a, b, **kwargs))
    a, b = to_list_numeric_aligned(a, b, **kwargs)
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
{_max_line(result, kwargs.get('d_max'),
           unbounded_note=f"unbounded on R^n (on [0,1]^n it would be n^(1/{p}) = {len(a) ** (1 / p):.4f})")}
    """.strip()


@Metric
def dist_minkowski(a, b, **kwargs) -> float:
    p = kwargs.get('p', 2)
    # p=inf must be taken as a limit, not evaluated: |d|^inf is inf for any gap
    # above 1 and 0 below it, and inf^(1/inf) collapses to 1.0 for almost every
    # input. Hand off to the metric that limit actually defines.
    if p == math.inf:
        return dist_chebyshev(a, b, **kwargs)
    a, b = to_list_numeric_aligned(a, b, **kwargs)
    if 'scipy' in sys.modules:
        from scipy.spatial import distance
        return float(distance.minkowski(a, b, p))
    return sum(abs(x - y) ** p for x, y in zip(a, b)) ** (1/p)


@Metric
def dif_minkowski(a, b, **kwargs) -> float:
    return _dif_from_dist(dist_minkowski(a, b, **kwargs), kwargs.get('d_max'))


@Metric
def sim_minkowski(a, b, **kwargs) -> float:
    return _sim_from_dist(dist_minkowski(a, b, **kwargs), kwargs.get('d_max'))


METRICS['minkowski'] = {
    'class': 'vector',
    'default': 'dist',
    'dist': dist_minkowski,
    'dif': dif_minkowski,
    'sim': sim_minkowski,
    'info': info_minkowski,
	'explain': explain_minkowski,
}
