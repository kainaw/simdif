import math
from ..simdif import Metric, METRICS, to_list_numeric
from ._helpers import _sim_from_dist, _dif_from_dist, _max_line


def info_mahalanobis() -> str:
    return """
Mahalanobis Distance
--------------------
Measures the distance between a point and a distribution, or between two 
points within a coordinate system defined by a covariance matrix.

It differs from Euclidean distance by accounting for the correlation 
between variables and scaling each variable by its standard deviation.

Formula:
    d = sqrt( (x - y)^T * S^-1 * (x - y) )
    Where S is the Covariance Matrix.

Roles:
    dist: the quadratic form above (>= 0, unbounded)
    sim:  1 / (1 + d), or 1 - dif when d_max is supplied
    dif:  1 - sim,     or d / d_max when d_max is supplied

Note: no maximum exists, so sim defaults to the 1/(1+d) squash. Unlike the
Minkowski family there is no bound to be had by range-normalizing the inputs
either: the covariance matrix rescales the space, so the ceiling depends on
S^-1 rather than on the coordinate ranges. If you need a linear dif, pick
d_max from the chi-square distribution -- d^2 is chi-square with n degrees of
freedom under normality, so sqrt of a high quantile is a defensible bound
(e.g. n=2: sqrt(chi2.ppf(0.999, 2)) ~ 3.72).

WARNING -- d_max must be a real bound. Distances above it clamp to dif=1.0,
so every pair beyond d_max scores identically. explain_ reports the clamp.

Note: In this implementation, the INVERSE covariance matrix (S^-1) is supplied
via the 'covariance_inv' keyword argument. If it is not provided, S^-1 defaults
to the Identity Matrix, rendering the result identical to Euclidean Distance.
    """.strip()


def explain_mahalanobis(a, b, **kwargs) -> str:
    a, b = to_list_numeric(a), to_list_numeric(b)
    dist = dist_mahalanobis(a, b, **kwargs)
    has_cov = "Provided" if 'covariance_inv' in kwargs else "Identity (Default)"
    return f"""
A: {a}
B: {b}
Covariance Matrix: {has_cov}
Mahalanobis Distance: {dist:.4f}
{_max_line(dist, kwargs.get('d_max'),
           unbounded_note="unbounded -- the ceiling depends on S^-1, not on the coordinate ranges")}
(Note: If this matches Euclidean, check if you passed a custom covariance matrix.)
    """.strip()


@Metric
def dist_mahalanobis(a, b, **kwargs) -> float:
    """
    Standard Mahalanobis requires the inverse of the covariance matrix (S^-1).
    For educational simplicity in a pairwise comparison, we look for
    'covariance_inv' in kwargs. If missing, we perform Euclidean.
    """
    a, b = to_list_numeric(a), to_list_numeric(b)
    if len(a) != len(b):
        raise ValueError("Vector length mismatch")
    diff = [x - y for x, y in zip(a, b)]
    s_inv = kwargs.get('covariance_inv')
    if s_inv is None:
        return math.sqrt(sum(d**2 for d in diff))
    size = len(a)
    result = 0.0
    for i in range(size):
        row_sum = 0.0
        for j in range(size):
            row_sum += diff[j] * s_inv[i][j]
        result += diff[i] * row_sum
    return math.sqrt(max(0, result)) # max(0) prevents precision errors


@Metric
def dif_mahalanobis(a, b, **kwargs) -> float:
    return _dif_from_dist(dist_mahalanobis(a, b, **kwargs), kwargs.get('d_max'))


@Metric
def sim_mahalanobis(a, b, **kwargs) -> float:
    return _sim_from_dist(dist_mahalanobis(a, b, **kwargs), kwargs.get('d_max'))


METRICS['mahalanobis'] = {
    'class': 'vector',
    'default': 'dist',
    'dist': dist_mahalanobis,
    'dif': dif_mahalanobis,
    'sim': sim_mahalanobis,
    'info': info_mahalanobis,
    'explain': explain_mahalanobis,
}
