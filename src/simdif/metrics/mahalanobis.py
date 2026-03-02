import math
from ..simdif import Metric, METRICS, to_list_numeric


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

Note: In this implementation, if a covariance matrix is not provided via 
the 'covariance' keyword argument, it defaults to the Identity Matrix, 
rendering the result identical to Euclidean Distance.
    """.strip()


def explain_mahalanobis(a, b, **kwargs) -> str:
    a, b = to_list_numeric(a), to_list_numeric(b)
    dist = dist_mahalanobis(a, b, **kwargs)
    has_cov = "Provided" if 'covariance' in kwargs else "Identity (Default)"
    return f"""
A: {a}
B: {b}
Covariance Matrix: {has_cov}
Mahalanobis Distance: {dist:.4f}
(Note: If this matches Euclidean, check if you passed a custom covariance matrix.)
    """.strip()


@Metric
def dist_mahalanobis(a, b, **kwargs) -> float:
    """
    Standard Mahalanobis requires the inverse of the covariance matrix (S_inv).
    For educational simplicity in a pairwise comparison, we look for 'S_inv'
    in kwargs. If missing, we perform Euclidean.
    """
    a, b = to_list_numeric(a), to_list_numeric(b)
    if len(a) != len(b):
        raise ValueError("Vectors must be the same length.")
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
def sim_mahalanobis(a, b, **kwargs) -> float:
    return 1.0 / (1.0 + dist_mahalanobis(a, b, **kwargs))


METRICS['mahalanobis'] = {
    'class': 'vector',
    'default': 'dist',
    'dist': dist_mahalanobis,
    'sim': sim_mahalanobis,
    'info': info_mahalanobis,
    'explain': explain_mahalanobis,
}
