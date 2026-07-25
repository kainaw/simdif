import math
from ..simdif import Metric, METRICS, to_list_numeric, _dp_matrix, _fill_dp_matrix


def info_dtw() -> str:
    return """
Dynamic Time Warping (DTW)
--------------------------
Finds the optimal alignment between two numeric sequences that may vary in
speed or length, by letting each element match one or more consecutive
elements of the other sequence ("warping" the time axis). Closely related to
Needleman-Wunsch, but every step -- diagonal, insertion, or deletion -- is
priced by the same pointwise distance between the aligned elements, rather
than a fixed match/mismatch/gap cost, and the alignment must run corner to
corner (no free leading/trailing gaps).

Formula:
    D(i,j) = dist(Ai,Bj) + min(D(i-1,j), D(i,j-1), D(i-1,j-1))

The `dist_fn(a, b)` keyword sets the pointwise distance used at each step
(default: absolute difference).

Range: [0, inf)
    0 = identical sequences (under dist_fn)
    """.strip()


def _pointwise(dist_fn):
    return dist_fn if dist_fn is not None else (lambda a, b: abs(a - b))


def _fmt(cell):
    return f"{cell:.2f}" if isinstance(cell, float) else str(cell)


def explain_dtw(a, b, **kwargs) -> str:
    grid = matrix_dtw(a, b, **kwargs)
    rows_display = ["  " + "  ".join(f"{_fmt(cell):>8}" for cell in row) for row in grid]
    return f"""
A: {to_list_numeric(a)}
B: {to_list_numeric(b)}
DTW Cost Matrix (rows = A, cols = B):
{chr(10).join(rows_display)}
DTW Distance: {dist_dtw(a, b, **kwargs):.4f}
    """.strip()


@Metric
def dist_dtw(a, b, dist_fn=None, **kwargs) -> float:
    s1, s2 = to_list_numeric(a, **kwargs), to_list_numeric(b, **kwargs)
    cost = _pointwise(dist_fn)
    return _dp_matrix(s1, s2, insert=cost, delete=cost, substitute=cost, match_score=cost,
                       boundary=(math.inf, math.inf), combine="min")[-1][-1]


@Metric
def sim_dtw(a, b, **kwargs) -> float:
    return 1.0 / (1.0 + dist_dtw(a, b, **kwargs))


@Metric
def dif_dtw(a, b, **kwargs) -> float:
    return 1 - sim_dtw(a, b, **kwargs)


def matrix_dtw(a, b, dist_fn=None, **kwargs):
    s1, s2 = to_list_numeric(a, **kwargs), to_list_numeric(b, **kwargs)
    cost = _pointwise(dist_fn)
    return _fill_dp_matrix(s1, s2, insert=cost, delete=cost, substitute=cost, match_score=cost,
                            boundary=(math.inf, math.inf), combine="min")


METRICS['dtw'] = {
    'class': 'sequence',
    'default': 'dist',
    'dist': dist_dtw,
    'sim': sim_dtw,
    'dif': dif_dtw,
    'matrix': matrix_dtw,
    'info': info_dtw,
    'explain': explain_dtw,
}
