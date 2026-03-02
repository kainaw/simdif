from ..simdif import METRICS, to_list_numeric, Metric


def info_chebyshev() -> str:
    return """
Chebyshev Distance (L-infinity Norm)
------------------------------------
Also known as the Maximum Metric or Chessboard Distance. It is the 
limit of the Minkowski distance as p approaches infinity.

Formula:
    D(A, B) = max(|Ai - Bi|)

Range: [0, inf]
    0 = identical
    """.strip()
info_chessboard = info_chebyshev
info_linf = info_chebyshev


def explain_chebyshev(a, b, **_) -> str:
    v1, v2 = to_list_numeric(a), to_list_numeric(b)
    diffs = [abs(x - y) for x, y in zip(v1, v2)]
    max_diff = max(diffs)
    idx = diffs.index(max_diff)
    return f"""
A: {v1}
B: {v2}
Absolute Differences: {diffs}
Maximum Difference: {max_diff:.4f} (at index {idx})
Chebyshev Distance: {max_diff:.4f}
    """.strip()
explain_chessboard = explain_chebyshev
explain_linf = explain_chebyshev


@Metric
def dist_chebyshev(a, b, **_) -> float:
    v1, v2 = to_list_numeric(a), to_list_numeric(b)
    if len(v1) != len(v2): 
        raise ValueError("Length mismatch")
    return float(max(abs(x - y) for x, y in zip(v1, v2)))
dist_chessboard = dist_chebyshev
dist_linf = dist_chebyshev


@Metric
def sim_chebyshev(a, b, **_) -> float:
    return 1.0 / (1.0 + dist_chebyshev(a, b))
sim_chessboard = sim_chebyshev
sim_linf = sim_chebyshev


METRICS['chebyshev'] = {
    'class': 'vector',
    'default': 'dist',
    'dist': dist_chebyshev,
    'sim': sim_chebyshev,
    'info': info_chebyshev,
    'explain': explain_chebyshev,
}
METRICS['chessboard'] = METRICS['chebyshev']
METRICS['linf'] = METRICS['chebyshev']