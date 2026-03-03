from ..simdif import Metric, METRICS, to_list_numeric_aligned


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


def explain_chebyshev(a, b, **kwargs) -> str:
    a, b = to_list_numeric_aligned(a, b, **kwargs)
    diffs = [abs(x - y) for x, y in zip(a, b)]
    max_diff = max(diffs)
    idx = diffs.index(max_diff)
    return f"""
A: {a}
B: {b}
Absolute Differences: {diffs}
Maximum Difference: {max_diff:.4f} (at index {idx})
Chebyshev Distance: {max_diff:.4f}
    """.strip()
explain_chessboard = explain_chebyshev
explain_linf = explain_chebyshev


@Metric
def dist_chebyshev(a, b, **kwargs) -> float:
    a, b = to_list_numeric_aligned(a, b, **kwargs)
    if len(a)==0 and len(b)==0:
        return 0.0
    return float(max(abs(x - y) for x, y in zip(a, b)))
dist_chessboard = dist_chebyshev
dist_linf = dist_chebyshev


@Metric
def sim_chebyshev(a, b, **kwargs) -> float:
    return 1.0 / (1.0 + dist_chebyshev(a, b, **kwargs))
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
