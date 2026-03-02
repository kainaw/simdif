from ..simdif import METRICS, to_list_numeric, _rank
from .pearson import sim_pearson
import sys

def info_spearman() -> str:
    return """
Spearman Rank Correlation Coefficient
--------------------------------------
A non-parametric version of Pearson correlation that operates on the ranks
of values rather than the values themselves. This makes it robust to outliers
and appropriate for ordinal data or non-linear but monotonic relationships.
Computed by ranking each vector and then applying Pearson correlation to
those ranks.
Formula:
    ρ(A,B) = Pearson(rank(A), rank(B))
Range: [-1, 1]
    1  = perfect positive monotonic relationship
    0  = no monotonic relationship
   -1  = perfect negative monotonic relationship
Distance: [0, 2] — 1 - similarity
Note: Requires at least 2 elements. Tied values are assigned their average
rank. Use Pearson when you expect a linear relationship; use Spearman when
you expect a monotonic but possibly non-linear one.
    """.strip()

def explain_spearman(a, b, **_) -> str:
    a, b = to_list_numeric(a), to_list_numeric(b)
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
    if len(a) < 2:
        raise ValueError(f"Spearman requires at least 2 elements, got {len(a)}")
    ranks_a = _rank(a)
    ranks_b = _rank(b)
    sim = sim_spearman(a, b)
    return f"""
A: ({", ".join(map(str, a))})
B: ({", ".join(map(str, b))})
Spearman Rank Correlation:
Ranks of A: ({", ".join(map(str, ranks_a))})
Ranks of B: ({", ".join(map(str, ranks_b))})
Pearson applied to ranks:
= {sim:.4f}
Distance: 1 - ρ = {dist_spearman(a, b):.4f}
    """.strip()

def sim_spearman(a, b, **_) -> float:
    a, b = to_list_numeric(a), to_list_numeric(b)
    if len(a) != len(b):
        raise ValueError(f"Sequences must be the same length, got {len(a)} and {len(b)}")
    if len(a) < 2:
        raise ValueError(f"Spearman requires at least 2 elements, got {len(a)}")
    return sim_pearson(_rank(a), _rank(b))

def dist_spearman(a, b, **_) -> float:
    return 1 - sim_spearman(a, b)

METRICS['spearman'] = {
    'class': 'vector',
    'default': 'sim',
    'sim': sim_spearman,
    'dist': dist_spearman,
    'info': info_spearman,
    'explain': explain_spearman,
}
