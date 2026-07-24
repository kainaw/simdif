import math
from ..simdif import Metric, METRICS, _aleph_counts, to_set


def info_mountford() -> str:
    return """
Mountford's Index of Similarity
-------------------------------
An ecological similarity index for presence/absence data, designed to be
(approximately) independent of sample size - unlike Jaccard or Sorensen,
which shrink as more of the fauna goes unsampled. Higher means more similar.

Formula (closed-form approximation):
    M(A, B) = 2*n11 / (n11 * (n10 + n01) + 2 * n10 * n01)
    Where n11 = shared species, n10/n01 = species unique to A / to B.
Range: [0, inf)
    0   = no shared species
    inf = identical species lists (n10 = n01 = 0)
Because it is unbounded above, Mountford is returned as a raw 'sim'; the
'dif' companion is the bounded 1 / (1 + M) (1 = disjoint, 0 = identical).

Note: Mountford's exact index is the root of a transcendental equation
    derived from Fisher's log-series; this implementation uses the standard
    closed-form approximation, which needs no numerical solver. Shared
    absences (n00) are deliberately ignored, so no n_universe is required.
    """.strip()


def _mountford_index(n11, n10, n01):
    """Raw Mountford index; +inf for identical non-empty lists, 0 if no overlap."""
    if n11 == 0:
        return 0.0
    denominator = n11 * (n10 + n01) + 2 * n10 * n01
    if denominator == 0:  # n10 = n01 = 0 -> identical species lists
        return math.inf
    return 2 * n11 / denominator


def explain_mountford(a, b, **_) -> str:
    a_set, b_set = to_set(a), to_set(b)
    n00, n01, n10, n11 = _aleph_counts(a, b)
    denominator = n11 * (n10 + n01) + 2 * n10 * n01
    m = _mountford_index(n11, n10, n01)
    intersection = sorted(map(str, a_set & b_set))
    dif = 1 / (1 + m)
    return f"""
A: ({", ".join(sorted(map(str, a_set)))})
B: ({", ".join(sorted(map(str, b_set)))})
Mountford's Index (sample-size independent):
Shared species (n11):   {n11} ({", ".join(intersection)})
Only in A (n10):        {n10}
Only in B (n01):        {n01}
Calculation:
  2*n11 / (n11*(n10 + n01) + 2*n10*n01)
= 2*{n11} / ({n11}*({n10} + {n01}) + 2*{n10}*{n01})
= {2 * n11} / {denominator}
= {m:.4f}
Similarity (raw, unbounded): {m:.4f}
Difference: 1 / (1 + M) = {dif:.4f}
    """.strip()


@Metric
def sim_mountford(a, b, **_) -> float:
    n00, n01, n10, n11 = _aleph_counts(a, b)
    return _mountford_index(n11, n10, n01)


@Metric
def dif_mountford(a, b, **_) -> float:
    return 1 / (1 + sim_mountford(a, b))


METRICS['mountford'] = {
    'class': 'set',
    'default': 'sim',
    'sim': sim_mountford,
    'dif': dif_mountford,
    'info': info_mountford,
    'explain': explain_mountford,
}
