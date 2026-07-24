import math
from ..simdif import Metric, METRICS, _aleph_counts, to_set


def info_baroni_urbani_buser() -> str:
    return """
Baroni-Urbani-Buser Coefficient
-------------------------------
A binary similarity coefficient that credits shared absences through the
geometric mean of shared presences and shared absences, sqrt(n11 * n00).
It is the only common set coefficient to use that geometric-mean term,
which makes shared absences count without letting them dominate the score.

Formula:
    BUB(A, B) = (sqrt(n11 * n00) + n11)
                / (sqrt(n11 * n00) + n11 + n10 + n01)
    Where n11 = shared presences, n10/n01 = mismatches,
          n00 = shared absences, N = n11 + n10 + n01 + n00
Range: [0, 1]
    1 = identical (no disagreements)
    0 = no shared presences and no shared absences
Note: Requires n_universe for shared absences. When n_universe = |A u B|,
    n00 = 0 and BUB reduces to the Jaccard index n11 / (n11 + n10 + n01).
Aliases: BUB, Baroni-Urbani
    """.strip()
info_bub = info_baroni_urbani_buser
info_baroni_urbani = info_baroni_urbani_buser


def explain_baroni_urbani_buser(a, b, n_universe=None, **_) -> str:
    a_set, b_set = to_set(a), to_set(b)
    n00, n01, n10, n11 = _aleph_counts(a, b, n_universe)
    n_total = n11 + n10 + n01 + n00
    g = math.sqrt(n11 * n00)
    denominator = g + n11 + n10 + n01
    sim = (g + n11) / denominator if denominator > 0 else 1.0
    intersection = sorted(map(str, a_set & b_set))
    return f"""
A: ({", ".join(sorted(map(str, a_set)))})
B: ({", ".join(sorted(map(str, b_set)))})
Baroni-Urbani-Buser (Universe-Aware):
Shared presences (n11): {n11} ({", ".join(intersection)})
Only in A (n10):        {n10}
Only in B (n01):        {n01}
Shared absences (n00):  {n00}
Total universe (N):     {n_total} {f'(corrected from {n_universe})' if n_universe is not None and n_total != n_universe else ''}
Geometric mean sqrt(n11 * n00): sqrt({n11} * {n00}) = {g:.4f}
Calculation:
  (sqrt(n11*n00) + n11) / (sqrt(n11*n00) + n11 + n10 + n01)
= ({g:.4f} + {n11}) / ({g:.4f} + {n11} + {n10} + {n01})
= {g + n11:.4f} / {denominator:.4f}
= {sim:.4f}
Difference: 1 - Sim = {1 - sim:.4f}
    """.strip()
explain_bub = explain_baroni_urbani_buser
explain_baroni_urbani = explain_baroni_urbani_buser


@Metric
def sim_baroni_urbani_buser(a, b, n_universe=None, **_) -> float:
    n00, n01, n10, n11 = _aleph_counts(a, b, n_universe)
    g = math.sqrt(n11 * n00)
    denominator = g + n11 + n10 + n01
    if denominator == 0:
        return 1.0
    return (g + n11) / denominator
sim_bub = sim_baroni_urbani_buser
sim_baroni_urbani = sim_baroni_urbani_buser


@Metric
def dif_baroni_urbani_buser(a, b, n_universe=None, **_) -> float:
    return 1 - sim_baroni_urbani_buser(a, b, n_universe)
dif_bub = dif_baroni_urbani_buser
dif_baroni_urbani = dif_baroni_urbani_buser


METRICS['baroni_urbani_buser'] = {
    'class': 'set',
    'default': 'sim',
    'sim': sim_baroni_urbani_buser,
    'dif': dif_baroni_urbani_buser,
    'info': info_baroni_urbani_buser,
    'explain': explain_baroni_urbani_buser,
}
METRICS['bub'] = METRICS['baroni_urbani_buser']
METRICS['baroni_urbani'] = METRICS['baroni_urbani_buser']
