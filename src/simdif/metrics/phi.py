import math
from ..simdif import Metric, METRICS, _aleph_counts, to_set


def info_phi() -> str:
    return """
Phi Coefficient (Mean Square Contingency / Matthews Correlation)
----------------------------------------------------------------
The Pearson correlation coefficient applied to two binary (present/absent)
variables. For 2x2 tables it is identical to the Matthews Correlation
Coefficient (MCC). Uses shared absences (d), so it requires the universe size.
Formula:
    phi(A, B) = (a*d - b*c) / sqrt((a+b)(a+c)(b+d)(c+d))
        a = |A ∩ B|             (in both)
        b = |A only|            (in A, not B)
        c = |B only|            (in B, not A)
        d = shared absences     (in neither; needs n_universe)
Range: [-1, 1]
    1  = perfect agreement
    0  = no correlation
    -1 = perfect disagreement
Difference: -1 * similarity
Note: Requires n_universe to define d. Without it, d = 0 and the coefficient is
skewed strongly negative - always pass n_universe. Because the range is
[-1, 1], 1 - sim would not be meaningful; negating similarity instead
preserves the signed range.
Aliases: MCC, Matthews Correlation Coefficient, Mean Square Contingency
    """.strip()
info_mcc = info_phi
info_matthews = info_phi


def explain_phi(a, b, n_universe=None, **_) -> str:
    a_set, b_set = to_set(a), to_set(b)
    n00, n01, n10, n11 = _aleph_counts(a, b, n_universe)
    n_total = n11 + n10 + n01 + n00
    intersection = sorted(map(str, a_set & b_set))
    only_a = sorted(map(str, a_set - b_set))
    only_b = sorted(map(str, b_set - a_set))
    numerator = n11 * n00 - n10 * n01
    denom = math.sqrt((n11 + n10) * (n11 + n01) * (n10 + n00) * (n01 + n00))
    sim = numerator / denom if denom != 0 else 0.0
    corrected = f'(corrected from {n_universe})' if n_universe is not None and n_total != n_universe else ''
    return f"""
A: ({", ".join(sorted(map(str, a_set)))})
B: ({", ".join(sorted(map(str, b_set)))})
Phi Coefficient (Universe-Aware):
a (in both):       {n11} ({", ".join(intersection)})
b (A only):        {n10} ({", ".join(only_a)})
c (B only):        {n01} ({", ".join(only_b)})
d (shared absent): {n00}
Total universe (N): {n_total} {corrected}
Calculation:
  (a*d - b*c) / sqrt((a+b)(a+c)(b+d)(c+d))
= ({n11}*{n00} - {n10}*{n01}) / sqrt({n11 + n10} * {n11 + n01} * {n10 + n00} * {n01 + n00})
= {numerator} / {denom:.4f}
= {sim:.4f}
Difference: -1 * sim = {-sim:.4f}
    """.strip()
explain_mcc = explain_phi
explain_matthews = explain_phi


@Metric
def sim_phi(a, b, n_universe=None, **_) -> float:
    n00, n01, n10, n11 = _aleph_counts(a, b, n_universe)
    denom = math.sqrt((n11 + n10) * (n11 + n01) * (n10 + n00) * (n01 + n00))
    if denom == 0:
        return 0.0
    return (n11 * n00 - n10 * n01) / denom
sim_mcc = sim_phi
sim_matthews = sim_phi


@Metric
def dif_phi(a, b, n_universe=None, **_) -> float:
    return -sim_phi(a, b, n_universe=n_universe)
dif_mcc = dif_phi
dif_matthews = dif_phi


METRICS['phi'] = {
    'class': 'set',
    'default': 'sim',
    'sim': sim_phi,
    'dif': dif_phi,
    'info': info_phi,
    'explain': explain_phi,
}
METRICS['mcc'] = METRICS['phi']
METRICS['matthews'] = METRICS['phi']
