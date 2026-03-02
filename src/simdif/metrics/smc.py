from ..simdif import Metric, METRICS, to_set, _aleph_counts


def info_smc() -> str:
    return """
Simple Matching Coefficient (SMC)
----------------------------------
Measures similarity by counting both shared presences and shared absences
relative to the total universe size. Unlike Jaccard, shared absences count
as evidence of similarity — making it a symmetrical measure appropriate when
the absence of an attribute is as meaningful as its presence.
Formula:
    SMC(A,B) = (n11 + n00) / N
             = (|A∩B| + shared absences) / n_universe
Range: [0, 1]
    1 = identical (same elements present and absent)
    0 = no agreement on any attribute
Aliases: SMC, Sokal-Michener
Note: Requires n_universe. When n_universe = |A∪B|, n00 = 0 and SMC reduces
to Jaccard similarity.
    """.strip()
info_sokal_michener = info_smc


def explain_smc(a, b, n_universe, **_) -> str:
    a_set, b_set = to_set(a), to_set(b)
    n00, n01, n10, n11 = _aleph_counts(a, b, n_universe)
    n_total = n11 + n10 + n01 + n00
    intersection = sorted(map(str, a_set & b_set))
    only_a = sorted(map(str, a_set - b_set))
    only_b = sorted(map(str, b_set - a_set))
    sim = (n11 + n00) / n_total if n_total > 0 else 1.0
    return f"""
A: ({", ".join(sorted(map(str, a_set)))})
B: ({", ".join(sorted(map(str, b_set)))})
Simple Matching Coefficient (Universe-Aware):
Shared presences (n11): {n11} ({", ".join(intersection)})
Only in A (n10):        {n10} ({", ".join(only_a)})
Only in B (n01):        {n01} ({", ".join(only_b)})
Shared absences (n00):  {n00}
Total universe (N):     {n_total} {f'(corrected from {n_universe})' if n_universe is not None and n_total != n_universe else ''}
Calculation:
  (n11 + n00) / N
= ({n11} + {n00}) / {n_total}
= {n11 + n00} / {n_total}
= {sim:.4f}
Difference: 1 - Sim = {1 - sim:.4f}
    """.strip()
explain_sokal_michener = explain_smc


@Metric
def sim_smc(a, b, n_universe=None) -> float:
    n00, n01, n10, n11 = _aleph_counts(a, b, n_universe)
    if (n11 + n10 + n01 + n00) == 0:
        return 1.0
    return (n11 + n00) / (n11 + n10 + n01 + n00)
sim_sokal_michener = sim_smc


@Metric
def dif_smc(a, b, n_universe=None) -> float:
    return 1.0 - sim_smc(a, b, n_universe)
dif_sokal_michener = dif_smc


METRICS['smc'] = {
    'class': 'set',
    'default': 'sim',
    'sim': sim_smc,
    'dif': dif_smc,
    'info': info_smc,
    'explain': explain_smc,
}
METRICS['sokal_michener'] = METRICS['smc']
