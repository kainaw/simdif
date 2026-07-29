from ..simdif import Metric, METRICS, _aleph_counts, to_set


def info_yule_q() -> str:
    return """
Yule's Q (Coefficient of Association)
-------------------------------------
A universe-aware association measure built from the 2x2 contingency table of
two sets. Uses shared absences (d), so it requires the total universe size.
Formula:
    Q(A, B) = (a*d - b*c) / (a*d + b*c)
        a = |A ∩ B|             (in both)
        b = |A only|            (in A, not B)
        c = |B only|            (in B, not A)
        d = shared absences     (in neither; needs n_universe)
Range: [-1, 1]
    1  = perfect positive association (b*c = 0)
    0  = independence
    -1 = perfect negative association (a*d = 0)
Difference: -1 * similarity
Note: Requires n_universe to define d. Without it, d = 0 and Q collapses to
-1 whenever b*c > 0, which is meaningless - always pass n_universe. Because the
range is [-1, 1], 1 - sim would not be meaningful; negating similarity instead
preserves the signed range.
Aliases: Yule Q
    """.strip()


def explain_yule_q(a, b, n_universe=None, **_) -> str:
    a_set, b_set = to_set(a), to_set(b)
    n00, n01, n10, n11 = _aleph_counts(a, b, n_universe)
    n_total = n11 + n10 + n01 + n00
    intersection = sorted(map(str, a_set & b_set))
    only_a = sorted(map(str, a_set - b_set))
    only_b = sorted(map(str, b_set - a_set))
    ad = n11 * n00
    bc = n10 * n01
    sim = (ad - bc) / (ad + bc) if (ad + bc) != 0 else 0.0
    corrected = f'(corrected from {n_universe})' if n_universe is not None and n_total != n_universe else ''
    return f"""
A: ({", ".join(sorted(map(str, a_set)))})
B: ({", ".join(sorted(map(str, b_set)))})
Yule's Q (Universe-Aware):
a (in both):       {n11} ({", ".join(intersection)})
b (A only):        {n10} ({", ".join(only_a)})
c (B only):        {n01} ({", ".join(only_b)})
d (shared absent): {n00}
Total universe (N): {n_total} {corrected}
Calculation:
  (a*d - b*c) / (a*d + b*c)
= ({n11}*{n00} - {n10}*{n01}) / ({n11}*{n00} + {n10}*{n01})
= ({ad} - {bc}) / ({ad} + {bc})
= {sim:.4f}
Difference: -1 * sim = {-sim:.4f}
    """.strip()


@Metric
def sim_yule_q(a, b, n_universe=None, **_) -> float:
    n00, n01, n10, n11 = _aleph_counts(a, b, n_universe)
    ad = n11 * n00
    bc = n10 * n01
    if (ad + bc) == 0:
        return 0.0
    return (ad - bc) / (ad + bc)


@Metric
def dif_yule_q(a, b, n_universe=None, **_) -> float:
    return -sim_yule_q(a, b, n_universe=n_universe)


METRICS['yule_q'] = {
    'class': 'set',
    'default': 'sim',
    'sim': sim_yule_q,
    'dif': dif_yule_q,
    'info': info_yule_q,
    'explain': explain_yule_q,
}
