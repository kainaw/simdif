from ..simdif import Metric, METRICS, _aleph_counts, to_set


def info_mcconnaughey() -> str:
    return """
McConnaughey Similarity Coefficient
--------------------------------------------------------------------
An association measure that rewards shared elements and penalizes
the product of the two mismatch counts, normalized by the two set
sizes. Shared absences are ignored, so no universe size is needed.
Formula:
    M(A, B) = (a^2 - b*c) / ((a + b) * (a + c))
        a = |A ∩ B|   (in both)
        b = |A only|  (in A, not B)
        c = |B only|  (in B, not A)
Range: [-1, 1]
     1 = identical sets
     0 = balance point
    -1 = disjoint sets
Difference: -1 * similarity
Note: Because the range is [-1, 1], 1 - sim would not be meaningful; negating
similarity instead preserves the signed range (1 = disjoint, -1 = identical).
Aliases: McConnaughey
    """.strip()


def explain_mcconnaughey(a, b, **_) -> str:
    a_set, b_set = to_set(a), to_set(b)
    n00, n01, n10, n11 = _aleph_counts(a, b)
    intersection = sorted(map(str, a_set & b_set))
    only_a = sorted(map(str, a_set - b_set))
    only_b = sorted(map(str, b_set - a_set))
    denom = (n11 + n10) * (n11 + n01)
    sim = (n11 * n11 - n10 * n01) / denom if denom > 0 else 0.0
    return f"""
A: ({", ".join(sorted(map(str, a_set)))})
B: ({", ".join(sorted(map(str, b_set)))})
McConnaughey:
a (in both): {n11} ({", ".join(intersection)})
b (A only):  {n10} ({", ".join(only_a)})
c (B only):  {n01} ({", ".join(only_b)})
Calculation:
  (a^2 - b*c) / ((a + b) * (a + c))
= ({n11}^2 - {n10}*{n01}) / (({n11} + {n10}) * ({n11} + {n01}))
= {n11 * n11 - n10 * n01} / {denom}
= {sim:.4f}
Difference: -1 * sim = {-sim:.4f}
    """.strip()


@Metric
def sim_mcconnaughey(a, b, **_) -> float:
    n00, n01, n10, n11 = _aleph_counts(a, b)
    denom = (n11 + n10) * (n11 + n01)
    if denom == 0:
        return 0.0  # at least one set empty -> association undefined
    return (n11 * n11 - n10 * n01) / denom


@Metric
def dif_mcconnaughey(a, b, **_) -> float:
    return -sim_mcconnaughey(a, b)


METRICS['mcconnaughey'] = {
    'class': 'set',
    'default': 'sim',
    'sim': sim_mcconnaughey,
    'dif': dif_mcconnaughey,
    'info': info_mcconnaughey,
    'explain': explain_mcconnaughey,
}
