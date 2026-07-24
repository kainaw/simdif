from ..simdif import Metric, METRICS, _aleph_counts, to_set


def info_braun_blanquet() -> str:
    return """
Braun-Blanquet Similarity Coefficient
-------------------------------------
The number of shared elements divided by the size of the LARGER set. Because it
normalizes by the larger of the two sets, it is a conservative, asymmetric-safe
measure of overlap. Shared absences are ignored, so no universe size is needed.
Formula:
    BB(A, B) = a / max(a + b, a + c)
        a = |A ∩ B|   (in both)
        b = |A only|  (in A, not B)
        c = |B only|  (in B, not A)
    Note: a + b = |A| and a + c = |B|, so the denominator is max(|A|, |B|).
Range: [0, 1]
    1 = the smaller set is fully contained and the sets are the same size
    0 = no shared elements
Aliases: Braun-Blanquet
    """.strip()


def explain_braun_blanquet(a, b, **_) -> str:
    a_set, b_set = to_set(a), to_set(b)
    n00, n01, n10, n11 = _aleph_counts(a, b)
    intersection = sorted(map(str, a_set & b_set))
    size_a = n11 + n10
    size_b = n11 + n01
    denom = max(size_a, size_b)
    sim = n11 / denom if denom > 0 else 1.0
    return f"""
A: ({", ".join(sorted(map(str, a_set)))})
B: ({", ".join(sorted(map(str, b_set)))})
Braun-Blanquet:
a (in both): {n11} ({", ".join(intersection)})
|A| (a + b): {size_a}
|B| (a + c): {size_b}
Calculation:
  a / max(|A|, |B|)
= {n11} / max({size_a}, {size_b})
= {n11} / {denom}
= {sim:.4f}
Difference: 1 - Sim = {1 - sim:.4f}
    """.strip()


@Metric
def sim_braun_blanquet(a, b, **_) -> float:
    n00, n01, n10, n11 = _aleph_counts(a, b)
    denom = max(n11 + n10, n11 + n01)
    if denom == 0:
        return 1.0  # both sets empty -> identical
    return n11 / denom


@Metric
def dif_braun_blanquet(a, b, **_) -> float:
    return 1.0 - sim_braun_blanquet(a, b)


METRICS['braun_blanquet'] = {
    'class': 'set',
    'default': 'sim',
    'sim': sim_braun_blanquet,
    'dif': dif_braun_blanquet,
    'info': info_braun_blanquet,
    'explain': explain_braun_blanquet,
}
