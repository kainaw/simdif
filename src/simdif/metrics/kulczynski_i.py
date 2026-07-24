from ..simdif import Metric, METRICS, _aleph_counts, to_set


def info_kulczynski_i() -> str:
    return """
Kulczynski Similarity Coefficient I (K1)
----------------------------------------
The ratio of shared elements to non-shared elements. Counts matches (elements
in both sets) against mismatches (elements in exactly one set). Shared absences
are ignored, so no universe size is needed.
Formula:
    K1(A, B) = a / (b + c)
        a = |A ∩ B|   (in both)
        b = |A only|  (in A, not B)
        c = |B only|  (in B, not A)
Range: [0, inf)
    inf = identical sets (no mismatches)
    0   = no shared elements
Note: Unlike Jaccard, K1 is unbounded above, so it is reported as a raw
similarity score. No difference (1 - sim) is provided because it would be
meaningless on an unbounded scale.
Aliases: Kulczynski I
    """.strip()


def explain_kulczynski_i(a, b, **_) -> str:
    a_set, b_set = to_set(a), to_set(b)
    n00, n01, n10, n11 = _aleph_counts(a, b)
    intersection = sorted(map(str, a_set & b_set))
    only_a = sorted(map(str, a_set - b_set))
    only_b = sorted(map(str, b_set - a_set))
    denom = n10 + n01
    if denom == 0:
        sim = float('inf') if n11 > 0 else 0.0
    else:
        sim = n11 / denom
    return f"""
A: ({", ".join(sorted(map(str, a_set)))})
B: ({", ".join(sorted(map(str, b_set)))})
Kulczynski I:
a (in both): {n11} ({", ".join(intersection)})
b (A only):  {n10} ({", ".join(only_a)})
c (B only):  {n01} ({", ".join(only_b)})
Calculation:
  a / (b + c)
= {n11} / ({n10} + {n01})
= {n11} / {denom}
= {sim}
    """.strip()


@Metric
def sim_kulczynski_i(a, b, **_) -> float:
    n00, n01, n10, n11 = _aleph_counts(a, b)
    denom = n10 + n01
    if denom == 0:
        return float('inf') if n11 > 0 else 0.0
    return n11 / denom


METRICS['kulczynski_i'] = {
    'class': 'set',
    'default': 'sim',
    'sim': sim_kulczynski_i,
    'info': info_kulczynski_i,
    'explain': explain_kulczynski_i,
}
