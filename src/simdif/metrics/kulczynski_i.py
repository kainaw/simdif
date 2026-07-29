from ..simdif import Metric, METRICS, _aleph_counts, to_set


def info_kulczynski_i() -> str:
    return """
Kulczynski Score I (K1)
------------------------
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
Note: K1 is a raw ratio of counts, not a coefficient calibrated to [0, 1] -
unlike Jaccard, it has no fixed maximum, so it is reported as a 'score'
rather than a 'sim'. Its distance is the plain reciprocal:
    dist(A, B) = (b + c) / a
Range: [0, inf)
    0   = identical sets
    inf = no shared elements
There is no 'sim' or 'dif' role: without knowing the largest score actually
possible for a given A/B, there is nothing to normalize dist against to get
a [0, 1] difference (and 1 minus that, a similarity).
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
        score = float('inf') if n11 > 0 else 0.0
    else:
        score = n11 / denom
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
= {score}
Distance (reciprocal of score):
  (b + c) / a
= {denom} / {n11}
= {'inf' if n11 == 0 else denom / n11}
    """.strip()


@Metric
def score_kulczynski_i(a, b, **_) -> float:
    n00, n01, n10, n11 = _aleph_counts(a, b)
    denom = n10 + n01
    if denom == 0:
        return float('inf') if n11 > 0 else 0.0
    return n11 / denom


@Metric
def dist_kulczynski_i(a, b, **_) -> float:
    n00, n01, n10, n11 = _aleph_counts(a, b)
    if n11 == 0:
        return float('inf')
    return (n10 + n01) / n11


METRICS['kulczynski_i'] = {
    'class': 'set',
    'default': 'score',
    'score': score_kulczynski_i,
    'dist': dist_kulczynski_i,
    'info': info_kulczynski_i,
    'explain': explain_kulczynski_i,
}
