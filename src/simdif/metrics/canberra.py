from ..simdif import Metric, METRICS, to_list_numeric_aligned
from ._helpers import _bounded_dif


def info_canberra() -> str:
    return """
Canberra Distance
-----------------
A weighted version of the Manhattan distance that is sensitive to small values
near the origin. It computes the sum of absolute differences between elements,
divided by the sum of their absolute magnitudes.

Formula:
               n    |Ai - Bi|
    d(A, B) = sum  -----------
              i=1  |Ai| + |Bi|

Range: [0, n]
    Where n is the length of the sequences.
    0 = identical
    n = maximum theoretical distance (if all pairs have opposite signs/zeros)

Roles:
    dist: the raw sum above (>= 0, at most n)
    dif:  dist / n   -- every term is at most 1, so n is a real bound
    sim:  1 - dif

Because the maximum is known, Canberra needs no 1/(1+d) squash and no
supplied bound: dif and sim are exact linear rescalings of dist, and
sim + dif == 1. Contrast manhattan, which Canberra otherwise resembles --
there the per-term magnitude is unbounded, so no such n exists.

Note: If both Ai and Bi are zero, the term is skipped (treated as 0). The
n bound is therefore not attainable when any coordinate is zero in both
vectors, so dif reads slightly low for sparse data; n is still the
published bound and, unlike a count of the non-skipped terms, it stays
constant across a fixed-width dataset.
    """.strip()


def explain_canberra(a, b, **kwargs) -> str:
    a, b = to_list_numeric_aligned(a, b, **kwargs)
    terms = []
    total = 0.0
    for i, (x, y) in enumerate(zip(a, b)):
        num = abs(x - y)
        den = abs(x) + abs(y)
        val = num / den if den > 0 else 0.0
        total += val
        terms.append(f"  idx {i}: |{x} - {y}| / (|{x}| + |{y}|) = {num} / {den} = {val:.4f}")
    terms_display = "\n".join(terms)
    n = len(a)
    dif = _bounded_dif(total, n)
    return f"""
A: {a}
B: {b}
Canberra Contributions:
{terms_display}
Total Canberra Distance: {total:.4f}
Maximum: n = {n} (derived -- each of the {n} terms is at most 1)
Difference (dist / n): {total:.4f} / {n} = {dif:.4f}
Similarity (1 - dif): {1.0 - dif:.4f}
    """.strip()


@Metric
def dist_canberra(a, b, **kwargs) -> float:
    a, b = to_list_numeric_aligned(a, b, **kwargs)
    score = 0.0
    for x, y in zip(a, b):
        denominator = abs(x) + abs(y)
        if denominator > 0:
            score += abs(x - y) / denominator
    return score


@Metric
def dif_canberra(a, b, **kwargs) -> float:
    a, b = to_list_numeric_aligned(a, b, **kwargs)
    return _bounded_dif(dist_canberra(a, b, **kwargs), len(a))


@Metric
def sim_canberra(a, b, **kwargs) -> float:
    return 1.0 - dif_canberra(a, b, **kwargs)


METRICS['canberra'] = {
    'class': 'vector',
    'default': 'dist',
    'dist': dist_canberra,
    'dif': dif_canberra,
    'sim': sim_canberra,
    'info': info_canberra,
    'explain': explain_canberra,
}
