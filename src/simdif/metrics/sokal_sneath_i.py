from ..simdif import Metric, METRICS, _aleph_counts, to_set


def info_sokal_sneath_i() -> str:
    return """
Sokal-Sneath I Similarity Coefficient (SSI)
-------------------------------------------
A stricter version of Jaccard similarity that applies double weight to 
differences (mismatches) between sets.

Formula:
    SSI(A, B) = |A ∩ B| / (|A ∩ B| + 2 * |A Δ B|)
    Alternatively: n11 / (n11 + 2 * (n10 + n01))

Range: [0, 1]
    1 = identical
    0 = no shared elements

Difference:
    Calculated as 1 - Similarity

Aliases: SSI, Sokal-Sneath I
    """.strip()
info_ssi = info_sokal_sneath_i


def explain_sokal_sneath_i(a, b, **_) -> str:
    a, b = to_set(a), to_set(b)
    i = sorted(map(str, a & b))
    diff = sorted(map(str, a ^ b)) # Symmetric difference (mismatches)
    ni = len(i)
    ndiff = len(diff)
    denominator = ni + (2 * ndiff)
    similarity = ni / denominator if denominator > 0 else 1.0
    return f"""
A: ({", ".join(sorted(map(str, a)))}), length {len(a)}
B: ({", ".join(sorted(map(str, b)))}), length {len(b)}
Intersection (n11): ({", ".join(i)}), length {ni}
Mismatches (n10 + n01): ({", ".join(diff)}), length {ndiff}
Similarity: {ni} / ({ni} + 2 * {ndiff}) = {similarity}
Difference: 1 - Similarity = {1 - similarity}
    """.strip()
explain_ssi = explain_sokal_sneath_i


@Metric
def sim_sokal_sneath_i(a, b, **_) -> float:
    n00, n01, n10, n11 = _aleph_counts(a, b)
    denominator = n11 + 2 * (n10 + n01)
    if denominator == 0:
        return 1.0
    return n11 / denominator
sim_ssi = sim_sokal_sneath_i


@Metric
def dif_sokal_sneath_i(a, b, **_) -> float:
    return 1 - sim_sokal_sneath_i(a, b)
dif_ssi = dif_sokal_sneath_i


METRICS['sokal_sneath_i'] = {
    'class': 'set',
    'default': 'sim',
    'sim': sim_sokal_sneath_i,
    'dif': dif_sokal_sneath_i,
    'info': info_sokal_sneath_i,
    'explain': explain_sokal_sneath_i,
}
METRICS['ssi'] = METRICS['sokal_sneath_i']
