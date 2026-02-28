from ..simdif import METRICS, _aleph_counts, to_set

def info_overlap() -> str:
    return """
Overlap Coefficient
-------------------
Measures the degree to which one set is a subset of the other.

Formula:
    O(A, B) = |A ∩ B| / min(|A|, |B|)

Range: [0, 1]
    1 = one set is a complete subset of the other
    0 = no shared elements

Difference:
    Calculated as 1 - Similarity

Note: Returns 1.0 when one set is entirely contained within the other, regardless of the size difference.

Aliases: Szymkiewicz–Simpson, Simpson
    """.strip()
info_szymkiewicz_simpson = info_overlap
info_simpson = info_overlap

def explain_overlap(a, b, **_) -> str:
    a, b = to_set(a), to_set(b)
    i = sorted(map(str, a & b))
    a = sorted(map(str, a))
    b = sorted(map(str, b))
    ni = len(i)
    na = len(a)
    nb = len(b)
    return f"""
A: ({", ".join(a)}), length {na}
B: ({", ".join(b)}), length {nb}
Intersection: ({", ".join(i)}), length {ni}
Similarity: {ni} / min({na}, {nb}) = {ni / min(na,nb) if min(na,nb)>0 else 'Division by Zero'}
Difference: 1 - Similarity = {1 - ni / min(na,nb) if min(na,nb)>0 else 'Division by Zero'}
    """.strip()
explain_szymkiewicz_simpson = explain_overlap
explain_simpson = explain_overlap


def sim_overlap(a, b, **_) -> float:
    a, b = to_set(a), to_set(b)
    if len(a) == 0 and len(b) == 0:
        return 1.0
    if len(a) == 0 or len(b) == 0:
        return 0.0
    intersection = len(a & b)
    return intersection / min(len(a), len(b))
sim_szymkiewicz_simpson = sim_overlap
sim_simpson = sim_overlap


def dif_overlap(a, b, **_) -> float:
    return 1 - sim_overlap(a, b)
dif_szymkiewicz_simpson = dif_overlap
dif_simpson = dif_overlap


METRICS['overlap'] = {
    'default': 'sim',
    'sim': sim_overlap,
    'dif': dif_overlap,
    'info': info_overlap,
    'explain': explain_overlap,
}
METRICS['szymkiewicz_simpson'] = METRICS['overlap']
METRICS['simpson'] = METRICS['overlap']
