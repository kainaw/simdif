from ..simdif import Metric, METRICS, _aleph_counts, to_set


def info_cosine_set() -> str:
    return """
Set Cosine Similarity
---------------------
Applies cosine similarity to sets by treating them as binary vectors.

Formula:
    C(A, B) = |A ∩ B| / √(|A| · |B|)

Range: [0, 1]
    1 = identical
    0 = no shared elements

Aliases: Ochiai
    """.strip()
info_ochiai = info_cosine_set


def explain_cosine_set(a, b, **_) -> str:
    a_set, b_set = to_set(a), to_set(b)
    i_list = sorted(map(str, a_set & b_set))
    a_list = sorted(map(str, a_set))
    b_list = sorted(map(str, b_set))
    ni, na, nb = len(i_list), len(a_list), len(b_list)
    geo_mean = (na * nb) ** 0.5
    if na == 0 and nb == 0:
        sim = 1.0
        calc_str = "Both sets are empty (Identical) = 1.0"
    elif geo_mean == 0:
        sim = 0.0
        calc_str = f"{ni} / sqrt({na} * {nb}) = 0.0"
    else:
        sim = ni / geo_mean
        calc_str = f"{ni} / sqrt({na} * {nb}) = {ni} / {geo_mean:.4f} = {sim:.4f}"

    return f"""
A: ({", ".join(a_list)}), length {na}
B: ({", ".join(b_list)}), length {nb}
Intersection: ({", ".join(i_list)}), length {ni}
Formula: |A ∩ B| / sqrt(|A| * |B|)
Calculation: {calc_str}
Similarity: {sim:.4f}
Difference: 1 - Similarity = {1.0 - sim:.4f}
    """.strip()
explain_ochiai = explain_cosine_set


@Metric
def sim_cosine_set(a, b, **_) -> float:
    a, b = to_set(a), to_set(b)
    if len(a) == 0 and len(b) == 0:
        return 1.0
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return len(a & b)/(len (a) * len(b)) ** 0.5
sim_ochiai = sim_cosine_set


@Metric
def dif_cosine_set(a, b, **_) -> float:
    return 1 - sim_cosine_set(a, b)
dif_ochiai = dif_cosine_set


METRICS['cosine_set'] = {
    'class': 'set',
    'default': 'sim',
    'sim': sim_cosine_set,
    'dif': dif_cosine_set,
    'info': info_cosine_set,
    'explain': explain_cosine_set,
}
METRICS['ochiai'] = METRICS['cosine_set']
