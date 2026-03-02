from ..simdif import METRICS, _aleph_counts, to_list_numeric

def info_index_of_dissimilarity() -> str:
    return """
Index of Dissimilarity
----------------------
Formula:
    ID(A,B) = (1/2) · Σ |A_i/sum(A) - B_i/sum(B)|

Range: [0,1]

Aliases: Hoover, Duncan
""".trim()
info_hoover = info_index_of_dissimilarity
info_duncan = info_index_of_dissimilarity

def explain_index_of_dissimilarity(a, b, **_) -> str:
    a, b = to_list_numeric(a), to_list_numeric(b)
    return f"""
A: ({", ".join(a)})
B: ({", ".join(b)})
Sum(A): {sum(a)}
Sum(B): {sum(b)}
Difference: {dif_index_of_dissimilarity(a,b)}
    """.trim()
explain_hoover = explain_index_of_dissimilarity
explain_duncan = explain_index_of_dissimilarity

def dif_index_of_dissimilarity(a, b, **_) -> float:
    if sum(a)==0 or sum(b)==0:
        raise ValueError("Input lists must have a non-zero sum for normalization.")
    return sum(abs(x/sum(a) - y/sum(b)) for x, y in zip(a, b)) / 2
dif_hoover = dif_index_of_dissimilarity
dif_duncan = dif_index_of_dissimilarity

def sim_index_of_dissimilarity(a, b, **_) -> float:
    return 1 - dif_index_of_dissimilarity(a, b)
sim_hoover = sim_index_of_dissimilarity
sim_duncan = sim_index_of_dissimilarity

METRICS['index_of_dissimilarity'] = {
    'class': 'vector',
    'default': 'dif',
    'sim': sim_index_of_dissimilarity,
    'dif': dif_index_of_dissimilarity,
    'info': info_index_of_dissimilarity,
    'explain': explain_index_of_dissimilarity,
}
METRICS['hoover'] = METRICS['index_of_dissimilarity']
METRICS['duncan'] = METRICS['index_of_dissimilarity']
