from ..simdif import Metric, METRICS, to_list_numeric_aligned


def info_tanimoto_continuous() -> str:
    return """
Continuous (Extended) Tanimoto Coefficient
------------------------------------------
The real-valued generalization of the Tanimoto/Jaccard coefficient, for
count or weighted vectors rather than 0/1 bit vectors. Common in
cheminformatics for count-based or feature-weighted fingerprints.

Formula:
    T(A, B) = (A . B) / (||A||^2 + ||B||^2 - A . B)
            = Σ(a_i b_i) / (Σa_i^2 + Σb_i^2 - Σ(a_i b_i))

Range: [0, 1] for non-negative vectors
    1 = identical vectors (two all-zero vectors are defined as 1.0)
    0 = no shared mass (e.g. orthogonal non-negative vectors)
    Can go slightly negative when vectors have opposing (signed) components.

Note: reduces EXACTLY to the binary Tanimoto (= Jaccard) when the inputs are
0/1 vectors, since then A.B = shared 1s and ||A||^2 = count of 1s in A. Unlike
cosine, Tanimoto is NOT scale-invariant: doubling one vector changes the score.

Aliases: continuous_tanimoto, extended_tanimoto
    """.strip()
info_continuous_tanimoto = info_extended_tanimoto = info_tanimoto_continuous


def _tanimoto_continuous(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    denom = sum(x * x for x in a) + sum(y * y for y in b) - dot
    if denom == 0:
        return 1.0  # both vectors all-zero: defined as identical
    return dot / denom


def explain_tanimoto_continuous(a, b, **kwargs) -> str:
    a, b = to_list_numeric_aligned(a, b, **kwargs)
    dot = sum(x * y for x, y in zip(a, b))
    na2 = sum(x * x for x in a)
    nb2 = sum(y * y for y in b)
    denom = na2 + nb2 - dot
    sim = sim_tanimoto_continuous(a, b)
    return f"""
A: ({", ".join(map(str, a))})
B: ({", ".join(map(str, b))})
Continuous Tanimoto:
A . B (dot):        {dot}
||A||^2:            {na2}
||B||^2:            {nb2}
Calculation:
  (A . B) / (||A||^2 + ||B||^2 - A . B)
= {dot} / ({na2} + {nb2} - {dot})
= {dot} / {denom}
= {sim:.4f}
Difference: 1 - Sim = {1 - sim:.4f}
    """.strip()
explain_continuous_tanimoto = explain_extended_tanimoto = explain_tanimoto_continuous


@Metric
def sim_tanimoto_continuous(a, b, **kwargs) -> float:
    a, b = to_list_numeric_aligned(a, b, **kwargs)
    if len(a) == 0 and len(b) == 0:
        return 1.0
    return _tanimoto_continuous(a, b)
sim_continuous_tanimoto = sim_extended_tanimoto = sim_tanimoto_continuous


@Metric
def dif_tanimoto_continuous(a, b, **kwargs) -> float:
    return 1 - sim_tanimoto_continuous(a, b, **kwargs)
dif_continuous_tanimoto = dif_extended_tanimoto = dif_tanimoto_continuous


METRICS['tanimoto_continuous'] = {
    'class': 'vector',
    'default': 'sim',
    'sim': sim_tanimoto_continuous,
    'dif': dif_tanimoto_continuous,
    'info': info_tanimoto_continuous,
    'explain': explain_tanimoto_continuous,
}
METRICS['continuous_tanimoto'] = METRICS['tanimoto_continuous']
METRICS['extended_tanimoto'] = METRICS['tanimoto_continuous']
