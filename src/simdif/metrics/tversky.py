from ..simdif import Metric, METRICS, _aleph_counts, to_set


def info_tversky() -> str:
    return """
Tversky Index
-------------
Generalization of both Jaccard and Dice-Sorensen that allows asymmetric weighting of the two sets via parameters α and β.

Formula:
    T(A, B) = |A ∩ B| / (|A ∩ B| + α|A - B| + β|B - A|)

Parameters:
    α controls the penalty for elements in A but not B
    β controls the penalty for elements in B but not A
    α = β = 1 → Jaccard
    α = β = 0.5 → Dice
    α and β cannot both be 0.

Range: [0, 1]

Difference:
    Calculated as 1 - Similarity

Note: This is an index, not a coefficient.
    Arithmetic operations such as averaging or subtraction are not meaningful.
    """.strip()


def explain_tversky(a, b, alpha=0.5, beta=0.5, **_) -> str:
    a, b = to_set(a), to_set(b)
    i = sorted(map(str, a & b))
    u = sorted(map(str, a | b))
    a = sorted(map(str, a))
    b = sorted(map(str, b))
    ua = sorted(map(str, a - b))
    ub = sorted(map(str, b - a))
    ni = len(i)
    nu = len(u)
    nua = len(ua)
    nub = len(ub)
    return f"""
A: ({", ".join(a)})
B: ({", ".join(b)})
α: {alpha}
β: {beta}
T(A, B) = |A ∩ B| / (|A ∩ B| + α|A - B| + β|B - A|)
Intersection: ({", ".join(i)}), length {ni}
Union: ({", ".join(u)}), length {nu}
Unique to A: ({", ".join(ua)})
Unique to B: ({", ".join(ub)})
Similarity: {ni} / ({nu} + {alpha}×{nua} + {beta}×{nub} = {ni / (nu+alpha*nua+beta*nub) if nu+alpha*nua+beta*nub>0 else 'Division by Zero'}
Difference: 1 - Similarity = {1 - ni / (nu+alpha*nua+beta*nub) if nu+alpha*nua+beta*nub>0 else 'Division by Zero'}
    """.strip()


@Metric
def sim_tversky(a, b, alpha=0.5, beta=0.5, **_) -> float:
    if alpha == 0 and beta == 0:
        raise ValueError("alpha and beta cannot both be 0")
    a, b = to_set(a), to_set(b)
    if len(a) == 0 and len(b) == 0:
        return 1.0
    intersection = len(a & b)
    return intersection / (intersection + alpha*len(a - b) + beta*len(b - a))


@Metric
def dif_tversky(a, b, alpha=0.5, beta=0.5, **_) -> float:
    return 1 - sim_tversky(a, b, alpha, beta)


METRICS['tversky'] = {
    'class': 'set',
    'default': 'sim',
    'sim': sim_tversky,
    'dif': dif_tversky,
    'info': info_tversky,
    'explain': explain_tversky,
}
