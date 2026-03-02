from ..simdif import METRICS, _aleph_counts, to_set

def info_jaccard() -> str:
    return """
Jaccard Similarity Index
------------------------
Measures similarity between two sets as intersection over union.

Formula:
    J(A, B) = |A ∩ B| / |A ∪ B|

Range: [0, 1]
    1 = identical sets
    0 = no shared elements

Difference:
    Calculated as 1 - Similarity

Note: This is an index, not a coefficient.
    Arithmetic operations such as averaging or subtraction are not meaningful.

Aliases: IoU (Intersection over Union), Tanimoto-Set
    """.strip()
info_iou = info_jaccard


def explain_jaccard(a, b, **_) -> str:
    a, b = to_set(a), to_set(b)
    i = sorted(map(str, a & b))
    u = sorted(map(str, a | b))
    a = sorted(map(str, a))
    b = sorted(map(str, b))
    ni = len(i)
    nu = len(u)
    return f"""
A: ({", ".join(a)})
B: ({", ".join(b)})
Intersection: ({", ".join(i)}), length {ni}
Union: ({", ".join(u)}), length {nu}
Similarity: {ni} / {nu} = {ni / nu if nu>0 else 'Division by Zero'}
Difference: 1 - Similarity = {1 - ni / nu if nu>0 else 'Division by Zero'}
    """.strip()
explain_iou = explain_jaccard


def sim_jaccard(a, b, **_) -> float:
    n00, n01, n10, n11 = _aleph_counts(a, b)
    if (n11 + n10 + n01) == 0:
        return 1.0
    return n11 / (n11 + n10 + n01)
sim_iou = sim_jaccard


def dif_jaccard(a, b, **_) -> float:
    return 1 - sim_jaccard(a, b)
dif_iou = dif_jaccard


METRICS['jaccard'] = {
    'class': 'set',
    'default': 'sim',
    'sim': sim_jaccard,
    'dif': dif_jaccard,
    'info': info_jaccard,
    'explain': explain_jaccard,
}
METRICS['iou'] = METRICS['jaccard']
