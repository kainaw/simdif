from ..simdif import METRICS, to_set

def info_kulczynski() -> str:
    return """
Kulczynski Similarity Coefficient (K2)
---------------------------------------
The arithmetic mean of the conditional probabilities that an element in the
intersection belongs to A given A, and belongs to B given B. Treats both sets
symmetrically and handles asymmetric set sizes more gracefully than Jaccard.
Formula:
    K2(A,B) = (|A∩B| / 2) * (1/|A| + 1/|B|)
           = (1/2) * (|A∩B|/|A| + |A∩B|/|B|)
Range: [0, 1]
    1 = identical
    0 = no shared elements (or one set is empty)
Aliases: Kulczynski, Kulczynski II
    """.strip()

info_kulczynski_ii = info_kulczynski

def explain_kulczynski(a, b, **_) -> str:
    a_set, b_set = to_set(a), to_set(b)
    intersection = sorted(map(str, a_set & b_set))
    only_a = sorted(map(str, a_set - b_set))
    only_b = sorted(map(str, b_set - a_set))
    ni = len(a_set & b_set)
    na = len(a_set)
    nb = len(b_set)
    if na == 0 and nb == 0:
        sim = 1.0
    elif na == 0 or nb == 0:
        sim = 0.0
    else:
        sim = (ni / 2) * (1/na + 1/nb)
    return f"""
A: ({", ".join(sorted(map(str, a_set)))})
B: ({", ".join(sorted(map(str, b_set)))})
Kulczynski II:
Intersection (|A∩B|): {ni} ({", ".join(intersection)})
Only in A: {len(only_a)} ({", ".join(only_a)})
Only in B: {len(only_b)} ({", ".join(only_b)})
|A|: {na}
|B|: {nb}
Calculation:
  (|A∩B| / 2) * (1/|A| + 1/|B|)
= ({ni} / 2) * (1/{na} + 1/{nb})
= {ni/2:.4f} * {(1/na + 1/nb) if na > 0 and nb > 0 else 'undefined'}
= {sim:.4f}
Difference: 1 - Sim = {1 - sim:.4f}
    """.strip()

explain_kulczynski_ii = explain_kulczynski

def sim_kulczynski(a, b, **_) -> float:
    a, b = to_set(a), to_set(b)
    if len(a) == 0 and len(b) == 0:
        return 1.0
    if len(a) == 0 or len(b) == 0:
        return 0.0
    i = len(a & b)
    return (i / 2) * (1/len(a) + 1/len(b))

sim_kulczynski_ii = sim_kulczynski

def dif_kulczynski(a, b, **_) -> float:
    return 1 - sim_kulczynski(a, b)

dif_kulczynski_ii = dif_kulczynski

METRICS['kulczynski'] = {
    'class': 'set',
    'default': 'sim',
    'sim': sim_kulczynski,
    'dif': dif_kulczynski,
    'info': info_kulczynski,
    'explain': explain_kulczynski,
}
METRICS['kulczynski_ii'] = METRICS['kulczynski']
