from ..simdif import METRICS, _aleph_counts, to_set

def info_dice_sorensen() -> str:
    return """
Dice-Sorensen Similarity Coefficient
------------------------------------
Measures set overlap by weighting the intersection double relative to the total size of both sets.

Formula:
    D(A, B) = 2|A ∩ B| / (|A| + |B|)

Range: [0, 1]
    1 = identical
    0 = no shared elements

Difference:
    Calculated as 1 - Similarity

Aliases: Dice, Sorensen, Dice-Sorensen, Sorensen-Dice

Note: Sørensen is the proper spelling, but 'Sorensen' is used throughout the code for keyboard convenience.
    """.strip()
info_sorensen_dice = info_dice_sorensen
info_dice = info_dice_sorensen
info_sorensen = info_dice_sorensen

def explain_dice_sorensen(a, b) -> str:
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
Similarity: 2 * {ni} / ({na + nb}) = {2*ni / (na+nb) if (na+nb)>0 else 'Division by Zero'}
Difference: 1 - Similarity = {1 - 2 * ni / (na+nb) if (na+nb)>0 else 'Division by Zero'}
    """.strip()
explain_sorensen_dice = explain_dice_sorensen
explain_dice = explain_dice_sorensen
explain_sorensen = explain_dice_sorensen

def sim_dice_sorensen(a, b) -> float:
    n00, n01, n10, n11 = _aleph_counts(a, b)
    if (n11 + n10 + n01) == 0:
        return 1.0
    return 2 * n11 / (2 * n11 + n10 + n01)
sim_sorensen_dice = sim_dice_sorensen
sim_dice = sim_dice_sorensen
sim_sorensen = sim_dice_sorensen

def dif_dice_sorensen(a, b) -> float:
    return 1 - sim_dice(a, b)
dif_sorensen_dice = dif_dice_sorensen
dif_dice = dif_dice_sorensen
dif_sorensen = dif_dice_sorensen

METRICS['dice_sorensen'] = {
    'default': 'sim',
    'sim': sim_dice_sorensen,
    'dif': dif_dice_sorensen,
    'info': info_dice_sorensen,
    'explain': explain_dice_sorensen,
}
METRICS['sorensen_dice'] = METRICS['dice_sorensen']
METRICS['dice'] = METRICS['dice_sorensen']
METRICS['sorensen'] = METRICS['dice_sorensen']
