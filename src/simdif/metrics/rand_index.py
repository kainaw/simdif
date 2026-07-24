from ..simdif import Metric, METRICS
from ._helpers import _labels, _pair_counts, _pair_table


def info_rand_index() -> str:
    return """
Rand Index
----------
The fraction of object PAIRS on which two clusterings agree - agree that a pair
belongs together, or agree that it is apart. Label-invariant: relabelling the
clusters does not change the score.

Formula:
    RI = (a + d) / (a + b + c + d)
    over the 2x2 pair-agreement table (a = together in both, b = together in A
    only, c = together in B only, d = apart in both). This is the Simple
    Matching Coefficient applied to object pairs.

Roles:
    sim - (a + d) / total pairs
    dif - 1 - sim = (b + c) / total pairs (the pair-disagreement fraction)

Range: [0, 1]
    1 = identical clusterings, 0 = disagree on every pair

Note: A and B are two clusterings of the SAME objects, given as equal-length
label sequences aligned by object index (object i has label A[i] and B[i]).
Raw Rand is not corrected for chance - see adjusted_rand.

Aliases: Rand
    """.strip()
info_rand = info_rand_index


def explain_rand_index(a, b, **kwargs) -> str:
    la, lb = _labels(a, b)
    aa, bb, cc, dd = _pair_counts(la, lb)
    total = aa + bb + cc + dd
    sim = (aa + dd) / total if total else 1.0
    return f"""
A (clustering): {la}
B (clustering): {lb}
{_pair_table(aa, bb, cc, dd)}
Rand Index = (a + d) / total = ({aa} + {dd}) / {total} = {sim:.4f}
Disagreement (dif) = (b + c) / total = {1.0 - sim:.4f}
    """.strip()
explain_rand = explain_rand_index


@Metric
def sim_rand_index(a, b, **kwargs) -> float:
    aa, bb, cc, dd = _pair_counts(*_labels(a, b))
    total = aa + bb + cc + dd
    if total == 0:
        return 1.0
    return (aa + dd) / total
sim_rand = sim_rand_index


@Metric
def dif_rand_index(a, b, **kwargs) -> float:
    return 1.0 - sim_rand_index(a, b, **kwargs)
dif_rand = dif_rand_index


METRICS['rand_index'] = {
    'class': 'sequence',
    'default': 'sim',
    'sim': sim_rand_index,
    'dif': dif_rand_index,
    'info': info_rand_index,
    'explain': explain_rand_index,
}
METRICS['rand'] = METRICS['rand_index']
