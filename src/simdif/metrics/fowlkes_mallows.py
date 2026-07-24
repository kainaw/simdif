import math
from ..simdif import Metric, METRICS
from ._helpers import _labels, _pair_counts, _pair_table


def info_fowlkes_mallows() -> str:
    return """
Fowlkes-Mallows Index
---------------------
The geometric mean of the precision and recall of co-clustered pairs: of the
pairs put together in A, how many are together in B, and vice versa.

Formula:
    FM = a / sqrt((a + b) * (a + c))
    over the pair-agreement table. This is the Ochiai (cosine-set) coefficient
    applied to object pairs.

Roles:
    sim - a / sqrt((a+b)(a+c))
    dif - 1 - sim

Range: [0, 1]
    1 = identical clusterings, 0 = no pair is together in both

Note: A and B are two clusterings of the SAME objects as equal-length label
sequences. Undefined when one clustering places no pair together (all
singletons); reported as 0.0 there.

Aliases: FM, fowlkes_mallows_index
    """.strip()
info_fm = info_fowlkes_mallows_index = info_fowlkes_mallows


def explain_fowlkes_mallows(a, b, **kwargs) -> str:
    la, lb = _labels(a, b)
    aa, bb, cc, dd = _pair_counts(la, lb)
    denom = math.sqrt((aa + bb) * (aa + cc))
    fm = aa / denom if denom else 0.0
    return f"""
A (clustering): {la}
B (clustering): {lb}
{_pair_table(aa, bb, cc, dd)}
FM = a / sqrt((a+b)(a+c)) = {aa} / sqrt(({aa}+{bb})({aa}+{cc})) = {aa} / {denom:.4f} = {fm:.4f}
    """.strip()
explain_fm = explain_fowlkes_mallows_index = explain_fowlkes_mallows


@Metric
def sim_fowlkes_mallows(a, b, **kwargs) -> float:
    aa, bb, cc, dd = _pair_counts(*_labels(a, b))
    denom = math.sqrt((aa + bb) * (aa + cc))
    if denom == 0:
        return 0.0
    return aa / denom
sim_fm = sim_fowlkes_mallows_index = sim_fowlkes_mallows


@Metric
def dif_fowlkes_mallows(a, b, **kwargs) -> float:
    return 1.0 - sim_fowlkes_mallows(a, b, **kwargs)
dif_fm = dif_fowlkes_mallows_index = dif_fowlkes_mallows


METRICS['fowlkes_mallows'] = {
    'class': 'sequence',
    'default': 'sim',
    'sim': sim_fowlkes_mallows,
    'dif': dif_fowlkes_mallows,
    'info': info_fowlkes_mallows,
    'explain': explain_fowlkes_mallows,
}
METRICS['fm'] = METRICS['fowlkes_mallows_index'] = METRICS['fowlkes_mallows']
