from ..simdif import Metric, METRICS
from ._helpers import _labels, _pair_counts, _pair_table


def info_adjusted_rand() -> str:
    return """
Adjusted Rand Index (ARI)
-------------------------
The Rand Index corrected for the agreement expected by chance. This matters
because raw Rand is inflated by the many pairs held apart in both clusterings.

Formula (from the pair-agreement table):
    ARI = 2(a*d - b*c) / ((a+b)(b+d) + (a+c)(c+d))

Range: (<= 1]
    1  = identical clusterings
    0  = agreement no better than random labelling
    <0 = agreement worse than chance

Roles:
    sim - the adjusted index (can be negative, so NOT bounded to [0, 1])

Note: A and B are two clusterings of the SAME objects as equal-length label
sequences. Prefer ARI over raw Rand when comparing clusterings, especially with
many clusters or unequal cluster sizes.

Aliases: ARI, adjusted_rand_index
    """.strip()
info_ari = info_adjusted_rand_index = info_adjusted_rand


def _ari(la, lb):
    a, b, c, d = _pair_counts(la, lb)
    denom = (a + b) * (b + d) + (a + c) * (c + d)
    if denom == 0:
        return 1.0  # degenerate (e.g. fewer than 2 objects): treat as identical
    return 2.0 * (a * d - b * c) / denom


def explain_adjusted_rand(a, b, **kwargs) -> str:
    la, lb = _labels(a, b)
    aa, bb, cc, dd = _pair_counts(la, lb)
    denom = (aa + bb) * (bb + dd) + (aa + cc) * (cc + dd)
    ari = _ari(la, lb)
    return f"""
A (clustering): {la}
B (clustering): {lb}
{_pair_table(aa, bb, cc, dd)}
ARI = 2(a*d - b*c) / ((a+b)(b+d) + (a+c)(c+d))
    = 2*({aa}*{dd} - {bb}*{cc}) / (({aa}+{bb})({bb}+{dd}) + ({aa}+{cc})({cc}+{dd}))
    = {2 * (aa * dd - bb * cc)} / {denom}
    = {ari:.4f}
    """.strip()
explain_ari = explain_adjusted_rand_index = explain_adjusted_rand


@Metric
def sim_adjusted_rand(a, b, **kwargs) -> float:
    return _ari(*_labels(a, b))
sim_ari = sim_adjusted_rand_index = sim_adjusted_rand


METRICS['adjusted_rand'] = {
    'class': 'sequence',
    'default': 'sim',
    'sim': sim_adjusted_rand,
    'info': info_adjusted_rand,
    'explain': explain_adjusted_rand,
}
METRICS['ari'] = METRICS['adjusted_rand_index'] = METRICS['adjusted_rand']
