import math
from ..simdif import Metric, METRICS, to_list


# ------------------------------------------------------------------
# Shared pair-counting adapter
# ------------------------------------------------------------------

def _labels(a, b):
    la, lb = to_list(a), to_list(b)
    if len(la) != len(lb):
        raise ValueError("Vector length mismatch")
    return la, lb


def _pair_counts(la, lb):
    """Over all object pairs, count agreements/disagreements between two
    clusterings given as aligned label sequences:
        a = together in both        b = together in A only
        c = together in B only      d = apart in both
    Returns (a, b, c, d). Label-invariant: only '==' of labels is used."""
    n = len(la)
    a = b = c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            same_a = la[i] == la[j]
            same_b = lb[i] == lb[j]
            if same_a and same_b:
                a += 1
            elif same_a:
                b += 1
            elif same_b:
                c += 1
            else:
                d += 1
    return a, b, c, d


def _pair_table(a, b, c, d):
    total = a + b + c + d
    return f"""Object pairs (C(n,2) = {total}):
  a  together in both   = {a}   (agreement)
  b  together in A only = {b}   (disagreement)
  c  together in B only = {c}   (disagreement)
  d  apart in both      = {d}   (agreement)"""


# ------------------------------------------------------------------
# Rand Index
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Adjusted Rand Index
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Fowlkes-Mallows Index
# ------------------------------------------------------------------

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
