import math
from ..simdif import Metric, METRICS, to_list_aligned
from ._helpers import _sim_from_dist, _dif_from_dist, _max_line


# Default transition/transversion partition for DNA (case-insensitive):
#   purines = A, G     pyrimidines = C, T
_DNA_GROUPS = [frozenset("AGag"), frozenset("CTct")]


def _group_of(sym, groups):
    for i, g in enumerate(groups):
        if sym in g:
            return i
    return None


def _transitions_transversions(a, b, groups, **kwargs):
    """(n_sites, transitions, transversions) under a symbol partition.

    A mismatch is a transition if both symbols share a group, a transversion if
    they fall in different groups. Sites where either symbol is outside every
    group (gaps, ambiguity codes, wrong case) are skipped (pairwise deletion),
    so n_sites counts only classifiable positions. Kimura needs this grouping
    because transition-vs-transversion is not expressible with '==' alone.
    """
    a, b = to_list_aligned(a, b, **kwargs)
    n_sites = transitions = transversions = 0
    for x, y in zip(a, b):
        gx, gy = _group_of(x, groups), _group_of(y, groups)
        if gx is None or gy is None:
            continue
        n_sites += 1
        if x != y:
            if gx == gy:
                transitions += 1
            else:
                transversions += 1
    return n_sites, transitions, transversions


def info_kimura() -> str:
    return """
Kimura 2-Parameter Distance (K80)
---------------------------------
An evolutionary distance that separates transitions (a substitution within a
symbol group, e.g. purine<->purine A<->G or pyrimidine<->pyrimidine C<->T) from
transversions (across groups), which typically occur at different rates.

Formula:
    d = -1/2 ln(1 - 2P - Q) - 1/4 ln(1 - 2Q)
    where P = transitions / sites and Q = transversions / sites.

Range: [0, inf)
    Saturates to inf when 1 - 2P - Q <= 0 or 1 - 2Q <= 0.

Roles:
    dist: the corrected distance above (>= 0, unbounded)
    sim:  1 / (1 + d), or 1 - dif when d_max is supplied
    dif:  1 - sim,     or d / d_max when d_max is supplied

Note: like jukes_cantor, the corrected substitutions-per-site count has no
ceiling, so sim defaults to the 1/(1+d) squash and a saturated pair
(dist = inf) gives dif=1.0 and sim=0.0 rather than nan. Use p_distance if you
want the bounded observed divergence instead; d_max here is a
substitutions-per-site cutoff you declare, not a derived bound.

WARNING -- d_max must be a real bound. Distances above it clamp to dif=1.0,
merging genuinely-different divergence levels. explain_ reports the clamp.

Note: unlike p-distance/JC69, K80 needs MORE than '=='; a mismatch must be
classifiable as within-group or across-group. The 'groups' parameter supplies
that partition and defaults to DNA purines {A,G} / pyrimidines {C,T}
(case-insensitive). Sites containing a symbol outside every group (gaps,
ambiguity codes) are skipped. Sequences must be the same length (assumed
already aligned).

Aliases: K80, K2P
    """.strip()
info_k80 = info_kimura
info_k2p = info_kimura


def explain_kimura(a, b, groups=None, **kwargs) -> str:
    if groups is None:
        groups = _DNA_GROUPS
    n_sites, ti, tv = _transitions_transversions(a, b, groups, **kwargs)
    P = ti / n_sites if n_sites else 0.0
    Q = tv / n_sites if n_sites else 0.0
    w1 = 1.0 - 2.0 * P - Q
    w2 = 1.0 - 2.0 * Q
    if n_sites == 0:
        d = 0.0
    elif w1 <= 0 or w2 <= 0:
        d = float('inf')
    else:
        d = -0.5 * math.log(w1) - 0.25 * math.log(w2)
    d_str = "inf (saturated)" if d == float('inf') else f"{d:.4f}"
    return f"""
Classifiable sites: {n_sites}
Transitions:  {ti}  -> P = {P:.4f}
Transversions: {tv}  -> Q = {Q:.4f}
d = -1/2 ln(1 - 2P - Q) - 1/4 ln(1 - 2Q)
  = -1/2 ln({w1:.4f}) - 1/4 ln({w2:.4f})
  = {d_str}
{_max_line(d, kwargs.get('d_max'),
           unbounded_note="unbounded -- substitutions per site have no ceiling (p_distance is the bounded observed form)")}
    """.strip()
explain_k80 = explain_kimura
explain_k2p = explain_kimura


@Metric
def dist_kimura(a, b, groups=None, **kwargs) -> float:
    if groups is None:
        groups = _DNA_GROUPS
    n_sites, ti, tv = _transitions_transversions(a, b, groups, **kwargs)
    if n_sites == 0:
        return 0.0
    P = ti / n_sites
    Q = tv / n_sites
    w1 = 1.0 - 2.0 * P - Q
    w2 = 1.0 - 2.0 * Q
    if w1 <= 0 or w2 <= 0:
        return float('inf')  # saturation
    return -0.5 * math.log(w1) - 0.25 * math.log(w2)
dist_k80 = dist_kimura
dist_k2p = dist_kimura


@Metric
def dif_kimura(a, b, **kwargs) -> float:
    return _dif_from_dist(dist_kimura(a, b, **kwargs), kwargs.get('d_max'))
dif_k80 = dif_kimura
dif_k2p = dif_kimura


@Metric
def sim_kimura(a, b, **kwargs) -> float:
    return _sim_from_dist(dist_kimura(a, b, **kwargs), kwargs.get('d_max'))
sim_k80 = sim_kimura
sim_k2p = sim_kimura


METRICS['kimura'] = {
    'class': 'sequence',
    'default': 'dist',
    'dist': dist_kimura,
    'dif': dif_kimura,
    'sim': sim_kimura,
    'info': info_kimura,
    'explain': explain_kimura,
}
METRICS['k80'] = METRICS['kimura']
METRICS['k2p'] = METRICS['kimura']
