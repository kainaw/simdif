import math
from ..simdif import Metric, METRICS, to_list_aligned


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------

def _seq_diffs(a, b, **kwargs):
    """Two aligned sequences of any '=='-comparable elements -> (n_sites, n_diff).

    Generic: only equality is used, so it applies to DNA, protein, words, or
    arbitrary tokens. Raises 'Vector length mismatch' unless a pad_value is
    supplied (same contract as hamming).
    """
    a, b = to_list_aligned(a, b, **kwargs)
    return len(a), sum(x != y for x, y in zip(a, b))


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


# ------------------------------------------------------------------
# p-distance
# ------------------------------------------------------------------

def info_p_distance() -> str:
    return """
p-distance (Proportion of Differing Sites)
------------------------------------------
The fraction of aligned positions at which two equal-length sequences differ -
the uncorrected observed distance. Generic: it uses only '==', so it applies to
DNA, protein, or any comparable tokens.

Formula:
    p = (number of differing sites) / (number of sites)

Range: [0, 1]
    0 = identical, 1 = differ at every site

Note: p-distance does NOT correct for multiple substitutions at the same site
(see Jukes-Cantor and Kimura). Sequences must be the same length (assumed
already aligned).

Aliases: p-dist
    """.strip()
info_p_dist = info_p_distance


def explain_p_distance(a, b, **kwargs) -> str:
    n_sites, n_diff = _seq_diffs(a, b, **kwargs)
    p = n_diff / n_sites if n_sites else 0.0
    return f"""
Sites compared:  {n_sites}
Differing sites: {n_diff}
p = {n_diff} / {n_sites} = {p:.4f}
Proportion identical (sim): {1.0 - p:.4f}
    """.strip()
explain_p_dist = explain_p_distance


@Metric
def dist_p_distance(a, b, **kwargs) -> float:
    n_sites, n_diff = _seq_diffs(a, b, **kwargs)
    if n_sites == 0:
        return 0.0
    return n_diff / n_sites
dist_p_dist = dist_p_distance


@Metric
def sim_p_distance(a, b, **kwargs) -> float:
    return 1.0 - dist_p_distance(a, b, **kwargs)
sim_p_dist = sim_p_distance


METRICS['p_distance'] = {
    'class': 'sequence',
    'default': 'dist',
    'dist': dist_p_distance,
    'sim': sim_p_distance,
    'info': info_p_distance,
    'explain': explain_p_distance,
}
METRICS['p_dist'] = METRICS['p_distance']


# ------------------------------------------------------------------
# Jukes-Cantor (JC69)
# ------------------------------------------------------------------

def info_jukes_cantor() -> str:
    return """
Jukes-Cantor Distance (JC69)
----------------------------
An evolutionary distance that corrects the observed p-distance for multiple
substitutions at the same site, under a k-state model with equal substitution
rates. For DNA, k = 4 (the default).

Formula (k states):
    d = -((k-1)/k) * ln( 1 - (k/(k-1)) * p )
    where p is the p-distance. k = 4 gives the classic DNA JC69; k = 20 suits
    proteins; k = 2 a binary alphabet.

Range: [0, inf)
    Saturates: once p >= (k-1)/k the log argument is <= 0 and the distance is
    reported as inf (too diverged to estimate).

Note: k (the number of possible states) is a parameter, not derived from A and B
- two sequences need not contain every state. Only '==' is used to count
differences, so JC69 stays generic. Sequences must be the same length (assumed
already aligned).

Aliases: JC, JC69
    """.strip()
info_jc = info_jukes_cantor
info_jc69 = info_jukes_cantor


def explain_jukes_cantor(a, b, k=4, **kwargs) -> str:
    n_sites, n_diff = _seq_diffs(a, b, **kwargs)
    p = n_diff / n_sites if n_sites else 0.0
    ratio = k / (k - 1)
    x = 1.0 - ratio * p
    d = float('inf') if x <= 0 else -(1.0 / ratio) * math.log(x)
    d_str = "inf (saturated: p >= (k-1)/k)" if x <= 0 else f"{d:.4f}"
    return f"""
Sites compared: {n_sites}, differing: {n_diff}
p-distance: {p:.4f}
States k: {k}
d = -((k-1)/k) ln(1 - (k/(k-1)) p)
  = -{(k - 1) / k:.4f} * ln(1 - {ratio:.4f} * {p:.4f})
  = -{(k - 1) / k:.4f} * ln({x:.4f})
  = {d_str}
Similarity 1/(1+d): {1.0 / (1.0 + d):.4f}
    """.strip()
explain_jc = explain_jukes_cantor
explain_jc69 = explain_jukes_cantor


@Metric
def dist_jukes_cantor(a, b, k=4, **kwargs) -> float:
    if k <= 1:
        raise ValueError("Jukes-Cantor requires k > 1 (number of states)")
    n_sites, n_diff = _seq_diffs(a, b, **kwargs)
    if n_sites == 0:
        return 0.0
    p = n_diff / n_sites
    ratio = k / (k - 1)
    x = 1.0 - ratio * p
    if x <= 0:
        return float('inf')  # saturation: too diverged to estimate
    return -(1.0 / ratio) * math.log(x)
dist_jc = dist_jukes_cantor
dist_jc69 = dist_jukes_cantor


@Metric
def sim_jukes_cantor(a, b, **kwargs) -> float:
    return 1.0 / (1.0 + dist_jukes_cantor(a, b, **kwargs))
sim_jc = sim_jukes_cantor
sim_jc69 = sim_jukes_cantor


METRICS['jukes_cantor'] = {
    'class': 'sequence',
    'default': 'dist',
    'dist': dist_jukes_cantor,
    'sim': sim_jukes_cantor,
    'info': info_jukes_cantor,
    'explain': explain_jukes_cantor,
}
METRICS['jc'] = METRICS['jukes_cantor']
METRICS['jc69'] = METRICS['jukes_cantor']


# ------------------------------------------------------------------
# Kimura 2-parameter (K80)
# ------------------------------------------------------------------

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
Similarity 1/(1+d): {1.0 / (1.0 + d):.4f}
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
def sim_kimura(a, b, **kwargs) -> float:
    return 1.0 / (1.0 + dist_kimura(a, b, **kwargs))
sim_k80 = sim_kimura
sim_k2p = sim_kimura


METRICS['kimura'] = {
    'class': 'sequence',
    'default': 'dist',
    'dist': dist_kimura,
    'sim': sim_kimura,
    'info': info_kimura,
    'explain': explain_kimura,
}
METRICS['k80'] = METRICS['kimura']
METRICS['k2p'] = METRICS['kimura']
