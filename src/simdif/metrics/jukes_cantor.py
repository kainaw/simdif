import math
from ..simdif import Metric, METRICS
from ._helpers import _seq_diffs, _sim_from_dist, _dif_from_dist, _max_line


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

Roles:
    dist: the corrected distance above (>= 0, unbounded)
    sim:  1 / (1 + d), or 1 - dif when d_max is supplied
    dif:  1 - sim,     or d / d_max when d_max is supplied

Note: the whole point of the JC correction is that the number of substitutions
per site is unbounded even though the observed p-distance is capped at
(k-1)/k -- multiple hits at the same site hide arbitrarily much divergence. So
there is no maximum to divide by and sim defaults to the 1/(1+d) squash. A
saturated pair (dist = inf) gives dif=1.0 and sim=0.0 rather than nan.

If you want the bounded observed divergence instead of the corrected one, use
p_distance: it is a proportion, so its range is [0, 1] with no bounding
needed. d_max here is a substitutions-per-site cutoff you declare (e.g.
d_max=2.0 for "2 substitutions per site is as diverged as we distinguish").

WARNING -- d_max must be a real bound. Distances above it clamp to dif=1.0,
which merges genuinely-different divergence levels, and saturated pairs are
already at 1.0. explain_ reports the clamp.

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
{_max_line(d, kwargs.get('d_max'),
           unbounded_note="unbounded -- substitutions per site have no ceiling (p_distance is the bounded observed form)")}
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
def dif_jukes_cantor(a, b, **kwargs) -> float:
    return _dif_from_dist(dist_jukes_cantor(a, b, **kwargs), kwargs.get('d_max'))
dif_jc = dif_jukes_cantor
dif_jc69 = dif_jukes_cantor


@Metric
def sim_jukes_cantor(a, b, **kwargs) -> float:
    return _sim_from_dist(dist_jukes_cantor(a, b, **kwargs), kwargs.get('d_max'))
sim_jc = sim_jukes_cantor
sim_jc69 = sim_jukes_cantor


METRICS['jukes_cantor'] = {
    'class': 'sequence',
    'default': 'dist',
    'dist': dist_jukes_cantor,
    'dif': dif_jukes_cantor,
    'sim': sim_jukes_cantor,
    'info': info_jukes_cantor,
    'explain': explain_jukes_cantor,
}
METRICS['jc'] = METRICS['jukes_cantor']
METRICS['jc69'] = METRICS['jukes_cantor']
