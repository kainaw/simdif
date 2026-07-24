import math
from ..simdif import Metric, METRICS
from ._helpers import _seq_diffs


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
