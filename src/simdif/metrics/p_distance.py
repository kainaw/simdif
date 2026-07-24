from ..simdif import Metric, METRICS
from ._helpers import _seq_diffs


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
