import math
from ..simdif import Metric, METRICS, to_list_numeric


def _avg_pairwise(x, y):
    """Mean absolute pairwise distance between every element of x and y."""
    return sum(abs(u - v) for u in x for v in y) / (len(x) * len(y))


def _energy_terms(va, vb):
    """(cross, within_a, within_b, d2, d) for two non-empty numeric samples."""
    if len(va) < 1 or len(vb) < 1:
        raise ValueError("Each sample needs at least 1 value")
    cross = _avg_pairwise(va, vb)
    within_a = _avg_pairwise(va, va)
    within_b = _avg_pairwise(vb, vb)
    d2 = 2 * cross - within_a - within_b
    return cross, within_a, within_b, d2, math.sqrt(max(0.0, d2))


def info_energy() -> str:
    return """
Energy Distance (Szekely & Rizzo)
---------------------------------
Distance between the DISTRIBUTIONS of two independent samples, built only
from pairwise Euclidean distances. Named by analogy to Newtonian potential
energy between two bodies.

Formula:
    E(A, B) = sqrt( 2*mean||a-b|| - mean||a-a'|| - mean||b-b'|| )
    i.e. twice the average cross-distance between the samples minus each
    sample's average internal distance. The quantity under the root (the
    energy statistic) is always >= 0; its square root is a true metric.

Roles:
    dist - E(A, B) (>= 0, unbounded)
    sim  - 1 / (1 + E)

Range (dist): [0, inf)
    0 = the two samples come from the same distribution

Note: A and B are two INDEPENDENT samples - unordered, duplicates significant,
unequal lengths OK, each with at least 1 value. Unlike Welch's t or Cohen's d
(which only see a shift in the mean), energy distance detects ANY distributional
difference - mean, variance, or shape. It is a close cousin of the Wasserstein
distance (in 1-D it integrates (F-G)^2 where Wasserstein integrates |F-G|). The
formula generalizes unchanged to vector-valued observations.

Aliases: Energy Distance, E-distance
    """.strip()
info_energy_distance = info_e_distance = info_energy


def explain_energy(a, b, **kwargs) -> str:
    va, vb = to_list_numeric(a), to_list_numeric(b)
    cross, within_a, within_b, d2, d = _energy_terms(va, vb)
    return f"""
A: {va}
B: {vb}
Average cross-distance   mean||a-b||:  {cross:.4f}
Average within-A distance mean||a-a'||: {within_a:.4f}
Average within-B distance mean||b-b'||: {within_b:.4f}
Energy statistic: 2*{cross:.4f} - {within_a:.4f} - {within_b:.4f} = {d2:.4f}
Energy distance: sqrt({max(0.0, d2):.4f}) = {d:.4f}
Similarity 1/(1+E): {1.0 / (1.0 + d):.4f}
    """.strip()
explain_energy_distance = explain_e_distance = explain_energy


@Metric
def dist_energy(a, b, **kwargs) -> float:
    va, vb = to_list_numeric(a), to_list_numeric(b)
    return _energy_terms(va, vb)[4]
dist_energy_distance = dist_e_distance = dist_energy


@Metric
def sim_energy(a, b, **kwargs) -> float:
    return 1.0 / (1.0 + dist_energy(a, b, **kwargs))
sim_energy_distance = sim_e_distance = sim_energy


METRICS['energy'] = {
    'class': 'vector',
    'default': 'dist',
    'dist': dist_energy,
    'sim': sim_energy,
    'info': info_energy,
    'explain': explain_energy,
}
METRICS['energy_distance'] = METRICS['e_distance'] = METRICS['energy']
