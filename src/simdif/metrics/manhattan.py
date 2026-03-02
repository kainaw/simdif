from .minkowski import dist_minkowski
from ..simdif import Metric, METRICS


def info_manhattan() -> str:
    return """
Manhattan Distance (L1 Norm / Taxicab)
--------------------------------------
The distance between two points measured along axes at right angles. 
Imagine a taxi driving through a grid-based city.

Formula:
    D(A, B) = sum(|Ai - Bi|)
    (Minkowski Distance where p=1)
    """.strip()
info_taxicab = info_manhattan
info_cityblock = info_manhattan


def explain_manhattan(a, b, **_) -> str:
    v1, v2 = to_list_numeric(a), to_list_numeric(b)
    if len(v1) != len(v2):
        return "Error: Vector length mismatch"
    steps = []
    total_dist = 0.0
    for i, (x, y) in enumerate(zip(v1, v2)):
        diff = abs(x - y)
        total_dist += diff
        steps.append(f"  idx {i}: |{x} - {y}| = {diff:.4f}")
    return f"""
A: {v1}
B: {v2}
Step-by-step Absolute Differences:
{chr(10).join(steps)}
Total Manhattan Distance (Sum): {total_dist:.4f}
Similarity (1 / (1 + d)): {1.0 / (1.0 + total_dist):.4f}
    """.strip()
explain_taxicab = explain_manhattan
explain_cityblock = explain_manhattan


@Metric
def dist_manhattan(a, b, **_) -> float:
    return dist_minkowski(a, b, p=1)
dist_taxicab = dist_manhattan
dist_cityblock = dist_manhattan


@Metric
def sim_manhattan(a, b, **_) -> float:
    return 1.0 / (1.0 + dist_manhattan(a, b))
sim_taxicab = sim_manhattan
sim_cityblock = sim_manhattan


METRICS['manhattan'] = {
    'class': 'vector',
    'default': 'dist',
    'dist': dist_manhattan,
    'sim': sim_manhattan,
    'info': info_manhattan,
	'explain': explain_manhattan,
}
METRICS['taxicab'] = METRICS['manhattan']
METRICS['cityblock'] = METRICS['manhattan']
