import math
from ..simdif import Metric, METRICS, to_distribution


def info_bhattacharyya() -> str:
    return """
Bhattacharyya Distance / Coefficient
------------------------------------
Measures the overlap between two probability distributions P (A) and Q (B).

The Bhattacharyya coefficient is the amount of overlap (a similarity):
    BC(P, Q) = sum( sqrt(p_i * q_i) )           range [0, 1]
        1 = identical distributions
        0 = disjoint support (no overlap)

The Bhattacharyya distance is derived from it:
    D_B(P, Q) = -ln( BC(P, Q) )                 range [0, inf)
        0   = identical distributions
        inf = disjoint support

Roles:
    sim  - Bhattacharyya coefficient BC (bounded [0, 1])
    dist - Bhattacharyya distance -ln(BC)

Note: The coefficient relates to the Hellinger distance by H^2 = 1 - BC. Unlike
Hellinger, the Bhattacharyya distance is NOT a true metric (it does not obey the
triangle inequality). Inputs are normalized into probability distributions
before comparison.
    """.strip()


def explain_bhattacharyya(a, b, **kwargs) -> str:
    a = to_distribution(a)
    b = to_distribution(b)
    if len(a) != len(b):
        raise ValueError(f"Distributions must have the same length, got {len(a)} and {len(b)}")
    steps = []
    bc = 0.0
    for i, (p, q) in enumerate(zip(a, b)):
        term = math.sqrt(p * q)
        bc += term
        steps.append(f"  idx {i}: sqrt({p:.4f} * {q:.4f}) = {term:.4f}")
    dist = -math.log(bc) if bc > 0 else float('inf')
    return f"""
P (A): {a}
Q (B): {b}
Step-by-step Overlap Terms:
{chr(10).join(steps)}
Bhattacharyya Coefficient (sim): {bc:.4f}
Bhattacharyya Distance -ln(BC): {dist:.4f}
    """.strip()


@Metric
def sim_bhattacharyya(a, b, **kwargs) -> float:
    a = to_distribution(a)
    b = to_distribution(b)
    if len(a) != len(b):
        raise ValueError(f"Distributions must have the same length, got {len(a)} and {len(b)}")
    return sum(math.sqrt(p * q) for p, q in zip(a, b))


@Metric
def dist_bhattacharyya(a, b, **kwargs) -> float:
    bc = sim_bhattacharyya(a, b, **kwargs)
    if bc >= 1.0:            # identical (guard float noise that could push BC slightly over 1)
        return 0.0
    if bc <= 0.0:            # disjoint support
        return float('inf')
    return -math.log(bc)


METRICS['bhattacharyya'] = {
    'class': 'vector',
    'default': 'dist',
    'sim': sim_bhattacharyya,
    'dist': dist_bhattacharyya,
    'info': info_bhattacharyya,
    'explain': explain_bhattacharyya,
}
