import math
from ..simdif import Metric, METRICS, to_distribution


def info_hellinger() -> str:
    return """
Hellinger Distance
------------------
A true metric between two probability distributions P (A) and Q (B). It is the
Euclidean distance between the elementwise square roots of the distributions,
scaled by 1/sqrt(2).

Formula:
    H(P, Q) = (1 / sqrt(2)) * sqrt( sum( (sqrt(p_i) - sqrt(q_i))^2 ) )

Range: [0, 1]
    0 = identical distributions
    1 = disjoint support (no overlap)

Note: Because each sqrt-distribution is a unit vector (sum(p_i) = 1), the raw
Euclidean distance maxes out at sqrt(2); the 1/sqrt(2) factor maps the result
onto [0, 1]. Unlike KL and JS divergence, Hellinger is symmetric and obeys the
triangle inequality. Inputs are normalized into probability distributions
before comparison.
    """.strip()


def explain_hellinger(a, b, **kwargs) -> str:
    a = to_distribution(a)
    b = to_distribution(b)
    if len(a) != len(b):
        raise ValueError(f"Distributions must have the same length, got {len(a)} and {len(b)}")
    steps = []
    sum_sq = 0.0
    for i, (p, q) in enumerate(zip(a, b)):
        diff = math.sqrt(p) - math.sqrt(q)
        diff_sq = diff ** 2
        sum_sq += diff_sq
        steps.append(f"  idx {i}: (sqrt({p:.4f}) - sqrt({q:.4f}))^2 = {diff_sq:.4f}")
    dist = math.sqrt(sum_sq) / math.sqrt(2)
    return f"""
P (A): {a}
Q (B): {b}
Step-by-step Squared Differences of Roots:
{chr(10).join(steps)}
Sum of Squares: {sum_sq:.4f}
Hellinger Distance: sqrt({sum_sq:.4f}) / sqrt(2) = {dist:.4f}
Similarity (1 - d): {1.0 - dist:.4f}
    """.strip()


@Metric
def dist_hellinger(a, b, **kwargs) -> float:
    a = to_distribution(a)
    b = to_distribution(b)
    if len(a) != len(b):
        raise ValueError(f"Distributions must have the same length, got {len(a)} and {len(b)}")
    sum_sq = sum((math.sqrt(p) - math.sqrt(q)) ** 2 for p, q in zip(a, b))
    return math.sqrt(sum_sq) / math.sqrt(2)


@Metric
def sim_hellinger(a, b, **kwargs) -> float:
    return 1.0 - dist_hellinger(a, b, **kwargs)


METRICS['hellinger'] = {
    'class': 'vector',
    'default': 'dist',
    'dist': dist_hellinger,
    'sim': sim_hellinger,
    'info': info_hellinger,
    'explain': explain_hellinger,
}
