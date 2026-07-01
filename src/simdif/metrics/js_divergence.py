import math
from ..simdif import Metric, METRICS, to_distribution


def info_js_divergence() -> str:
    return """
Jensen-Shannon Divergence
-------------------------
A symmetric, smoothed measure of the similarity between two probability
distributions P (A) and Q (B). It is based on the Kullback-Leibler divergence
but, unlike KL, it is symmetric and always finite.

It compares each distribution against the pointwise mean M = (P + Q) / 2:

Formula:
    M      = (P + Q) / 2
    JSD    = (1/2) * D_KL(P || M) + (1/2) * D_KL(Q || M)

Range: [0, ln(2)]  (using natural log)
    0 = the distributions are identical

Note: JSD is symmetric (JSD(P, Q) == JSD(Q, P)) and its square root is a true
metric (the Jensen-Shannon distance). Inputs are normalized into probability
distributions before comparison.

Aliases: Jensen-Shannon
    """.strip()
info_jensen_shannon = info_js_divergence


def explain_js_divergence(a, b, **kwargs) -> str:
    a = to_distribution(a)
    b = to_distribution(b)
    if len(a) != len(b):
        raise ValueError(f"Distributions must have the same length, got {len(a)} and {len(b)}")
    m = [(x + y) / 2 for x, y in zip(a, b)]
    kl_a = sum(p * math.log(p / q) for p, q in zip(a, m) if p > 0)
    kl_b = sum(p * math.log(p / q) for p, q in zip(b, m) if p > 0)
    total = (kl_a + kl_b) / 2
    return f"""
P (A): {a}
Q (B): {b}
Mean M = (P + Q) / 2: {m}
D_KL(P || M): {kl_a:.4f}
D_KL(Q || M): {kl_b:.4f}
JS Divergence ((KL_P + KL_Q) / 2): {total:.4f}
Similarity (1 / (1 + d)): {1.0 / (1.0 + total):.4f}
    """.strip()
explain_jensen_shannon = explain_js_divergence


@Metric
def dist_js_divergence(a, b, **kwargs) -> float:
    a = to_distribution(a)
    b = to_distribution(b)
    if len(a) != len(b):
        raise ValueError(f"Distributions must have the same length, got {len(a)} and {len(b)}")
    m = [(x + y) / 2 for x, y in zip(a, b)]
    kl_a = sum(p * math.log(p / q) for p, q in zip(a, m) if p > 0)
    kl_b = sum(p * math.log(p / q) for p, q in zip(b, m) if p > 0)
    return (kl_a + kl_b) / 2
dist_jensen_shannon = dist_js_divergence


@Metric
def sim_js_divergence(a, b, **kwargs) -> float:
    return 1.0 / (1.0 + dist_js_divergence(a, b, **kwargs))
sim_jensen_shannon = sim_js_divergence


METRICS['js_divergence'] = {
    'class': 'vector',
    'default': 'dist',
    'dist': dist_js_divergence,
    'sim': sim_js_divergence,
    'info': info_js_divergence,
    'explain': explain_js_divergence,
}
METRICS['jensen_shannon'] = METRICS['js_divergence']
