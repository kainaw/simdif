import math
from ..simdif import Metric, METRICS, to_distribution


def info_kl_divergence() -> str:
    return """
Kullback-Leibler Divergence (Relative Entropy)
----------------------------------------------
A measure of how one probability distribution P (A) diverges from a second,
reference distribution Q (B). It represents the extra information (in nats)
needed to encode samples from P using a code optimized for Q.

Formula:
    D_KL(P || Q) = sum( P_i * log(P_i / Q_i) )

Range: [0, inf)
    0 = the distributions are identical

Note: KL divergence is NOT symmetric (D_KL(P||Q) != D_KL(Q||P)) and does not
obey the triangle inequality, so it is a divergence rather than a true metric.
Inputs are normalized into probability distributions before comparison.

Aliases: Kullback-Leibler
    """.strip()
info_kullback_leibler = info_kl_divergence


def explain_kl_divergence(a, b, **kwargs) -> str:
    a = to_distribution(a)
    b = to_distribution(b)
    if len(a) != len(b):
        raise ValueError(f"Distributions must have the same length, got {len(a)} and {len(b)}")
    steps = []
    total = 0.0
    for i, (p, q) in enumerate(zip(a, b)):
        if p > 0:
            if q <= 0:
                total = float('inf')
                steps.append(f"  idx {i}: P={p:.4f}, Q=0 -> divergence is infinite")
            else:
                term = p * math.log(p / q)
                total += term
                steps.append(f"  idx {i}: {p:.4f} * log({p:.4f} / {q:.4f}) = {term:.4f}")
        else:
            steps.append(f"  idx {i}: P={p:.4f} -> skipped (contributes 0)")
    return f"""
P (A): {a}
Q (B): {b}
Step-by-step Contributions:
{chr(10).join(steps)}
KL Divergence (Sum): {total:.4f}
Similarity (1 / (1 + d)): {1.0 / (1.0 + total):.4f}
    """.strip()
explain_kullback_leibler = explain_kl_divergence


@Metric
def dist_kl_divergence(a, b, **kwargs) -> float:
    a = to_distribution(a)
    b = to_distribution(b)
    if len(a) != len(b):
        raise ValueError(f"Distributions must have the same length, got {len(a)} and {len(b)}")
    total = 0.0
    for p, q in zip(a, b):
        if p > 0:
            if q <= 0:
                # P has support where Q does not: divergence is infinite.
                return float('inf')
            total += p * math.log(p / q)
    return total
dist_kullback_leibler = dist_kl_divergence


@Metric
def sim_kl_divergence(a, b, **kwargs) -> float:
    return 1.0 / (1.0 + dist_kl_divergence(a, b, **kwargs))
sim_kullback_leibler = sim_kl_divergence


METRICS['kl_divergence'] = {
    'class': 'vector',
    'default': 'dist',
    'dist': dist_kl_divergence,
    'sim': sim_kl_divergence,
    'info': info_kl_divergence,
    'explain': explain_kl_divergence,
}
METRICS['kullback_leibler'] = METRICS['kl_divergence']
