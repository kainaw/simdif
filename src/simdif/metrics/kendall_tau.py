from ..simdif import Metric, METRICS, to_list_numeric
import sys


def info_kendall_tau() -> str:
    return """
Kendall's Tau-a
---------------
Measures the ordinal association between two sequences by comparing all
possible pairs and counting how many are concordant (same relative order
in both sequences) versus discordant (opposite order). Ties are ignored
entirely — if either sequence has tied values, Tau-b is more appropriate
as it adjusts the denominator to account for them.
Formula:
    τ_a(A,B) = (C - D) / (n*(n-1)/2)
Where:
    C = number of concordant pairs
    D = number of discordant pairs
    n*(n-1)/2 = total number of pairs
Range: [-1, 1]
    1  = perfect agreement in ordering
    0  = no association
   -1  = perfect disagreement in ordering
Note: Tau-a cannot reach ±1 when ties are present. Use Tau-b for data
with tied values.
    """.strip()
info_kendall_tau_a = info_kendall_tau
info_tau_a = info_kendall_tau


def explain_kendall_tau(a, b, **_) -> str:
    a, b = to_list_numeric(a), to_list_numeric(b)
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
    if len(a) < 2:
        raise ValueError(f"Kendall's Tau requires at least 2 elements, got {len(a)}")
    n = len(a)
    concordant = 0
    discordant = 0
    ties = 0
    for i in range(n):
        for j in range(i + 1, n):
            a_dir = a[i] - a[j]
            b_dir = b[i] - b[j]
            if a_dir * b_dir > 0:
                concordant += 1
            elif a_dir * b_dir < 0:
                discordant += 1
            else:
                ties += 1
    total = n * (n - 1) // 2
    sim = sim_kendall_tau(a, b)
    return f"""
A: ({", ".join(map(str, a))})
B: ({", ".join(map(str, b))})
Kendall's Tau-a:
Total pairs n*(n-1)/2: {total}
Concordant pairs (C):  {concordant}
Discordant pairs (D):  {discordant}
Tied pairs (ignored):  {ties}
Calculation:
  (C - D) / (n*(n-1)/2)
= ({concordant} - {discordant}) / {total}
= {concordant - discordant} / {total}
= {sim:.4f}
Distance: 1 - τ_a = {dist_kendall_tau(a, b):.4f}
    """.strip()
explain_kendall_tau_a = explain_kendall_tau
explain_tau_a = explain_kendall_tau


@Metric
def sim_kendall_tau(a, b, **_) -> float:
    a, b = to_list_numeric(a), to_list_numeric(b)
    if len(a) != len(b):
        raise ValueError(f"Sequences must be the same length, got {len(a)} and {len(b)}")
    if len(a) < 2:
        raise ValueError(f"Kendall's Tau requires at least 2 elements, got {len(a)}")
    n = len(a)
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            a_dir = a[i] - a[j]
            b_dir = b[i] - b[j]
            if a_dir * b_dir > 0:
                concordant += 1
            elif a_dir * b_dir < 0:
                discordant += 1
            # if either is 0, it's a tie — we ignore it (Tau-b handles ties differently)
    total = n * (n - 1) // 2
    return (concordant - discordant) / total
sim_kendall_tau_a = sim_kendall_tau
sim_tau_a = sim_kendall_tau


@Metric
def dist_kendall_tau(a, b, **_) -> float:
    return 1 - sim_kendall_tau(a, b)
dist_kendall_tau_a = dist_kendall_tau
dist_tau_a = dist_kendall_tau


METRICS['kendall_tau'] = {
    'class': 'sequence',
    'default': 'sim',
    'sim': sim_kendall_tau,
    'dist': dist_kendall_tau,
    'info': info_kendall_tau,
    'explain': explain_kendall_tau,
}
METRICS['kendall_tau_a'] = METRICS['kendall_tau']
METRICS['tau_a'] = METRICS['kendall_tau']
