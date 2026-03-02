from ..simdif import METRICS, to_list_numeric
import sys

def info_kendall_tau_b() -> str:
    return """
Kendall's Tau-b
---------------
Measures ordinal association between two sequences by comparing all possible
pairs as concordant or discordant. Unlike Tau-a, Tau-b adjusts the denominator
to account for tied values in either sequence, allowing it to reach ±1 even
when ties are present. Preferred over Tau-a whenever the data may contain
repeated values.
Formula:
    τ_b(A,B) = (C - D) / sqrt((C + D + T_a) * (C + D + T_b))
Where:
    C   = number of concordant pairs
    D   = number of discordant pairs
    T_a = pairs tied in A only
    T_b = pairs tied in B only
Range: [-1, 1]
    1  = perfect agreement in ordering
    0  = no association
   -1  = perfect disagreement in ordering
Note: A pair can be tied in both A and B simultaneously — T_a and T_b are
counted independently. If scipy is available, scipy.stats.kendalltau is used.
    """.strip()
info_tau_b = info_kendall_tau_b

def explain_kendall_tau_b(a, b, **_) -> str:
    a, b = to_list_numeric(a), to_list_numeric(b)
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
    if len(a) < 2:
        raise ValueError(f"Kendall's Tau-b requires at least 2 elements, got {len(a)}")
    n = len(a)
    concordant = 0
    discordant = 0
    ties_a = 0
    ties_b = 0
    for i in range(n):
        for j in range(i + 1, n):
            a_dir = a[i] - a[j]
            b_dir = b[i] - b[j]
            if a_dir * b_dir > 0:
                concordant += 1
            elif a_dir * b_dir < 0:
                discordant += 1
            if a_dir == 0:
                ties_a += 1
            if b_dir == 0:
                ties_b += 1
    denom_a = concordant + discordant + ties_a
    denom_b = concordant + discordant + ties_b
    denom = (denom_a * denom_b) ** 0.5
    sim = sim_kendall_tau_b(a, b)
    return f"""
A: ({", ".join(map(str, a))})
B: ({", ".join(map(str, b))})
Kendall's Tau-b:
Total pairs n*(n-1)/2:      {n * (n - 1) // 2}
Concordant pairs (C):       {concordant}
Discordant pairs (D):       {discordant}
Tied in A (T_a):            {ties_a}
Tied in B (T_b):            {ties_b}
sqrt((C+D+T_a)*(C+D+T_b)): {denom:.4f}
Calculation:
  (C - D) / sqrt((C+D+T_a) * (C+D+T_b))
= ({concordant} - {discordant}) / sqrt({denom_a} * {denom_b})
= {concordant - discordant} / {denom:.4f}
= {sim:.4f}
Distance: 1 - τ_b = {dist_kendall_tau_b(a, b):.4f}
    """.strip()

explain_tau_b = explain_kendall_tau_b

def sim_kendall_tau_b(a, b, **_) -> float:
    a, b = to_list_numeric(a), to_list_numeric(b)
    if len(a) != len(b):
        raise ValueError(f"Sequences must be the same length, got {len(a)} and {len(b)}")
    if len(a) < 2:
        raise ValueError(f"Kendall's Tau-b requires at least 2 elements, got {len(a)}")
    if 'scipy' in sys.modules:
        return float(sys.modules['scipy'].stats.kendalltau(a, b).statistic)
    n = len(a)
    concordant = 0
    discordant = 0
    ties_a = 0
    ties_b = 0
    for i in range(n):
        for j in range(i + 1, n):
            a_dir = a[i] - a[j]
            b_dir = b[i] - b[j]
            if a_dir * b_dir > 0:
                concordant += 1
            elif a_dir * b_dir < 0:
                discordant += 1
            if a_dir == 0:
                ties_a += 1
            if b_dir == 0:
                ties_b += 1
    denom = ((concordant + discordant + ties_a) * (concordant + discordant + ties_b)) ** 0.5
    if denom == 0:
        return 1.0
    return (concordant - discordant) / denom
sim_tau_b = sim_kendall_tau_b

def dist_kendall_tau_b(a, b, **_) -> float:
    return 1 - sim_kendall_tau_b(a, b)
dist_tau_b = dist_kendall_tau_b

METRICS['kendall_tau_b'] = {
    'class': 'vector',
    'default': 'sim',
    'sim': sim_kendall_tau_b,
    'dist': dist_kendall_tau_b,
    'info': info_kendall_tau_b,
    'explain': explain_kendall_tau_b,
}
METRICS['tau_b'] = METRICS['kendall_tau_b']
