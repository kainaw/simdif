from ..simdif import METRICS, to_list_numeric
import sys

def info_pearson() -> str:
    return """
Pearson Correlation Coefficient
--------------------------------
Measures the linear correlation between two numeric vectors. Equivalent to
cosine similarity applied to mean-centered vectors — the mean of each vector
is subtracted before computing the angle between them. This makes it
insensitive to shifts in scale or offset, unlike cosine similarity.
Formula:
    r(A,B) = Σ((a_i - ā)(b_i - b̄)) / (√Σ(a_i - ā)² * √Σ(b_i - b̄)²)
Where ā and b̄ are the means of A and B respectively.
Range: [-1, 1]
    1  = perfect positive linear correlation
    0  = no linear correlation
   -1  = perfect negative linear correlation
Distance: [0, 2] — 1 - similarity (standard correlation distance)
Note: Requires at least 2 elements. Returns 0.0 if either vector has zero
variance (all elements identical). Pearson is to cosine what mean-centering
is to raw vectors — use cosine if you want magnitude-sensitive comparison,
Pearson if you want to compare shape/trend regardless of offset.
    """.strip()

def explain_pearson(a, b, **_) -> str:
    a, b = to_list_numeric(a), to_list_numeric(b)
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
    if len(a) < 2:
        raise ValueError(f"Pearson requires at least 2 elements, got {len(a)}")
    n = len(a)
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    centered_a = [x - mean_a for x in a]
    centered_b = [y - mean_b for y in b]
    numerator = sum(x * y for x, y in zip(centered_a, centered_b))
    denom_a = sum(x ** 2 for x in centered_a) ** 0.5
    denom_b = sum(y ** 2 for y in centered_b) ** 0.5
    sim = sim_pearson(a, b)
    return f"""
A: ({", ".join(map(str, a))})
B: ({", ".join(map(str, b))})
Pearson Correlation:
Mean(A): {mean_a:.4f}
Mean(B): {mean_b:.4f}
Centered A: ({", ".join(f"{x:.4f}" for x in centered_a)})
Centered B: ({", ".join(f"{y:.4f}" for y in centered_b)})
Numerator Σ((a_i - ā)(b_i - b̄)): {numerator:.4f}
||A - ā||: {denom_a:.4f}
||B - b̄||: {denom_b:.4f}
||A - ā|| * ||B - b̄||: {denom_a * denom_b:.4f}
Calculation:
  {numerator:.4f} / {denom_a * denom_b:.4f}
= {sim:.4f}
Distance: 1 - r = {dist_pearson(a, b):.4f}
    """.strip()

def sim_pearson(a, b, **_) -> float:
    a, b = to_list_numeric(a), to_list_numeric(b)
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
    if len(a) < 2:
        raise ValueError(f"Pearson requires at least 2 elements, got {len(a)}")
    if 'scipy' in sys.modules:
        from scipy.stats import pearsonr
        result = pearsonr(a, b)
        return float(result[0])
    n = len(a)
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    numerator = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    denom_a = sum((x - mean_a) ** 2 for x in a) ** 0.5
    denom_b = sum((y - mean_b) ** 2 for y in b) ** 0.5
    if denom_a == 0 or denom_b == 0:
        return 0.0
    return max(-1.0, min(1.0, numerator / (denom_a * denom_b)))

def dist_pearson(a, b, **_) -> float:
    return 1 - sim_pearson(a, b)

METRICS['pearson'] = {
    'class': 'vector',
    'default': 'sim',
    'sim': sim_pearson,
    'dist': dist_pearson,
    'info': info_pearson,
    'explain': explain_pearson,
}
