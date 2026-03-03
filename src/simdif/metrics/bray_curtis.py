from ..simdif import Metric, METRICS, to_list_numeric_aligned


def info_bray_curtis() -> str:
    return """
Bray-Curtis Dissimilarity
-------------------------
A non-metric distance used primarily in ecology and biology to measure the 
dissimilarity between two samples (e.g., species counts in two sites). 

Unlike Euclidean distance, it is not affected by "double zeros" (where a 
species is absent from both sites) and is dominated by the most abundant 
elements.

Formula:
               sum(|Ai - Bi|)
    d(A, B) = ----------------
               sum(|Ai + Bi|)

Range: [0, 1]
    0 = identical
    1 = no shared elements (completely disjoint)

Note: This is technically a "dissimilarity" and not a "distance" because 
it does not necessarily obey the triangle inequality.
    """.strip()


def explain_bray_curtis(a, b, **kwargs) -> str:
    a, b = to_list_numeric_aligned(a, b, **kwargs)
    diffs = [abs(x - y) for x, y in zip(a, b)]
    sums = [abs(x + y) for x, y in zip(a, b)]
    numerator = sum(diffs)
    denominator = sum(sums)
    score = numerator / denominator if denominator != 0 else 0.0
    return f"""
A: {a}
B: {b}
Absolute Differences (|Ai - Bi|): {diffs} -> Sum: {numerator}
Absolute Totals (|Ai + Bi|): {sums} -> Sum: {denominator}
Bray-Curtis Dissimilarity: {numerator} / {denominator} = {score:.4f}
Similarity (1 - d): {1.0 - score:.4f}
    """.strip()


@Metric
def dist_bray_curtis(a, b, **kwargs) -> float:
    a, b = to_list_numeric_aligned(a, b, **kwargs)
    diff_sum = sum(abs(x - y) for x, y in zip(a, b))
    total_sum = sum(abs(x + y) for x, y in zip(a, b))
    return diff_sum / total_sum if total_sum != 0 else 0.0


@Metric
def sim_bray_curtis(a, b, **kwargs) -> float:
    return 1.0 - dist_bray_curtis(a, b, **kwargs)


METRICS['bray_curtis'] = {
    'class': 'vector',
    'default': 'dist',
    'dist': dist_bray_curtis,
    'sim': sim_bray_curtis,
    'info': info_bray_curtis,
    'explain': explain_bray_curtis,
}
