from ..simdif import Metric, METRICS, to_list_numeric


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


def explain_bray_curtis(a, b, **_) -> str:
    v1, v2 = to_list_numeric(a), to_list_numeric(b)
    if len(v1) != len(v2):
        raise ValueError("Vector length mismatch")
    diffs = [abs(x - y) for x, y in zip(v1, v2)]
    sums = [abs(x + y) for x, y in zip(v1, v2)]
    numerator = sum(diffs)
    denominator = sum(sums)
    score = numerator / denominator if denominator != 0 else 0.0
    return f"""
A: {v1}
B: {v2}
Absolute Differences (|Ai - Bi|): {diffs} -> Sum: {numerator}
Absolute Totals (|Ai + Bi|): {sums} -> Sum: {denominator}
Bray-Curtis Dissimilarity: {numerator} / {denominator} = {score:.4f}
Similarity (1 - d): {1.0 - score:.4f}
    """.strip()


@Metric
def dist_bray_curtis(a, b, **_) -> float:
    v1, v2 = to_list_numeric(a), to_list_numeric(b)
    if len(v1) != len(v2):
        raise ValueError("Vector length mismatch")
    diff_sum = sum(abs(x - y) for x, y in zip(v1, v2))
    total_sum = sum(abs(x + y) for x, y in zip(v1, v2))
    return diff_sum / total_sum if total_sum != 0 else 0.0


@Metric
def sim_bray_curtis(a, b, **_) -> float:
    return 1.0 - dist_bray_curtis(a, b)


METRICS['bray_curtis'] = {
    'class': 'vector',
    'default': 'dist',
    'dist': dist_bray_curtis,
    'sim': sim_bray_curtis,
    'info': info_bray_curtis,
    'explain': explain_bray_curtis,
}
