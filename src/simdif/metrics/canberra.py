from ..simdif import Metric, METRICS, to_list_numeric_aligned


def info_canberra() -> str:
    return """
Canberra Distance
-----------------
A weighted version of the Manhattan distance that is sensitive to small values
near the origin. It computes the sum of absolute differences between elements,
divided by the sum of their absolute magnitudes.

Formula:
               n    |Ai - Bi|
    d(A, B) = sum  -----------
              i=1  |Ai| + |Bi|

Range: [0, n]
    Where n is the length of the sequences.
    0 = identical
    n = maximum theoretical distance (if all pairs have opposite signs/zeros)

Note: If both Ai and Bi are zero, the term is skipped (treated as 0).
    """.strip()


def explain_canberra(a, b, **kwargs) -> str:
    a, b = to_list_numeric_aligned(a, b, **kwargs)
    terms = []
    total = 0.0
    for i, (x, y) in enumerate(zip(a, b)):
        num = abs(x - y)
        den = abs(x) + abs(y)
        val = num / den if den > 0 else 0.0
        total += val
        terms.append(f"  idx {i}: |{x} - {y}| / (|{x}| + |{y}|) = {num} / {den} = {val:.4f}")
    terms_display = "\n".join(terms)
    return f"""
A: {a}
B: {b}
Canberra Contributions:
{terms_display}
Total Canberra Distance: {total:.4f}
Similarity (1 / (1+d)): {1 / (1 + total):.4f}
    """.strip()


@Metric
def dist_canberra(a, b, **kwargs) -> float:
    a, b = to_list_numeric_aligned(a, b, **kwargs)
    score = 0.0
    for x, y in zip(a, b):
        denominator = abs(x) + abs(y)
        if denominator > 0:
            score += abs(x - y) / denominator
    return score


@Metric
def sim_canberra(a, b, **kwargs) -> float:
    d = dist_canberra(a, b, **kwargs)
    return 1.0 / (1.0 + d)


METRICS['canberra'] = {
    'class': 'vector',
    'default': 'dist',
    'dist': dist_canberra,
    'sim': sim_canberra,
    'info': info_canberra,
    'explain': explain_canberra,
}
