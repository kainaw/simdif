from ..simdif import Metric, METRICS, to_list, _dp_matrix, _fill_dp_matrix
import sys


def info_levenshtein() -> str:
    return """
Levenshtein Distance (Edit Distance)
------------------------------------
Counts the minimum number of single-character insertions, deletions, and
substitutions needed to transform one sequence into another. Unlike OSA and
Damerau-Levenshtein, it does not treat transposition as a single operation
(a swap of two adjacent characters costs 2).

Formula:
    LEV(A,B) = minimum edit operations using:
        insertion, deletion, substitution

Range: [0, max(|A|, |B|)]
    0 = identical sequences

Note: If the optional `Levenshtein` package is installed, it is used to
compute distances on strings for speed; otherwise a dynamic-programming
matrix is filled locally.
    """.strip()


def explain_levenshtein(a, b, **kwargs) -> str:
    grid = matrix_levenshtein(a, b)
    rows_display = ["  " + "  ".join(f"{str(cell):>3}" for cell in row) for row in grid]
    return f"""
A: ({", ".join(f"'{x}'" for x in to_list(a))})
B: ({", ".join(f"'{y}'" for y in to_list(b))})
Edit-Distance Matrix (rows = A, cols = B):
{chr(10).join(rows_display)}
Levenshtein Distance: {dist_levenshtein(a, b, **kwargs)}
    """.strip()


@Metric
def dist_levenshtein(a, b, **kwargs) -> int:
    if isinstance(a, str) and isinstance(b, str) and 'Levenshtein' in sys.modules:
        return float(sys.modules['Levenshtein'].distance(a, b))
    s1, s2 = to_list(a), to_list(b)
    return _dp_matrix(s1, s2, insert=1, delete=1, substitute=1, transpose=None, boundary=(1, 1), combine="min")[-1][-1]


@Metric
def sim_levenshtein(a, b, **kwargs) -> float:
    s1, s2 = to_list(a), to_list(b)
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    return 1 - (dist_levenshtein(s1, s2, **kwargs) / max_len)


@Metric
def dif_levenshtein(a, b, **kwargs) -> float:
    return 1 - sim_levenshtein(a, b, **kwargs)


def matrix_levenshtein(a, b, **kwargs):
    return _fill_dp_matrix(a, b, insert=1, delete=1, substitute=1, transpose=None, boundary=(1, 1), combine="min")


METRICS['levenshtein'] = {
    'class': 'sequence',
    'default': 'dist',
    'dist': dist_levenshtein,
    'sim': sim_levenshtein,
    'dif': dif_levenshtein,
    'matrix': matrix_levenshtein,
    'info': info_levenshtein,
    'explain': explain_levenshtein,
}
