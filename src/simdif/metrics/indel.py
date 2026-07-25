from ..simdif import Metric, METRICS, to_list, _dp_matrix, _fill_dp_matrix
import sys


def info_indel() -> str:
    return """
Indel Distance (Insertions & Deletions)
---------------------------------------
Counts the minimum number of single-character insertions and deletions needed
to transform one sequence into another. Unlike Levenshtein, substitution is
NOT allowed — a changed character costs 2 (one delete + one insert).

It is directly related to the longest common subsequence (LCS):
    INDEL(A,B) = |A| + |B| - 2 * LCS(A, B)

Range: [0, |A| + |B|]
    0 = identical sequences

Note: If the optional `rapidfuzz` package is installed, its `Indel` distance
is used on strings for speed; otherwise a dynamic-programming matrix is filled
locally.
    """.strip()


def explain_indel(a, b, **kwargs) -> str:
    grid = matrix_indel(a, b)
    rows_display = ["  " + "  ".join(f"{str(cell):>3}" for cell in row) for row in grid]
    return f"""
A: ({", ".join(f"'{x}'" for x in to_list(a))})
B: ({", ".join(f"'{y}'" for y in to_list(b))})
Indel Matrix (rows = A, cols = B):
{chr(10).join(rows_display)}
Indel Distance: {dist_indel(a, b, **kwargs)}
    """.strip()


@Metric
def dist_indel(a, b, **kwargs) -> int:
    if isinstance(a, str) and isinstance(b, str) and 'rapidfuzz' in sys.modules:
        return float(sys.modules['rapidfuzz'].distance.Indel.distance(a, b))
    s1, s2 = to_list(a), to_list(b)
    return _dp_matrix(s1, s2, insert=1, delete=1, substitute=None, transpose=None, boundary=(1, 1), combine="min")[-1][-1]


def matrix_indel(a, b, **kwargs):
    return _fill_dp_matrix(a, b, insert=1, delete=1, substitute=None, transpose=None, boundary=(1, 1), combine="min")


METRICS['indel'] = {
    'class': 'sequence',
    'default': 'dist',
    'dist': dist_indel,
    'matrix': matrix_indel,
    'info': info_indel,
    'explain': explain_indel,
}
