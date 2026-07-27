from ..simdif import Metric, METRICS, to_list, _dp_matrix, _fill_dp_matrix
import sys


def info_osa() -> str:
    return """
Optimal String Alignment (OSA)
------------------------------
A restricted version of Damerau-Levenshtein. It counts insertions, deletions,
substitutions, and transpositions, but with the constraint that no substring
can be edited more than once.
Note: Because of this restriction, it does NOT satisfy the triangle inequality.
Example: "CA" → "ABC" results in 3 (not 2).

Note: If the optional `rapidfuzz` package is installed, its `OSA` distance is
used on strings for speed; otherwise a dynamic-programming matrix is filled
locally. (rapidfuzz's `OSA` matches this implementation exactly; its
`DamerauLevenshtein` is the *unrestricted* true DL: see damerau_levenshtein.)
    """.strip()


def explain_osa(a, b, **kwargs) -> str:
    grid = matrix_osa(a, b)
    rows_display = ["  " + "  ".join(f"{str(cell):>3}" for cell in row) for row in grid]
    return f"""
A: ({", ".join(f"'{x}'" for x in to_list(a))})
B: ({", ".join(f"'{y}'" for y in to_list(b))})
OSA Matrix (rows = A, cols = B):
{chr(10).join(rows_display)}
OSA Distance: {dist_osa(a, b, **kwargs)}
    """.strip()


@Metric
def dist_osa(a, b, **kwargs) -> float:
    if isinstance(a, str) and isinstance(b, str) and 'rapidfuzz' in sys.modules:
        return float(sys.modules['rapidfuzz'].distance.OSA.distance(a, b))
    s1, s2 = to_list(a), to_list(b)
    return _dp_matrix(s1, s2, insert=1, delete=1, substitute=1, transpose=1, boundary=(1, 1), combine="min")[-1][-1]


def matrix_osa(a, b, **kwargs):
    return _fill_dp_matrix(a, b, insert=1, delete=1, substitute=1, transpose=1, boundary=(1, 1), combine="min")


METRICS['osa'] = {
    'class': 'sequence',
    'default': 'dist',
    'dist': dist_osa,
    'matrix': matrix_osa,
    'info': info_osa,
    'explain': explain_osa,
}
