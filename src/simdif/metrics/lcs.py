from ..simdif import Metric, METRICS, to_list, _dp_matrix, _fill_dp_matrix
import sys


def info_lcs() -> str:
    return """
Longest Common Subsequence (LCS)
--------------------------------
The length of the longest sequence of characters that appears in both inputs
in the same relative order, but not necessarily contiguously. For example, the
LCS of "ABCBDAB" and "BDCABA" is "BCBA" (length 4).

Roles:
    score - length of the longest common subsequence (larger = more similar)
    dist  - |A| + |B| - 2 * LCS(A, B)  (the indel distance)

Range (score): [0, min(|A|, |B|)]

Note: If the optional `rapidfuzz` package is installed, its `LCSseq` similarity
is used on strings for speed; otherwise a dynamic-programming matrix is filled
locally.
    """.strip()


def explain_lcs(a, b, **kwargs) -> str:
    grid = matrix_lcs(a, b)
    rows_display = ["  " + "  ".join(f"{str(cell):>3}" for cell in row) for row in grid]
    return f"""
A: ({", ".join(f"'{x}'" for x in to_list(a))})
B: ({", ".join(f"'{y}'" for y in to_list(b))})
LCS Matrix (rows = A, cols = B):
{chr(10).join(rows_display)}
LCS Length (score): {score_lcs(a, b, **kwargs)}
Indel Distance (dist): {dist_lcs(a, b, **kwargs)}
    """.strip()


@Metric
def score_lcs(a, b, **kwargs) -> int:
    if isinstance(a, str) and isinstance(b, str) and 'rapidfuzz' in sys.modules:
        return int(sys.modules['rapidfuzz'].distance.LCSseq.similarity(a, b))
    s1, s2 = to_list(a), to_list(b)
    return _dp_matrix(s1, s2, insert=0, delete=0, substitute=None, match_score=1, local=False, maximize=True)[-1][-1]


@Metric
def dist_lcs(a, b, **kwargs) -> int:
    s1, s2 = to_list(a), to_list(b)
    return len(s1) + len(s2) - 2 * score_lcs(s1, s2, **kwargs)


def matrix_lcs(a, b, **kwargs):
    return _fill_dp_matrix(a, b, insert=0, delete=0, substitute=None, match_score=1, local=False, maximize=True)


METRICS['lcs'] = {
    'class': 'sequence',
    'default': 'score',
    'score': score_lcs,
    'dist': dist_lcs,
    'matrix': matrix_lcs,
    'info': info_lcs,
    'explain': explain_lcs,
}
