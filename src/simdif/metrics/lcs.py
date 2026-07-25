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
    dist  - |A| + |B| - 2 * LCS(A, B)  (the indel distance: the number of
            single-character insertions/deletions needed to turn A into B,
            using no substitutions)
    sim   - 1 - dist / (|A| + |B|), i.e. 2*LCS(A,B) / (|A|+|B|). Ranges
            [0, 1]; 1.0 iff A and B are identical, 0.0 iff they share no
            common subsequence at all. Note this is tied to the `dist` role
            above (indel distance), not to score's own [0, min(|A|,|B|)]
            range -- a short string fully contained in a much longer one
            still scores well below 1.0 here, same as sim_levenshtein.
    dif   - 1 - sim
    trace - the actual subsequence itself (a str for string inputs, else a
            list). When several subsequences tie for longest, one canonical
            path is returned.

Range (score): [0, min(|A|, |B|)]
Range (sim/dif): [0, 1]

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
Longest Common Subsequence (trace): {trace_lcs(a, b, **kwargs)!r}
Indel Distance (dist): {dist_lcs(a, b, **kwargs)}
Similarity (sim): {sim_lcs(a, b, **kwargs):.4f}
Difference (dif): {dif_lcs(a, b, **kwargs):.4f}
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


@Metric
def sim_lcs(a, b, **kwargs) -> float:
    s1, s2 = to_list(a), to_list(b)
    total = len(s1) + len(s2)
    if total == 0:
        return 1.0
    return 1 - (dist_lcs(s1, s2, **kwargs) / total)


@Metric
def dif_lcs(a, b, **kwargs) -> float:
    return 1 - sim_lcs(a, b, **kwargs)


def matrix_lcs(a, b, **kwargs):
    return _fill_dp_matrix(a, b, insert=0, delete=0, substitute=None, match_score=1, local=False, maximize=True)


def trace_lcs(a, b, **kwargs):
    """The actual longest common subsequence, recovered by backtracking the DP
    matrix. Returns a str when both inputs are strings, else a list of elements.
    When several subsequences tie for longest, one canonical path is returned
    (preferring a match, then the higher-scoring neighbour)."""
    s1, s2 = to_list(a), to_list(b)
    matrix = _dp_matrix(s1, s2, insert=0, delete=0, substitute=None, match_score=1, local=False, maximize=True)
    i, j = len(s1), len(s2)
    subseq = []
    while i > 0 and j > 0:
        if s1[i-1] == s2[j-1]:
            subseq.append(s1[i-1])
            i -= 1
            j -= 1
        elif matrix[i-1][j] >= matrix[i][j-1]:
            i -= 1
        else:
            j -= 1
    subseq.reverse()
    if isinstance(a, str) and isinstance(b, str):
        return "".join(str(x) for x in subseq)
    return subseq


METRICS['lcs'] = {
    'class': 'sequence',
    'default': 'score',
    'score': score_lcs,
    'dist': dist_lcs,
    'sim': sim_lcs,
    'dif': dif_lcs,
    'matrix': matrix_lcs,
    'trace': trace_lcs,
    'info': info_lcs,
    'explain': explain_lcs,
}
