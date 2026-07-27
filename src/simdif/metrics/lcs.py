from ..simdif import Metric, METRICS, to_list, _dp_matrix, _fill_dp_matrix
import sys


def info_lcs() -> str:
    return """
Longest Common Subsequence (LC subsequence)
-------------------------------------------
The length of the longest sequence of characters that appears in both inputs
in the same relative order, but NOT necessarily contiguously -- gaps are
allowed. For example, the longest common subsequence of "ABCBDAB" and
"BDCABA" is "BCBA" (length 4).

Not to be confused with the Longest Common SUBSTRING ('lc_substring'), which
requires the run to be contiguous. The literature abbreviates both as "LCS",
so check which recurrence a paper prints before comparing numbers. The
difference is not cosmetic: for "ABCDE" vs "AXBXCXDXE" the subsequence score
is 5 and the substring score is 1.

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
            ('lc_substring' normalizes by max(|A|,|B|) instead, since it has
            no indel distance to stay consistent with.)
    dif   - 1 - sim
    trace - the actual subsequence itself (a str for string inputs, else a
            list). When several subsequences tie for longest, one canonical
            path is returned.

Range (score): [0, min(|A|, |B|)]
Range (sim/dif): [0, 1]

Note: In the DP grid, a cell holds the best score over all prefixes up to that
pair of positions, so the grid is non-decreasing and the answer is the
bottom-right corner. ('lc_substring' instead stores the run ending exactly at
each cell and takes the largest cell anywhere.)

Note: If the optional `rapidfuzz` package is installed, its `LCSseq` similarity
is used on strings for speed; otherwise a dynamic-programming matrix is filled
locally.

Aliases: lc_subsequence, lcsubseq, longest_common_subsequence
    """.strip()
info_lc_subsequence = info_lcs
info_lcsubseq = info_lcs
info_longest_common_subsequence = info_lcs


def explain_lcs(a, b, **kwargs) -> str:
    grid = matrix_lcs(a, b)
    rows_display = ["  " + "  ".join(f"{str(cell):>3}" for cell in row) for row in grid]
    return f"""
A: ({", ".join(f"'{x}'" for x in to_list(a))})
B: ({", ".join(f"'{y}'" for y in to_list(b))})
Longest Common Subsequence -- skips allowed, matches need not be contiguous.
(For the contiguous variant, see 'lc_substring'.)
LC Subsequence Matrix (rows = A, cols = B):
{chr(10).join(rows_display)}
Subsequence Length (score): {score_lcs(a, b, **kwargs)}
Longest Common Subsequence (trace): {trace_lcs(a, b, **kwargs)!r}
Indel Distance (dist): {dist_lcs(a, b, **kwargs)}
Similarity (sim): {sim_lcs(a, b, **kwargs):.4f}
Difference (dif): {dif_lcs(a, b, **kwargs):.4f}
    """.strip()
explain_lc_subsequence = explain_lcs
explain_lcsubseq = explain_lcs
explain_longest_common_subsequence = explain_lcs


@Metric
def score_lcs(a, b, **kwargs) -> int:
    if isinstance(a, str) and isinstance(b, str) and 'rapidfuzz' in sys.modules:
        return int(sys.modules['rapidfuzz'].distance.LCSseq.similarity(a, b))
    s1, s2 = to_list(a), to_list(b)
    return _dp_matrix(s1, s2, insert=0, delete=0, substitute=None, match_score=1, boundary=(0, 0), combine="max")[-1][-1]
score_lc_subsequence = score_lcs
score_lcsubseq = score_lcs
score_longest_common_subsequence = score_lcs


@Metric
def dist_lcs(a, b, **kwargs) -> int:
    s1, s2 = to_list(a), to_list(b)
    return len(s1) + len(s2) - 2 * score_lcs(s1, s2, **kwargs)
dist_lc_subsequence = dist_lcs
dist_lcsubseq = dist_lcs
dist_longest_common_subsequence = dist_lcs


@Metric
def sim_lcs(a, b, **kwargs) -> float:
    s1, s2 = to_list(a), to_list(b)
    total = len(s1) + len(s2)
    if total == 0:
        return 1.0
    return 1 - (dist_lcs(s1, s2, **kwargs) / total)
sim_lc_subsequence = sim_lcs
sim_lcsubseq = sim_lcs
sim_longest_common_subsequence = sim_lcs


@Metric
def dif_lcs(a, b, **kwargs) -> float:
    return 1 - sim_lcs(a, b, **kwargs)
dif_lc_subsequence = dif_lcs
dif_lcsubseq = dif_lcs
dif_longest_common_subsequence = dif_lcs


def matrix_lcs(a, b, **kwargs):
    return _fill_dp_matrix(a, b, insert=0, delete=0, substitute=None, match_score=1, boundary=(0, 0), combine="max")
matrix_lc_subsequence = matrix_lcs
matrix_lcsubseq = matrix_lcs
matrix_longest_common_subsequence = matrix_lcs


def trace_lcs(a, b, **kwargs):
    """The actual longest common subsequence, recovered by backtracking the DP
    matrix. Returns a str when both inputs are strings, else a list of elements.
    When several subsequences tie for longest, one canonical path is returned
    (preferring a match, then the higher-scoring neighbour)."""
    s1, s2 = to_list(a), to_list(b)
    matrix = _dp_matrix(s1, s2, insert=0, delete=0, substitute=None, match_score=1, boundary=(0, 0), combine="max")
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
trace_lc_subsequence = trace_lcs
trace_lcsubseq = trace_lcs
trace_longest_common_subsequence = trace_lcs


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
# 'lcs' is kept as the canonical registry key for backward compatibility, but
# it is an ambiguous abbreviation in the literature (see info_lcs). These
# explicit aliases let callers say which variant they mean without relying on
# simdif's choice of default; the contiguous variant lives in lc_substring.py.
METRICS['lc_subsequence'] = METRICS['lcs']
METRICS['lcsubseq'] = METRICS['lcs']
METRICS['longest_common_subsequence'] = METRICS['lcs']
