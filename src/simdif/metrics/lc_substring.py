from ..simdif import Metric, METRICS, to_list, _dp_matrix, _fill_dp_matrix
import difflib

# Indels are forbidden outright rather than merely made expensive: a -inf step
# cost plus floor=0 is exactly the "reset the run to zero on a mismatch" rule.
_NO_INDEL = -float('inf')
_DP_ARGS = dict(insert=_NO_INDEL, delete=_NO_INDEL, substitute=None,
                match_score=1, boundary=(0, 0), floor=0, combine="max")


def info_lc_substring() -> str:
    return """
Longest Common Substring (LC substring)
---------------------------------------
The length of the longest run of elements that appears in both inputs
CONTIGUOUSLY -- no skipping. For example, the longest common substring of
"ABABC" and "BABCA" is "BABC" (length 4).

Not to be confused with the Longest Common SUBSEQUENCE ('lc_subsequence'),
which allows gaps. The literature abbreviates both as "LCS", so check which
recurrence a paper prints before comparing numbers. The difference is not
cosmetic: for "ABCDE" vs "AXBXCXDXE" the subsequence score is 5 and the
substring score is 1.

Roles:
    score - length of the longest common substring (larger = more similar)
    sim   - score / max(|A|, |B|). Ranges [0, 1]; 1.0 iff A and B are
            identical. Normalized by the LONGER input, so a short string
            fully contained in a much longer one still scores well below 1.0
            (same convention as sim_prefix / sim_suffix).
    dif   - 1 - sim
    trace - the actual substring itself (a str for string inputs, else a
            list). When several substrings tie for longest, the earliest one
            in A is returned.
    matrix - the filled DP grid

There is deliberately no `dist` role. The indel-distance identity that gives
'lc_subsequence' its `dist` (|A| + |B| - 2*LCS) does not carry over here:
|A| + |B| - 2*LCSubstr counts no sequence of edit operations and is not a
metric, so offering it under the name `dist` would be misleading.

Note: In the DP grid, a cell holds the length of the common run ending exactly
at that pair of positions -- NOT the best score over all prefixes as in
'lc_subsequence'. Mismatches reset the cell to 0 and the answer is the largest
cell anywhere in the grid, not the bottom-right corner. This is the same
clamp-to-zero trick that turns Needleman-Wunsch (global) into Smith-Waterman
(local); LC substring is exactly smith_waterman with mismatch and gap
penalties of -inf.

Range (score): [0, min(|A|, |B|)]
Range (sim/dif): [0, 1]

Aliases: lcstr, lcsubstr, longest_common_substring
    """.strip()
info_lcstr = info_lc_substring
info_lcsubstr = info_lc_substring
info_longest_common_substring = info_lc_substring


def explain_lc_substring(a, b, **kwargs) -> str:
    grid = matrix_lc_substring(a, b)
    rows_display = ["  " + "  ".join(f"{str(cell):>3}" for cell in row) for row in grid]
    return f"""
A: ({", ".join(f"'{x}'" for x in to_list(a))})
B: ({", ".join(f"'{y}'" for y in to_list(b))})
Longest Common Substring -- runs must be contiguous, no skipping allowed.
(For the variant that permits gaps, see 'lc_subsequence'.)
LC Substring Matrix (rows = A, cols = B; each cell = length of the common
run ending exactly there, reset to 0 on a mismatch):
{chr(10).join(rows_display)}
Substring Length (score): {score_lc_substring(a, b, **kwargs)} (the largest cell above, not the corner)
Longest Common Substring (trace): {trace_lc_substring(a, b, **kwargs)!r}
Similarity (sim): {sim_lc_substring(a, b, **kwargs):.4f}
Difference (dif): {dif_lc_substring(a, b, **kwargs):.4f}
    """.strip()
explain_lcstr = explain_lc_substring
explain_lcsubstr = explain_lc_substring
explain_longest_common_substring = explain_lc_substring


@Metric
def score_lc_substring(a, b, **kwargs) -> int:
    if isinstance(a, str) and isinstance(b, str):
        # difflib is stdlib, so this fast path is always available for strings.
        # autojunk=False keeps it exact: the heuristic otherwise ignores
        # elements that are "popular" in long inputs, which would understate
        # the match.
        return difflib.SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b)).size
    s1, s2 = to_list(a), to_list(b)
    matrix = _dp_matrix(s1, s2, **_DP_ARGS)
    return max(max(row) for row in matrix)


@Metric
def sim_lc_substring(a, b, **kwargs) -> float:
    s1, s2 = to_list(a), to_list(b)
    longest = max(len(s1), len(s2))
    if longest == 0:
        return 1.0
    return score_lc_substring(a, b, **kwargs) / longest


@Metric
def dif_lc_substring(a, b, **kwargs) -> float:
    return 1 - sim_lc_substring(a, b, **kwargs)


def matrix_lc_substring(a, b, **kwargs):
    return _fill_dp_matrix(a, b, **_DP_ARGS)


def trace_lc_substring(a, b, **kwargs):
    """The actual longest common substring. Because each cell already holds the
    length of the run ending there, no backtracking is needed -- the best cell
    names both the length and its own end position, and the run is the slice
    reaching back from it. Ties are broken toward the earliest occurrence in A.
    Returns a str when both inputs are strings, else a list of elements."""
    s1, s2 = to_list(a), to_list(b)
    matrix = _dp_matrix(s1, s2, **_DP_ARGS)
    best, end_i = 0, 0
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if matrix[i][j] > best:
                best, end_i = matrix[i][j], i
    run = s1[end_i - best:end_i]
    if isinstance(a, str) and isinstance(b, str):
        return "".join(str(x) for x in run)
    return run


METRICS['lc_substring'] = {
    'class': 'sequence',
    'default': 'score',
    'score': score_lc_substring,
    'sim': sim_lc_substring,
    'dif': dif_lc_substring,
    'matrix': matrix_lc_substring,
    'trace': trace_lc_substring,
    'info': info_lc_substring,
    'explain': explain_lc_substring,
}
# Unambiguous aliases only. 'lcs' is deliberately NOT registered here -- it
# resolves to the subsequence variant, and silently answering a different
# question under a familiar name is worse than an unknown-metric error.
METRICS['lcstr'] = METRICS['lc_substring']
METRICS['lcsubstr'] = METRICS['lc_substring']
METRICS['longest_common_substring'] = METRICS['lc_substring']
