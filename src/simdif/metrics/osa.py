from ..simdif import Metric, METRICS, to_list
import sys


def _osa_matrix(s1, s2) -> list:
    """
    Optimal String Alignment (OSA) distance matrix.
    Restricted edit distance: a substring can only be edited once.
    Does NOT satisfy the triangle inequality.
    """
    len1, len2 = len(s1), len(s2)
    matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        matrix[i][0] = i
    for j in range(len2 + 1):
        matrix[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,      # deletion
                matrix[i][j - 1] + 1,      # insertion
                matrix[i - 1][j - 1] + cost # substitution
            )

            if i > 1 and j > 1 and s1[i - 1] == s2[j - 2] and s1[i - 2] == s2[j - 1]:
                matrix[i][j] = min(matrix[i][j], matrix[i - 2][j - 2] + cost)

    return matrix


def info_osa() -> str:
    return """
Optimal String Alignment (OSA)
------------------------------
A restricted version of Damerau-Levenshtein. It counts insertions, deletions, 
substitutions, and transpositions, but with the constraint that no substring 
can be edited more than once.
Note: Because of this restriction, it does NOT satisfy the triangle inequality.
Example: "CA" → "ABC" results in 3 (not 2).
    """.strip()


@Metric
def dist_osa(a, b, **_) -> float:
    # RapidFuzz uses True DL for its DamerauLevenshtein, 
    # so we use our local matrix for OSA correctness.
    s1, s2 = to_list(a), to_list(b)
    matrix = _osa_matrix(s1, s2)
    return float(matrix[len(s1)][len(s2)])


METRICS['osa'] = {
    'class': 'sequence',
    'default': 'dist',
    'dist': dist_osa,
    'info': info_osa,
}