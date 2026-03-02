from ..simdif import Metric, METRICS, to_list
import sys


def _dl_matrix(s1, s2) -> list:
    """
    True Damerau-Levenshtein distance matrix using the unrestricted algorithm.
    Tracks the last seen position of each character to correctly handle
    overlapping transpositions. Satisfies the triangle inequality, unlike OSA.
    """
    len1, len2 = len(s1), len(s2)
    last_row = {}
    INF = len1 + len2 + 1
    matrix = [[0] * (len2 + 2) for _ in range(len1 + 2)]
    matrix[0][0] = INF
    for i in range(len1 + 1):
        matrix[i + 1][0] = INF
        matrix[i + 1][1] = i
    for j in range(len2 + 1):
        matrix[0][j + 1] = INF
        matrix[1][j + 1] = j

    for i in range(1, len1 + 1):
        last_col = 0
        for j in range(1, len2 + 1):
            i1 = last_row.get(s2[j - 1], 0)
            j1 = last_col
            if s1[i - 1] == s2[j - 1]:
                cost = 0
                last_col = j
            else:
                cost = 1
            matrix[i + 1][j + 1] = min(
                matrix[i][j] + cost,           # substitute (or match)
                matrix[i + 1][j] + 1,          # insert
                matrix[i][j + 1] + 1,          # delete
                matrix[i1][j1]                 # transpose
                    + (i - i1 - 1)             # deletions before transposition
                    + 1                        # the transposition itself
                    + (j - j1 - 1)             # insertions after transposition
            )

        last_row[s1[i - 1]] = i

    return matrix


def info_damerau_levenshtein() -> str:
    return """
Damerau-Levenshtein Distance
-----------------------------
Counts the minimum number of single-character insertions, deletions,
substitutions, and transpositions (swapping two adjacent characters) needed
to transform one sequence into another. Unlike OSA (Optimal String Alignment),
this is the true Damerau-Levenshtein algorithm — it allows a substring to be
edited more than once and satisfies the triangle inequality.
Formula:
    DL(A,B) = minimum edit operations using:
        insertion, deletion, substitution, transposition
Range: [0, max(|A|, |B|)]
    0 = identical sequences
Note: Differs from OSA only on strings requiring overlapping transpositions,
e.g. "CA" → "ABC" (DL=2, OSA=3). If your data is unlikely to contain such
cases, OSA is a reasonable approximation. Use this when correctness and the
triangle inequality are required.
Aliases: Damerau-Levenshtein, DL
    """.strip()
info_dl = info_damerau_levenshtein


def explain_damerau_levenshtein(a, b, **_) -> str:
    s1, s2 = to_list(a), to_list(b)
    matrix = _dl_matrix(s1, s2)
    dist = matrix[len(s1) + 1][len(s2) + 1]
    header = "      " + "  ".join(f"'{c}'" for c in ['_'] + list(s2))
    rows_display = []
    labels = ['_'] + list(s1)
    for i, label in enumerate(labels):
        row = matrix[i + 1][1:]
        rows_display.append(f"  '{label}'  " + "  ".join(f"{v:3}" for v in row))
    return f"""
A: ({", ".join(f"'{x}'" for x in s1)})
B: ({", ".join(f"'{y}'" for y in s2)})
Damerau-Levenshtein Distance (true, unrestricted):
{header}
""" + "\n".join(rows_display) + f"""
Distance: {dist}
    """.strip()
explain_dl = explain_damerau_levenshtein


@Metric
def dist_damerau_levenshtein(a, b, **_) -> float:
    if isinstance(a, str) and isinstance(b, str) and 'rapidfuzz' in sys.modules:
        # rapidfuzz implements true DL, not OSA
        return float(sys.modules['rapidfuzz'].distance.DamerauLevenshtein.distance(a, b))
    s1, s2 = to_list(a), to_list(b)
    return float(_dl_matrix(s1, s2)[len(s1) + 1][len(s2) + 1])
dist_dl = dist_damerau_levenshtein


METRICS['damerau_levenshtein'] = {
    'class': 'sequence',
    'default': 'dist',
    'dist': dist_damerau_levenshtein,
    'info': info_damerau_levenshtein,
    'explain': explain_damerau_levenshtein,
}
METRICS['dl'] = METRICS['damerau_levenshtein']
