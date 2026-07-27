from ..simdif import Metric, METRICS, to_list, _dp_matrix, _backtrack, _fill_dp_matrix


def info_smith_waterman() -> str:
    return """
Smith-Waterman (Local Alignment)
--------------------------------
A dynamic-programming algorithm that finds the optimal LOCAL alignment of two
sequences, the highest-scoring pair of subsequences, rather than aligning
them end-to-end. Negative running scores are clamped to zero, so the alignment
is free to start and stop anywhere.

    match_score      reward for aligning two equal symbols (default  2)
    mismatch_penalty cost of aligning two different symbols  (default -1)
    gap_penalty      cost of inserting a gap ('-')           (default -1)

Roles:
    score - the best local alignment score (max cell in the matrix)
    trace - the aligned (A, B) pair for that best local region
    matrix - the filled scoring matrix

Note: There is no standard optimized third-party implementation for arbitrary
scoring schemes, so the local dynamic-programming implementation is always used.

Aliases: Smith, Waterman
    """.strip()
info_smith = info_smith_waterman
info_waterman = info_smith_waterman


def explain_smith_waterman(a, b, match_score=2, mismatch_penalty=-1, gap_penalty=-1, gap_symbol="-", **kwargs) -> str:
    aln_a, aln_b = trace_smith_waterman(a, b, match_score, mismatch_penalty, gap_penalty, gap_symbol)
    result = score_smith_waterman(a, b, match_score, mismatch_penalty, gap_penalty)
    markers = "".join("|" if x == y and x != gap_symbol else " " for x, y in zip(aln_a, aln_b))
    return f"""
A: ({", ".join(f"'{x}'" for x in to_list(a))})
B: ({", ".join(f"'{y}'" for y in to_list(b))})
Scoring: match={match_score}, mismatch={mismatch_penalty}, gap={gap_penalty}
Best Local Alignment:
  {"".join(str(x) for x in aln_a)}
  {markers}
  {"".join(str(y) for y in aln_b)}
Alignment Score: {result}
    """.strip()
explain_smith = explain_smith_waterman
explain_waterman = explain_smith_waterman


@Metric
def score_smith_waterman(a, b, match_score=2, mismatch_penalty=-1, gap_penalty=-1) -> int:
    s1, s2 = to_list(a), to_list(b)
    matrix = _dp_matrix(s1, s2, insert=gap_penalty, delete=gap_penalty, substitute=mismatch_penalty, match_score=match_score, boundary=(0, 0), floor=0, combine="max")
    return max(matrix[i][j] for i in range(len(s1)+1) for j in range(len(s2)+1))
score_smith = score_smith_waterman
score_waterman = score_smith_waterman


def trace_smith_waterman(a, b, match_score=2, mismatch_penalty=-1, gap_penalty=-1, gap_symbol="-"):
    s1, s2 = to_list(a), to_list(b)
    matrix = _dp_matrix(s1, s2, insert=gap_penalty, delete=gap_penalty, substitute=mismatch_penalty, match_score=match_score, boundary=(0, 0), floor=0, combine="max")
    return _backtrack(matrix, s1, s2, match_score, mismatch_penalty, gap_penalty, local=True, gap_symbol=gap_symbol)
trace_smith = trace_smith_waterman
trace_waterman = trace_smith_waterman


def matrix_smith_waterman(a, b, match_score=2, mismatch_penalty=-1, gap_penalty=-1):
    return _fill_dp_matrix(a, b, insert=gap_penalty, delete=gap_penalty, substitute=mismatch_penalty, match_score=match_score, boundary=(0, 0), floor=0, combine="max")
matrix_smith = matrix_smith_waterman
matrix_waterman = matrix_smith_waterman


METRICS['smith_waterman'] = {
    'class': 'sequence',
    'default': 'score',
    'score': score_smith_waterman,
    'trace': trace_smith_waterman,
    'matrix': matrix_smith_waterman,
    'info': info_smith_waterman,
    'explain': explain_smith_waterman,
}
METRICS['smith'] = METRICS['smith_waterman']
METRICS['waterman'] = METRICS['smith_waterman']
