from ..simdif import Metric, METRICS, to_list, _dp_matrix, _backtrack, _fill_dp_matrix


def info_needleman_wunsch() -> str:
    return """
Needleman-Wunsch (Global Alignment)
-----------------------------------
A dynamic-programming algorithm that finds the optimal GLOBAL alignment of two
sequences end-to-end, maximizing a score built from three tunable weights:

    match_score      reward for aligning two equal symbols (default  1)
    mismatch_penalty cost of aligning two different symbols  (default -1)
    gap_penalty      cost of inserting a gap ('-')           (default -1)

Roles:
    score - the optimal global alignment score
    trace - the aligned (A, B) pair, gaps marked with gap_symbol
    matrix - the filled scoring matrix

Note: There is no standard optimized third-party implementation for arbitrary
scoring schemes, so the local dynamic-programming implementation is always used.

Aliases: Needleman, Wunsch
    """.strip()
info_needleman = info_needleman_wunsch
info_wunsch = info_needleman_wunsch


def explain_needleman_wunsch(a, b, match_score=1, mismatch_penalty=-1, gap_penalty=-1, gap_symbol="-", **kwargs) -> str:
    aln_a, aln_b = trace_needleman_wunsch(a, b, match_score, mismatch_penalty, gap_penalty, gap_symbol)
    result = score_needleman_wunsch(a, b, match_score, mismatch_penalty, gap_penalty)
    markers = "".join("|" if x == y and x != gap_symbol else " " for x, y in zip(aln_a, aln_b))
    return f"""
A: ({", ".join(f"'{x}'" for x in to_list(a))})
B: ({", ".join(f"'{y}'" for y in to_list(b))})
Scoring: match={match_score}, mismatch={mismatch_penalty}, gap={gap_penalty}
Optimal Global Alignment:
  {"".join(str(x) for x in aln_a)}
  {markers}
  {"".join(str(y) for y in aln_b)}
Alignment Score: {result}
    """.strip()
explain_needleman = explain_needleman_wunsch
explain_wunsch = explain_needleman_wunsch


@Metric
def score_needleman_wunsch(a, b, match_score=1, mismatch_penalty=-1, gap_penalty=-1) -> int:
    s1, s2 = to_list(a), to_list(b)
    return _dp_matrix(s1, s2, insert=gap_penalty, delete=gap_penalty, substitute=mismatch_penalty, match_score=match_score, local=False, maximize=True)[-1][-1]
score_needleman = score_needleman_wunsch
score_wunsch = score_needleman_wunsch


def trace_needleman_wunsch(a, b, match_score=1, mismatch_penalty=-1, gap_penalty=-1, gap_symbol="-"):
    s1, s2 = to_list(a), to_list(b)
    matrix = _dp_matrix(s1, s2, insert=gap_penalty, delete=gap_penalty, substitute=mismatch_penalty, match_score=match_score, local=False, maximize=True)
    return _backtrack(matrix, s1, s2, match_score, mismatch_penalty, gap_penalty, local=False, gap_symbol=gap_symbol)
trace_needleman = trace_needleman_wunsch
trace_wunsch = trace_needleman_wunsch


def matrix_needleman_wunsch(a, b, match_score=1, mismatch_penalty=-1, gap_penalty=-1):
    return _fill_dp_matrix(a, b, insert=gap_penalty, delete=gap_penalty, substitute=mismatch_penalty, match_score=match_score, local=False, maximize=True)
matrix_needleman = matrix_needleman_wunsch
matrix_wunsch = matrix_needleman_wunsch


METRICS['needleman_wunsch'] = {
    'class': 'sequence',
    'default': 'score',
    'score': score_needleman_wunsch,
    'trace': trace_needleman_wunsch,
    'matrix': matrix_needleman_wunsch,
    'info': info_needleman_wunsch,
    'explain': explain_needleman_wunsch,
}
METRICS['needleman'] = METRICS['needleman_wunsch']
METRICS['wunsch'] = METRICS['needleman_wunsch']
