from ..simdif import (Metric, METRICS, to_list, _dp_matrix_affine,
                       _backtrack_affine, _fill_dp_matrix_affine)


def info_affine_gap() -> str:
    return """
Affine Gap Alignment
---------------------
A dynamic-programming alignment (Gotoh's algorithm) that scores gaps with
two separate costs instead of one flat per-position cost:

    match_score      reward for aligning two equal symbols     (default  1)
    mismatch_penalty cost of aligning two different symbols    (default -1)
    gap_open         cost of STARTING a new gap                (default -10)
    gap_extend       cost of each additional position in that
                      same gap run                             (default -1)

A run of k consecutive gap positions costs gap_open + (k-1)*gap_extend,
rather than k*gap_penalty as in Needleman-Wunsch/Smith-Waterman. This makes
one long contiguous gap cheaper than many scattered single-position gaps,
which is the behavior you want whenever a single "edit episode" (e.g. an
indel, a dropped phrase, a missing subsequence) is more plausible than an
equal number of scattered edits.

    local=False (default)  - global alignment, end-to-end (like Needleman-
                             Wunsch, but with affine gaps)
    local=True             - local alignment, best-scoring subsequence pair
                             (like Smith-Waterman, but with affine gaps)

Roles:
    score - the optimal alignment score
    trace - the aligned (A, B) pair, gaps marked with gap_symbol
    matrix - a display matrix (best of the three internal DP matrices per
             cell -- see explain_affine_gap for the underlying M/Ix/Iy
             breakdown if you need it)

Note: Uses a local O(n*m) dynamic-programming implementation (Gotoh's
three-matrix recurrence); there is no third-party fast-path since gap_open/
gap_extend are user-tunable.

Aliases: Gotoh
    """.strip()
info_gotoh = info_affine_gap


def explain_affine_gap(a, b, match_score=1, mismatch_penalty=-1, gap_open=-10,                         gap_extend=-1, local=False, gap_symbol="-", **kwargs) -> str:
    aln_a, aln_b = trace_affine_gap(a, b, match_score, mismatch_penalty, gap_open,                                      gap_extend, local, gap_symbol)
    result = score_affine_gap(a, b, match_score, mismatch_penalty, gap_open, gap_extend, local)
    markers = "".join("|" if x == y and x != gap_symbol else " " for x, y in zip(aln_a, aln_b))
    mode = "Local" if local else "Global"
    return f"""
A: ({", ".join(f"'{x}'" for x in to_list(a))})
B: ({", ".join(f"'{y}'" for y in to_list(b))})
Scoring: match={match_score}, mismatch={mismatch_penalty}, gap_open={gap_open}, gap_extend={gap_extend}
Best {mode} Affine-Gap Alignment:
  {"".join(str(x) for x in aln_a)}
  {markers}
  {"".join(str(y) for y in aln_b)}
Alignment Score: {result}
    """.strip()
explain_gotoh = explain_affine_gap


@Metric
def score_affine_gap(a, b, match_score=1, mismatch_penalty=-1, gap_open=-10,                       gap_extend=-1, local=False) -> float:
    s1, s2 = to_list(a), to_list(b)
    M, Ix, Iy = _dp_matrix_affine(s1, s2, gap_open=gap_open, gap_extend=gap_extend,                                    mismatch_penalty=mismatch_penalty, match_score=match_score,                                    local=local, maximize=True)
    if local:
        return max(M[i][j] for i in range(len(s1)+1) for j in range(len(s2)+1))
    return max(M[-1][-1], Ix[-1][-1], Iy[-1][-1])
score_gotoh = score_affine_gap


def trace_affine_gap(a, b, match_score=1, mismatch_penalty=-1, gap_open=-10,                       gap_extend=-1, local=False, gap_symbol="-"):
    s1, s2 = to_list(a), to_list(b)
    M, Ix, Iy = _dp_matrix_affine(s1, s2, gap_open=gap_open, gap_extend=gap_extend,                                    mismatch_penalty=mismatch_penalty, match_score=match_score,                                    local=local, maximize=True)
    return _backtrack_affine(M, Ix, Iy, s1, s2, gap_open, gap_extend, match_score,                               mismatch_penalty, local=local, maximize=True, gap_symbol=gap_symbol)
trace_gotoh = trace_affine_gap


def matrix_affine_gap(a, b, match_score=1, mismatch_penalty=-1, gap_open=-10,                        gap_extend=-1, local=False):
    return _fill_dp_matrix_affine(a, b, gap_open=gap_open, gap_extend=gap_extend,                                    mismatch_penalty=mismatch_penalty, match_score=match_score,                                    local=local, maximize=True)
matrix_gotoh = matrix_affine_gap


METRICS['affine_gap'] = {
    'class': 'sequence',
    'default': 'score',
    'score': score_affine_gap,
    'trace': trace_affine_gap,
    'matrix': matrix_affine_gap,
    'info': info_affine_gap,
    'explain': explain_affine_gap,
}
METRICS['gotoh'] = METRICS['affine_gap']
