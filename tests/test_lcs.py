import pytest
from simdif.metrics.lcs import score_lcs, dist_lcs, matrix_lcs, trace_lcs
from simdif import lcs, score, dist, trace, simdif

def test_lcs():
    # Edge cases.
    assert score_lcs("", "") == 0
    assert score_lcs("abc", "abc") == 3
    assert dist_lcs("abc", "abc") == 0
    # "cat"/"cot": LCS = "ct" (length 2).
    assert score_lcs("cat", "cot") == 2
    # Docstring example: LCS("ABCBDAB", "BDCABA") = "BCBA" (length 4).
    assert score_lcs("ABCBDAB", "BDCABA") == 4
    # dist role is the indel distance |A| + |B| - 2*LCS.
    assert dist_lcs("cat", "cot") == 3 + 3 - 2 * 2
    assert dist_lcs("ABCBDAB", "BDCABA") == 7 + 6 - 2 * 4
    # matrix role: bottom-right cell equals the LCS length.
    assert matrix_lcs("cat", "cot")[-1][-1] == 2
    # Convenience name (default role is 'score').
    assert lcs("cat", "cot") == 2
    assert score("cat", "cot", "lcs") == 2
    assert dist("cat", "cot", "lcs") == 2
    assert simdif("cat", "cot", ["lcs"]) == {"lcs": 2}


def test_lcs_trace():
    # trace returns the actual subsequence; a str for string inputs.
    assert trace_lcs("cat", "cot") == "ct"
    assert trace_lcs("ABCBDAB", "BDCABA") == "BCBA"
    # No common subsequence -> empty.
    assert trace_lcs("abc", "xyz") == ""
    assert trace_lcs("", "") == ""
    # List inputs -> list output (elements preserved, not joined into a string).
    assert trace_lcs(["a", "b", "c"], ["a", "x", "c"]) == ["a", "c"]
    # The recovered subsequence's length always equals the LCS score, and it is
    # a genuine subsequence of both inputs.
    result = trace_lcs("ABCBDAB", "BDCABA")
    assert len(result) == score_lcs("ABCBDAB", "BDCABA")
    # trace() dispatcher and simdif('trace_lcs') resolve the role.
    assert trace("cat", "cot", "lcs") == "ct"
    assert simdif("ABCBDAB", "BDCABA", ["trace_lcs"]) == {"trace_lcs": "BCBA"}
