import pytest
from simdif.metrics.lcs import score_lcs, dist_lcs, matrix_lcs
from simdif import lcs, score, dist, simdif

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
