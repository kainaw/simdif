import pytest
from simdif.metrics.lcs import score_lcs, dist_lcs, sim_lcs, dif_lcs, trace_lcs, matrix_lcs
from simdif import lcs, sim, dif, dist, simdif

def test_lcs_sim_dif():
    # Edge case: two empty sequences are identical -> sim=1, dif=0.
    assert sim_lcs("", "") == 1.0
    assert dif_lcs("", "") == 0.0

    # Identical strings: dist_lcs is 0 regardless of length, so sim is
    # always 1.0 / dif always 0.0 for any A == B.
    assert sim_lcs("abc", "abc") == 1.0
    assert dif_lcs("abc", "abc") == 0.0

    # No common subsequence at all: sim=0, dif=1.
    assert sim_lcs("XYZ", "ABC") == 0.0
    assert dif_lcs("XYZ", "ABC") == 1.0

    # Docstring example: LCS("ABCBDAB","BDCABA")=4, |A|=7,|B|=6,
    # dist_lcs = 7+6-2*4 = 5, sim = 1 - 5/13 = 8/13.
    assert sim_lcs("ABCBDAB", "BDCABA") == pytest.approx(8/13)
    assert dif_lcs("ABCBDAB", "BDCABA") == pytest.approx(5/13)

    # sim and dif are always complementary.
    for a, b in [("cat", "cot"), ("kitten", "sitting"), ("AB", "ABCDEFG"), ("", "xyz")]:
        assert sim_lcs(a, b) + dif_lcs(a, b) == pytest.approx(1.0)

    # A short string fully contained (as a subsequence) in a much longer one
    # is NOT sim=1 -- sim is tied to dist_lcs's (|A|+|B|) normalization, not
    # to score's own [0, min(|A|,|B|)] range, so length mismatch still
    # penalizes it (mirrors sim_levenshtein's behavior).
    assert sim_lcs("AB", "ABCDEFG") == pytest.approx(1 - 5/9)

    # Convenience/dispatcher access, matching the pattern used for the other
    # metrics' sim/dif roles.
    assert sim("ABCBDAB", "BDCABA", "lcs") == pytest.approx(8/13)
    assert dif("ABCBDAB", "BDCABA", "lcs") == pytest.approx(5/13)
    assert simdif("cat", "cot", ["lcs"]) == {"lcs": 2}  # default role is still 'score'
    assert simdif("cat", "cot", "sim_lcs") == sim_lcs("cat", "cot")
