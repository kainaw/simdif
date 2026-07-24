import pytest
from simdif.metrics.indel import dist_indel, matrix_indel
from simdif import indel, dist, simdif

def test_indel():
    # Edge cases.
    assert dist_indel("", "") == 0
    assert dist_indel("abc", "abc") == 0
    assert dist_indel("", "abc") == 3
    # "cat"/"cot": no substitution allowed, delete 'a' + insert 'o' -> 2.
    # Equivalently |A| + |B| - 2*LCS = 3 + 3 - 2*2 = 2.
    assert dist_indel("cat", "cot") == 2
    # "cat"/"cats": one insertion -> 3 + 4 - 2*3 = 1.
    assert dist_indel("cat", "cats") == 1
    # matrix role: bottom-right cell equals the distance.
    assert matrix_indel("cat", "cot")[-1][-1] == 2
    # Convenience name (default role is 'dist').
    assert indel("cat", "cot") == 2
    assert dist("cat", "cot", "indel") == 2
    assert simdif("cat", "cot", ["indel"]) == {"indel": 2}
