import pytest
from simdif.metrics.damerau_levenshtein import dist_damerau_levenshtein
from simdif import damerau_levenshtein, dist, simdif

def test_damerau_levenshtein():
    assert dist_damerau_levenshtein("CA", "ABC") == 2.0
    assert dist_damerau_levenshtein("CA", "ABCA") == 2.0
    assert dist_damerau_levenshtein("BCA", "ABC") == 2.0
    assert dist_damerau_levenshtein("ABC", "CDE") <= dist_damerau_levenshtein("ABC", "BCD") + dist_damerau_levenshtein("BCD", "CDE")
    assert damerau_levenshtein("CA", "ABC") == 2.0
    assert dist("CA", "ABC","damerau_levenshtein") == 2.0
    assert simdif("CA","ABC",["damerau_levenshtein"]) == {"damerau_levenshtein": 2.0}
