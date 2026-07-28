import pytest
from simdif.metrics.damerau_levenshtein import dist_damerau_levenshtein, explain_damerau_levenshtein
from simdif import damerau_levenshtein, dist, simdif

def test_damerau_levenshtein():
    assert dist_damerau_levenshtein("CA", "ABC") == 2.0
    assert dist_damerau_levenshtein("CA", "ABCA") == 2.0
    assert dist_damerau_levenshtein("BCA", "ABC") == 2.0
    assert dist_damerau_levenshtein("ABC", "CDE") <= dist_damerau_levenshtein("ABC", "BCD") + dist_damerau_levenshtein("BCD", "CDE")
    assert damerau_levenshtein("CA", "ABC") == 2.0
    assert dist("CA", "ABC","damerau_levenshtein") == 2.0
    assert simdif("CA","ABC",["damerau_levenshtein"]) == {"damerau_levenshtein": 2.0}


def test_damerau_levenshtein_optimized_lib(optimized_lib):
    optimized_lib('rapidfuzz')
    assert dist_damerau_levenshtein("CA", "ABC") == 2.0
    assert "Note:" not in explain_damerau_levenshtein("CA", "ABC")


def test_damerau_levenshtein_explain_formatting():
    # The trailing "Distance: N" line must stay on its own line -- a stray
    # .strip() that bound to only the last f-string fragment used to eat the
    # newline separating it from the last matrix row.
    out = explain_damerau_levenshtein("CA", "ABC")
    assert "\nDistance: 2" in out
