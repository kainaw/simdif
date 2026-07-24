import pytest
from simdif.metrics.levenshtein import (
    dist_levenshtein, sim_levenshtein, dif_levenshtein, matrix_levenshtein,
)
from simdif import levenshtein, dist, sim, simdif

def test_levenshtein():
    # Edge cases.
    assert dist_levenshtein("", "") == 0
    assert dist_levenshtein("abc", "abc") == 0
    # "cat"/"cot": a single substitution -> 1.
    assert dist_levenshtein("cat", "cot") == 1
    # Classic "kitten"/"sitting" -> 3 (k->s, e->i, insert g).
    assert dist_levenshtein("kitten", "sitting") == 3
    # sim role: 1 - dist/max_len.
    assert sim_levenshtein("cat", "cot") == pytest.approx(1 - 1/3)
    assert sim_levenshtein("", "") == 1.0
    assert dif_levenshtein("cat", "cot") == pytest.approx(1/3)
    # matrix role: bottom-right cell equals the distance.
    assert matrix_levenshtein("cat", "cot")[-1][-1] == 1
    # Convenience name (default role is 'dist').
    assert levenshtein("cat", "cot") == 1
    assert dist("cat", "cot", "levenshtein") == 1
    assert sim("cat", "cot", "levenshtein") == pytest.approx(1 - 1/3)
    assert simdif("cat", "cot", ["levenshtein"]) == {"levenshtein": 1}
