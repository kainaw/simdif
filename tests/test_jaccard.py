import pytest
from simdif.metrics.jaccard import sim_jaccard, dif_jaccard
from simdif import jaccard, sim, simdif

def test_jaccard():
    assert sim_jaccard([], []) == 1.0
    assert sim_jaccard([1,2,3],[1,3,5]) == pytest.approx(2/4)
    assert sim_jaccard([1,2,3],[]) == pytest.approx(0.0)
    assert dif_jaccard([1,2,3],[1,3,5]) == pytest.approx(2/4)
    assert jaccard([1,2,3],[1,3,5]) == pytest.approx(2/4)
    assert simdif([1,2,3],[1,3,5],['jaccard']) == {
        'jaccard': pytest.approx(2/4)
    }
