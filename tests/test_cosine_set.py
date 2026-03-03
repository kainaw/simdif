import pytest
from simdif.metrics.cosine_set import sim_cosine_set, dif_cosine_set
from simdif import cosine_set, sim, simdif

def test_cosine_set():
    assert sim_cosine_set([], []) == 1.0
    assert sim_cosine_set([1,2,3],[1,3,5]) == pytest.approx(2/3)
    assert sim_cosine_set([1,2,3],[]) == pytest.approx(0.0)
    assert dif_cosine_set([1,2,3],[1,3,5]) == pytest.approx(1/3)
    assert cosine_set([1,2,3],[1,3,5]) == pytest.approx(2/3)
    assert sim([1,2,3],[1,3,5],'cosine_set') == pytest.approx(2/3)
    assert sim([1,2,3],[1,3,5],'ochiai') == pytest.approx(2/3)
    assert simdif([1,2,3],[1,3,5],['cosine_set','ochiai']) == {'cosine_set': pytest.approx(2/3), 'ochiai': pytest.approx(2/3)}
