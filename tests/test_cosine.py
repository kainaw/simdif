import pytest
from simdif.metrics.cosine import dist_cosine, sim_cosine
from simdif import cosine, sim, simdif

def test_cosine_basic():
    assert dist_cosine([], []) == pytest.approx(0.0)
    assert sim_cosine([0, 2], [2, 0]) == pytest.approx(0.0)
    assert sim_cosine([2, 2], [4, 4]) == pytest.approx(1.0)
    assert sim_cosine([-2, -2], [2, 2]) == pytest.approx(-1.0)
    assert sim_cosine([0, 2], [2], pad_value="0") == pytest.approx(0.0)
    with pytest.raises(ValueError, match="Vector length mismatch"):
        sim_cosine([1, 2, 3], [1, 2], pad_value=None)
    assert dist_cosine([0, 2], [2, 0]) == pytest.approx(1.0)
    assert cosine([0, 2], [2, 0]) == pytest.approx(0.0)
    assert sim([0, 2], [2, 0], 'cosine') == pytest.approx(0.0)
    assert simdif([0,2], [2,0], ['cosine']) == {'cosine': pytest.approx(0.0)}
