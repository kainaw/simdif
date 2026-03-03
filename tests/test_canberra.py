import pytest
from simdif.metrics.canberra import dist_canberra, sim_canberra
from simdif import canberra, dist, simdif

def test_canberra_basic():
    assert dist_canberra([], []) == pytest.approx(0.0)
    assert dist_canberra([1, 2], [1, 6]) == pytest.approx(0.5)
    assert dist_canberra([1, 2], [1], pad_value="6") == pytest.approx(0.5)
    with pytest.raises(ValueError, match="Vector length mismatch"):
        dist_canberra([1, 2, 3], [1, 2], pad_value=None)
    assert sim_canberra([1, 2], [1, 6]) == pytest.approx(0.666666, rel=1e-5)
    assert canberra([1, 2], [1, 6]) == pytest.approx(0.5)
    assert dist([1, 2], [1, 6], 'canberra') == pytest.approx(0.5)
    assert simdif([1,2], [1,6], ['canberra']) == {'canberra': pytest.approx(0.5)}
