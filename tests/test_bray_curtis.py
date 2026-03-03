import pytest
from simdif.metrics.bray_curtis import dist_bray_curtis, sim_bray_curtis
from simdif import bray_curtis, dist, simdif

def test_bray_curtis():
    assert dist_bray_curtis([], []) == 0.0
    assert dist_bray_curtis([1,2,3],[1,3,5]) == pytest.approx(0.2)
    assert dist_bray_curtis([1,2,3],[1,3],pad_value='5') == pytest.approx(0.2)
    assert dist_bray_curtis([0,0], [0,0]) == 0.0
    assert dist_bray_curtis([1,1], [0,0]) == 1.0
    assert dist_bray_curtis([-1,-2,-3],[1,3,5]) == pytest.approx(5.0)
    assert sim_bray_curtis([1,2,3],[1,3,5]) == pytest.approx(0.8)
    assert bray_curtis([1,2,3],[1,3,5]) == pytest.approx(0.2)
    assert dist([1,2,3],[1,3,5],'bray_curtis') == pytest.approx(0.2)
    assert simdif([1,2,3],[1,3,5],['bray_curtis']) == {'bray_curtis': pytest.approx(0.2)}
    with pytest.raises(ValueError, match="Vector length mismatch"):
        dist_bray_curtis([1, 2, 3], [1, 2], pad_value=None)
