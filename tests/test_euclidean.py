import pytest
from simdif.metrics.euclidean import dist_euclidean
from simdif import euclidean, dist, simdif

def test_euclidean_basic():
    assert dist_euclidean([], []) == pytest.approx(0.0)
    assert dist_euclidean([0, 3], [4, 0]) == pytest.approx(5.0)
    assert dist_euclidean([0, 3], [4], pad_value="0") == pytest.approx(5.0)
    with pytest.raises(ValueError, match="Vector length mismatch"):
        dist_euclidean([1, 2, 3], [1, 2], pad_value=None)
    assert euclidean([0, 3], [4, 0]) == pytest.approx(5.0)
    assert dist([0, 3], [4, 0], 'euclidean') == pytest.approx(5.0)
    assert simdif([0,3], [4,0], ['euclidean']) == {'euclidean': pytest.approx(5.0)}
