import pytest
from simdif.metrics.minkowski import dist_minkowski, sim_minkowski
from simdif import minkowski, dist, simdif

def test_minkowski_basic():
    assert dist_minkowski([], []) == pytest.approx(0.0)
    # p=1 reduces to Manhattan: |1-2| + |10-3| + |3-5| = 10
    assert dist_minkowski([1, 10, 3], [2, 3, 5], p=1) == pytest.approx(10.0)
    # p=2 reduces to Euclidean: sqrt(4^2 + 3^2) = 5
    assert dist_minkowski([0, 3], [4, 0], p=2) == pytest.approx(5.0)
    # default p is 2 (Euclidean)
    assert dist_minkowski([0, 3], [4, 0]) == pytest.approx(5.0)
    # sim = 1 / (1 + d) = 1 / 6 for the Euclidean case
    assert sim_minkowski([0, 3], [4, 0]) == pytest.approx(1.0 / 6.0)
    assert dist_minkowski([0, 3], [4], pad_value="0") == pytest.approx(5.0)
    with pytest.raises(ValueError, match="Vector length mismatch"):
        dist_minkowski([1, 2, 3], [1, 2], pad_value=None)
    assert minkowski([0, 3], [4, 0]) == pytest.approx(5.0)
    assert dist([0, 3], [4, 0], 'minkowski') == pytest.approx(5.0)
    assert simdif([0, 3], [4, 0], ['minkowski']) == {'minkowski': pytest.approx(5.0)}
