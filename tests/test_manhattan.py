import pytest
from simdif.metrics.manhattan import dist_manhattan, sim_manhattan
from simdif import manhattan, dist, simdif

def test_manhattan_basic():
    assert dist_manhattan([], []) == pytest.approx(0.0)
    # |1-2| + |10-3| + |3-5| = 1 + 7 + 2 = 10
    assert dist_manhattan([1, 10, 3], [2, 3, 5]) == pytest.approx(10.0)
    # |0-4| + |3-0| = 4 + 3 = 7
    assert dist_manhattan([0, 3], [4, 0]) == pytest.approx(7.0)
    # sim = 1 / (1 + d) = 1 / 11
    assert sim_manhattan([1, 10, 3], [2, 3, 5]) == pytest.approx(1.0 / 11.0)
    # pad_value is forwarded to the underlying Minkowski alignment:
    # b padded to [2, 3, 5] -> |1-2| + |10-3| + |3-5| = 1 + 7 + 2 = 10
    assert dist_manhattan([1, 10, 3], [2, 3], pad_value="5") == pytest.approx(10.0)
    # Without pad_value a length mismatch still raises.
    with pytest.raises(ValueError, match="Vector length mismatch"):
        dist_manhattan([1, 2, 3], [1, 2], pad_value=None)
    assert manhattan([1, 10, 3], [2, 3, 5]) == pytest.approx(10.0)
    assert dist([1, 10, 3], [2, 3, 5], 'manhattan') == pytest.approx(10.0)
    assert simdif([1, 10, 3], [2, 3, 5], ['manhattan']) == {'manhattan': pytest.approx(10.0)}
