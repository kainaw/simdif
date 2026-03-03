import pytest
from simdif.metrics.chebyshev import dist_chebyshev, sim_chebyshev
from simdif import chebyshev, dist, simdif

def test_chebyshev_basic():
    assert dist_chebyshev([], []) == pytest.approx(0.0)
    assert dist_chebyshev([1, 10, 3], [2, 3, 5]) == pytest.approx(7.0)
    assert dist_chebyshev([1, 10, 3], [2, 3], pad_value="5") == pytest.approx(7.0)
    with pytest.raises(ValueError, match="Vector length mismatch"):
        dist_chebyshev([1, 2, 3], [1, 2], pad_value=None)
    assert sim_chebyshev([1, 10, 3], [2, 3, 5]) == pytest.approx(0.125)
    assert chebyshev([1, 10, 3], [2, 3, 5]) == pytest.approx(7.0)
    assert dist([1, 10, 3], [2, 3, 5], 'chebyshev') == pytest.approx(7.0)
    assert simdif([1, 10, 3], [2, 3, 5], ['chebyshev']) == {'chebyshev': pytest.approx(7.0)}
