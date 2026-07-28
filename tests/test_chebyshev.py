import math
import pytest
from simdif.metrics.chebyshev import dist_chebyshev, sim_chebyshev, explain_chebyshev
from simdif.metrics.minkowski import dist_minkowski
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


def test_chebyshev_reads_only_the_worst_coordinate():
    # The documented failure mode: every dimension but the max is discarded, so
    # a shared offset in the other coordinates changes nothing.
    assert dist_chebyshev([0, 0, 0, 0], [0, 0, 0, 9]) == pytest.approx(9.0)
    assert dist_chebyshev([8, 8, 8, 8], [8, 8, 8, 17]) == pytest.approx(9.0)
    # ...until another coordinate overtakes it.
    assert dist_chebyshev([8, 8, 8, 8], [8, 8, 20, 17]) == pytest.approx(12.0)


def test_chebyshev_is_the_minkowski_limit():
    a, b = [1, 10, 3], [2, 3, 5]
    assert dist_minkowski(a, b, p=math.inf) == pytest.approx(dist_chebyshev(a, b))
    # Evaluating |d|^inf literally would collapse to 1.0; the limit is 7.0.
    assert dist_minkowski(a, b, p=math.inf) == pytest.approx(7.0)
    assert dist_minkowski([], [], p=math.inf) == pytest.approx(0.0)
    # Rising p walks from the sum toward the single largest gap.
    assert dist_minkowski(a, b, p=1) == pytest.approx(10.0)
    assert dist_minkowski(a, b, p=64) == pytest.approx(7.0, abs=1e-3)


def test_chebyshev_explain():
    out = explain_chebyshev([1, 10, 3], [2, 3, 5])
    assert "Chebyshev Distance: 7.0000" in out
    assert "<- the max" in out
    assert "falls to 2.0000" in out          # runner-up makes the discard visible
    # Empty input returned 0.0 from dist but crashed explain (max of empty seq).
    assert "Chebyshev Distance: 0.0000" in explain_chebyshev([], [])
    assert "only one dimension" in explain_chebyshev([1], [5])


def test_chebyshev_optimized_lib(optimized_lib):
    optimized_lib('scipy')
    assert dist_chebyshev([1, 10, 3], [2, 3, 5]) == pytest.approx(7.0)
    assert dist_chebyshev([], []) == pytest.approx(0.0)  # empty guard precedes the scipy call
    assert "Note:" not in explain_chebyshev([1, 10, 3], [2, 3, 5])
