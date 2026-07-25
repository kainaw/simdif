import math
import pytest
from simdif.metrics.dtw import dist_dtw, sim_dtw, dif_dtw, matrix_dtw
from simdif import dtw, dist, sim, dif, simdif

def test_dtw():
    # Edge case: two empty sequences align to a distance of 0.
    assert dist_dtw([], []) == 0
    # Edge case: empty vs. non-empty has no valid warping path (boundary is
    # blocked to inf), so the distance is infinite.
    assert dist_dtw([], [1, 2, 3]) == math.inf

    # Identical sequences: every step costs 0.
    assert dist_dtw([1, 2, 3], [1, 2, 3]) == 0.0

    # Single-element sequences: default dist_fn is absolute difference.
    assert dist_dtw([1], [3]) == 2.0

    # Reference DTW example, cross-checked against a plain min-plus
    # recurrence D[i,j] = |Ai-Bj| + min(D[i-1,j], D[i,j-1], D[i-1,j-1]).
    a = [1, 2, 3, 5, 5, 5, 6]
    b = [1, 1, 2, 2, 3, 5]
    assert dist_dtw(a, b) == 1.0

    # Custom dist_fn kwarg: pointwise squared difference instead of |a-b|.
    assert dist_dtw([0, 1], [0, 2], dist_fn=lambda x, y: (x - y) ** 2) == 1.0

    # sim/dif follow the 1/(1+d) convention used by the other unbounded
    # distance metrics (e.g. manhattan/euclidean), and stay complementary.
    assert sim_dtw([1, 2, 3], [1, 2, 3]) == 1.0
    assert dif_dtw([1, 2, 3], [1, 2, 3]) == 0.0
    assert sim_dtw([1], [3]) == pytest.approx(1 / 3)
    for x, y in [([1, 2, 3], [1, 2, 3]), ([1], [3]), (a, b)]:
        assert sim_dtw(x, y) + dif_dtw(x, y) == pytest.approx(1.0)

    # matrix_dtw: row/col 0 is blocked to inf except the corner, matching
    # the boundary=(inf, inf) semi-global constraint.
    grid = matrix_dtw([1, 3], [1, 2, 4])
    assert grid[1][1] == 0
    assert grid[1][2] == math.inf
    assert grid[2][1] == math.inf

    # Convenience name maps to the default 'dist' role.
    assert dtw([1, 2, 3], [1, 2, 3]) == 0.0
    assert dist([1, 2, 3], [1, 2, 3], "dtw") == 0.0
    assert sim([1, 2, 3], [1, 2, 3], "dtw") == 1.0
    assert dif([1, 2, 3], [1, 2, 3], "dtw") == 0.0
    assert simdif([1, 2, 3], [1, 2, 3], ["dtw"]) == {"dtw": 0.0}
