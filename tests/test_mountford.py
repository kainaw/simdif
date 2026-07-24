import math
import pytest
from simdif.metrics.mountford import sim_mountford, dif_mountford
from simdif import mountford, sim, dif, simdif


def test_mountford():
    # n11=2, n10=1, n01=1 -> 2*2 / (2*(1+1) + 2*1*1) = 4 / 6
    assert sim_mountford([1, 2, 3], [1, 3, 5]) == pytest.approx(4 / 6)
    # dif = 1 / (1 + M) = 1 / (1 + 2/3) = 0.6
    assert dif_mountford([1, 2, 3], [1, 3, 5]) == pytest.approx(0.6)

    # Identical non-empty lists -> unbounded -> +inf, dif -> 0
    assert sim_mountford([1, 2, 3], [1, 2, 3]) == math.inf
    assert dif_mountford([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)

    # No shared species -> 0.0, dif -> 1.0
    assert sim_mountford([1, 2], [3, 4]) == pytest.approx(0.0)
    assert dif_mountford([1, 2], [3, 4]) == pytest.approx(1.0)

    # Empty inputs -> no shared species -> 0.0 (no NaN/inf)
    assert sim_mountford([], []) == pytest.approx(0.0)

    # Sample-size independence: adding species unique to B in equal measure
    # to a proportional resample keeps the index stable relative to Jaccard.
    # Here just assert it stays finite and positive with partial overlap.
    assert sim_mountford([1, 2, 3, 4], [1, 2, 5, 6]) > 0

    # Convenience name + role dispatch + simdif forwarding
    assert mountford([1, 2, 3], [1, 3, 5]) == pytest.approx(4 / 6)
    assert sim([1, 2, 3], [1, 3, 5], 'mountford') == pytest.approx(4 / 6)
    assert dif([1, 2, 3], [1, 3, 5], 'mountford') == pytest.approx(0.6)
    assert simdif([1, 2, 3], [1, 3, 5], ['mountford']) == {
        'mountford': pytest.approx(4 / 6)
    }
