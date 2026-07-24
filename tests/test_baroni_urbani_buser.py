import math
import pytest
from simdif.metrics.baroni_urbani_buser import (
    sim_baroni_urbani_buser,
    dif_baroni_urbani_buser,
)
from simdif import baroni_urbani_buser, sim, dif, simdif


def test_baroni_urbani_buser():
    # Two empty sets -> identical -> 1.0
    assert sim_baroni_urbani_buser([], []) == 1.0

    # No n_universe -> n00 = 0 -> reduces to Jaccard.
    # n11=2, n10=1, n01=1 -> (0 + 2) / (0 + 2 + 1 + 1) = 0.5
    assert sim_baroni_urbani_buser([1, 2, 3], [1, 3, 5]) == pytest.approx(0.5)

    # With n_universe=10 -> n00 = 10 - |A U B| = 10 - 4 = 6.
    # g = sqrt(2*6) = sqrt(12); (g + 2) / (g + 2 + 1 + 1)
    g = math.sqrt(12)
    expected = (g + 2) / (g + 4)
    assert sim_baroni_urbani_buser([1, 2, 3], [1, 3, 5], n_universe=10) == pytest.approx(expected)

    # dif = 1 - sim
    assert dif_baroni_urbani_buser([1, 2, 3], [1, 3, 5]) == pytest.approx(0.5)
    assert dif_baroni_urbani_buser([1, 2, 3], [1, 3, 5], n_universe=10) == pytest.approx(1 - expected)

    # Identical non-empty sets -> 1.0 regardless of universe
    assert sim_baroni_urbani_buser([1, 2, 3], [1, 2, 3], n_universe=10) == pytest.approx(1.0)

    # Disjoint sets, no universe -> 0.0
    assert sim_baroni_urbani_buser([1, 2], [3, 4]) == pytest.approx(0.0)

    # Convenience name + role dispatch + aliases + simdif forwarding
    assert baroni_urbani_buser([1, 2, 3], [1, 3, 5]) == pytest.approx(0.5)
    assert sim([1, 2, 3], [1, 3, 5], 'baroni_urbani_buser') == pytest.approx(0.5)
    assert sim([1, 2, 3], [1, 3, 5], 'bub') == pytest.approx(0.5)
    assert dif([1, 2, 3], [1, 3, 5], 'baroni_urbani') == pytest.approx(0.5)
    assert simdif([1, 2, 3], [1, 3, 5], ['bub'], n_universe=10) == {
        'bub': pytest.approx(expected)
    }
