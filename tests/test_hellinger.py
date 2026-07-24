import math
import pytest
from simdif.metrics.hellinger import dist_hellinger, sim_hellinger
from simdif import hellinger, dist, sim, simdif

def test_hellinger():
    # Identical distributions -> distance 0, similarity 1.
    assert dist_hellinger([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)
    assert sim_hellinger([0.5, 0.5], [0.5, 0.5]) == pytest.approx(1.0)
    # Disjoint support: sum_sq = 1 + 1 = 2; H = sqrt(2)/sqrt(2) = 1 (max).
    assert dist_hellinger([1, 0], [0, 1]) == pytest.approx(1.0)
    # Known value: sum_sq = (sqrt(0.5)-1)^2 + (sqrt(0.5)-0)^2; H = sqrt(sum_sq)/sqrt(2).
    sum_sq = (math.sqrt(0.5) - 1) ** 2 + (math.sqrt(0.5)) ** 2
    expected = math.sqrt(sum_sq) / math.sqrt(2)
    assert dist_hellinger([0.5, 0.5], [1, 0]) == pytest.approx(expected)
    assert sim_hellinger([0.5, 0.5], [1, 0]) == pytest.approx(1.0 - expected)
    # Convenience name (default role is 'dist') + role dispatchers + simdif dict form.
    assert hellinger([0.5, 0.5], [1, 0]) == pytest.approx(expected)
    assert dist([0.5, 0.5], [1, 0], 'hellinger') == pytest.approx(expected)
    assert sim([0.5, 0.5], [1, 0], 'hellinger') == pytest.approx(1.0 - expected)
    assert simdif([1, 0], [0, 1], ['hellinger']) == {'hellinger': pytest.approx(1.0)}
