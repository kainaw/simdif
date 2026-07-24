import math
import pytest
from simdif.metrics.bhattacharyya import dist_bhattacharyya, sim_bhattacharyya
from simdif import bhattacharyya, dist, sim, simdif

def test_bhattacharyya():
    # Identical distributions: coefficient (sim) = 1, distance = 0.
    assert sim_bhattacharyya([0.5, 0.5], [0.5, 0.5]) == pytest.approx(1.0)
    assert dist_bhattacharyya([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)
    # Disjoint support: BC = 0 -> coefficient 0, distance inf.
    assert sim_bhattacharyya([1, 0], [0, 1]) == pytest.approx(0.0)
    assert dist_bhattacharyya([1, 0], [0, 1]) == float('inf')
    # Known value: BC = sqrt(0.5*1) + sqrt(0.5*0) = sqrt(0.5); D = -ln(sqrt(0.5)) = 0.5*ln(2).
    assert sim_bhattacharyya([0.5, 0.5], [1, 0]) == pytest.approx(math.sqrt(0.5))
    assert dist_bhattacharyya([0.5, 0.5], [1, 0]) == pytest.approx(0.5 * math.log(2))
    # Convenience name (default role is 'dist') + role dispatchers + simdif dict form.
    assert bhattacharyya([0.5, 0.5], [1, 0]) == pytest.approx(0.5 * math.log(2))
    assert dist([0.5, 0.5], [1, 0], 'bhattacharyya') == pytest.approx(0.5 * math.log(2))
    assert sim([0.5, 0.5], [1, 0], 'bhattacharyya') == pytest.approx(math.sqrt(0.5))
    assert simdif([0.5, 0.5], [1, 0], ['bhattacharyya']) == {'bhattacharyya': pytest.approx(0.5 * math.log(2))}
