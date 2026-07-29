import pytest
from simdif.metrics.kulczynski_i import score_kulczynski_i, dist_kulczynski_i
from simdif import kulczynski_i, simdif, score, dist, sim, dif

def test_kulczynski_i():
    # Empty sets: b+c == 0 and a == 0 -> 0.0
    assert score_kulczynski_i([], []) == pytest.approx(0.0)
    # A={1,2,3,4}, B={1,2,5} -> a=2, b=2, c=1; K1 = a/(b+c) = 2/3
    assert score_kulczynski_i([1,2,3,4],[1,2,5]) == pytest.approx(2/3)
    # Disjoint: a=0 -> 0/(b+c) = 0.0
    assert score_kulczynski_i([1,2],[3,4]) == pytest.approx(0.0)
    # Identical sets: no mismatches (b+c=0), a>0 -> unbounded -> inf
    assert score_kulczynski_i([1,2,3],[1,2,3]) == float('inf')
    # convenience name resolves to the default (score) role
    assert kulczynski_i([1,2,3,4],[1,2,5]) == pytest.approx(2/3)
    assert simdif([1,2,3,4],[1,2,5],['kulczynski_i']) == {
        'kulczynski_i': pytest.approx(2/3)
    }
    assert score([1,2,3,4],[1,2,5],'kulczynski_i') == pytest.approx(2/3)
    # dist is the reciprocal of score: (b+c)/a
    assert dist_kulczynski_i([], []) == float('inf')
    assert dist_kulczynski_i([1,2,3,4],[1,2,5]) == pytest.approx(3/2)
    assert dist_kulczynski_i([1,2],[3,4]) == float('inf')
    assert dist_kulczynski_i([1,2,3],[1,2,3]) == pytest.approx(0.0)
    assert dist([1,2,3,4],[1,2,5],'kulczynski_i') == pytest.approx(3/2)
    # no sim or dif role: score has no known maximum to normalize against
    with pytest.raises(ValueError):
        sim([1,2,3,4],[1,2,5],'kulczynski_i')
    with pytest.raises(ValueError):
        dif([1,2,3,4],[1,2,5],'kulczynski_i')
