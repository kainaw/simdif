import pytest
from simdif.metrics.kulczynski_i import sim_kulczynski_i
from simdif import kulczynski_i, simdif, dif

def test_kulczynski_i():
    # Empty sets: b+c == 0 and a == 0 -> 0.0
    assert sim_kulczynski_i([], []) == pytest.approx(0.0)
    # A={1,2,3,4}, B={1,2,5} -> a=2, b=2, c=1; K1 = a/(b+c) = 2/3
    assert sim_kulczynski_i([1,2,3,4],[1,2,5]) == pytest.approx(2/3)
    # Disjoint: a=0 -> 0/(b+c) = 0.0
    assert sim_kulczynski_i([1,2],[3,4]) == pytest.approx(0.0)
    # Identical sets: no mismatches (b+c=0), a>0 -> unbounded -> inf
    assert sim_kulczynski_i([1,2,3],[1,2,3]) == float('inf')
    # convenience name resolves to the default (sim) role
    assert kulczynski_i([1,2,3,4],[1,2,5]) == pytest.approx(2/3)
    assert simdif([1,2,3,4],[1,2,5],['kulczynski_i']) == {
        'kulczynski_i': pytest.approx(2/3)
    }
    # sim-only metric: there is no difference role
    with pytest.raises(ValueError):
        dif([1,2,3,4],[1,2,5],'kulczynski_i')
