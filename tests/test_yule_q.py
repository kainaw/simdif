import pytest
from simdif.metrics.yule_q import sim_yule_q
from simdif import yule_q, simdif, dif

def test_yule_q():
    # Empty sets: ad+bc == 0 -> 0.0 (no universe)
    assert sim_yule_q([], []) == pytest.approx(0.0)
    # A={1,2,3}, B={1,3,5}, n_universe=10 -> a=2,b=1,c=1,d=6
    # Q = (a*d - b*c)/(a*d + b*c) = (12-1)/(12+1) = 11/13
    assert sim_yule_q([1,2,3],[1,3,5], n_universe=10) == pytest.approx(11/13)
    # Identical sets with universe -> perfect positive association +1
    assert sim_yule_q([1,2,3],[1,2,3], n_universe=10) == pytest.approx(1.0)
    # Disjoint sets with universe -> perfect negative association -1
    assert sim_yule_q([1,2],[3,4], n_universe=10) == pytest.approx(-1.0)
    # convenience name resolves to the default (sim) role
    assert yule_q([1,2,3],[1,3,5], n_universe=10) == pytest.approx(11/13)
    assert simdif([1,2,3],[1,3,5],['yule_q'], n_universe=10) == {
        'yule_q': pytest.approx(11/13)
    }
    # sim-only metric: there is no difference role
    with pytest.raises(ValueError):
        dif([1,2,3],[1,3,5],'yule_q', n_universe=10)
