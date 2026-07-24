import pytest
from simdif.metrics.phi import sim_phi
from simdif import phi, simdif, dif

def test_phi():
    # Empty sets: denom == 0 -> 0.0
    assert sim_phi([], []) == pytest.approx(0.0)
    # A={1,2,3}, B={1,3,5}, n_universe=10 -> a=2,b=1,c=1,d=6
    # phi = (a*d - b*c)/sqrt((a+b)(a+c)(b+d)(c+d))
    #     = (12-1)/sqrt(3*3*7*7) = 11/21
    assert sim_phi([1,2,3],[1,3,5], n_universe=10) == pytest.approx(11/21)
    # Identical sets with universe -> perfect agreement +1
    assert sim_phi([1,2,3],[1,2,3], n_universe=10) == pytest.approx(1.0)
    # Disjoint sets, n_universe=10 -> a=0,b=2,c=2,d=6
    # phi = (0-4)/sqrt(2*2*8*8) = -4/16 = -0.25
    assert sim_phi([1,2],[3,4], n_universe=10) == pytest.approx(-0.25)
    # convenience name resolves to the default (sim) role
    assert phi([1,2,3],[1,3,5], n_universe=10) == pytest.approx(11/21)
    assert simdif([1,2,3],[1,3,5],['phi'], n_universe=10) == {
        'phi': pytest.approx(11/21)
    }
    # alias 'mcc' resolves to the same (Matthews Correlation Coefficient)
    assert simdif([1,2,3],[1,3,5],['mcc'], n_universe=10) == {
        'mcc': pytest.approx(11/21)
    }
    # sim-only metric: there is no difference role
    with pytest.raises(ValueError):
        dif([1,2,3],[1,3,5],'phi', n_universe=10)
