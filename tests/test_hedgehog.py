import pytest
from simdif.metrics.hedgehog import dif_hedgehog, sim_hedgehog, dist_hedgehog
from simdif import hedgehog, sim, dif, dist, simdif

def test_hedgehog_basic():
    # No pairs -> d=0 -> dif=0, sim=1.
    assert dif_hedgehog([], []) == pytest.approx(0.0)
    assert sim_hedgehog([], []) == pytest.approx(1.0)
    # A=[1,2], B=[3]. Odd indices are negated.
    #   i=0(x=1): j=0(y=3) -> |1 - 3| = 2
    #   i=1(x=-2): j=0(y=3) -> |-2 - 3| = 5
    #   d = 7; dif = 7/(1+7) = 0.875; sim = 1 - 0.875 = 0.125
    assert dif_hedgehog([1, 2], [3]) == pytest.approx(0.875)
    assert sim_hedgehog([1, 2], [3]) == pytest.approx(0.125)
    # Distance is always 0.0 -- hedgehogs always go together perfectly.
    assert dist_hedgehog([1, 2], [3]) == pytest.approx(0.0)
    assert dist_hedgehog([9, 9, 9], [1]) == pytest.approx(0.0)
    assert hedgehog([1, 2], [3]) == pytest.approx(0.125)
    assert sim([1, 2], [3], 'hedgehog') == pytest.approx(0.125)
    assert dif([1, 2], [3], 'hedgehog') == pytest.approx(0.875)
    assert dist([1, 2], [3], 'hedgehog') == pytest.approx(0.0)
    assert simdif([1, 2], [3], ['hedgehog']) == {'hedgehog': pytest.approx(0.125)}
