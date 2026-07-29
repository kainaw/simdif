import pytest
from simdif.metrics.mcconnaughey import sim_mcconnaughey
from simdif import mcconnaughey, simdif, dif

def test_mcconnaughey():
    # Empty sets: denom == 0 -> association undefined -> 0.0
    assert sim_mcconnaughey([], []) == pytest.approx(0.0)
    # A={1,2,3}, B={1,3,5} -> a=2,b=1,c=1
    # M = (a^2 - b*c)/((a+b)(a+c)) = (4-1)/(3*3) = 3/9 = 1/3
    assert sim_mcconnaughey([1,2,3],[1,3,5]) == pytest.approx(1/3)
    # Identical sets -> 1.0
    assert sim_mcconnaughey([1,2,3],[1,2,3]) == pytest.approx(1.0)
    # Disjoint sets -> -1.0
    assert sim_mcconnaughey([1,2],[3,4]) == pytest.approx(-1.0)
    # convenience name resolves to the default (sim) role
    assert mcconnaughey([1,2,3],[1,3,5]) == pytest.approx(1/3)
    assert simdif([1,2,3],[1,3,5],['mcconnaughey']) == {
        'mcconnaughey': pytest.approx(1/3)
    }
    # difference is -1 * similarity, preserving the signed range
    assert dif([1,2,3],[1,3,5],'mcconnaughey') == pytest.approx(-1/3)
