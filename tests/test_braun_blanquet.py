import pytest
from simdif.metrics.braun_blanquet import sim_braun_blanquet, dif_braun_blanquet
from simdif import braun_blanquet, simdif

def test_braun_blanquet():
    # Empty sets: denom == 0 -> identical
    assert sim_braun_blanquet([], []) == 1.0
    # A={1,2,3}, B={1,3,5} -> a=2, |A|=3, |B|=3; BB = a/max(|A|,|B|) = 2/3
    assert sim_braun_blanquet([1,2,3],[1,3,5]) == pytest.approx(2/3)
    # Different sizes: A has 4, B has 2, a=2 -> 2/max(4,2) = 2/4 = 0.5
    assert sim_braun_blanquet([1,2,3,4],[1,2]) == pytest.approx(0.5)
    # Disjoint sets -> 0.0
    assert sim_braun_blanquet([1,2],[3,4]) == pytest.approx(0.0)
    # dif = 1 - sim = 1/3
    assert dif_braun_blanquet([1,2,3],[1,3,5]) == pytest.approx(1/3)
    # convenience name resolves to the default (sim) role
    assert braun_blanquet([1,2,3],[1,3,5]) == pytest.approx(2/3)
    assert simdif([1,2,3],[1,3,5],['braun_blanquet']) == {
        'braun_blanquet': pytest.approx(2/3)
    }
