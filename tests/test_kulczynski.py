import pytest
from simdif.metrics.kulczynski import sim_kulczynski, dif_kulczynski
from simdif import kulczynski, sim, dif, simdif

def test_kulczynski():
    # Edge cases
    assert sim_kulczynski([], []) == 1.0
    assert sim_kulczynski([1, 2, 3], []) == pytest.approx(0.0)
    # Known values: K2 = (i/2)*(1/|A| + 1/|B|)
    # i=2, |A|=|B|=3 -> (2/2)*(1/3+1/3) = 2/3
    assert sim_kulczynski([1, 2, 3], [1, 3, 5]) == pytest.approx(2/3)
    # i=2, |A|=4, |B|=2 -> (2/2)*(1/4+1/2) = 0.75
    assert sim_kulczynski([1, 2, 3, 4], [3, 4]) == pytest.approx(0.75)
    # dif = 1 - sim
    assert dif_kulczynski([1, 2, 3], [1, 3, 5]) == pytest.approx(1/3)
    # Convenience name (default role sim)
    assert kulczynski([1, 2, 3], [1, 3, 5]) == pytest.approx(2/3)
    assert sim([1, 2, 3], [1, 3, 5], 'kulczynski') == pytest.approx(2/3)
    assert dif([1, 2, 3], [1, 3, 5], 'kulczynski') == pytest.approx(1/3)
    # simdif list/dict form
    assert simdif([1, 2, 3], [1, 3, 5], ['kulczynski']) == {
        'kulczynski': pytest.approx(2/3)
    }
