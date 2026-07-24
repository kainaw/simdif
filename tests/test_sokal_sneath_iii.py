import pytest
from simdif.metrics.sokal_sneath_iii import sim_sokal_sneath_iii, dif_sokal_sneath_iii
from simdif import sokal_sneath_iii, sim, dif, simdif

def test_sokal_sneath_iii():
    # SS3 = (n11 + n00) / (n10 + n01); no n_universe -> n00 = 0
    # Identical sets: mismatches = 0 -> inf
    assert sim_sokal_sneath_iii([1, 2], [1, 2]) == float('inf')
    # n11=2, n00=0, n10=1, n01=1 -> (2+0)/(1+1) = 1.0
    assert sim_sokal_sneath_iii([1, 2, 3], [1, 3, 5]) == pytest.approx(1.0)
    # n11=3, n10=2, n01=0 -> (3+0)/(2+0) = 1.5
    assert sim_sokal_sneath_iii([1, 2, 3, 4, 5], [1, 2, 3]) == pytest.approx(1.5)
    # No shared elements: n11=0, n00=0 -> matches=0
    # sim: mismatches=4 -> 0/4 = 0.0
    assert sim_sokal_sneath_iii([1, 2], [3, 4]) == pytest.approx(0.0)
    # dif = (n10+n01)/(n11+n00); matches=0 -> inf
    assert dif_sokal_sneath_iii([1, 2], [3, 4]) == float('inf')
    # dif for [1,2,3]/[1,3,5]: (1+1)/(2+0) = 1.0
    assert dif_sokal_sneath_iii([1, 2, 3], [1, 3, 5]) == pytest.approx(1.0)
    # Convenience name (default role sim)
    assert sokal_sneath_iii([1, 2, 3, 4, 5], [1, 2, 3]) == pytest.approx(1.5)
    assert sim([1, 2, 3, 4, 5], [1, 2, 3], 'sokal_sneath_iii') == pytest.approx(1.5)
    assert dif([1, 2, 3], [1, 3, 5], 'sokal_sneath_iii') == pytest.approx(1.0)
    # simdif list/dict form
    assert simdif([1, 2, 3, 4, 5], [1, 2, 3], ['sokal_sneath_iii']) == {
        'sokal_sneath_iii': pytest.approx(1.5)
    }
