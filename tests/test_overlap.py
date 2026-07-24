import pytest
from simdif.metrics.overlap import sim_overlap, dif_overlap
from simdif import overlap, sim, dif, simdif

def test_overlap():
    # Edge cases
    assert sim_overlap([], []) == 1.0
    assert sim_overlap([1, 2, 3], []) == pytest.approx(0.0)
    # Known values: O = i / min(|A|, |B|)
    # i=2, min(3,3)=3 -> 2/3
    assert sim_overlap([1, 2, 3], [1, 3, 5]) == pytest.approx(2/3)
    # Subset: i=2, min(4,2)=2 -> 1.0
    assert sim_overlap([1, 2, 3, 4], [2, 3]) == pytest.approx(1.0)
    # dif = 1 - sim
    assert dif_overlap([1, 2, 3], [1, 3, 5]) == pytest.approx(1/3)
    # Convenience name (default role sim)
    assert overlap([1, 2, 3], [1, 3, 5]) == pytest.approx(2/3)
    assert sim([1, 2, 3], [1, 3, 5], 'overlap') == pytest.approx(2/3)
    assert dif([1, 2, 3], [1, 3, 5], 'overlap') == pytest.approx(1/3)
    # simdif list/dict form
    assert simdif([1, 2, 3], [1, 3, 5], ['overlap']) == {
        'overlap': pytest.approx(2/3)
    }
