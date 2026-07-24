import pytest
from simdif.metrics.tversky import sim_tversky, dif_tversky
from simdif import tversky, sim, dif, simdif

def test_tversky():
    # Edge case
    assert sim_tversky([], []) == 1.0
    # T = i / (i + a*|A-B| + b*|B-A|)
    # alpha=beta=1 -> Jaccard: i=2, |A-B|=1, |B-A|=1 -> 2/(2+1+1) = 0.5
    assert sim_tversky([1, 2, 3], [1, 3, 5], alpha=1, beta=1) == pytest.approx(0.5)
    # Default alpha=beta=0.5 -> Dice: 2/(2+0.5+0.5) = 2/3
    assert sim_tversky([1, 2, 3], [1, 3, 5]) == pytest.approx(2/3)
    # Asymmetric alpha=1, beta=0: 2/(2 + 1*1 + 0*1) = 2/3
    assert sim_tversky([1, 2, 3], [1, 3, 5], alpha=1, beta=0) == pytest.approx(2/3)
    # alpha and beta cannot both be 0
    with pytest.raises(ValueError, match="both be 0"):
        sim_tversky([1, 2, 3], [1, 3, 5], alpha=0, beta=0)
    # dif = 1 - sim (default Dice) -> 1 - 2/3 = 1/3
    assert dif_tversky([1, 2, 3], [1, 3, 5]) == pytest.approx(1/3)
    # Convenience name (default role sim, default Dice weights)
    assert tversky([1, 2, 3], [1, 3, 5]) == pytest.approx(2/3)
    assert sim([1, 2, 3], [1, 3, 5], 'tversky') == pytest.approx(2/3)
    assert dif([1, 2, 3], [1, 3, 5], 'tversky') == pytest.approx(1/3)
    # simdif list/dict form, with kwargs forwarded (alpha=beta=1 -> Jaccard)
    assert simdif([1, 2, 3], [1, 3, 5], ['tversky'], alpha=1, beta=1) == {
        'tversky': pytest.approx(0.5)
    }
