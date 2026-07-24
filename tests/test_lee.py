import pytest
from simdif.metrics.lee import dist_lee, sim_lee
from simdif import lee, dist, simdif

def test_lee_basic():
    # Identical vectors -> distance 0 (q auto = max + 1).
    assert dist_lee([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)
    # q=4: per position min(|a-b|, q-|a-b|):
    #   |0-0|=0 -> 0; |1-0|=1 -> min(1,3)=1; |2-0|=2 -> min(2,2)=2; |3-0|=3 -> min(3,1)=1
    #   sum = 0 + 1 + 2 + 1 = 4  (note the wrap-around on the last term)
    assert dist_lee([0, 1, 2, 3], [0, 0, 0, 0], q=4) == pytest.approx(4.0)
    # q auto = max(3,3)+1 = 4: |1-3|=2->2, |2-2|=0->0, |3-1|=2->2 => 4
    assert dist_lee([1, 2, 3], [3, 2, 1]) == pytest.approx(4.0)
    # sim = 1 / (1 + d) = 1 / 5
    assert sim_lee([0, 1, 2, 3], [0, 0, 0, 0], q=4) == pytest.approx(0.2)
    # A length mismatch raises the standard message (like other vector metrics),
    # rather than silently truncating to the shorter vector.
    with pytest.raises(ValueError, match="Vector length mismatch"):
        dist_lee([1, 2, 3], [1, 2])
    # pad_value is honored: b padded to [1, 2, 0], q=4:
    #   |1-1|=0->0, |2-2|=0->0, |3-0|=3->min(3,1)=1  => 1
    assert dist_lee([1, 2, 3], [1, 2], q=4, pad_value="0") == pytest.approx(1.0)
    assert lee([1, 2, 3], [3, 2, 1]) == pytest.approx(4.0)
    assert dist([1, 2, 3], [3, 2, 1], 'lee') == pytest.approx(4.0)
    assert simdif([1, 2, 3], [3, 2, 1], ['lee']) == {'lee': pytest.approx(4.0)}
