import pytest
from simdif.metrics.hamming import dist_hamming
from simdif import hamming, dist, simdif

def test_hamming_basic():
    assert dist_hamming([], []) == pytest.approx(0.0)
    assert dist_hamming(["12345"], ["12445"]) == pytest.approx(1.0)
    assert dist_hamming(7, 3, binary=True) == pytest.approx(1.0)
    assert dist_hamming("abcde", "abcd", pad_value="e") == pytest.approx(0.0)
    with pytest.raises(ValueError, match="Vector length mismatch"):
        dist_hamming([1, 2, 3], [1, 2], pad_value=None)
    assert hamming("abcde", "abxde") == pytest.approx(1.0)
    assert dist("abcde", "abxde", 'hamming') == pytest.approx(1.0)
    assert simdif("abcde", "abxde", ['hamming']) == {'hamming': pytest.approx(1.0)}
