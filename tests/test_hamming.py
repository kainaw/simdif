import pytest
from simdif.metrics.hamming import dist_hamming, dif_hamming, sim_hamming
from simdif import hamming, dist, dif, sim, simdif

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


def test_hamming_dif_sim():
    # dif = mismatches / length; sim = 1 - dif.
    assert dif_hamming("abcde", "abxde") == pytest.approx(1 / 5)
    assert sim_hamming("abcde", "abxde") == pytest.approx(4 / 5)
    assert dif_hamming("abcde", "abcde") == pytest.approx(0.0)
    assert sim_hamming("abcde", "abcde") == pytest.approx(1.0)
    # Empty inputs -> no differences.
    assert dif_hamming([], []) == pytest.approx(0.0)
    # binary=True normalizes by bit width: 7=111 vs 3=011 -> 1 diff / 3 bits.
    assert dif_hamming(7, 3, binary=True) == pytest.approx(1 / 3)
    # pad_value participates in the denominator: 'abcd' padded to 'abcde',
    # 0 mismatches over 5 positions.
    assert dif_hamming("abcde", "abcd", pad_value="e") == pytest.approx(0.0)
    # Dispatch forms for the previously-broken roles.
    assert dif("abcde", "abxde", 'hamming') == pytest.approx(1 / 5)
    assert sim("abcde", "abxde", 'hamming') == pytest.approx(4 / 5)
