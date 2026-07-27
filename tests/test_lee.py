import pytest
from simdif.metrics.lee import dist_lee, dif_lee, sim_lee
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
    # dif = dist / (floor(q/2) * n) = 4 / (2 * 4) = 0.5, and sim = 1 - dif.
    assert dif_lee([0, 1, 2, 3], [0, 0, 0, 0], q=4) == pytest.approx(0.5)
    assert sim_lee([0, 1, 2, 3], [0, 0, 0, 0], q=4) == pytest.approx(0.5)
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


def test_lee_bound_is_tight():
    # One position tops out at floor(q/2): past the half-way point of the
    # circle the wrap-around direction is the shorter one. q=16 -> 8 per
    # position, so [0,0] vs [8,8] is maximally distant.
    assert dist_lee([0, 0], [8, 8], q=16) == pytest.approx(16.0)
    assert dif_lee([0, 0], [8, 8], q=16) == pytest.approx(1.0)
    assert sim_lee([0, 0], [8, 8], q=16) == pytest.approx(0.0)
    # Odd q floors: q=5 -> 2 per position, and |0-2| = 2 attains it.
    assert dif_lee([0], [2], q=5) == pytest.approx(1.0)
    assert dif_lee([1, 2], [1, 2], q=16) == pytest.approx(0.0)


def test_lee_inferred_q_scales_dif():
    # The default q is one past the largest symbol *present*, not the alphabet
    # the caller had in mind. dist barely notices; dif is scaled by it directly,
    # which is why info_lee tells you to pass q when the data may not exercise
    # the whole alphabet.
    assert dist_lee([1], [2]) == pytest.approx(1.0)        # inferred q = 3
    assert dist_lee([1], [2], q=16) == pytest.approx(1.0)  # same distance
    assert dif_lee([1], [2]) == pytest.approx(1.0)         # max = floor(3/2)*1 = 1
    assert dif_lee([1], [2], q=16) == pytest.approx(0.125)  # max = 8


def test_lee_sim_dif_complementary():
    cases = (([0, 1, 2, 3], [0, 0, 0, 0], 4), ([1, 2, 3], [3, 2, 1], None),
             ([0, 0], [8, 8], 16), ([1, 2], [1, 2], 16))
    for a, b, q in cases:
        assert sim_lee(a, b, q=q) + dif_lee(a, b, q=q) == pytest.approx(1.0)
