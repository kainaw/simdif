import pytest
from simdif.metrics.tanimoto_continuous import (
    sim_tanimoto_continuous, dif_tanimoto_continuous,
)
from simdif.metrics.tanimoto import sim_tanimoto
from simdif import tanimoto_continuous, sim, dif, simdif


def test_tanimoto_continuous():
    # Identical vectors -> 1.0.
    assert sim_tanimoto_continuous([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)

    # A.B=28, ||A||^2=14, ||B||^2=56 -> 28/(14+56-28) = 28/42 = 2/3.
    # NOT scale-invariant: B = 2A still scores < 1 (unlike cosine).
    assert sim_tanimoto_continuous([1, 2, 3], [2, 4, 6]) == pytest.approx(2 / 3)

    # Orthogonal non-negative vectors -> 0.
    assert sim_tanimoto_continuous([1, 0], [0, 1]) == pytest.approx(0.0)

    # Both all-zero vectors -> defined as 1.0 (identical).
    assert sim_tanimoto_continuous([0, 0], [0, 0]) == pytest.approx(1.0)

    # dif = 1 - sim.
    assert dif_tanimoto_continuous([1, 2, 3], [2, 4, 6]) == pytest.approx(1 / 3)

    # Reduces EXACTLY to binary Tanimoto on 0/1 input.
    for a, b in ([[1, 0, 1, 1], [1, 1, 0, 1]], [[1, 1, 0], [1, 0, 1]]):
        assert sim_tanimoto_continuous(a, b) == pytest.approx(sim_tanimoto(a, b))

    # Length mismatch raises unless a pad_value is supplied.
    with pytest.raises(ValueError, match="Vector length mismatch"):
        sim_tanimoto_continuous([1, 2, 3], [1, 2])

    # Convenience name + role dispatch + aliases + simdif forwarding.
    assert tanimoto_continuous([1, 2, 3], [2, 4, 6]) == pytest.approx(2 / 3)
    assert sim([1, 2, 3], [2, 4, 6], 'tanimoto_continuous') == pytest.approx(2 / 3)
    assert dif([1, 2, 3], [2, 4, 6], 'extended_tanimoto') == pytest.approx(1 / 3)
    assert simdif([1, 2, 3], [2, 4, 6], ['continuous_tanimoto']) == {
        'continuous_tanimoto': pytest.approx(2 / 3)}
