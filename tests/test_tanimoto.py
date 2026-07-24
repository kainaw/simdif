import pytest
from simdif.metrics.tanimoto import sim_tanimoto, dif_tanimoto
from simdif import tanimoto, sim, dif, simdif

def test_tanimoto():
    # This is the binary / bit-vector Tanimoto (no distribution normalization).
    # Two empty vectors are defined as 1.0.
    assert sim_tanimoto([], []) == pytest.approx(1.0)
    # Identical bit patterns -> 1.0.
    assert sim_tanimoto([1, 0, 1], [1, 0, 1]) == pytest.approx(1.0)
    # Known value: c=1 shared 1-bit, a=2, b=2 -> 1 / (2 + 2 - 1) = 1/3.
    assert sim_tanimoto([1, 1, 0], [1, 0, 1]) == pytest.approx(1 / 3)
    assert dif_tanimoto([1, 1, 0], [1, 0, 1]) == pytest.approx(1 - 1 / 3)
    # No shared 1-bits -> 0.
    assert sim_tanimoto([1, 0], [0, 1]) == pytest.approx(0.0)
    # binary=True interprets ints as bitmasks: 0b110 vs 0b101 == [1,1,0] vs [1,0,1].
    assert sim_tanimoto(0b110, 0b101, binary=True) == pytest.approx(1 / 3)
    with pytest.raises(ValueError, match="Vector length mismatch"):
        sim_tanimoto([1, 1, 0], [1, 0])
    # Convenience name (default role is 'sim') + role dispatchers + simdif dict form.
    assert tanimoto([1, 1, 0], [1, 0, 1]) == pytest.approx(1 / 3)
    assert sim([1, 1, 0], [1, 0, 1], 'tanimoto') == pytest.approx(1 / 3)
    assert dif([1, 1, 0], [1, 0, 1], 'tanimoto') == pytest.approx(1 - 1 / 3)
    assert simdif([1, 1, 0], [1, 0, 1], ['tanimoto']) == {'tanimoto': pytest.approx(1 / 3)}
