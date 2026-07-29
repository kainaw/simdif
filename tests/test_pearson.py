import pytest
from simdif.metrics.pearson import sim_pearson, dist_pearson, dif_pearson
from simdif import pearson, sim, dif, simdif

def test_pearson_basic():
    # Perfect positive linear correlation -> 1.
    assert sim_pearson([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    # Perfect negative linear correlation -> -1.
    assert sim_pearson([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    # means both 2.5; centered A=[-1.5,-0.5,0.5,1.5], B=[-1.5,0.5,-0.5,1.5]
    # num = 2.25 - 0.25 - 0.25 + 2.25 = 4.0; denom = sqrt(5)*sqrt(5) = 5 -> r = 0.8
    assert sim_pearson([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(0.8)
    assert dist_pearson([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(0.2)
    assert dif_pearson([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(-0.8)
    assert dif([1, 2, 3, 4], [1, 3, 2, 4], 'pearson') == pytest.approx(-0.8)
    # Zero variance in one vector -> 0.0 by definition here.
    assert sim_pearson([5, 5, 5], [1, 2, 3]) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="Pearson requires at least 2 elements"):
        sim_pearson([1], [1])
    with pytest.raises(ValueError, match="Vector length mismatch"):
        sim_pearson([1, 2, 3], [1, 2])
    assert pearson([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(0.8)
    assert sim([1, 2, 3, 4], [1, 3, 2, 4], 'pearson') == pytest.approx(0.8)
    assert simdif([1, 2, 3, 4], [1, 3, 2, 4], ['pearson']) == {'pearson': pytest.approx(0.8)}


def test_pearson_optimized_lib(optimized_lib):
    optimized_lib('scipy')
    assert sim_pearson([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(0.8)
    # scipy's pearsonr returns nan on a constant input; the zero-variance
    # check runs before the library gate so both paths agree on 0.0.
    assert sim_pearson([5, 5, 5], [1, 2, 3]) == pytest.approx(0.0)
