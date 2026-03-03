import pytest
from simdif.metrics.index_of_dissimilarity import dif_index_of_dissimilarity
from simdif import index_of_dissimilarity, dif, simdif

def test_index_of_dissimilarity_basic():
    with pytest.raises(ValueError, match="Input lists must have a non-zero sum for normalization."):
        dif_index_of_dissimilarity([], [])
        dif_index_of_dissimilarity([0,0,0], [0,0,0])
    assert dif_index_of_dissimilarity([0, 2], [2, 0]) == pytest.approx(1.0)
    assert dif_index_of_dissimilarity([2, 2], [4, 4]) == pytest.approx(0.0)
    assert dif_index_of_dissimilarity([-2, -2], [2, 2]) == pytest.approx(0.0)
    assert dif_index_of_dissimilarity([0, 2], [2], pad_value="0") == pytest.approx(1.0)
    with pytest.raises(ValueError, match="Vector length mismatch"):
        dif_index_of_dissimilarity([1, 2, 3], [1, 2], pad_value=None)
    assert dif_index_of_dissimilarity([0, 2], [2, 0]) == pytest.approx(1.0)
    assert index_of_dissimilarity([0, 2], [2, 0]) == pytest.approx(1.0)
    assert dif([0, 2], [2, 0], 'index_of_dissimilarity') == pytest.approx(1.0)
    assert simdif([0,2], [2,0], ['index_of_dissimilarity','hoover','duncan']) == {
        'index_of_dissimilarity': pytest.approx(1.0),
        'hoover': pytest.approx(1.0),
        'duncan': pytest.approx(1.0)
    }
