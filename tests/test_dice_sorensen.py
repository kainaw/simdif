import pytest
from simdif.metrics.dice_sorensen import sim_dice_sorensen, dif_dice_sorensen
from simdif import dice_sorensen, sim, simdif

def test_dice_sorensen():
    assert sim_dice_sorensen([], []) == 1.0
    assert sim_dice_sorensen([1,2,3],[1,3,5]) == pytest.approx(2/3)
    assert sim_dice_sorensen([1,2,3],[]) == pytest.approx(0.0)
    assert dif_dice_sorensen([1,2,3],[1,3,5]) == pytest.approx(1/3)
    assert dice_sorensen([1,2,3],[1,3,5]) == pytest.approx(2/3)
    assert sim([1,2,3],[1,3,5],'dice_sorensen') == pytest.approx(2/3)
    assert sim([1,2,3],[1,3,5],'ochiai') == pytest.approx(2/3)
    assert simdif([1,2,3],[1,3,5],['dice_sorensen','dice','sorensen_dice','sorensen']) == {
        'dice_sorensen': pytest.approx(2/3),
        'dice': pytest.approx(2/3),
        'sorensen_dice': pytest.approx(2/3),
        'sorensen': pytest.approx(2/3),
    }
