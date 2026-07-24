import math
import pytest
from simdif.metrics.jukes_cantor import dist_jukes_cantor, sim_jukes_cantor
from simdif import jukes_cantor, simdif


def test_jukes_cantor():
    assert dist_jukes_cantor("ATGC", "ATGC") == pytest.approx(0.0)
    # p = 0.5, k = 4: d = -(3/4) ln(1 - (4/3)*0.5)
    expected = -(3 / 4) * math.log(1 - (4 / 3) * 0.5)
    assert dist_jukes_cantor("AAAA", "AAGG") == pytest.approx(expected)
    assert sim_jukes_cantor("AAAA", "AAGG") == pytest.approx(1 / (1 + expected))
    # k is a parameter: same p=0.25, different k -> different distance.
    p = 0.25  # "AAAA" vs "AAAG"
    assert dist_jukes_cantor("AAAA", "AAAG", k=4) == pytest.approx(-(3 / 4) * math.log(1 - (4 / 3) * p))
    assert dist_jukes_cantor("AAAA", "AAAG", k=2) == pytest.approx(-(1 / 2) * math.log(1 - 2 * p))
    # Saturation: p = 1.0 (>= 3/4) -> inf.
    assert math.isinf(dist_jukes_cantor("AAAA", "GGGG"))
    assert sim_jukes_cantor("AAAA", "GGGG") == pytest.approx(0.0)
    # Dispatch + alias.
    assert jukes_cantor("AAAA", "AAGG") == pytest.approx(expected)
    assert simdif("AAAA", "AAGG", ['jc69']) == {'jc69': pytest.approx(expected)}
