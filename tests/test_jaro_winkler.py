import pytest
from simdif.metrics.jaro_winkler import sim_jaro_winkler, dif_jaro_winkler
from simdif import jaro_winkler, sim, simdif

def test_jaro_winkler():
    # Edge cases.
    assert sim_jaro_winkler("", "") == 1.0
    assert sim_jaro_winkler("abc", "abc") == 1.0
    # "cat"/"cot": base jaro=7/9 (>0.7), prefix l=1 -> 7/9 + 1*0.1*(1 - 7/9) = 0.8.
    assert sim_jaro_winkler("cat", "cot") == pytest.approx(0.8)
    # "MARTHA"/"MARHTA": base jaro=17/18, prefix l=3 -> 17/18 + 3*0.1*(1 - 17/18).
    assert sim_jaro_winkler("MARTHA", "MARHTA") == pytest.approx(17/18 + 0.3 * (1 - 17/18))
    # p above 0.25 must raise.
    with pytest.raises(ValueError):
        sim_jaro_winkler("cat", "cot", p=0.3)
    # Difference role.
    assert dif_jaro_winkler("cat", "cot") == pytest.approx(1 - 0.8)
    # Convenience name (default role is 'sim').
    assert jaro_winkler("cat", "cot") == pytest.approx(0.8)
    assert sim("cat", "cot", "jaro_winkler") == pytest.approx(0.8)
    assert simdif("cat", "cot", ["jaro_winkler"]) == {"jaro_winkler": pytest.approx(0.8)}
