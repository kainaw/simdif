import pytest
from simdif.metrics.jaro import sim_jaro, dif_jaro, explain_jaro
from simdif import jaro, sim, simdif

def test_jaro():
    # Edge cases.
    assert sim_jaro("", "") == 1.0
    assert sim_jaro("abc", "") == 0.0
    assert sim_jaro("abc", "abc") == 1.0
    # "cat"/"cot": window=0, m=2 (c,t), t=0 -> (2/3 + 2/3 + 2/2)/3 = 7/9.
    assert sim_jaro("cat", "cot") == pytest.approx(7/9)
    # Classic "MARTHA"/"MARHTA": m=6, one transposition -> 17/18.
    assert sim_jaro("MARTHA", "MARHTA") == pytest.approx(17/18)
    # Difference role.
    assert dif_jaro("cat", "cot") == pytest.approx(1 - 7/9)
    # Convenience name (default role is 'sim').
    assert jaro("cat", "cot") == pytest.approx(7/9)
    assert sim("cat", "cot", "jaro") == pytest.approx(7/9)
    assert simdif("cat", "cot", ["jaro"]) == {"jaro": pytest.approx(7/9)}


def test_explain_jaro_window():
    # explain replays the match logic; its reported similarity must agree with
    # sim_jaro, and it must actually render the sliding window legend/markers.
    out = explain_jaro("MARTHA", "MARHTA")
    assert "Match window radius w" in out
    assert "'-' in window, '#' matched" in out
    assert "0.9444" in out  # matches sim_jaro("MARTHA","MARHTA") = 17/18
    # A window that lands a match shows the '#' marker.
    assert "#" in out
    # Empty-input branch stays graceful.
    assert "empty" in explain_jaro("", "abc")
