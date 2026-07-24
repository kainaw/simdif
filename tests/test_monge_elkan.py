import pytest
from simdif.metrics.monge_elkan import sim_monge_elkan
from simdif import monge_elkan, sim, simdif

def test_monge_elkan():
    # Edge case: either side without tokens => 0.0.
    assert sim_monge_elkan("", "") == 0.0
    assert sim_monge_elkan("hello", "") == 0.0
    # Identical strings: every token's best match is 1.0 (default jaro_winkler).
    assert sim_monge_elkan("foo bar", "foo bar") == pytest.approx(1.0)
    # Known value with jaccard inner metric:
    # jaccard({a,b,c},{a,b,d}) = 2/4 = 0.5, single token => average 0.5.
    assert sim_monge_elkan("abc", "abd", method="jaccard") == pytest.approx(0.5)
    # Averaging over two tokens of A:
    # "abc" vs "abd" = 2/4 = 0.5 ; "ab" vs "abd" = 2/3.
    assert sim_monge_elkan("abc ab", "abd", method="jaccard") == pytest.approx((0.5 + 2/3) / 2)
    # Asymmetry: ME(A,B) may differ from ME(B,A).
    assert sim_monge_elkan("foo", "foo bar") == pytest.approx(1.0)
    assert sim_monge_elkan("foo bar", "foo") < 1.0
    assert sim_monge_elkan("foo", "foo bar") != sim_monge_elkan("foo bar", "foo")
    # Convenience name (default 'sim' role) and dispatchers.
    assert monge_elkan("foo bar", "foo bar") == pytest.approx(1.0)
    assert sim("abc", "abd", "monge_elkan", method="jaccard") == pytest.approx(0.5)
    assert simdif("abc", "abd", ["monge_elkan"], method="jaccard") == {
        "monge_elkan": pytest.approx(0.5)
    }
