import pytest
from simdif.metrics.monge_elkan import sim_monge_elkan, dist_monge_elkan, dif_monge_elkan, score_monge_elkan
from simdif import monge_elkan, sim, simdif

def test_monge_elkan():
    # Edge case: either side without tokens => 0.0.
    assert sim_monge_elkan("", "") == 0.0
    assert sim_monge_elkan("hello", "") == 0.0
    # Identical strings: every token's best match is 1.0 (default method: levenshtein).
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


def test_monge_elkan_dist_role():
    # dist_monge_elkan takes the best (min) Levenshtein distance per token of A.
    assert dist_monge_elkan("", "") == 0.0
    assert dist_monge_elkan("foo bar", "foo bar") == pytest.approx(0.0)
    # "cat" -> best in {"cats", "dog"} is "cats" (edit distance 1).
    assert dist_monge_elkan("cat", "cats dog", method="levenshtein") == pytest.approx(1.0)


def test_monge_elkan_dif_role():
    # dif_monge_elkan uses levenshtein's dif role (1 - sim), best (min) per token.
    assert dif_monge_elkan("foo bar", "foo bar") == pytest.approx(0.0)
    assert dif_monge_elkan("foo", "foo bar") == pytest.approx(0.0)


def test_monge_elkan_combine():
    # Two tokens of A vs jaccard: per-token best scores are 0.5 and 2/3.
    scores = (0.5, 2 / 3)
    assert sim_monge_elkan("abc ab", "abd", method="jaccard", combine="avg") == pytest.approx(sum(scores) / 2)
    assert sim_monge_elkan("abc ab", "abd", method="jaccard", combine="sum") == pytest.approx(sum(scores))
    assert sim_monge_elkan("abc ab", "abd", method="jaccard", combine="min") == pytest.approx(min(scores))
    assert sim_monge_elkan("abc ab", "abd", method="jaccard", combine="max") == pytest.approx(max(scores))


def test_monge_elkan_missing_role_raises():
    # jaccard has no 'dist' role, so dist_monge_elkan must fail rather than
    # silently falling back to something else.
    with pytest.raises(ValueError):
        dist_monge_elkan("foo bar", "foo bar", method="jaccard")


def test_monge_elkan_score_role():
    # score_monge_elkan uses the `score` role and takes the best (max) per
    # token, same as sim. "cat" -> best of {"cats", "dog"} under
    # needleman_wunsch is "cats" (score 2) over "dog" (score -3).
    assert score_monge_elkan("cat", "cats dog", method="needleman_wunsch") == pytest.approx(2.0)
    # levenshtein has no 'score' role.
    with pytest.raises(ValueError):
        score_monge_elkan("cat", "cats dog")
