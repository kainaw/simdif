from difflib import SequenceMatcher

import pytest
from simdif.metrics.ratcliff_obershelp import (
    sim_ratcliff_obershelp, dif_ratcliff_obershelp, trace_ratcliff_obershelp,
)
from simdif import ratcliff_obershelp, sim, dif, trace, simdif


def test_ratcliff_obershelp_sim():
    # Identical / disjoint / empty.
    assert sim_ratcliff_obershelp("abc", "abc") == pytest.approx(1.0)
    assert sim_ratcliff_obershelp("abc", "xyz") == pytest.approx(0.0)
    assert sim_ratcliff_obershelp("", "") == pytest.approx(1.0)
    # "abcd"/"bcde": one block "bcd" (M=3) -> 2*3/8 = 0.75
    assert sim_ratcliff_obershelp("abcd", "bcde") == pytest.approx(0.75)
    # "cat"/"cot": blocks "c" + "t" (M=2) -> 2*2/6 = 2/3
    assert sim_ratcliff_obershelp("cat", "cot") == pytest.approx(2 / 3)
    assert dif_ratcliff_obershelp("cat", "cot") == pytest.approx(1 - 2 / 3)


def test_matches_difflib():
    # The pure algorithm equals difflib.SequenceMatcher.ratio() on short inputs
    # (below difflib's autojunk threshold).
    pairs = [
        ("WIKIMEDIA", "WIKIMANIA"),
        ("GESTALT", "GESTAPO"),
        ("kitten", "sitting"),
        ("ratcliff", "obershelp"),
        ("aaaa", "aa"),
    ]
    for a, b in pairs:
        assert sim_ratcliff_obershelp(a, b) == pytest.approx(
            SequenceMatcher(None, a, b).ratio()), (a, b)


def test_ratcliff_obershelp_trace():
    # trace returns matching blocks (piece, a_index, b_index, length); str piece
    # for string inputs.
    assert trace_ratcliff_obershelp("abcd", "bcde") == [("bcd", 1, 0, 3)]
    # List inputs -> list pieces.
    assert trace_ratcliff_obershelp(["a", "b", "c"], ["x", "b", "c"]) == [(["b", "c"], 1, 1, 2)]
    # Total matched length in the trace reconstructs the similarity.
    blocks = trace_ratcliff_obershelp("cat", "cot")
    m = sum(k for _, _, _, k in blocks)
    assert 2 * m / (3 + 3) == pytest.approx(sim_ratcliff_obershelp("cat", "cot"))


def test_dispatch_and_aliases():
    assert ratcliff_obershelp("abcd", "bcde") == pytest.approx(0.75)   # convenience name
    assert sim("abcd", "bcde", "gestalt") == pytest.approx(0.75)
    assert dif("cat", "cot", "ro") == pytest.approx(1 - 2 / 3)
    assert trace("abcd", "bcde", "ratcliff") == [("bcd", 1, 0, 3)]
    # Each alias resolves to the same metric.
    for name in ("gestalt", "ro", "ratcliff", "obershelp"):
        assert simdif("abcd", "bcde", [name]) == {name: pytest.approx(0.75)}
