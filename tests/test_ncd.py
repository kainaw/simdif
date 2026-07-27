import pytest
from simdif.metrics.ncd import (
    dist_ncd, sim_ncd, dif_ncd, _to_bytes, _compressed_sizes,
)
from simdif import ncd, sim, dif, dist, simdif, METRICS

# NCD needs inputs long enough that compressor header overhead stops dominating
# (see info_ncd), so the fixtures here are repeated phrases rather than the
# short words used elsewhere in the suite. Assertions are written as orderings
# and bounds rather than exact floats, because compressed sizes shift between
# zlib/bz2/lzma versions.
SAME = "the quick brown fox jumps over the lazy dog " * 12
NEAR = "the quick brown fox leaps over the lazy dog " * 12
OTHER = "shall i compare thee to a summers day thou art " * 12

ALL_COMPRESSORS = ['zlib', 'bz2', 'lzma']


@pytest.mark.parametrize("compressor", ALL_COMPRESSORS)
def test_ncd_orders_by_shared_structure(compressor):
    # The core property: more shared structure -> smaller distance. This holds
    # for every compressor even though the absolute values differ widely.
    identical = dist_ncd(SAME, SAME, compressor)
    near = dist_ncd(SAME, NEAR, compressor)
    different = dist_ncd(SAME, OTHER, compressor)
    assert identical < near < different


@pytest.mark.parametrize("compressor", ALL_COMPRESSORS)
def test_ncd_bounds(compressor):
    # dist is reported raw and is only approximately bounded by 1 -- an ideal
    # compressor would guarantee [0, 1], real ones leak slightly past it.
    for a, b in [(SAME, SAME), (SAME, NEAR), (SAME, OTHER), ("", ""), ("", SAME)]:
        assert 0.0 <= dist_ncd(a, b, compressor) <= 1.2

    # sim/dif are clamped and always complementary.
    for a, b in [(SAME, SAME), (SAME, OTHER), ("", ""), ("", SAME), ("x", "y")]:
        s, d = sim_ncd(a, b, compressor), dif_ncd(a, b, compressor)
        assert 0.0 <= s <= 1.0
        assert 0.0 <= d <= 1.0
        assert s + d == pytest.approx(1.0)


def test_ncd_identical_inputs_are_not_zero():
    # The documented gotcha, pinned so it cannot regress into a false promise:
    # identical inputs do NOT score 0, because the concatenation still pays a
    # second copy's worth of encoded output plus header overhead.
    assert dist_ncd(SAME, SAME) > 0.0
    assert sim_ncd(SAME, SAME) < 1.0

    # Still, a string against its own copy must beat any unrelated pair.
    assert dist_ncd(SAME, SAME) < dist_ncd(SAME, OTHER)


def test_ncd_is_asymmetric():
    # C(AB) != C(BA) in general, so NCD is directional (same caveat as
    # monge_elkan). Documented rather than papered over.
    assert dist_ncd(SAME, OTHER) != dist_ncd(OTHER, SAME)


def test_ncd_empty_inputs():
    # Two empty inputs compress to identical headers -> distance 0.
    assert dist_ncd("", "") == 0.0
    assert sim_ncd("", "") == 1.0
    assert dif_ncd("", "") == 0.0


def test_ncd_input_serialization():
    # str is encoded UTF-8, bytes pass through, so the two agree.
    assert dist_ncd("hello world", "hello world") == dist_ncd(b"hello world", b"hello world")
    assert _to_bytes("abc") == b"abc"
    assert _to_bytes(b"abc") == b"abc"
    assert _to_bytes(bytearray(b"abc")) == b"abc"

    # Non-string sequences are NUL-joined, so ['ab','c'] and ['a','bc'] cannot
    # collide -- and list("abc") is deliberately not the same as "abc".
    assert _to_bytes(["ab", "c"]) == b"ab\x00c"
    assert _to_bytes(["a", "bc"]) == b"a\x00bc"
    assert _to_bytes(list("abc")) != _to_bytes("abc")

    # Numeric tokens work too.
    assert _to_bytes([1, 2, 3]) == b"1\x002\x003"
    assert 0.0 <= dist_ncd([1, 2, 3] * 50, [1, 2, 3] * 50) <= 1.2


def test_ncd_compressor_selection():
    # The compressor genuinely changes the answer, which is why it is an
    # explicit parameter rather than an implementation detail.
    scores = {c: dist_ncd(SAME, OTHER, c) for c in ALL_COMPRESSORS}
    assert len(set(scores.values())) == len(ALL_COMPRESSORS)

    # A callable is accepted directly.
    import zlib
    assert dist_ncd(SAME, OTHER, lambda data: zlib.compress(data, 1)) > 0

    # An unknown name fails loudly rather than silently picking a default.
    with pytest.raises(ValueError, match="Unknown compressor"):
        dist_ncd(SAME, OTHER, "gzip2000")


def test_ncd_compressed_sizes_helper():
    # explain_ncd's arithmetic must reconcile with dist_ncd exactly.
    c_a, c_b, c_ab = _compressed_sizes(SAME, OTHER)
    expected = (c_ab - min(c_a, c_b)) / max(c_a, c_b)
    assert dist_ncd(SAME, OTHER) == pytest.approx(expected)

    # Concatenating similar inputs saves bytes versus compressing apart.
    c_a, c_b, c_ab = _compressed_sizes(SAME, NEAR)
    assert c_ab < c_a + c_b


def test_ncd_registry_and_aliases():
    assert METRICS['ncd']['class'] == 'sequence'
    assert METRICS['ncd']['default'] == 'dist'
    # No alignment or DP grid exists to expose.
    assert 'matrix' not in METRICS['ncd']
    assert 'trace' not in METRICS['ncd']
    assert 'score' not in METRICS['ncd']

    for alias in ['normalized_compression_distance', 'compression', 'compression_distance']:
        assert METRICS[alias] is METRICS['ncd']

    # Convenience/dispatcher access; the bare name resolves to 'dist'.
    assert ncd(SAME, OTHER) == pytest.approx(dist_ncd(SAME, OTHER))
    assert dist(SAME, OTHER, "ncd") == pytest.approx(dist_ncd(SAME, OTHER))
    assert sim(SAME, OTHER, "ncd") == pytest.approx(sim_ncd(SAME, OTHER))
    assert dif(SAME, OTHER, "ncd") == pytest.approx(dif_ncd(SAME, OTHER))
    assert simdif(SAME, OTHER, ["ncd"]) == {"ncd": pytest.approx(dist_ncd(SAME, OTHER))}

    # The compressor kwarg threads through the dispatcher.
    assert simdif(SAME, OTHER, ["ncd"], compressor="lzma") == {
        "ncd": pytest.approx(dist_ncd(SAME, OTHER, "lzma"))
    }
