import math
import pytest
from simdif.metrics.hausdorff import (
    dist_hausdorff, sim_hausdorff, dif_hausdorff, directed_hausdorff,
    _nearest_distances, _reduce,
)
from simdif.metrics.dtw import dist_dtw
from simdif.metrics.jaccard import sim_jaccard
from simdif import hausdorff, sim, dif, dist, simdif, METRICS


def test_hausdorff_basic():
    # Worked example: every point of A is within 0.5 of some point of B and
    # vice versa, so the worst-case nearest miss is 0.5.
    assert dist_hausdorff([0, 1, 2], [0.5, 1.5]) == 0.5

    # Identical sets score 0.
    assert dist_hausdorff([1, 2, 3], [1, 2, 3]) == 0.0

    # Duplicates are irrelevant -- a repeated point adds no new nearest miss.
    assert dist_hausdorff([1, 2, 3], [1, 1, 2, 2, 3, 3]) == 0.0

    # A constant offset shows up directly.
    assert dist_hausdorff([1, 2, 3], [1.01, 2.01, 3.01]) == pytest.approx(0.01)


def test_hausdorff_is_order_blind():
    # THE defining property, and the contrast with dtw: reversed inputs are the
    # same point set, so Hausdorff is 0 while dtw sees a large mismatch.
    assert dist_hausdorff([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) == 0.0
    assert dist_dtw([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) > 0

    # Any permutation is free.
    assert dist_hausdorff([1, 2, 3], [3, 1, 2]) == 0.0

    # And unlike the set metrics, near misses still count: jaccard sees no
    # shared members at all here, Hausdorff sees a 0.01 gap.
    assert sim_jaccard([1, 2, 3], [1.01, 2.01, 3.01]) == 0.0
    assert dist_hausdorff([1, 2, 3], [1.01, 2.01, 3.01]) == pytest.approx(0.01)


def test_hausdorff_unequal_lengths():
    # No alignment and no padding needed -- lengths may differ freely.
    assert dist_hausdorff([1], [1, 2, 3]) == 2.0
    assert dist_hausdorff([0, 5, 10], [5]) == 5.0


def test_hausdorff_needs_both_directions():
    # A single point inside a cloud: ~0 going toward the cloud, large coming
    # back. Only the max of the two directions is symmetric.
    assert directed_hausdorff([1], [0, 1, 2]) == 0.0
    assert directed_hausdorff([0, 1, 2], [1]) == 1.0
    assert dist_hausdorff([1], [0, 1, 2]) == 1.0

    # Symmetry of the combined metric, on an asymmetric pair.
    a, b = [0, 1, 2], [1]
    assert dist_hausdorff(a, b) == dist_hausdorff(b, a)


def test_hausdorff_is_symmetric_and_a_metric():
    sets = [[0, 1, 2], [1], [0.5, 1.5], [10, 20], [1, 2, 3, 4]]
    for a in sets:
        for b in sets:
            # Symmetry.
            assert dist_hausdorff(a, b) == pytest.approx(dist_hausdorff(b, a))
            # Identity of indiscernibles.
            assert (dist_hausdorff(a, b) == 0.0) == (sorted(set(a)) == sorted(set(b)))
    # Triangle inequality (which dtw does not satisfy).
    for a in sets:
        for b in sets:
            for c in sets:
                assert dist_hausdorff(a, c) <= dist_hausdorff(a, b) + dist_hausdorff(b, c) + 1e-9


def test_hausdorff_outlier_sensitivity_and_percentile():
    # The documented warning, pinned: one stray point dominates the maximum.
    assert dist_hausdorff([0, 1, 2], [0.5, 1.5]) == 0.5
    assert dist_hausdorff([0, 1, 2], [0.5, 1.5, 100]) == 98.0

    # ...and the percentile is the cure. B tracks A at a 0.2 offset except for
    # one wild point at 500.
    A = list(range(20))
    B = [i + 0.2 for i in range(19)] + [500]
    assert dist_hausdorff(A, B) == pytest.approx(481.0)          # percentile=100
    assert dist_hausdorff(A, B, percentile=95) == pytest.approx(0.2)
    assert dist_hausdorff(A, B, percentile=90) == pytest.approx(0.2)

    # Lowering the percentile can only ever lower the distance.
    scores = [dist_hausdorff(A, B, percentile=p) for p in (100, 99, 95, 90, 50, 10)]
    assert scores == sorted(scores, reverse=True)

    # percentile=100 is exactly the max, with no interpolation.
    assert dist_hausdorff(A, B, percentile=100) == max(
        max(_nearest_distances(A, B, lambda p, q: abs(p - q))),
        max(_nearest_distances(B, A, lambda p, q: abs(p - q))),
    )


def test_hausdorff_percentile_needs_enough_points():
    # Nearest-rank means a high percentile cannot drop anything from a small
    # set: with n points, percentile p discards nothing unless p < 100*(n-1)/n.
    # So HD95 needs 20+ points, and on 6 points it equals percentile=100.
    small_a = list(range(6))
    small_b = [i + 0.2 for i in range(5)] + [500]
    assert dist_hausdorff(small_a, small_b, percentile=95) == dist_hausdorff(small_a, small_b)

    # Correct, but easy to mistake for an ignored parameter -- so explain_ must
    # call it out rather than silently printing an unchanged number.
    from simdif.metrics.hausdorff import explain_hausdorff
    report = explain_hausdorff(small_a, small_b, percentile=95)
    assert "drops NOTHING here" in report
    assert "percentile < 83.3" in report

    # With enough points the same percentile does bite, and explain_ then
    # reports the drop instead.
    big_a = list(range(20))
    big_b = [i + 0.2 for i in range(19)] + [500]
    assert dist_hausdorff(big_a, big_b, percentile=95) < dist_hausdorff(big_a, big_b)
    report = explain_hausdorff(big_a, big_b, percentile=95)
    assert "dropping 1" in report
    assert "drops NOTHING" not in report


def test_hausdorff_aggregate_mean():
    # percentile=100 + mean is the modified Hausdorff (Dubuisson & Jain).
    # Averaging pulls the score below the max whenever the gaps differ.
    A, B = [0, 1, 2], [0.5, 1.5, 100]
    assert dist_hausdorff(A, B, aggregate='mean') < dist_hausdorff(A, B, aggregate='max')

    # On uniform gaps, mean and max coincide.
    assert dist_hausdorff([0, 1, 2], [0.5, 1.5], aggregate='mean') == pytest.approx(0.5)

    # percentile composes with mean rather than being ignored by it.
    A2 = list(range(20))
    B2 = [i + 0.2 for i in range(19)] + [500]
    assert dist_hausdorff(A2, B2, percentile=95, aggregate='mean') < dist_hausdorff(A2, B2, aggregate='mean')

    with pytest.raises(ValueError, match="aggregate must be"):
        dist_hausdorff(A, B, aggregate='median')
    for bad in (-1, 101):
        with pytest.raises(ValueError, match="percentile must be"):
            dist_hausdorff(A, B, percentile=bad)


def test_hausdorff_reduce_conventions():
    # Nearest-rank: index = ceil(p/100 * n) - 1, never interpolating.
    gaps = [1.0, 2.0, 3.0, 4.0]
    assert _reduce(gaps, 100, 'max') == 4.0
    assert _reduce(gaps, 75, 'max') == 3.0
    assert _reduce(gaps, 50, 'max') == 2.0
    assert _reduce(gaps, 25, 'max') == 1.0
    # A percentile too small to keep anything still keeps one value.
    assert _reduce(gaps, 0, 'max') == 1.0
    assert _reduce(gaps, 100, 'mean') == pytest.approx(2.5)
    assert _reduce([], 100, 'max') == 0.0


def test_hausdorff_empty_inputs():
    # Two empty sets are identical; an empty set cannot be reached from a
    # non-empty one, so the distance is infinite.
    assert dist_hausdorff([], []) == 0.0
    assert dist_hausdorff([], [1, 2]) == math.inf
    assert dist_hausdorff([1, 2], []) == math.inf
    assert sim_hausdorff([], [1, 2]) == 0.0
    assert directed_hausdorff([], [1, 2]) == 0.0
    assert directed_hausdorff([1, 2], []) == math.inf


def test_hausdorff_custom_dist_fn():
    # A dist_fn lifts the numeric requirement, so tuples act as n-D points.
    euclid = lambda p, q: math.dist(p, q)
    assert dist_hausdorff([(0, 0), (1, 1)], [(0, 0), (1, 1)], dist_fn=euclid) == 0.0
    assert dist_hausdorff([(0, 0), (1, 1)], [(0, 1), (1, 0)], dist_fn=euclid) == pytest.approx(1.0)

    # Squared distance is a valid pointwise cost too.
    assert dist_hausdorff([0, 1, 2], [0.5, 1.5], dist_fn=lambda p, q: (p - q) ** 2) == 0.25


def test_hausdorff_sim_dif():
    # sim = 1/(1+H), matching the dtw/energy convention for unbounded distances.
    assert sim_hausdorff([1, 2, 3], [1, 2, 3]) == 1.0
    assert dif_hausdorff([1, 2, 3], [1, 2, 3]) == 0.0
    assert sim_hausdorff([0, 1, 2], [0.5, 1.5]) == pytest.approx(1 / 1.5)

    for a, b in [([0, 1, 2], [0.5, 1.5]), ([1], [0, 1, 2]), ([1, 2], [10, 20]), ([], [])]:
        s, d = sim_hausdorff(a, b), dif_hausdorff(a, b)
        assert 0.0 <= s <= 1.0
        assert s + d == pytest.approx(1.0)


def test_hausdorff_registry_and_aliases():
    assert METRICS['hausdorff']['class'] == 'vector'
    assert METRICS['hausdorff']['default'] == 'dist'
    for role in ('matrix', 'trace', 'score'):
        assert role not in METRICS['hausdorff']

    for alias in ['hausdorff_distance', 'hd']:
        assert METRICS[alias] is METRICS['hausdorff']

    # Convenience/dispatcher access.
    assert hausdorff([0, 1, 2], [0.5, 1.5]) == 0.5
    assert dist([0, 1, 2], [0.5, 1.5], "hausdorff") == 0.5
    assert sim([0, 1, 2], [0.5, 1.5], "hausdorff") == pytest.approx(1 / 1.5)
    assert dif([0, 1, 2], [0.5, 1.5], "hausdorff") == pytest.approx(1 - 1 / 1.5)
    assert simdif([0, 1, 2], [0.5, 1.5], ["hausdorff"]) == {"hausdorff": 0.5}

    # The percentile kwarg threads through the dispatcher -- the side-by-side
    # comparison a student runs to see what one outlier was carrying.
    A = list(range(20))
    B = [i + 0.2 for i in range(19)] + [500]
    assert simdif(A, B, ["hausdorff"], percentile=95) == {"hausdorff": pytest.approx(0.2)}

    # And the headline contrast against an order-sensitive metric in one call.
    result = simdif([1, 2, 3, 4, 5], [5, 4, 3, 2, 1], ["hausdorff", "dtw"])
    assert result["hausdorff"] == 0.0
    assert result["dtw"] > 0
