import math
import pytest
from simdif.metrics.cohens_d import dist_cohens_d, sim_cohens_d
from simdif.metrics.welch_t import dist_welch_t
from simdif import cohens_d, simdif


def test_cohens_d():
    # Same samples: pooled sd = sqrt((4*2.5 + 4*2.5)/8) = sqrt(2.5);
    # d = |3-5| / sqrt(2.5)
    expected = 2 / math.sqrt(2.5)
    assert dist_cohens_d([1, 2, 3, 4, 5], [3, 4, 5, 6, 7]) == pytest.approx(expected)
    assert sim_cohens_d([1, 2, 3, 4, 5], [3, 4, 5, 6, 7]) == pytest.approx(1 / (1 + expected))
    # Identical means -> d = 0.
    assert dist_cohens_d([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)
    # Zero pooled SD, different means -> inf.
    assert math.isinf(dist_cohens_d([5, 5, 5], [8, 8, 8]))
    # Welch's t and Cohen's d are genuinely different measures on the same data:
    # t divides by the standard error (includes /n), d by the pooled SD.
    assert dist_welch_t([1, 2, 3, 4, 5], [3, 4, 5, 6, 7]) != pytest.approx(
        dist_cohens_d([1, 2, 3, 4, 5], [3, 4, 5, 6, 7]))
    # Dispatch + aliases.
    assert cohens_d([1, 2, 3, 4, 5], [3, 4, 5, 6, 7]) == pytest.approx(expected)
    assert simdif([1, 2, 3, 4, 5], [3, 4, 5, 6, 7], ['cohen_d']) == {
        'cohen_d': pytest.approx(expected)}
