import math
import pytest
from simdif.metrics.energy import dist_energy, sim_energy
from simdif.metrics.cohens_d import dist_cohens_d
from simdif import energy, dist, simdif


def test_energy():
    # A=[0,0], B=[1,1]: cross=1, within_a=within_b=0 -> d2=2 -> E=sqrt(2)
    assert dist_energy([0, 0], [1, 1]) == pytest.approx(math.sqrt(2))
    assert sim_energy([0, 0], [1, 1]) == pytest.approx(1 / (1 + math.sqrt(2)))

    # Identical samples -> 0 (same distribution).
    assert dist_energy([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)
    assert sim_energy([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)

    # Sees a spread difference that a mean-only statistic misses:
    # equal means (both 3), different variance -> energy distance > 0,
    # but Cohen's d = 0.
    same_mean_a = [3, 3, 3, 3]
    same_mean_b = [0, 2, 4, 6]
    assert dist_cohens_d(same_mean_a, same_mean_b) == pytest.approx(0.0)
    assert dist_energy(same_mean_a, same_mean_b) > 0

    # A=[0,1], B=[0,10]: cross=5, within_a=0.5, within_b=5 -> d2=4.5
    assert dist_energy([0, 1], [0, 10]) == pytest.approx(math.sqrt(4.5))

    # Unequal lengths OK; single-value samples OK (no variance needed).
    assert dist_energy([1, 2, 3, 4], [10, 11]) > 0
    # Single points: d2 = 2*|0-4| = 8 -> E = sqrt(8)
    assert dist_energy([0], [4]) == pytest.approx(math.sqrt(8))

    # Empty sample is rejected.
    with pytest.raises(ValueError, match="at least 1 value"):
        dist_energy([], [1, 2])

    # Dispatch + aliases.
    assert energy([0, 0], [1, 1]) == pytest.approx(math.sqrt(2))
    assert dist([0, 0], [1, 1], 'energy_distance') == pytest.approx(math.sqrt(2))
    assert simdif([0, 0], [1, 1], ['e_distance']) == {
        'e_distance': pytest.approx(math.sqrt(2))}
