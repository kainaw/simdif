import math
import pytest
from simdif.metrics.welch_t import dist_welch_t, sim_welch_t
from simdif import welch_t, dist, simdif


def test_welch_t():
    # A=[1..5] mean 3 var 2.5; B=[3..7] mean 5 var 2.5.
    # sed = sqrt(2.5/5 + 2.5/5) = 1.0 ; |t| = |3-5|/1 = 2.0
    assert dist_welch_t([1, 2, 3, 4, 5], [3, 4, 5, 6, 7]) == pytest.approx(2.0)
    assert sim_welch_t([1, 2, 3, 4, 5], [3, 4, 5, 6, 7]) == pytest.approx(1 / 3)
    # Identical means -> t = 0.
    assert dist_welch_t([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)
    assert sim_welch_t([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    # Zero standard error but different means -> inf (the [100,100,100] case).
    assert math.isinf(dist_welch_t([5, 5, 5], [8, 8, 8]))
    assert sim_welch_t([5, 5, 5], [8, 8, 8]) == pytest.approx(0.0)
    # Zero standard error and equal means -> 0 (identical constant samples).
    assert dist_welch_t([5, 5], [5, 5]) == pytest.approx(0.0)
    # Unequal sample sizes are fine.
    assert dist_welch_t([1, 2, 3, 4], [10, 11]) > 0
    # Each sample needs at least 2 values.
    with pytest.raises(ValueError, match="at least 2 values"):
        dist_welch_t([1], [2, 3])
    # Dispatch + aliases.
    assert welch_t([1, 2, 3, 4, 5], [3, 4, 5, 6, 7]) == pytest.approx(2.0)
    assert dist([1, 2, 3, 4, 5], [3, 4, 5, 6, 7], 'welch') == pytest.approx(2.0)
    assert simdif([1, 2, 3, 4, 5], [3, 4, 5, 6, 7], ['two_sample_t']) == {
        'two_sample_t': pytest.approx(2.0)}
    # 'sed' is an alias of welch_t, so it returns |t| (2.0), not the raw SED (1.0).
    assert simdif([1, 2, 3, 4, 5], [3, 4, 5, 6, 7], ['sed']) == {'sed': pytest.approx(2.0)}
    assert dist([1, 2, 3, 4, 5], [3, 4, 5, 6, 7], 'sed') == pytest.approx(2.0)
