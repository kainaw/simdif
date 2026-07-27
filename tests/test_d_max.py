"""The d_max contract, which spans metrics rather than living in any one of them.

Metrics split into two disjoint sets. Where a maximum is derivable the metric
uses it and d_max is meaningless (see test_canberra / test_lee / test_geodesic /
test_js_divergence). Where no maximum exists, sim defaults to the 1/(1+d) squash
and d_max is the only way to get a linear rescaling. These tests pin the second
set, and pin that the first set ignores d_max rather than honoring it.
"""
import math
import pytest
from simdif import sim, dif, dist
from simdif.simdif import METRICS

# No derivable maximum: d_max is the only route to a linear dif.
UNBOUNDED = ['chebyshev', 'euclidean', 'manhattan', 'minkowski', 'mahalanobis',
             'energy', 'cohens_d', 'welch_t', 'kl_divergence', 'jukes_cantor',
             'kimura', 'dtw', 'hausdorff']

# Maximum is derivable, so d_max must be ignored.
DERIVED = ['canberra', 'js_divergence', 'lee', 'geodesic', 'earth']

# Inputs that produce a finite, nonzero distance for each metric.
CASES = {
    'cohens_d': ([1, 2, 3, 4], [10, 11, 12, 13]),
    'welch_t': ([1, 2, 3, 4], [10, 11, 12, 13]),
    'energy': ([1, 2, 3], [10, 11, 12]),
    'kl_divergence': ([0.7, 0.3], [0.3, 0.7]),
    # One transition (A->G) and one transversion (A->T) over 8 sites keeps the
    # Kimura correction away from its saturation point, so both branches get
    # real coverage rather than skipping.
    'jukes_cantor': ('AAAAAAAA', 'AAAAAAGT'),
    'kimura': ('AAAAAAAA', 'AAAAAAGT'),
    'dtw': ([1, 2, 3], [2, 3, 4]),
    'hausdorff': ([1, 2, 3], [8, 9, 10]),
}
DEFAULT_CASE = ([0, 0, 0], [1, 1, 1])


@pytest.mark.parametrize('metric', UNBOUNDED)
def test_unbounded_metrics_expose_all_three_roles(metric):
    assert 'dist' in METRICS[metric]
    assert 'sim' in METRICS[metric]
    assert 'dif' in METRICS[metric]


@pytest.mark.parametrize('metric', UNBOUNDED)
def test_squash_is_the_default(metric):
    # With no d_max the primitive is sim = 1/(1+d) and dif is derived from it,
    # so neither role can reach its extreme for a finite distance: "completely
    # different" is undefined without a bound, and saying so is the point.
    a, b = CASES.get(metric, DEFAULT_CASE)
    d = dist(a, b, metric)
    if d == math.inf:
        pytest.skip(f'{metric} saturates to inf for this input')
    assert sim(a, b, metric) == pytest.approx(1.0 / (1.0 + d))
    assert dif(a, b, metric) == pytest.approx(1.0 - 1.0 / (1.0 + d))
    assert 0.0 < sim(a, b, metric) <= 1.0
    assert 0.0 <= dif(a, b, metric) < 1.0


@pytest.mark.parametrize('metric', UNBOUNDED)
def test_d_max_rescales_linearly(metric):
    # With d_max the primitive flips: dif = d/d_max, sim = 1 - dif. Picking a
    # d_max above the actual distance keeps it off the clamp.
    a, b = CASES.get(metric, DEFAULT_CASE)
    d = dist(a, b, metric)
    if d == math.inf:
        pytest.skip(f'{metric} saturates to inf for this input')
    d_max = d * 4
    assert dif(a, b, metric, d_max=d_max) == pytest.approx(d / d_max)
    assert dif(a, b, metric, d_max=d_max) == pytest.approx(0.25)
    assert sim(a, b, metric, d_max=d_max) == pytest.approx(0.75)
    # Exactly at the bound, dif is exactly 1.
    assert dif(a, b, metric, d_max=d) == pytest.approx(1.0)
    assert sim(a, b, metric, d_max=d) == pytest.approx(0.0)


@pytest.mark.parametrize('metric', UNBOUNDED)
def test_d_max_clamps_and_that_is_lossy(metric):
    # Beyond d_max everything collapses onto 1.0. This is the documented cost
    # of a wrong bound: two very different distances become one score.
    a, b = CASES.get(metric, DEFAULT_CASE)
    tiny = 1e-9
    assert dif(a, b, metric, d_max=tiny) == pytest.approx(1.0)
    assert sim(a, b, metric, d_max=tiny) == pytest.approx(0.0)


@pytest.mark.parametrize('metric', UNBOUNDED)
def test_sim_plus_dif_is_one_in_both_branches(metric):
    a, b = CASES.get(metric, DEFAULT_CASE)
    for kwargs in ({}, {'d_max': 0.5}, {'d_max': 100.0}):
        assert sim(a, b, metric, **kwargs) + dif(a, b, metric, **kwargs) == pytest.approx(1.0)


@pytest.mark.parametrize('metric,a,b', [
    ('cohens_d', [1, 1, 1], [5, 5, 5]),      # zero pooled SD, nonzero mean gap
    ('welch_t', [1, 1, 1], [5, 5, 5]),       # zero standard error
    ('kl_divergence', [1, 0], [0, 1]),       # disjoint supports
    ('jukes_cantor', 'AAAA', 'CGTG'),        # saturated correction
    ('kimura', 'AAAA', 'CGTG'),
])
def test_infinite_distance_gives_one_not_nan(metric, a, b):
    # An unbounded metric that actually reaches inf must land on dif=1.0 rather
    # than nan. Computing dif as d/(1+d) directly would give inf/inf here, which
    # is why the squash branch derives dif from sim instead.
    assert dist(a, b, metric) == math.inf
    for kwargs in ({}, {'d_max': 2.0}):
        s, f = sim(a, b, metric, **kwargs), dif(a, b, metric, **kwargs)
        assert s == pytest.approx(0.0)
        assert f == pytest.approx(1.0)
        assert s == s and f == f  # not nan


def test_identical_inputs_score_zero_difference():
    for metric in UNBOUNDED:
        a, _ = CASES.get(metric, DEFAULT_CASE)
        if metric in ('cohens_d', 'welch_t', 'energy'):
            continue  # two identical samples have zero spread; covered per-metric
        for kwargs in ({}, {'d_max': 10.0}):
            assert dif(a, a, metric, **kwargs) == pytest.approx(0.0)
            assert sim(a, a, metric, **kwargs) == pytest.approx(1.0)


@pytest.mark.parametrize('metric', DERIVED)
def test_derived_maximum_metrics_ignore_d_max(metric):
    # These already know their maximum, so d_max is meaningless and must be
    # dropped -- not honored, and not an error. It cannot raise, because kwargs
    # broadcast to every metric in a list: dif(a, b, ['euclidean', 'canberra'],
    # d_max=10) is a legitimate call where only euclidean should use the bound.
    a, b = ([1, 2], [1, 6]) if metric != 'earth' else ([40.7, -74.0], [51.5, -0.1])
    if metric in ('geodesic', 'earth'):
        a, b = ([0.0, 0.0], [0.0, 1.0]) if metric == 'geodesic' else (a, b)
    if metric == 'js_divergence':
        a, b = [0.7, 0.3], [0.3, 0.7]
    baseline_dif = dif(a, b, metric)
    baseline_sim = sim(a, b, metric)
    for d_max in (1e-9, 0.5, 1000.0):
        assert dif(a, b, metric, d_max=d_max) == pytest.approx(baseline_dif)
        assert sim(a, b, metric, d_max=d_max) == pytest.approx(baseline_sim)


def test_d_max_broadcasts_across_a_metric_list():
    # The design point: one parameter, sent to every metric, honored only by the
    # ones that need it. Nothing raises and nothing is silently wrong.
    a, b = [0, 0, 0], [1, 1, 1]
    mixed = ['euclidean', 'manhattan', 'chebyshev', 'canberra']
    got = dif(a, b, mixed, d_max=3.0)
    assert got['manhattan'] == pytest.approx(1.0)              # dist 3 / 3
    assert got['euclidean'] == pytest.approx(math.sqrt(3) / 3)
    assert got['chebyshev'] == pytest.approx(1.0 / 3)
    assert got['canberra'] == pytest.approx(1.0)               # derived n=3, ignored d_max
    assert got['canberra'] == pytest.approx(dif(a, b, 'canberra'))


def test_range_normalized_inputs_give_derivable_bounds():
    # The reason info_ tells you to range-normalize: it converts "no bound
    # exists" into d_max = n^(1/p), and opposite corners of the unit cube then
    # land exactly on dif = 1.0.
    n = 4
    lo, hi = [0] * n, [1] * n
    assert dif(lo, hi, 'manhattan', d_max=n) == pytest.approx(1.0)
    assert dif(lo, hi, 'euclidean', d_max=math.sqrt(n)) == pytest.approx(1.0)
    assert dif(lo, hi, 'chebyshev', d_max=1.0) == pytest.approx(1.0)
    for p in (1, 2, 3, 5):
        assert dif(lo, hi, 'minkowski', p=p, d_max=n ** (1 / p)) == pytest.approx(1.0)
    # Chebyshev's bound is 1 regardless of dimension -- it needs no n at all.
    for width in (1, 2, 10, 50):
        assert dif([0] * width, [1] * width, 'chebyshev', d_max=1.0) == pytest.approx(1.0)


def test_minkowski_p_inf_honors_d_max_via_chebyshev():
    # p=inf delegates to chebyshev as a limit; the bound must survive the handoff.
    a, b = [0, 0], [5, 1]
    assert dif(a, b, 'minkowski', p=math.inf, d_max=10.0) == pytest.approx(0.5)
    assert dif(a, b, 'chebyshev', d_max=10.0) == pytest.approx(0.5)


def test_squash_preserves_ordering_where_clamping_does_not():
    # Why d_max must be a real bound rather than a guess: the squash keeps
    # distant pairs distinguishable, a too-small d_max does not.
    near, far = ([0], [10]), ([0], [1000])
    assert dif(*near, 'euclidean') < dif(*far, 'euclidean')
    assert dif(*near, 'euclidean', d_max=5.0) == dif(*far, 'euclidean', d_max=5.0) == 1.0
