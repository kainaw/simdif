import math
from ..simdif import Metric, METRICS, to_list_numeric
from ._helpers import _mean_var, _sim_from_dist, _dif_from_dist, _max_line


def info_welch_t() -> str:
    return """
Welch's Two-Sample t-statistic
------------------------------
Measures how far apart the means of two independent numeric samples are, in
units of the standard error of their difference. Larger |t| = more clearly
distinguishable samples.

Formula:
    t = (mean_A - mean_B) / sqrt(s_A^2/n_A + s_B^2/n_B)
    where s^2 is the sample variance (divided by n-1). The denominator is the
    standard error of the difference; this is Welch's form (unequal variances
    allowed).

Roles:
    dist - |t|, magnitude of the standardized difference (>= 0, unbounded)
    sim  - 1 / (1 + |t|), or 1 - dif when d_max is supplied
    dif  - 1 - sim,       or |t| / d_max when d_max is supplied

Note: |t| has no maximum -- it grows without bound as the samples get larger
or the standard error shrinks -- so sim defaults to the 1/(1+|t|) squash.
d_max here is a critical value rather than a data bound: d_max=1.96 makes
dif=1.0 mean "significant at the two-sided 5% level or beyond". Note that
this makes dif a function of sample size as well as effect size, which is a
property of t itself, not of the bounding -- see cohens_d for a dif that
depends only on the effect.

An infinite |t| (zero standard error, nonzero mean gap) gives dif=1.0 and
sim=0.0 in both branches rather than nan.

WARNING -- d_max must be a real bound. Anything above it clamps to dif=1.0,
so a t of 2 and a t of 200 become indistinguishable. explain_ reports the
clamp.

Range (dist): [0, inf)
    0   = identical sample means
    inf = a nonzero mean difference with zero standard error

Note: A and B are two INDEPENDENT samples - unordered, duplicates significant,
and NOT required to be the same length. Each needs at least 2 values. (Standard
error alone measures the precision of the difference, not its size; dividing the
mean difference by it is what makes this a difference measure.)

Note on the 'sed' alias: it resolves to THIS metric, so it returns |t| (the mean
difference in units of the standard error). The standard error of the difference
itself, sqrt(s_A^2/n_A + s_B^2/n_B), is the denominator - shown in explain().

Aliases: Welch, two-sample t, SED
    """.strip()
info_welch = info_two_sample_t = info_sed = info_welch_t


def explain_welch_t(a, b, **kwargs) -> str:
    va, vb = to_list_numeric(a), to_list_numeric(b)
    na, ma, s2a = _mean_var(va)
    nb, mb, s2b = _mean_var(vb)
    sed = math.sqrt(s2a / na + s2b / nb)
    diff = abs(ma - mb)
    t = (0.0 if diff == 0 else float('inf')) if sed == 0 else diff / sed
    return f"""
A: {va}
B: {vb}
Sample A: n={na}, mean={ma:.4f}, var={s2a:.4f}
Sample B: n={nb}, mean={mb:.4f}, var={s2b:.4f}
Standard error of difference: sqrt({s2a:.4f}/{na} + {s2b:.4f}/{nb}) = {sed:.4f}
|t| = |{ma:.4f} - {mb:.4f}| / {sed:.4f} = {t:.4f}
{_max_line(t, kwargs.get('d_max'),
           unbounded_note="unbounded -- |t| grows with sample size (1.96 is the two-sided 5% critical value)")}
    """.strip()
explain_welch = explain_two_sample_t = explain_sed = explain_welch_t


@Metric
def dist_welch_t(a, b, **kwargs) -> float:
    na, ma, s2a = _mean_var(to_list_numeric(a))
    nb, mb, s2b = _mean_var(to_list_numeric(b))
    sed = math.sqrt(s2a / na + s2b / nb)
    diff = abs(ma - mb)
    if sed == 0:
        return 0.0 if diff == 0 else float('inf')
    return diff / sed
dist_welch = dist_two_sample_t = dist_sed = dist_welch_t


@Metric
def dif_welch_t(a, b, **kwargs) -> float:
    return _dif_from_dist(dist_welch_t(a, b, **kwargs), kwargs.get('d_max'))
dif_welch = dif_two_sample_t = dif_sed = dif_welch_t


@Metric
def sim_welch_t(a, b, **kwargs) -> float:
    return _sim_from_dist(dist_welch_t(a, b, **kwargs), kwargs.get('d_max'))
sim_welch = sim_two_sample_t = sim_sed = sim_welch_t


METRICS['welch_t'] = {
    'class': 'vector',
    'default': 'dist',
    'dist': dist_welch_t,
    'dif': dif_welch_t,
    'sim': sim_welch_t,
    'info': info_welch_t,
    'explain': explain_welch_t,
}
METRICS['welch'] = METRICS['two_sample_t'] = METRICS['sed'] = METRICS['welch_t']
