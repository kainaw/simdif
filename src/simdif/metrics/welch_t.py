import math
from ..simdif import Metric, METRICS, to_list_numeric
from ._helpers import _mean_var


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
    sim  - 1 / (1 + |t|)

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
Similarity 1/(1+|t|): {1.0 / (1.0 + t):.4f}
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
def sim_welch_t(a, b, **kwargs) -> float:
    return 1.0 / (1.0 + dist_welch_t(a, b, **kwargs))
sim_welch = sim_two_sample_t = sim_sed = sim_welch_t


METRICS['welch_t'] = {
    'class': 'vector',
    'default': 'dist',
    'dist': dist_welch_t,
    'sim': sim_welch_t,
    'info': info_welch_t,
    'explain': explain_welch_t,
}
METRICS['welch'] = METRICS['two_sample_t'] = METRICS['sed'] = METRICS['welch_t']
