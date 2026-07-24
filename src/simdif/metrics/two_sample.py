import math
from ..simdif import Metric, METRICS, to_list_numeric


# ------------------------------------------------------------------
# Shared helper
# ------------------------------------------------------------------

def _mean_var(x):
    """(n, mean, sample_variance) for a numeric sample. Variance uses n-1."""
    n = len(x)
    if n < 2:
        raise ValueError("Each sample needs at least 2 values to estimate variance")
    mean = sum(x) / n
    var = sum((v - mean) ** 2 for v in x) / (n - 1)
    return n, mean, var


# ------------------------------------------------------------------
# Welch's two-sample t-statistic
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Cohen's d (standardized effect size)
# ------------------------------------------------------------------

def info_cohens_d() -> str:
    return """
Cohen's d (Standardized Effect Size)
------------------------------------
The difference between two sample means expressed in units of their pooled
standard deviation - a scale-free measure of how far apart two samples are.

Formula:
    d = (mean_A - mean_B) / s_pooled
    s_pooled = sqrt( ((n_A-1) s_A^2 + (n_B-1) s_B^2) / (n_A + n_B - 2) )

Roles:
    dist - |d| (>= 0, unbounded)
    sim  - 1 / (1 + |d|)

Range (dist): [0, inf)
    Cohen's conventional benchmarks: ~0.2 small, ~0.5 medium, ~0.8 large.
    inf = a nonzero mean difference with zero pooled standard deviation.

Note: A and B are two INDEPENDENT samples - unordered, duplicates significant,
unequal lengths OK, each with at least 2 values. Unlike Welch's t, d does not
depend on sample size, so it stays comparable across studies.

Aliases: Cohen d, Cohens
    """.strip()
info_cohen_d = info_cohens = info_cohens_d


def explain_cohens_d(a, b, **kwargs) -> str:
    va, vb = to_list_numeric(a), to_list_numeric(b)
    na, ma, s2a = _mean_var(va)
    nb, mb, s2b = _mean_var(vb)
    sp = math.sqrt(((na - 1) * s2a + (nb - 1) * s2b) / (na + nb - 2))
    diff = abs(ma - mb)
    d = (0.0 if diff == 0 else float('inf')) if sp == 0 else diff / sp
    return f"""
A: {va}
B: {vb}
Sample A: n={na}, mean={ma:.4f}, var={s2a:.4f}
Sample B: n={nb}, mean={mb:.4f}, var={s2b:.4f}
Pooled SD: sqrt((({na}-1)*{s2a:.4f} + ({nb}-1)*{s2b:.4f}) / ({na}+{nb}-2)) = {sp:.4f}
|d| = |{ma:.4f} - {mb:.4f}| / {sp:.4f} = {d:.4f}
Similarity 1/(1+|d|): {1.0 / (1.0 + d):.4f}
    """.strip()
explain_cohen_d = explain_cohens = explain_cohens_d


@Metric
def dist_cohens_d(a, b, **kwargs) -> float:
    na, ma, s2a = _mean_var(to_list_numeric(a))
    nb, mb, s2b = _mean_var(to_list_numeric(b))
    sp = math.sqrt(((na - 1) * s2a + (nb - 1) * s2b) / (na + nb - 2))
    diff = abs(ma - mb)
    if sp == 0:
        return 0.0 if diff == 0 else float('inf')
    return diff / sp
dist_cohen_d = dist_cohens = dist_cohens_d


@Metric
def sim_cohens_d(a, b, **kwargs) -> float:
    return 1.0 / (1.0 + dist_cohens_d(a, b, **kwargs))
sim_cohen_d = sim_cohens = sim_cohens_d


METRICS['cohens_d'] = {
    'class': 'vector',
    'default': 'dist',
    'dist': dist_cohens_d,
    'sim': sim_cohens_d,
    'info': info_cohens_d,
    'explain': explain_cohens_d,
}
METRICS['cohen_d'] = METRICS['cohens'] = METRICS['cohens_d']
