import math
from ..simdif import Metric, METRICS, to_list_numeric
from ._helpers import _mean_var


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
