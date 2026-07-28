import sys
from bisect import bisect_left
from ..simdif import Metric, METRICS, to_distribution
from ._helpers import _sim_from_dist, _dif_from_dist, _max_line, _lib_note


def info_wasserstein() -> str:
    return """
Wasserstein Distance (Earth Mover's Distance)
---------------------------------------------
The minimum "work" needed to reshape one distribution into the other, where
work = (mass moved) x (distance moved)^p. Unlike KL/JS/Hellinger, which compare
distributions bin-by-bin, Wasserstein accounts for HOW FAR mass must travel, so
the ordering of the bins matters.

This is the exact 1-D solution, computed in quantile space:
    W_p(P, Q) = ( integral_0^1 | F_P^{-1}(t) - F_Q^{-1}(t) |^p dt )^(1/p)

where F^{-1} is the quantile (inverse-CDF) function. For p = 1 this equals the
integral of the absolute CDF difference (the classic Earth Mover's Distance).

Parameters:
    p - the order of the distance (default 1). p = 1 is the Earth Mover's
        Distance; p = 2 is the quadratic Wasserstein distance.

Roles:
    dist - W_p(P, Q) (>= 0, unbounded)
    sim  - 1 / (1 + W_p), or 1 - dif when d_max is supplied
    dif  - 1 - sim,       or W_p / d_max when d_max is supplied

Range (dist): [0, inf)
    0 = identical distributions. Not clamped to [0, 1]: a small amount of mass
    moved a long way can outweigh a large amount moved a short way.

Note: W_p is unbounded because it carries units of bin position, so sim
defaults to the 1/(1+W_p) squash. There is no distribution-free ceiling to
divide by; if you need a linear dif, pass d_max (e.g. the number of bins),
which makes it a declared convention rather than a derived bound.

WARNING -- d_max must be a real bound. Values above it clamp to dif=1.0,
so every pair beyond d_max scores identically. explain_ reports the clamp.

Assumptions: bins are ordered and equally spaced, so the ground distance
between bin i and bin j is |i - j|. Inputs are normalized into probability
distributions before comparison. For p >= 1 this is a true metric.

Aliases: Earth Mover, EMD
    """.strip()
info_earth_mover = info_wasserstein
info_emd = info_wasserstein


def _wasserstein(a, b, p):
    """Exact 1-D p-Wasserstein distance between two probability distributions.

    Walk through cumulative probability t in [0, 1]. The CDFs of P and Q are
    step functions; their jump heights partition [0, 1] into intervals on which
    both quantile functions are constant. On each interval of width w, mass of
    amount w sits at bin F_P^{-1} in P and must move to bin F_Q^{-1} in Q, a
    ground distance of |i_P - i_Q|. Summing w * |i_P - i_Q|^p and taking the
    p-th root gives W_p.
    """
    cp, cq = [], []
    s = 0.0
    for x in a:
        s += x
        cp.append(s)
    s = 0.0
    for y in b:
        s += y
        cq.append(s)

    # Cumulative-probability breakpoints where a quantile position can change.
    levels = sorted(set(cp) | set(cq))
    prev = 0.0
    total = 0.0
    for t in levels:
        width = t - prev
        if width > 0:
            # First bin whose cumulative mass has reached level t (the quantile).
            i_p = bisect_left(cp, t)
            i_q = bisect_left(cq, t)
            total += width * abs(i_p - i_q) ** p
        prev = t
    return total ** (1.0 / p)


def explain_wasserstein(a, b, p=1, **kwargs) -> str:
    a = to_distribution(a)
    b = to_distribution(b)
    if len(a) != len(b):
        raise ValueError(f"Distributions must have the same length, got {len(a)} and {len(b)}")
    cp, cq, s = [], [], 0.0
    for x in a:
        s += x
        cp.append(s)
    s = 0.0
    for y in b:
        s += y
        cq.append(s)
    levels = sorted(set(cp) | set(cq))
    steps = []
    prev = 0.0
    for t in levels:
        width = t - prev
        if width > 0:
            i_p = bisect_left(cp, t)
            i_q = bisect_left(cq, t)
            steps.append(f"  t in ({prev:.4f}, {t:.4f}]  width {width:.4f}: |bin {i_p} - bin {i_q}|^{p} = {abs(i_p - i_q) ** p}")
        prev = t
    d = _wasserstein(a, b, p)
    live = dist_wasserstein(a, b, p, **kwargs)
    if 'ot' in sys.modules:
        lib_name = 'ot (POT)'
    elif p == 1 and 'scipy' in sys.modules:
        lib_name = 'scipy'
    else:
        lib_name = None
    return f"""
P (A): {a}
Q (B): {b}
CDF(P): {[round(v, 4) for v in cp]}
CDF(Q): {[round(v, 4) for v in cq]}
Quantile-space contributions (order p={p}):
{chr(10).join(steps)}
Wasserstein Distance W_{p}: {d:.4f}{_lib_note(d, live, lib_name, 'dist_wasserstein')}
{_max_line(live, kwargs.get('d_max'),
           unbounded_note="unbounded -- W_p carries units of bin position")}
    """.strip()
explain_earth_mover = explain_wasserstein
explain_emd = explain_wasserstein


@Metric
def dist_wasserstein(a, b, p=1, **kwargs) -> float:
    if p <= 0:
        raise ValueError("p (the Wasserstein order) must be positive")
    a = to_distribution(a)
    b = to_distribution(b)
    if len(a) != len(b):
        raise ValueError(f"Distributions must have the same length, got {len(a)} and {len(b)}")
    # Optimized fallbacks, used only if the caller already imported the library.
    if 'ot' in sys.modules:
        import numpy as np
        from ot import wasserstein_1d
        x = np.arange(len(a), dtype=float)
        cost = wasserstein_1d(x, x, np.asarray(a), np.asarray(b), p=p)  # returns W_p ** p
        return float(cost) ** (1.0 / p)
    if p == 1 and 'scipy' in sys.modules:
        from scipy.stats import wasserstein_distance
        idx = list(range(len(a)))
        return float(wasserstein_distance(idx, idx, a, b))
    return _wasserstein(a, b, p)
dist_earth_mover = dist_wasserstein
dist_emd = dist_wasserstein


@Metric
def dif_wasserstein(a, b, p=1, **kwargs) -> float:
    return _dif_from_dist(dist_wasserstein(a, b, p, **kwargs), kwargs.get('d_max'))
dif_earth_mover = dif_wasserstein
dif_emd = dif_wasserstein


@Metric
def sim_wasserstein(a, b, p=1, **kwargs) -> float:
    return _sim_from_dist(dist_wasserstein(a, b, p, **kwargs), kwargs.get('d_max'))
sim_earth_mover = sim_wasserstein
sim_emd = sim_wasserstein


METRICS['wasserstein'] = {
    'class': 'vector',
    'default': 'dist',
    'dist': dist_wasserstein,
    'dif': dif_wasserstein,
    'sim': sim_wasserstein,
    'info': info_wasserstein,
    'explain': explain_wasserstein,
}
METRICS['earth_mover'] = METRICS['wasserstein']
METRICS['emd'] = METRICS['wasserstein']
