import sys
from ..simdif import Metric, METRICS, to_list_numeric_aligned
from ._helpers import _sim_from_dist, _dif_from_dist, _max_line, _lib_note


def info_chebyshev() -> str:
    return """
Chebyshev Distance (L-infinity Norm)
------------------------------------
Also known as the Maximum Metric or Chessboard Distance. Only the
single worst-matching coordinate is read; every other dimension is
discarded.

Formula:
    D(A,B) = max(|Ai - Bi|)

It is the limit of the Minkowski distance as p -> infinity. Raising p
weights the larger gaps more heavily, and in the limit the largest
gap is the only one that survives.

The chessboard name is literal: a king moves one square in any
direction, including diagonally, so the minimum number of king moves
between two squares is the Chebyshev distance between their (file,
rank) coordinates.

Roles:
    dist: max |Ai - Bi| (>= 0, unbounded)
    sim:  1 / (1 + D), or 1 - dif when d_max is supplied
    dif:  1 - sim,     or D / d_max when d_max is supplied

Range (dist): [0, inf)
    0 = the vectors are identical

Note: there is no maximum Chebyshev distance on R^n -- coordinates can be
arbitrarily far apart -- so sim defaults to the 1/(1+D) squash and never
quite reaches 0. Supply d_max to rescale linearly instead:

    dif(a, b, 'chebyshev', d_max=1.0)   -> D / 1.0, clamped at 1.0

Chebyshev is the one member of the Lp family where a bound comes for free
once the inputs are bounded: on [0,1]^n the largest possible coordinate gap
is exactly 1, so d_max=1 needs no scaling by n. Range-normalize your columns
(which you should do anyway -- see the scale warning below) and d_max=1 is
exact rather than a guess. For the others the bound grows with the
dimension: n for manhattan, sqrt(n) for euclidean, n^(1/p) for minkowski.

WARNING -- d_max must be a real bound, not a guess. Distances above it all
clamp to dif=1.0, so every pair beyond d_max becomes indistinguishable and
the ordering the metric computed is thrown away. explain_ reports when a
clamp happened.

Note: Aligned and index-by-index, like the rest of the Lp family.
Unequal lengths raise 'Vector length mismatch' unless a pad_value is
supplied (same contract as hamming). Two empty vectors are identical
and score 0.

Note: A true metric -- symmetric, zero only for identical vectors,
and satisfies the triangle inequality, since it is a genuine norm.

WARNING -- one coordinate decides everything. Every dimension except
the worst one is ignored, so [0,0,0,0] vs [0,0,0,9] and [8,8,8,8] vs
[8,8,8,17] both score 9, and you can move all the other coordinates
freely without changing the result until one of them overtakes the
max. That makes Chebyshev brutally outlier-sensitive in exactly the
way hausdorff is at percentile=100: a single bad dimension (one
broken sensor, one mis-scaled feature) becomes the entire answer.
When you want the other dimensions to have a say, use manhattan (all
gaps count equally) or euclidean (larger gaps count more, but
smaller ones still count). explain_ prints the runner-up gap so you
can see how much the winning coordinate is carrying on its own.

Note: Unlike hausdorff, there is deliberately no percentile knob to
trim outliers away. Hausdorff reduces a list of *nearest-neighbour*
distances, and trimming it at 95 is a published metric (HD95);
Chebyshev reduces a list of *aligned coordinate* gaps, where the
trimmed version has no published meaning and stops being a metric at
all. Taking the 75th percentile of the gaps for [0,0,0,5] vs
[0,0,0,0] would report 0 for two vectors that differ, breaking both
identity-of-indiscernibles and the triangle inequality. Reach for
manhattan or euclidean instead: they discount the outlier by
averaging against the other dimensions rather than by deleting it.

WARNING -- scale-sensitive, and more so than the summing metrics.
Whichever dimension carries the largest units wins by default, so a
feature in millimetres will outvote one in metres no matter what the
data says. Standardize the columns first unless they already share a
unit.

Aliases: chessboard, linf
    """.strip()
info_chessboard = info_chebyshev
info_linf = info_chebyshev


def explain_chebyshev(a, b, **kwargs) -> str:
    d_max = kwargs.get('d_max')
    a, b = to_list_numeric_aligned(a, b, **kwargs)
    if not a:
        return f"""
A: {a}
B: {b}
Both vectors are empty, so there are no coordinates to compare.
Chebyshev Distance: 0.0000
{_max_line(0.0, d_max)}
        """.strip()

    diffs = [abs(x - y) for x, y in zip(a, b)]
    max_diff = max(diffs)
    idx = diffs.index(max_diff)
    lines = []
    for i, (x, y) in enumerate(zip(a, b)):
        marker = "  <- the max, this is the whole answer" if i == idx else ""
        lines.append(f"  idx {i}: |{x} - {y}| = {diffs[i]:.4f}{marker}")

    # Everything below the max is discarded, so showing the runner-up makes the
    # size of that discard visible rather than leaving it as a claim in info_.
    rest = diffs[:idx] + diffs[idx + 1:]
    if rest:
        runner_up = max(rest)
        carrying = (f"\nDrop that one coordinate and the distance falls to {runner_up:.4f}"
                    f" -- idx {idx} is carrying {max_diff - runner_up:.4f} of the score"
                    f" on its own.\nEvery other dimension is ignored entirely.")
    else:
        carrying = "\nThere is only one dimension, so it is trivially the max."

    live = dist_chebyshev(a, b, **kwargs)
    lib_name = 'scipy' if 'scipy' in sys.modules else None
    return f"""
A: {a}
B: {b}
Absolute differences per coordinate (only the largest survives):
{chr(10).join(lines)}
{carrying}
Chebyshev Distance: {max_diff:.4f}{_lib_note(max_diff, live, lib_name, 'dist_chebyshev')}
{_max_line(live, d_max)}
    """.strip()
explain_chessboard = explain_chebyshev
explain_linf = explain_chebyshev


@Metric
def dist_chebyshev(a, b, **kwargs) -> float:
    a, b = to_list_numeric_aligned(a, b, **kwargs)
    if not a:
        return 0.0   # two empty vectors are identical
    if 'scipy' in sys.modules:
        from scipy.spatial import distance
        return float(distance.chebyshev(a, b))
    return float(max(abs(x - y) for x, y in zip(a, b)))
dist_chessboard = dist_chebyshev
dist_linf = dist_chebyshev


@Metric
def dif_chebyshev(a, b, **kwargs) -> float:
    return _dif_from_dist(dist_chebyshev(a, b, **kwargs), kwargs.get('d_max'))
dif_chessboard = dif_chebyshev
dif_linf = dif_chebyshev


@Metric
def sim_chebyshev(a, b, **kwargs) -> float:
    return _sim_from_dist(dist_chebyshev(a, b, **kwargs), kwargs.get('d_max'))
sim_chessboard = sim_chebyshev
sim_linf = sim_chebyshev


METRICS['chebyshev'] = {
    'class': 'vector',
    'default': 'dist',
    'dist': dist_chebyshev,
    'dif': dif_chebyshev,
    'sim': sim_chebyshev,
    'info': info_chebyshev,
    'explain': explain_chebyshev,
}
METRICS['chessboard'] = METRICS['chebyshev']
METRICS['linf'] = METRICS['chebyshev']
