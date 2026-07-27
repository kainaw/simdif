from ..simdif import Metric, METRICS, to_list_numeric_aligned


def info_chebyshev() -> str:
    return """
Chebyshev Distance (L-infinity Norm)
------------------------------------
Also known as the Maximum Metric or Chessboard Distance. Only the single
worst-matching coordinate is read; every other dimension is discarded.

Formula:
    D(A,B) = max over i of |Ai - Bi|

It is the limit of the Minkowski distance as p -> infinity. Raising p weights
the larger gaps more heavily, and in the limit the largest gap is the only one
that survives:

    p=1   Manhattan   sum of all gaps        (every dimension counts equally)
    p=2   Euclidean   root of squared gaps   (larger gaps count more)
    p=inf Chebyshev   the single worst gap   (nothing else counts at all)

The chessboard name is literal: a king moves one square in any direction,
including diagonally, so the minimum number of king moves between two squares
is the Chebyshev distance between their (file, rank) coordinates.

Roles:
    dist - max |Ai - Bi| (>= 0, unbounded)
    sim  - 1 / (1 + D)

Range (dist): [0, inf)
    0 = the vectors are identical

Note: Aligned and index-by-index, like the rest of the Lp family. Unequal
lengths raise 'Vector length mismatch' unless a pad_value is supplied (same
contract as hamming). Two empty vectors are identical and score 0.

Note: A true metric -- symmetric, zero only for identical vectors, and it
satisfies the triangle inequality, since it is a genuine norm.

WARNING -- one coordinate decides everything. Every dimension except the
worst one is ignored, so [0,0,0,0] vs [0,0,0,9] and [8,8,8,8] vs [8,8,8,17]
both score 9, and you can move all the other coordinates freely without
changing the result until one of them overtakes the max. That makes Chebyshev
brutally outlier-sensitive in exactly the way hausdorff is at percentile=100:
a single bad dimension -- one broken sensor, one mis-scaled feature -- becomes
the entire answer. When you want the other dimensions to have a say, use
manhattan (all gaps count equally) or euclidean (larger gaps count more, but
smaller ones still count). explain_ prints the runner-up gap so you can see how
much the winning coordinate is carrying on its own.

Note: Unlike hausdorff, there is deliberately no percentile knob to trim that
outlier away. Hausdorff reduces a list of *nearest-neighbour* distances, and
trimming it at 95 is a published metric (HD95); Chebyshev reduces a list of
*aligned coordinate* gaps, where the trimmed version has no published meaning
and stops being a metric at all. Taking the 75th percentile of the gaps for
[0,0,0,5] vs [0,0,0,0] would report 0 for two vectors that differ, breaking
both identity-of-indiscernibles and the triangle inequality. Reach for
manhattan or euclidean instead: they discount the outlier by averaging it
against the other dimensions rather than by deleting it.

WARNING -- scale-sensitive, and more so than the summing metrics. Whichever
dimension carries the largest units wins by default, so a feature in
millimetres will outvote one in metres no matter what the data says.
Standardize the columns first unless they already share a unit.

Aliases: chessboard, linf
    """.strip()
info_chessboard = info_chebyshev
info_linf = info_chebyshev


def explain_chebyshev(a, b, **kwargs) -> str:
    a, b = to_list_numeric_aligned(a, b, **kwargs)
    if not a:
        return f"""
A: {a}
B: {b}
Both vectors are empty, so there are no coordinates to compare.
Chebyshev Distance: 0.0000
Similarity (1 / (1 + d)): 1.0000
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

    return f"""
A: {a}
B: {b}
Absolute differences per coordinate (only the largest survives):
{chr(10).join(lines)}
{carrying}
Chebyshev Distance: {max_diff:.4f}
Similarity (1 / (1 + d)): {1.0 / (1.0 + max_diff):.4f}
    """.strip()
explain_chessboard = explain_chebyshev
explain_linf = explain_chebyshev


@Metric
def dist_chebyshev(a, b, **kwargs) -> float:
    a, b = to_list_numeric_aligned(a, b, **kwargs)
    if not a:
        return 0.0   # two empty vectors are identical
    return float(max(abs(x - y) for x, y in zip(a, b)))
dist_chessboard = dist_chebyshev
dist_linf = dist_chebyshev


@Metric
def sim_chebyshev(a, b, **kwargs) -> float:
    return 1.0 / (1.0 + dist_chebyshev(a, b, **kwargs))
sim_chessboard = sim_chebyshev
sim_linf = sim_chebyshev


METRICS['chebyshev'] = {
    'class': 'vector',
    'default': 'dist',
    'dist': dist_chebyshev,
    'sim': sim_chebyshev,
    'info': info_chebyshev,
    'explain': explain_chebyshev,
}
METRICS['chessboard'] = METRICS['chebyshev']
METRICS['linf'] = METRICS['chebyshev']
