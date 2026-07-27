import math
from ..simdif import Metric, METRICS, to_list, to_list_numeric


def info_hausdorff() -> str:
    return """
Hausdorff Distance
------------------
The worst-case nearest miss between two point sets: for every point in A find
its nearest point in B and record that gap, then report the largest such gap.
Repeat from B to A and take the larger of the two directions.

Formula:
    h(A,B) = max over a in A of ( min over b in B of d(a,b) )   [directed]
    H(A,B) = max( h(A,B), h(B,A) )                              [symmetric]

Geometrically, H(A,B) is the smallest radius r such that growing A into a blob
of thickness r would swallow B, and vice versa.

    percentile  which order statistic of the per-point nearest distances to
                report, 0-100 (default 100 = the true maximum). 95 gives the
                HD95 used in image segmentation.
    aggregate   'max' (default) or 'mean' over the retained distances.
                percentile=100 with aggregate='mean' is the modified
                Hausdorff distance (Dubuisson & Jain, 1994).
    dist_fn     pointwise distance between elements (default: absolute
                difference, as in dtw). Supplying one also lifts the
                numeric-input requirement, so tuples work as n-D points:
                dist_fn=lambda p, q: math.dist(p, q)

Roles:
    dist - H(A,B) (>= 0, unbounded)
    sim  - 1 / (1 + H)
    dif  - 1 - sim

Range (dist): [0, inf)
    0 = the two sets contain the same points

Note: BOTH directions are required. A single point sitting inside a large cloud
has a directed distance of ~0 toward the cloud but a large one coming back, so
only the max of the two is symmetric. The module-level helper
directed_hausdorff(a, b, ...) exposes one direction at a time, and explain_
prints both.

Note: Order-blind, and this is the point. Unlike dtw or the alignment metrics,
each point picks its nearest neighbour independently -- the choices need not be
monotone, consistent, or one-to-one. So [1,2,3,4,5] and [5,4,3,2,1] are the
same point set and score 0, where dtw scores 12. Conversely, unlike jaccard and
the other set metrics, near misses count: jaccard([1,2,3],[1.01,2.01,3.01])
sees no shared members at all, while Hausdorff reports 0.01. It is the only
metric here that ignores order yet still measures *how far off* the mismatches
are, and it needs neither equal lengths nor shared elements.

Note: A true metric on compact sets -- it satisfies the triangle inequality and
is 0 only for equal sets (dtw satisfies neither).

WARNING -- the plain maximum is brutally outlier-sensitive. One stray point
sets the whole score: [0,1,2] against [0.5,1.5] is 0.5, but adding a single
far-away point to B makes it 98. That is why practice rarely uses percentile
100. Use percentile=95 (HD95, standard alongside Dice for segmentation
boundaries) or aggregate='mean' (shape matching) on noisy data. Running the
same pair at 100 and then at 95 shows exactly how much one point was carrying.

Note: Percentiles use the nearest-rank convention on the sorted nearest-
distance list (index = ceil(p/100 * n) - 1), so percentile=100 is exactly the
maximum and no interpolation between neighbouring values ever occurs. This may
differ slightly from numpy's default linear interpolation.

Note: A percentile needs enough points to bite. With n points, percentile p
discards nothing unless p < 100*(n-1)/n -- so HD95 requires at least 20 points
per set, and on 6 points percentile=95 is identical to percentile=100. This is
correct behaviour, not a bug (you cannot drop 1 of 6 values without discarding
17% of them), but it does mean small examples can look like the parameter is
being ignored. explain_ says so explicitly when it happens.

Note: An empty set has no nearest neighbours, so H(empty, non-empty) is
infinity. Two empty sets are identical and score 0.

Aliases: hausdorff_distance, hd
    """.strip()
info_hausdorff_distance = info_hd = info_hausdorff


def _points(val, dist_fn=None, **kwargs):
    """Elements to compare. Without a dist_fn the default pointwise distance is
    numeric subtraction, so inputs are coerced to numbers; with one, elements
    pass through untouched so that tuples can act as n-D points."""
    if dist_fn is None:
        return to_list_numeric(val, **kwargs)
    return to_list(val)


def _pointwise(dist_fn):
    return dist_fn if dist_fn is not None else (lambda a, b: abs(a - b))


def _nearest_distances(source, target, cost) -> list:
    """For each point in source, the distance to its closest point in target."""
    return [min(cost(p, q) for q in target) for p in source]


def _reduce(distances, percentile=100, aggregate='max') -> float:
    """Collapse the per-point nearest distances to one number. The percentile
    trims the largest outliers off the sorted list first, then aggregate
    reduces what remains -- so percentile composes with either aggregate rather
    than being silently ignored by one of them."""
    if not distances:
        return 0.0
    if not 0 <= percentile <= 100:
        raise ValueError(f"percentile must be between 0 and 100, got {percentile}")
    if aggregate not in ('max', 'mean'):
        raise ValueError(f"aggregate must be 'max' or 'mean', got {aggregate!r}")
    ordered = sorted(distances)
    # Nearest-rank: percentile=100 lands exactly on the largest value, and no
    # value is ever interpolated into existence.
    cutoff = max(1, math.ceil(percentile / 100 * len(ordered)))
    kept = ordered[:cutoff]
    return max(kept) if aggregate == 'max' else sum(kept) / len(kept)


def directed_hausdorff(a, b, percentile=100, aggregate='max', dist_fn=None, **kwargs) -> float:
    """h(A,B) -- one direction only: how far A's points are from B, ignoring how
    far B's points are from A. Asymmetric by construction; dist_hausdorff takes
    the max of both directions."""
    pa, pb = _points(a, dist_fn, **kwargs), _points(b, dist_fn, **kwargs)
    if not pa:
        return 0.0        # every point of an empty A is trivially covered
    if not pb:
        return math.inf   # ...but no point of A can reach an empty B
    return _reduce(_nearest_distances(pa, pb, _pointwise(dist_fn)), percentile, aggregate)


@Metric
def dist_hausdorff(a, b, percentile=100, aggregate='max', dist_fn=None, **kwargs) -> float:
    forward = directed_hausdorff(a, b, percentile, aggregate, dist_fn, **kwargs)
    backward = directed_hausdorff(b, a, percentile, aggregate, dist_fn, **kwargs)
    return max(forward, backward)
dist_hausdorff_distance = dist_hd = dist_hausdorff


@Metric
def sim_hausdorff(a, b, **kwargs) -> float:
    distance = dist_hausdorff(a, b, **kwargs)
    return 1.0 / (1.0 + distance)
sim_hausdorff_distance = sim_hd = sim_hausdorff


@Metric
def dif_hausdorff(a, b, **kwargs) -> float:
    return 1 - sim_hausdorff(a, b, **kwargs)
dif_hausdorff_distance = dif_hd = dif_hausdorff


def explain_hausdorff(a, b, percentile=100, aggregate='max', dist_fn=None, **kwargs) -> str:
    pa, pb = _points(a, dist_fn, **kwargs), _points(b, dist_fn, **kwargs)
    cost = _pointwise(dist_fn)

    def describe(source, target, label_s, label_t):
        if not source:
            return f"  {label_s} is empty -> h({label_s},{label_t}) = 0 (nothing to cover)"
        if not target:
            return f"  {label_t} is empty -> h({label_s},{label_t}) = inf (no nearest neighbour exists)"
        gaps = _nearest_distances(source, target, cost)
        lines = [f"  {p!r} -> nearest in {label_t} at {gap:.4f}" for p, gap in zip(source, gaps)]
        ordered = sorted(gaps)
        cutoff = max(1, math.ceil(percentile / 100 * len(ordered)))
        dropped = len(ordered) - cutoff
        detail = f"  sorted gaps: {[round(g, 4) for g in ordered]}"
        if dropped:
            detail += f"\n  percentile {percentile} keeps the lowest {cutoff} of {len(ordered)}, dropping {dropped}"
        elif percentile < 100:
            # Silent no-op otherwise: with few points, a high percentile cannot
            # drop anything, so a student would see percentile= change nothing
            # and have no way to know why.
            need = 100 * (len(ordered) - 1) / len(ordered)
            detail += (f"\n  percentile {percentile} drops NOTHING here: with only {len(ordered)} points,"
                       f"\n  dropping even one requires percentile < {need:.1f}")
        result = _reduce(gaps, percentile, aggregate)
        return "\n".join(lines + [detail, f"  h({label_s},{label_t}) = {aggregate} of kept gaps = {result:.4f}"])

    forward = directed_hausdorff(a, b, percentile, aggregate, dist_fn, **kwargs)
    backward = directed_hausdorff(b, a, percentile, aggregate, dist_fn, **kwargs)
    result = dist_hausdorff(a, b, percentile, aggregate, dist_fn, **kwargs)
    settings = f"percentile={percentile}, aggregate={aggregate!r}"
    if percentile == 100 and aggregate == 'max':
        settings += " (the true Hausdorff distance)"
    return f"""
A: {pa}
B: {pb}
Settings: {settings}
Each point takes its OWN nearest neighbour -- no ordering or consistency is
required between those choices, which is what makes this order-blind.

Direction A -> B:
{describe(pa, pb, 'A', 'B')}

Direction B -> A:
{describe(pb, pa, 'B', 'A')}

Hausdorff = max(h(A,B), h(B,A)) = max({forward:.4f}, {backward:.4f}) = {result:.4f}
Similarity 1/(1+H): {sim_hausdorff(a, b, percentile=percentile, aggregate=aggregate, dist_fn=dist_fn, **kwargs):.4f}
Difference (dif): {dif_hausdorff(a, b, percentile=percentile, aggregate=aggregate, dist_fn=dist_fn, **kwargs):.4f}
    """.strip()
explain_hausdorff_distance = explain_hd = explain_hausdorff


METRICS['hausdorff'] = {
    # 'vector' matches energy/welch_t: numeric inputs that are NOT aligned
    # index-by-index and need not share a length.
    'class': 'vector',
    'default': 'dist',
    'dist': dist_hausdorff,
    'sim': sim_hausdorff,
    'dif': dif_hausdorff,
    'info': info_hausdorff,
    'explain': explain_hausdorff,
}
METRICS['hausdorff_distance'] = METRICS['hd'] = METRICS['hausdorff']
