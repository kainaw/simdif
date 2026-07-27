import math
from ..simdif import Metric, METRICS, to_list_numeric_aligned
from ._helpers import _bounded_dif


def _resolve_q(va, vb, q):
    """The alphabet size, defaulting to one past the largest symbol present."""
    return q if q is not None else max(max(va), max(vb)) + 1


def _lee_max(va, q):
    """Largest possible Lee distance: every position maximally wrapped.

    A single position tops out at floor(q/2) -- the half-way point of the
    circle, since beyond it the wrap-around direction is the shorter one.
    """
    return math.floor(q / 2) * len(va)


def info_lee() -> str:
    return """
Lee Distance
------------
A distance between two equal-length strings over a q-ary alphabet {0, ..., q-1},
used in coding theory. For each position it takes the circular (wrap-around)
difference of the symbols and sums them.

Formula:
    D(A, B) = sum( min(|Ai - Bi|, q - |Ai - Bi|) )

Range: [0, floor(q/2) * n]  for length-n vectors

Roles:
    dist: the raw sum above
    dif:  dist / (floor(q/2) * n)
    sim:  1 - dif

The maximum is known in closed form -- one position can differ by at most
floor(q/2), because past the half-way point of the circle the wrap-around
direction is the shorter one -- so no 1/(1+d) squash and no supplied bound
are needed: sim + dif == 1.

Note: `q` (the alphabet size) defaults to max(A, B) + 1 when not supplied. For
q = 2 or q = 3 the Lee distance coincides with the Hamming distance.

WARNING -- the default q is inferred from the data, so it is the largest
symbol that actually appears, not the alphabet you designed. Over Z_16, the
pair [1] vs [2] infers q = 3 and reports dif = 1.0 (fully different) when
against the real alphabet it is 1/8. Pass q explicitly whenever the vectors
may not exercise the full alphabet -- dist is only mildly affected, but dif
and sim are scaled by it directly.
    """.strip()


def explain_lee(a, b, q=None, **kwargs) -> str:
    va, vb = to_list_numeric_aligned(a, b, **kwargs)
    qq = _resolve_q(va, vb, q)
    steps = []
    total = 0
    for i, (x, y) in enumerate(zip(va, vb)):
        diff = abs(x - y)
        d = min(diff, qq - diff)
        total += d
        steps.append(f"  idx {i}: min(|{x} - {y}|, {qq} - |{x} - {y}|) = {d}")
    d_max = _lee_max(va, qq)
    source = "supplied" if q is not None else "inferred from the data"
    dif = _bounded_dif(total, d_max)
    return f"""
A: {va}
B: {vb}
q (alphabet size): {qq} ({source})
Per-position circular differences:
{chr(10).join(steps)}
Lee Distance (sum): {total}
Maximum: floor({qq}/2) * {len(va)} = {d_max} (derived)
Difference (dist / max): {total} / {d_max} = {dif:.4f}
Similarity (1 - dif): {1.0 - dif:.4f}
    """.strip()


@Metric
def dist_lee(a, b, q=None, **kwargs) -> float:
    a, b = to_list_numeric_aligned(a, b, **kwargs)
    q = _resolve_q(a, b, q)
    distance = 0
    for va, vb in zip(a, b):
        diff = abs(va - vb)
        distance += min(diff, q - diff)
    return distance


@Metric
def dif_lee(a, b, q=None, **kwargs) -> float:
    a, b = to_list_numeric_aligned(a, b, **kwargs)
    qq = _resolve_q(a, b, q)
    return _bounded_dif(dist_lee(a, b, q=qq, **kwargs), _lee_max(a, qq))


@Metric
def sim_lee(a, b, q=None, **kwargs) -> float:
    return 1.0 - dif_lee(a, b, q=q, **kwargs)


METRICS['lee'] = {
    'class': 'vector',
    'default': 'dist',
    'dist': dist_lee,
    'dif': dif_lee,
    'sim': sim_lee,
    'info': info_lee,
    'explain': explain_lee,
}
