from ..simdif import Metric, METRICS, to_list_numeric


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

Note: `q` (the alphabet size) defaults to max(A, B) + 1 when not supplied. For
q = 2 or q = 3 the Lee distance coincides with the Hamming distance.
    """.strip()


def explain_lee(a, b, q=None, **kwargs) -> str:
    va, vb = to_list_numeric(a), to_list_numeric(b)
    qq = q if q is not None else max(max(va), max(vb)) + 1
    steps = []
    total = 0
    for i, (x, y) in enumerate(zip(va, vb)):
        diff = abs(x - y)
        d = min(diff, qq - diff)
        total += d
        steps.append(f"  idx {i}: min(|{x} - {y}|, {qq} - |{x} - {y}|) = {d}")
    return f"""
A: {va}
B: {vb}
q (alphabet size): {qq}
Per-position circular differences:
{chr(10).join(steps)}
Lee Distance (sum): {total}
    """.strip()


@Metric
def dist_lee(a, b, q=None, **kwargs) -> float:
    a, b = to_list_numeric(a), to_list_numeric(b)
    if q is None:
        q = max(max(a), max(b)) + 1
    distance = 0
    for va, vb in zip(a, b):
        diff = abs(va - vb)
        distance += min(diff, q - diff)
    return distance


@Metric
def sim_lee(a, b, q=None, **kwargs) -> float:
    return 1.0 / (1.0 + dist_lee(a, b, q=q))


METRICS['lee'] = {
    'class': 'vector',
    'default': 'dist',
    'dist': dist_lee,
    'sim': sim_lee,
    'info': info_lee,
    'explain': explain_lee,
}
