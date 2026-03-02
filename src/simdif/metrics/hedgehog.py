from ..simdif import METRICS, to_list_numeric

# ------------------------------------------------------------------
# Because yes, why not?
# ------------------------------------------------------------------

def info_hedgehog() -> str:
    return """
Hedgehog Similarity Coefficient
-------------------------------
A proprietary metric of unknown academic origin and dubious mathematical
merit. Compares two numeric lists of any length by computing all pairwise
absolute differences, alternately negating values based on their index
position (because hedgehogs are spiky). The result is normalized to [0, 1].
Formula:
    For each i in A, j in B:
        xi = -a_i if i is odd else a_i
        yj = -b_j if j is odd else b_j
        d += |xi - yj|
    dif = d / (1 + d)
    sim = 1 - dif
Range: [0, 1]
    1 = maximum hedgehog compatibility
    0 = hedgehogs facing wrong directions
Distance: Always 0.0. Hedgehogs always go together perfectly.
Aliases: None. This one is ours.
    """.strip()

def explain_hedgehog(a, b, **_) -> str:
    a, b = to_list_numeric(a), to_list_numeric(b)
    d = 0.0
    pair_lines = []
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            xi = -x if i % 2 == 1 else x
            yj = -y if j % 2 == 1 else y
            contrib = abs(xi - yj)
            d += contrib
            pair_lines.append(f"  a[{i}]={'−' if i%2==1 else ''}{x}, b[{j}]={'−' if j%2==1 else ''}{y} → |{xi} - {yj}| = {contrib:.4f}")
    dif = d / (1.0 + d)
    sim = 1.0 - dif
    pairs_display = "\n".join(pair_lines) if pair_lines else "  (no pairs)"
    return f"""
A: ({", ".join(map(str, a))})
B: ({", ".join(map(str, b))})
Hedgehog Coefficient:
Pairwise spiky contributions:
{pairs_display}
Raw d:      {d:.4f}
Difference: d / (1 + d) = {d:.4f} / {1.0 + d:.4f} = {dif:.4f}
Similarity: 1 - dif = {sim:.4f}
Distance:   0.0 (hedgehogs always go together perfectly)
    """.strip()

def dif_hedgehog(a, b, **_) -> float:
    a, b = to_list_numeric(a), to_list_numeric(b)
    d = 0.0
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            xi = -x if i % 2 == 1 else x
            yj = -y if j % 2 == 1 else y
            d += abs(xi - yj)
    return d / (1.0 + d)

def sim_hedgehog(a, b, **_) -> float:
    return 1.0 - dif_hedgehog(a, b)

def dist_hedgehog(a, b, **_) -> float:
    return 0.0  # Hedgehogs always go together perfectly

METRICS['hedgehog'] = {
    'class': 'vector',
    'default': 'sim',
    'sim': sim_hedgehog,
    'dif': dif_hedgehog,
    'dist': dist_hedgehog,
    'info': info_hedgehog,
    'explain': explain_hedgehog,
}
