from ..simdif import Metric, METRICS, to_list, to_binary


def info_tanimoto() -> str:
    return """
Tanimoto Coefficient (Binary / Bit-vector)
-------------------------------------------
Measures the similarity of two equal-length bit vectors as the ratio of shared
"on" bits to the total number of distinct "on" bits. With binary=True the inputs
are integers interpreted as bitmasks; otherwise they are treated as aligned
0/1 vectors.

Formula:
    T(A, B) = c / (a + b - c)
        a = number of 1s in A
        b = number of 1s in B
        c = number of positions where both are 1

Range: [0, 1]
    1 = identical bit patterns (two empty vectors are defined as 1.0)

Note: this is algebraically IDENTICAL to the Jaccard index - c/(a+b-c) equals
|A n B| / |A u B|. It is the same coefficient computed on 0/1 vectors (or
integer bitmasks) instead of sets. For count or weighted vectors, use
tanimoto_continuous, which generalizes this and reduces back to it on 0/1 input.

Aliases: binary_tanimoto, tanimoto_binary
    """.strip()
info_binary_tanimoto = info_tanimoto_binary = info_tanimoto


def explain_tanimoto(a, b, binary=False, **kwargs) -> str:
    if binary:
        width = max(a.bit_length(), b.bit_length())
        va, vb = to_binary(a, width), to_binary(b, width)
    else:
        va, vb = to_list(a), to_list(b)
    intersection = sum(x == y == 1 for x, y in zip(va, vb))
    denom = sum(va) + sum(vb) - intersection
    return f"""
A bits: {va}
B bits: {vb}
Shared 1-bits (c): {intersection}
Sum A (a): {sum(va)}   Sum B (b): {sum(vb)}
Tanimoto: c / (a + b - c) = {intersection} / {denom} = {sim_tanimoto(a, b, binary):.4f}
    """.strip()
explain_binary_tanimoto = explain_tanimoto_binary = explain_tanimoto


@Metric
def sim_tanimoto(a, b, binary=False) -> float:
    if binary:
        if not isinstance(a, int) or not isinstance(b, int):
            raise TypeError("binary=True requires integer inputs")
        width = max(a.bit_length(), b.bit_length())
        a, b = to_binary(a, width), to_binary(b, width)
    else:
        a, b = to_list(a), to_list(b)
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
    if len(a) == 0:
        return 1.0
    intersection = sum(x == y == 1 for x, y in zip(a, b))
    return intersection / (sum(a) + sum(b) - intersection)
sim_binary_tanimoto = sim_tanimoto_binary = sim_tanimoto


@Metric
def dif_tanimoto(a, b, binary=False) -> float:
    return 1 - sim_tanimoto(a, b, binary)
dif_binary_tanimoto = dif_tanimoto_binary = dif_tanimoto


METRICS['tanimoto'] = {
    'class': 'vector',
    'default': 'sim',
    'sim': sim_tanimoto,
    'dif': dif_tanimoto,
    'info': info_tanimoto,
    'explain': explain_tanimoto,
}
METRICS['binary_tanimoto'] = METRICS['tanimoto_binary'] = METRICS['tanimoto']
