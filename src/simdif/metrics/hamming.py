from ..simdif import Metric, METRICS, to_list_aligned, to_binary


def info_hamming() -> str:
    return """
Hamming Distance
----------------
Measures how many positions in two ordered sets have different values.

Formula:
    H(A, B) = Σ[A <> B]

Range: [0, ∞)
    0 = identical sets
    """.strip()


def explain_hamming(a, b, **kwargs) -> str:
    binary = kwargs.get('binary', False)
    if binary:
        if not isinstance(a, int) or not isinstance(b, int):
            raise TypeError("binary=True requires integer inputs")
        width = max(a.bit_length(), b.bit_length())
        a, b = to_binary(a, width), to_binary(b, width)
    else:
        a, b = to_list_aligned(a, b, **kwargs)
    a_str = [str(x) for x in a]
    b_str = [str(x) for x in b]
    mismatches = ["1" if x != y else "0" for x, y in zip(a, b)]
    col_width = max(
        max(len(s) for s in a_str),
        max(len(s) for s in b_str),
        1
    )
    def join_row(values):
        return " ".join(s.rjust(col_width) for s in values)
    row_a = join_row(a_str)
    row_b = join_row(b_str)
    row_m = join_row(mismatches)
    total = sum(int(x) for x in mismatches)
    return (
        "A: " + row_a + "\n"
        "B: " + row_b + "\n"
        "M: " + row_m + "  (1 = mismatch, 0 = match)\n"
        f"Sum of mismatches = {total}"
    )


@Metric
def dist_hamming(a, b, **kwargs) -> float:
    binary = kwargs.get('binary', False)
    if binary:
        if not isinstance(a, int) or not isinstance(b, int):
            raise TypeError("binary=True requires integer inputs")
        width = max(a.bit_length(), b.bit_length())
        a, b = to_binary(a, width), to_binary(b, width)
    else:
        a, b = to_list_aligned(a, b, **kwargs)
    return sum(x != y for x, y in zip(a, b))


@Metric
def dif_hamming(a, b, **kwargs) -> float:
    if binary:
        n = len(to_binary(a))
    else:
        n = len(to_list(a))
    return dist_hamming(a, b, **kwargs) / n


@Metric
def sim_hamming(a, b, **kwargs) -> float:
    return 1 - dif_hamming(a, b, **kwargs)


METRICS['hamming'] = {
    'class': 'sequence',
    'default': 'dist',
    'dist': dist_hamming,
    'dif': dif_hamming,
    'sim': sim_hamming,
    'info': info_hamming,
    'explain': explain_hamming,
}
