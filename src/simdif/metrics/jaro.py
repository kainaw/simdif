from ..simdif import Metric, METRICS, to_list
import sys


def info_jaro() -> str:
    return """
Jaro Similarity
---------------
A string similarity based on the number of matching characters and the number
of transpositions between them. Two characters are considered matching only if
they are equal and no farther apart than floor(max(|A|, |B|) / 2) - 1.

Formula:
    sim = 1/3 * ( m/|A| + m/|B| + (m - t) / m )
        m = number of matching characters
        t = number of transpositions (half the out-of-order matches, floored)

Range: [0, 1]
    1 = identical strings, 0 = no matching characters

Note: If the optional `rapidfuzz` package is installed, its `Jaro` similarity
is used on strings for speed; otherwise it is computed locally.
    """.strip()


def explain_jaro(a, b, **kwargs) -> str:
    s1, s2 = to_list(a), to_list(b)
    result = sim_jaro(a, b, **kwargs)
    return f"""
A: ({", ".join(f"'{x}'" for x in s1)})
B: ({", ".join(f"'{y}'" for y in s2)})
Match window: max(0, max({len(s1)}, {len(s2)}) // 2 - 1) = {max(0, max(len(s1), len(s2)) // 2 - 1)}
Jaro Similarity: {result:.4f}
    """.strip()


@Metric
def sim_jaro(a, b, **kwargs) -> float:
    if isinstance(a, str) and isinstance(b, str) and 'rapidfuzz' in sys.modules:
        return float(sys.modules['rapidfuzz'].distance.Jaro.similarity(a, b))
    s1, s2 = to_list(a), to_list(b)
    len1, len2 = len(s1), len(s2)
    if len1 == 0 and len2 == 0:
        return 1.0
    if len1 == 0 or len2 == 0:
        return 0.0

    match_window = max(0, max(len1, len2) // 2 - 1)
    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    for i in range(len1):
        start = max(0, i - match_window)
        end = min(i + match_window + 1, len2)
        for j in range(start, end):
            if not s2_matches[j] and s1[i] == s2[j]:
                s1_matches[i] = s2_matches[j] = True
                matches += 1
                break

    if matches == 0:
        return 0.0

    # Count transpositions (out-of-order matched characters).
    k = 0
    transpositions = 0
    for i in range(len1):
        if s1_matches[i]:
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1

    return (matches / len1 + matches / len2 + (matches - transpositions // 2) / matches) / 3


@Metric
def dif_jaro(a, b, **kwargs) -> float:
    return 1.0 - sim_jaro(a, b, **kwargs)


METRICS['jaro'] = {
    'class': 'sequence',
    'default': 'sim',
    'sim': sim_jaro,
    'dif': dif_jaro,
    'info': info_jaro,
    'explain': explain_jaro,
}
