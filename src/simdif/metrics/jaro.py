from ..simdif import Metric, METRICS, to_list
from ._helpers import _lib_note
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
    len1, len2 = len(s1), len(s2)

    if len1 == 0 or len2 == 0:
        sim = 1.0 if (len1 == 0 and len2 == 0) else 0.0
        return f"""
A: ({", ".join(f"'{x}'" for x in s1)})
B: ({", ".join(f"'{y}'" for y in s2)})
One sequence is empty -> Jaro Similarity: {sim:.4f}
    """.strip()

    w = max(0, max(len1, len2) // 2 - 1)

    # Replay the matching pass, recording each window and where it matched.
    s1_matches = [False] * len1
    s2_matches = [False] * len2
    windows = []  # (start, end, matched_j or None) per position i in A
    matches = 0
    for i in range(len1):
        start = max(0, i - w)
        end = min(i + w + 1, len2)
        matched_j = None
        for j in range(start, end):
            if not s2_matches[j] and s1[i] == s2[j]:
                s1_matches[i] = s2_matches[j] = True
                matched_j = j
                matches += 1
                break
        windows.append((start, end, matched_j))

    # Column-aligned view of the window ('-') sliding over B, '#' where it lands.
    b_strs = [str(y) for y in s2]
    col_w = max(max((len(s) for s in b_strs), default=1), len(str(len2 - 1)), 1)
    def row(cells):
        return " ".join(str(c).rjust(col_w) for c in cells)
    viz = [f"{'':<12}{row(range(len2))}   (B index)",
           f"{'':<12}{row(b_strs)}   (B)"]
    for i, (start, end, mj) in enumerate(windows):
        marks = ['#' if j == mj else '-' if start <= j < end else '.' for j in range(len2)]
        note = f"match B[{mj}]={s2[mj]!r}" if mj is not None else "no match in window"
        label = f"A[{i}]={s1[i]!r}"
        viz.append(f"{label:<12}{row(marks)}   {note}")

    # Transpositions: matched characters that appear out of order (halved).
    k = 0
    out_of_order = 0
    for i in range(len1):
        if s1_matches[i]:
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                out_of_order += 1
            k += 1
    t = out_of_order // 2

    sim = (matches / len1 + matches / len2 + (matches - t) / matches) / 3 if matches else 0.0
    matched_a = " ".join(str(s1[i]) if s1_matches[i] else "_" for i in range(len1))
    matched_b = " ".join(str(s2[j]) if s2_matches[j] else "_" for j in range(len2))
    live = sim_jaro(a, b, **kwargs)
    lib_name = 'rapidfuzz' if isinstance(a, str) and isinstance(b, str) and 'rapidfuzz' in sys.modules else None

    return f"""
A: ({", ".join(f"'{x}'" for x in s1)})
B: ({", ".join(f"'{y}'" for y in s2)})
Match window radius w = max(0, max({len1}, {len2}) // 2 - 1) = {w}
  A[i] may match B[j] only for j in [i-w, i+w].  '-' in window, '#' matched, '.' outside:

{chr(10).join(viz)}

Matched in A: {matched_a}
Matched in B: {matched_b}
m (matches): {matches}
t (transpositions): {out_of_order} out-of-order // 2 = {t}
sim = 1/3 * (m/|A| + m/|B| + (m - t)/m)
    = 1/3 * ({matches}/{len1} + {matches}/{len2} + ({matches} - {t})/{matches})
    = {sim:.4f}{_lib_note(sim, live, lib_name, 'sim_jaro')}
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
