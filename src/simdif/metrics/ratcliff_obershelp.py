from ..simdif import Metric, METRICS, to_list


def info_ratcliff_obershelp() -> str:
    return """
Ratcliff-Obershelp (Gestalt Pattern Matching)
---------------------------------------------
A similarity based on shared contiguous runs rather than edit operations. It
finds the longest matching substring, then recurses into the unmatched pieces on
each side (left of the match and right of it), summing the lengths of every
matching block found.

Formula:
    sim = 2 * M / (|A| + |B|)
    where M is the total number of matched elements across all blocks.

Range: [0, 1]
    1 = identical, 0 = no shared elements

Roles:
    sim   - 2M / (|A| + |B|)
    dif   - 1 - sim
    trace - the matching blocks as (piece, a_index, b_index, length) tuples

Note: This is the algorithm behind Python's difflib.SequenceMatcher.ratio()
(without its 'autojunk' heuristic). It is GREEDY - longest-match-first recursion
can miss a globally larger set of matches - which makes a nice contrast with the
DP-optimal edit distances. Uses only '==', so it applies to any comparable
elements. On ties the earliest-starting longest match is chosen.

Origin: Ratcliff & Metzener, Dr. Dobb's Journal, 1988.

Aliases: Gestalt, RO, Ratcliff, Obershelp
    """.strip()
info_gestalt = info_ro = info_ratcliff = info_obershelp = info_ratcliff_obershelp


def _longest_match(s1, s2, alo, ahi, blo, bhi):
    """Longest matching contiguous block within s1[alo:ahi] and s2[blo:bhi].

    Returns (best_i, best_j, best_size). On ties, the match with the smallest
    best_i (then smallest best_j) wins. Mirrors difflib's core, without junk."""
    best_i, best_j, best_size = alo, blo, 0
    j2len = {}
    for i in range(alo, ahi):
        newj2len = {}
        si = s1[i]
        for j in range(blo, bhi):
            if s2[j] == si:
                k = j2len.get(j - 1, 0) + 1
                newj2len[j] = k
                if k > best_size:
                    best_i, best_j, best_size = i - k + 1, j - k + 1, k
        j2len = newj2len
    return best_i, best_j, best_size


def _matching_blocks(s1, s2):
    """All matching blocks as (a_index, b_index, size), in position order."""
    blocks = []
    queue = [(0, len(s1), 0, len(s2))]
    while queue:
        alo, ahi, blo, bhi = queue.pop()
        i, j, k = _longest_match(s1, s2, alo, ahi, blo, bhi)
        if k > 0:
            blocks.append((i, j, k))
            if alo < i and blo < j:
                queue.append((alo, i, blo, j))
            if i + k < ahi and j + k < bhi:
                queue.append((i + k, ahi, j + k, bhi))
    blocks.sort()
    return blocks


def explain_ratcliff_obershelp(a, b, **kwargs) -> str:
    s1, s2 = to_list(a), to_list(b)
    blocks = _matching_blocks(s1, s2)
    m = sum(k for _, _, k in blocks)
    total = len(s1) + len(s2)
    sim = 2.0 * m / total if total else 1.0
    as_str = isinstance(a, str) and isinstance(b, str)
    lines = []
    for i, j, k in blocks:
        piece = s1[i:i + k]
        piece = "".join(str(x) for x in piece) if as_str else piece
        lines.append(f"  {piece!r}  (A[{i}:{i + k}], B[{j}:{j + k}], length {k})")
    body = "\n".join(lines) if lines else "  (no matching blocks)"
    return f"""
A: ({", ".join(f"'{x}'" for x in s1)})
B: ({", ".join(f"'{y}'" for y in s2)})
Matching blocks (in position order):
{body}
Total matched (M): {m}
Similarity: 2*M / (|A| + |B|) = 2*{m} / ({len(s1)} + {len(s2)}) = {sim:.4f}
    """.strip()
explain_gestalt = explain_ro = explain_ratcliff = explain_obershelp = explain_ratcliff_obershelp


@Metric
def sim_ratcliff_obershelp(a, b, **kwargs) -> float:
    s1, s2 = to_list(a), to_list(b)
    total = len(s1) + len(s2)
    if total == 0:
        return 1.0  # two empty sequences are identical
    m = sum(k for _, _, k in _matching_blocks(s1, s2))
    return 2.0 * m / total
sim_gestalt = sim_ro = sim_ratcliff = sim_obershelp = sim_ratcliff_obershelp


@Metric
def dif_ratcliff_obershelp(a, b, **kwargs) -> float:
    return 1.0 - sim_ratcliff_obershelp(a, b, **kwargs)
dif_gestalt = dif_ro = dif_ratcliff = dif_obershelp = dif_ratcliff_obershelp


def trace_ratcliff_obershelp(a, b, **kwargs):
    """The matching blocks as (piece, a_index, b_index, length) tuples. Each
    piece is a str when both inputs are strings, else a list of elements."""
    s1, s2 = to_list(a), to_list(b)
    as_str = isinstance(a, str) and isinstance(b, str)
    out = []
    for i, j, k in _matching_blocks(s1, s2):
        piece = s1[i:i + k]
        if as_str:
            piece = "".join(str(x) for x in piece)
        out.append((piece, i, j, k))
    return out
trace_gestalt = trace_ro = trace_ratcliff = trace_obershelp = trace_ratcliff_obershelp


METRICS['ratcliff_obershelp'] = {
    'class': 'sequence',
    'default': 'sim',
    'sim': sim_ratcliff_obershelp,
    'dif': dif_ratcliff_obershelp,
    'trace': trace_ratcliff_obershelp,
    'info': info_ratcliff_obershelp,
    'explain': explain_ratcliff_obershelp,
}
METRICS['gestalt'] = METRICS['ro'] = METRICS['ratcliff'] = METRICS['obershelp'] = METRICS['ratcliff_obershelp']
