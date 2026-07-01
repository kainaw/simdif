from ..simdif import Metric, METRICS, to_list
from .jaro import sim_jaro
import sys

# Winkler's boost threshold: the common-prefix bonus is only applied when the
# base Jaro similarity already exceeds this value.
BOOST_THRESHOLD = 0.7


def info_jaro_winkler() -> str:
    return """
Jaro-Winkler Similarity
-----------------------
An extension of Jaro similarity that rewards strings sharing a common prefix,
which makes it well suited to matching human names.

Formula:
    sim = jaro + (l * p * (1 - jaro))       if jaro > 0.7
        = jaro                              otherwise
        l = length of the common prefix (capped at max_l, default 4)
        p = prefix scaling factor (default 0.1, must be <= 0.25)

Range: [0, 1]

Note: The prefix bonus is only applied when the base Jaro score exceeds the
Winkler boost threshold (0.7). If the optional `rapidfuzz` package is installed
and default parameters are used, its `JaroWinkler` similarity is used on strings
for speed; otherwise it is computed locally.
    """.strip()


def explain_jaro_winkler(a, b, p=0.1, max_l=4, **kwargs) -> str:
    s1, s2 = to_list(a), to_list(b)
    j = sim_jaro(a, b)
    l = 0
    for c1, c2 in zip(s1[:max_l], s2[:max_l]):
        if c1 == c2:
            l += 1
        else:
            break
    boosted = j > BOOST_THRESHOLD
    return f"""
A: ({", ".join(f"'{x}'" for x in s1)})
B: ({", ".join(f"'{y}'" for y in s2)})
Base Jaro: {j:.4f}
Common prefix length (max {max_l}): {l}
Boost applied (Jaro > {BOOST_THRESHOLD}): {boosted}
Jaro-Winkler Similarity: {sim_jaro_winkler(a, b, p, max_l, **kwargs):.4f}
    """.strip()


@Metric
def sim_jaro_winkler(a, b, p=0.1, max_l=4, **kwargs) -> float:
    if p > 0.25:
        raise ValueError("p should not exceed 0.25 to keep score within [0, 1]")
    # rapidfuzz caps the prefix length at 4; only delegate when that matches max_l.
    if isinstance(a, str) and isinstance(b, str) and max_l == 4 and 'rapidfuzz' in sys.modules:
        return float(sys.modules['rapidfuzz'].distance.JaroWinkler.similarity(a, b, prefix_weight=p))
    s1, s2 = to_list(a), to_list(b)
    j = sim_jaro(s1, s2)
    if j <= BOOST_THRESHOLD:
        return j
    l = 0
    for c1, c2 in zip(s1[:max_l], s2[:max_l]):
        if c1 == c2:
            l += 1
        else:
            break
    return j + (l * p * (1 - j))


@Metric
def dif_jaro_winkler(a, b, p=0.1, max_l=4, **kwargs) -> float:
    return 1 - sim_jaro_winkler(a, b, p, max_l, **kwargs)


METRICS['jaro_winkler'] = {
    'class': 'sequence',
    'default': 'sim',
    'sim': sim_jaro_winkler,
    'dif': dif_jaro_winkler,
    'info': info_jaro_winkler,
    'explain': explain_jaro_winkler,
}
