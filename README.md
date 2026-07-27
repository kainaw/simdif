# simdif

**simdif** is a pure-Python library for computing, comparing, and understanding similarity, difference, and distance metrics. It started as a collection of similarity and difference metrics, and has grown to include distance metrics and alignment scores (Smith-Waterman, Needleman-Wunsch, Affine Gap). The design goal is **education first**: every metric ships with a step-by-step explanation function and a plain-English definition function so you can see exactly how a score is derived.

> ⚠️ **Not intended for large-scale or production workloads.** simdif is pure Python and prioritises clarity over speed. It is well-suited for classroom use, small experiments, and learning how metrics work - not for processing large datasets or performance-critical pipelines.

---

## Highlights

- **50+ metrics** under one unified interface
- **Three input classes**: sets, sequences, and vectors
- **Broad input support**: plain Python lists, strings, sets, numeric lists, and NumPy / SciPy / TensorFlow arrays/tensors
- **Compare metrics side-by-side** by passing a list of metric names to `simdif()`
- **`explain_<metric>()`** - walks through the calculation step by step and returns the score
- **`info_<metric>()`** - returns a plain-English definition of the metric
- **Alias-friendly**: most metrics have multiple accepted names (e.g. `'dice'`, `'sorensen'`, `'dice_sorensen'`, `'sorensen_dice'` all work)
- **Optional fast paths**: if a faster library (e.g. `python-Levenshtein`) is installed, simdif will lazy-import and use it automatically - no configuration needed

---

## Installation

```bash
pip install simdif
```

For development (editable install from a clone):

```bash
git clone https://github.com/kainaw/simdif.git
cd simdif
pip install -e .
```

> Optional speedup libraries (simdif will use them automatically if present):
> ```bash
> pip install python-Levenshtein
> ```

---

## Quickstart

### Compare multiple metrics at once

```python
from simdif import simdif

result = simdif(
    "Hedge", "Hog",
    ['jaccard', 'dice', 'cosine', 'levenshtein', 'pearson'],
    ascii=True,
    pad_value='0'
)
print(result)
```

`simdif()` returns a single dictionary keyed by metric name so you can compare results directly:

```python
{
    'jaccard':      0.375,
    'dice':         0.545,
    'cosine':       0.756,
    'levenshtein':  3,
    'pearson':      0.823
}
```

### Understand a metric step by step

```python
from simdif import explain_jaccard

explain_jaccard("Hedge", "Hog", ascii=True, pad_value='0')
```

```
Input A (as set): {'H', 'e', 'd', 'g'}
Input B (as set): {'H', 'o', 'g'}

Intersection |A ∩ B|: {'H', 'g'} → 2
Union        |A ∪ B|: {'H', 'e', 'd', 'g', 'o'} → 5

Jaccard = |A ∩ B| / |A ∪ B| = 2 / 5 = 0.4
```

### Look up a metric definition

```python
from simdif import info_jaccard

info_jaccard()
```

```
Jaccard Similarity (Jaccard Index / IoU)
----------------------------------------
Measures the overlap between two sets as a fraction of their union.
Range: 0 (no overlap) to 1 (identical sets).
Common uses: document similarity, recommendation systems, clustering evaluation.
```

---

## Input Classes

simdif organises metrics into three classes based on what kind of input they operate on. Passing the wrong type of input for a metric will raise an informative error.

| Class | What it operates on | Example inputs |
|---|---|---|
| **Set** | Unordered collections; duplicates ignored | `{'a','b','c'}`, string→set, list→set |
| **Sequence** | Ordered elements; position matters | strings, lists, tuples |
| **Vector** | Numeric arrays; magnitude and direction matter | lists of numbers, NumPy arrays, SciPy sparse, TensorFlow tensors |

### Input helpers

**simdif** is designed to be "input-agnostic." It features a deterministic casting engine that transparently translates complex data structures into the primitive types required by each metric. Whether your data lives in a standard library collection or a specialized scientific framework, simdif ensures a consistent interface.

If a single word is provided, such as "hedgehog", it will be converted to a list of characters: ['h','e','d','g','e','h','o','g'].
If a word is provided as part of a list, such as ["hedge","hog"], each value will remain a string.

Many metrics require numeric input. If the optional parameter `ascii` is set to True, characters will be converted to ASCII values for the characters.
For example "HELLO" becomes [72, 69, 76, 76, 79].

Many metrics require the lists to be the same length. If the optional parameter `pad_value` is given a value, that will be used to pad a short list so it is the same length as a long list.
For example, if A=[1, 2, 3] and B=[4, 5]. If `pad_value` is 0, B will become [4, 5, 0].

If the optional parameter `binary` is set to True and an integer value is provided by itself, not in a list, the integer will be converted to binary.
For example, 42 becomes [1, 0, 1, 0, 1, 0].

Input helpers are useful for comparing multiple metrics with conflicting input requirements in one function call:

```
a = 'hedge'
b = 'hog'
print(simdif.simdif(
    a, b,
    ['jaccard','dice','levenshtein','soundex', 'cosine','hamming'],
    pad_value="0", ascii=True
))

```

The two strings are turned into lists. Where appropriate, the ASCII characters are turned into integers. Where needed, the shorter list is padded with zero.

```
{
    'jaccard': 0.4,
    'dice': 0.5714285714285714,
    'levenshtein': 3,
    'soundex': 0.0,
    'cosine': 0.7729941672513129,
    'hamming': 4
}
```

---

## Gram Utilities

Not every useful variant needs its own metric file. `to_qgram`, `to_skipgram`, and `to_count_vector` are **tokenizers, not metrics** - they turn a sequence into grams, and you compose the result with whatever `set` or `vector` metric already does what you want (`jaccard`, `cosine_set`, `cosine`, `manhattan`, ...). This is how simdif covers q-gram distance, n-gram cosine, skip-gram similarity, and Jaccard-on-q-grams without four near-duplicate metric files.

> ⚠️ **`to_skipgram` is a brute-force, unoptimized enumeration** (consistent with simdif's overall design goal of clarity over speed). Fine for classroom-length strings; combinatorially expensive for long sequences or a large `k`.

### `to_qgram(val, n=2, pad=None)`

Breaks a sequence into contiguous n-grams ("shingles"). Returns a list of joined strings for string input, or tuples otherwise. Sequences shorter than `n` return an empty list unless padded long enough to reach it.

```python
from simdif import to_qgram

to_qgram("hello", 2)          # ['he', 'el', 'll', 'lo']
to_qgram("hi", 2, pad="$")    # ['$h', 'hi', 'i$']
                               # (pad both ends so edge characters appear in as many
                               #  grams as interior ones)
```

### `to_skipgram(val, n=2, k=1)`

Like `to_qgram`, but allows up to `k` skipped elements between each consecutively-chosen position (k-skip-n-grams). `k=0` is identical to `to_qgram(val, n)`.

```python
from simdif import to_skipgram

to_skipgram("ABCD", 2, 1)     # ['AB', 'AC', 'BC', 'BD', 'CD']
```

### `to_count_vector(a, b)`

Turns two gram lists (or any two lists of hashable tokens) into two aligned numeric count vectors over their combined vocabulary. This is the piece `cosine` / `manhattan` / `minkowski` need - they compare counts position by position - that `jaccard` / `cosine_set` don't, since those only care about set membership.

```python
from simdif import to_count_vector

to_count_vector(['ni', 'ig', 'gh', 'ht'], ['na', 'ac', 'ch', 'ht'])
# ([0, 0, 1, 1, 1, 0, 1], [1, 1, 0, 1, 0, 1, 0])
```

### Composing gram-based metrics

| Want | Compose as |
|---|---|
| Jaccard on q-grams | `sim(to_qgram(a, n), to_qgram(b, n), 'jaccard')` |
| Skip-gram similarity | `sim(to_skipgram(a, n, k), to_skipgram(b, n, k), 'jaccard')` |
| N-gram cosine | `sim(*to_count_vector(to_qgram(a, n), to_qgram(b, n)), 'cosine')` |
| Q-gram distance | `dist(*to_count_vector(to_qgram(a, n), to_qgram(b, n)), 'manhattan')` |

```python
from simdif import to_qgram, to_skipgram, to_count_vector, sim, dist

a, b = "night", "nacht"
qa, qb = to_qgram(a, 2), to_qgram(b, 2)

sim(qa, qb, 'jaccard')                # Jaccard on q-grams   -> 0.1429
va, vb = to_count_vector(qa, qb)
sim(va, vb, 'cosine')                 # N-gram cosine        -> 0.25
dist(va, vb, 'manhattan')             # Q-gram distance      -> 6

ska, skb = to_skipgram(a, 2, 1), to_skipgram(b, 2, 1)
sim(ska, skb, 'jaccard')              # Skip-gram similarity -> 0.0769
```

---

## Metrics Reference

Metrics marked with an alias share their implementation with the canonical name. All aliases are fully supported in both `simdif()` calls and standalone `explain_` / `info_` functions.

### Set Metrics

These metrics treat inputs as unordered collections. Element frequency is ignored; only membership matters.

| Canonical Name | Aliases | Default Output |
|---|---|---|
| `baroni_urbani_buser` | `bub`, `baroni_urbani` | sim |
| `braun_blanquet` | - | sim |
| `cosine_set` | `ochiai` | sim |
| `dice_sorensen` | `dice`, `sorensen`, `sorensen_dice` | sim |
| `jaccard` | `iou` | sim |
| `kulczynski` | `kulczynski_ii` | sim |
| `kulczynski_i` | - | sim |
| `mcconnaughey` | - | sim |
| `mountford` | - | sim |
| `overlap` | `szymkiewicz_simpson`, `simpson` | sim |
| `phi` | `mcc`, `matthews` | sim |
| `rogers_tanimoto` | `sokal_ii`, `sokal_michener_ii`, `sokal_sneath_ii` | sim |
| `russel_rao` | `russell_rao`, `rr` | sim |
| `smc` | `sokal_michener` | sim |
| `sokal_sneath_i` | `ssi` | sim |
| `sokal_sneath_iii` | `ssiii` | sim |
| `tversky` | - | sim |
| `yule_q` | - | sim |

> **Universe-aware metrics.** `smc`, `rogers_tanimoto`, `phi`, `yule_q`, and `baroni_urbani_buser` use "shared absences" (elements in neither set), so they take an optional `n_universe` parameter giving the size of the total element space. Without it, shared absences count as 0, which makes the association measures (`phi`, `yule_q`) degenerate - always pass `n_universe` for those. `baroni_urbani_buser` reduces to Jaccard when `n_universe` is omitted, and credits shared absences through the geometric mean `sqrt(n11 * n00)` - the only set coefficient here that uses that term. Example: `simdif(a, b, ['phi', 'yule_q'], n_universe=26)`.

> **Mountford is unbounded and sample-size independent.** `mountford` is an ecological presence/absence index built to stay roughly constant as sampling effort changes - unlike `jaccard`/`dice_sorensen`, which shrink when more of the fauna goes unobserved. It ignores shared absences (no `n_universe`) and its `sim` is a raw value in `[0, inf)`: `0` for disjoint species lists, `+inf` for identical ones. The bounded companion is `dif = 1 / (1 + M)` (`1` = disjoint, `0` = identical). This implementation uses Mountford's standard closed-form approximation, not the transcendental exact form. Example: `simdif(site_a, site_b, ['mountford'], output='dif')`.

### Sequence Metrics

These metrics treat inputs as ordered. The position of elements matters (e.g. `"abc" ≠ "bca"`).

| Canonical Name | Aliases | Default Output |
|---|---|---|
| `affine_gap` | `gotoh` | score |
| `bm25` | `okapi`, `okapi_bm25` | score |
| `damerau_levenshtein` | `dl` | dist |
| `dtw` | - | dist |
| `indel` | - | dist |
| `jaro` | - | sim |
| `jaro_winkler` | - | sim |
| `jukes_cantor` | `jc`, `jc69` | dist |
| `kendall_tau` | `kendall_tau_a`, `tau_a` | sim |
| `kendall_tau_b` | `tau_b` | sim |
| `kimura` | `k80`, `k2p` | dist |
| `lc_subsequence` | `lcs`, `lcsubseq`, `longest_common_subsequence` | score |
| `lc_substring` | `lcstr`, `lcsubstr`, `longest_common_substring` | score |
| `levenshtein` | - | dist |
| `monge_elkan` | - | sim |
| `ncd` | `normalized_compression_distance`, `compression`, `compression_distance` | dist |
| `needleman_wunsch` | `needleman`, `wunsch` | score |
| `osa` | - | dist |
| `p_distance` | `p_dist` | dist |
| `prefix` | - | score |
| `ratcliff_obershelp` | `gestalt`, `ro`, `ratcliff`, `obershelp` | sim |
| `smith_waterman` | `smith`, `waterman` | score |
| `soundex` | - | sim |
| `spearman` | - | sim |
| `suffix` | - | score |

> **BM25 takes a corpus.** `bm25` scores a query A against a document B and is the one sequence metric that needs external context: an optional `corpus` keyword (a list of documents, each a list of terms) supplies the IDF and average document length - the same pattern as `n_universe` for the set metrics. Without it, A and B are used as a degenerate 2-document corpus. Tune `k1` (term-frequency saturation) and `b_norm` (length normalization; named to avoid clashing with the document argument B). Example: `simdif(query, doc, ['bm25'], corpus=[doc1, doc2, ...])`.

> **NCD compares no elements at all.** `ncd` (Cilibrasi & Vitányi, *Clustering by Compression*, 2005) measures similarity by asking whether knowing A helps a compressor compress B: `NCD(A,B) = (C(AB) − min(C(A),C(B))) / max(C(A),C(B))`, where `C(x)` is the compressed byte length. There is no alignment, no token overlap, and no vector arithmetic - it is the only metric here that needs zero knowledge of what the data *is*, so the same call works on text, DNA, or raw binaries. Choose the compressor with `compressor=` (`'zlib'` default, `'bz2'`, `'lzma'`, or any `bytes -> bytes` callable); prefer `bz2`/`lzma` past zlib's 32 KB window, since zlib cannot see redundancy across a wider gap and will silently call two similar large files dissimilar. Asymmetric, like `monge_elkan`: `C(AB) ≠ C(BA)`.
>
> ⚠️ **NCD is unreliable on short inputs - the one metric here whose classroom-sized example lies to you.** Every compressor emits a fixed header (~11 bytes for zlib, ~38 for bz2, ~60 for lzma), so below a few hundred bytes the formula measures format overhead instead of shared information. Identical inputs do **not** score 0 (`dist_ncd("hello world", "hello world")` ≈ `0.105` under zlib), and rankings can invert outright: under `bz2`, `"hedge"` vs `"hog"` scores `0.075` while `"hello world"` against its own copy scores `0.0625` - two unrelated words looking nearly as close as a string to itself. `explain_ncd` prints a warning when either input is under 100 bytes. Give it real documents; for short strings use an edit-distance or token metric instead.

> **Evolutionary distances** (`p_distance`, `jukes_cantor`, `kimura`) estimate how far two aligned sequences have diverged - a progression from raw observation to biological correction. `p_distance` is the observed proportion of differing sites (generic, `==` only). `jukes_cantor` corrects it for multiple substitutions under a k-state model (`k=4` for DNA by default; `k=20` for protein). `kimura` (K80) additionally splits transitions from transversions, so it needs a symbol partition via the `groups` keyword (defaults to DNA purines `{A,G}` / pyrimidines `{C,T}`, case-insensitive); sites with out-of-group symbols are skipped. Both corrected distances saturate to `inf` when sequences are too diverged to estimate, and both assume inputs are already aligned (pair them with `needleman_wunsch` / `smith_waterman`).

> **"LCS" is ambiguous - simdif ships both.** In common writing "LCS" names two different metrics: the longest common **subsequence** (gaps allowed) and the longest common **substring** (contiguous only). simdif resolves the bare name `lcs` to the subsequence variant, matching the algorithms-textbook default and simdif's own earlier releases, but every unambiguous name is registered too - use `lc_subsequence` / `lc_substring` when it matters, and check which recurrence a paper prints before comparing numbers. The difference is not cosmetic:
>
> ```python
> simdif("ABCDE", "AXBXCXDXE", ['lc_subsequence', 'lc_substring'])
> # {'lc_subsequence': 5, 'lc_substring': 1}
> ```
>
> The two share a DP scaffold but not its meaning. In `lc_subsequence` a cell holds the best score over all prefixes, so the grid is non-decreasing and the answer is the bottom-right corner. In `lc_substring` a cell holds the length of the run *ending exactly there*, mismatches reset it to `0`, and the answer is the largest cell anywhere - the same clamp-to-zero trick that turns `needleman_wunsch` (global) into `smith_waterman` (local). `lc_substring` is in fact `smith_waterman` with mismatch and gap penalties of `-inf`.

> **LC subsequence's `sim`/`dif` are tied to `dist`, not to `score`'s own range.** `score_lcs` naturally ranges `[0, min(|A|,|B|)]`, but `sim_lcs` is defined as `1 - dist_lcs/(|A|+|B|)` (equivalently `2·LCS(A,B)/(|A|+|B|)`), so it stays consistent with the already-registered `dist_lcs` (indel distance) rather than with `score`'s range. Concretely: `sim_lcs=1.0` iff `A == B` exactly - a short string that's fully contained as a subsequence of a much longer one (e.g. `"AB"` vs `"ABCDEFG"`) does **not** score 1.0, since the length mismatch still counts against it (same behavior as `sim_levenshtein`).

> **`lc_substring` has no `dist` role, and normalizes by the longer input.** The indel-distance identity that gives `lc_subsequence` its `dist` doesn't carry over: `|A| + |B| - 2·LCSubstr` counts no sequence of edit operations and isn't a metric, so it is deliberately not offered under that name. With no `dist` to stay consistent with, `sim_lc_substring` is `score / max(|A|,|B|)` - the `prefix`/`suffix` convention - so a strictly contained run still scores below 1.0 (`sim_lc_substring("BCD", "ABCDE")` is `3/5`).

> **`prefix`/`suffix` are blind to the other end of the string**, and their `sim` is normalized by the *longer* string's length, not the shorter one - so a string that's a strict prefix (or suffix) of another still doesn't score `sim=1.0` (e.g. `sim_prefix("test", "testing")` is `4/7`, not `1.0`). `prefix` ignores everything after the first mismatch from the start; `suffix` ignores everything before the first mismatch counting backward from the end - two strings can share nothing on `prefix` while sharing everything on `suffix`, or vice versa (compare `simdif("prefix", "preheat", ['prefix','suffix'])` against `simdif("suffix", "postfix", ['prefix','suffix'])`).

### Vector Metrics

These metrics require input to be ordered and same length.

Most vector metrics require numeric input for mathematical operations.

| Canonical Name | Aliases | Default Output |
|---|---|---|
| `bray_curtis` | - | dist |
| `canberra` | - | dist |
| `chebyshev` | `chessboard`, `linf` | dist |
| `cosine` | - | sim |
| `euclidean` | - | dist |
| `geodesic` | `earth` | dist |
| `hausdorff` | `hausdorff_distance`, `hd` | dist |
| `index_of_dissimilarity` | `hoover`, `duncan` | dif |
| `lee` | - | dist |
| `mahalanobis` | - | dist |
| `manhattan` | `taxicab`, `cityblock` | dist |
| `minkowski` | - | dist |
| `pearson` | - | sim |
| `tanimoto` | `binary_tanimoto`, `tanimoto_binary` | sim |
| `tanimoto_continuous` | `continuous_tanimoto`, `extended_tanimoto` | sim |

> **Hausdorff measures the worst-case nearest miss, and ignores order entirely.** For every point in A, `hausdorff` finds its nearest point in B and records that gap; the distance is the largest such gap, taken over both directions: `H(A,B) = max(h(A,B), h(B,A))` where `h(A,B) = maxₐ minᵦ d(a,b)`. Equivalently, the smallest radius by which you'd have to fatten each set to engulf the other. Inputs need not be the same length and are not compared index by index (like `energy` and `welch_t`). Both directions are required - a lone point inside a large cloud is ~0 away from the cloud but far from it coming back. `directed_hausdorff(a, b, ...)` exposes one direction, and `explain_hausdorff` prints both.
>
> Two contrasts make it worth its own file. Against `dtw`: both compare collections through a pointwise distance, but DTW forces a monotone order-preserving path and *sums* the costs, while Hausdorff lets every point choose its nearest neighbour independently and takes the *max*. So `[1,2,3,4,5]` and `[5,4,3,2,1]` are the same point set - Hausdorff `0`, DTW `12`. Against `jaccard`: both ignore order, but the set metrics need exact shared members, so `sim_jaccard([1,2,3], [1.01,2.01,3.01])` is `0.0` while Hausdorff reports `0.01`. It is simdif's only order-blind metric where near misses still count. It is also a true metric (triangle inequality, zero iff equal) where DTW is not. Pass `dist_fn=` for a different pointwise distance - supplying one also lifts the numeric requirement, so tuples work as n-D points.
>
> ⚠️ **The plain maximum is brutally outlier-sensitive - use `percentile`.** `[0,1,2]` vs `[0.5,1.5]` is `0.5`; add one far-off point to B and it becomes `98`. `percentile` (default `100`, the true Hausdorff) reports a lower order statistic instead, and running the same pair twice shows exactly how much one point was carrying:
>
> ```python
> A = list(range(20))
> B = [i + 0.2 for i in range(19)] + [500]      # tracks A at 0.2, plus one wild point
>
> simdif(A, B, ['hausdorff'])                    # {'hausdorff': 481.0}
> simdif(A, B, ['hausdorff'], percentile=95)     # {'hausdorff': 0.2}    <- HD95
> ```
>
> `percentile=95` is the HD95 reported alongside Dice for segmentation boundaries. `aggregate='mean'` gives the modified Hausdorff (Dubuisson & Jain, 1994) used in shape matching; it composes with `percentile` rather than overriding it. Percentiles use the nearest-rank convention, so `percentile=100` is exactly the maximum and no value is ever interpolated into existence - this can differ slightly from NumPy's default.

> ⚠️ **Chebyshev reads exactly one coordinate - the worst one.** `chebyshev` is `max|Aᵢ - Bᵢ|`, the p → ∞ limit of Minkowski: `p=1` sums every gap (`manhattan`), `p=2` weights larger gaps more (`euclidean`), and in the limit the largest gap is the *only* one that counts. So `[0,0,0,0]` vs `[0,0,0,9]` and `[8,8,8,8]` vs `[8,8,8,17]` both score `9`, and the other three coordinates can move freely without changing a thing until one of them overtakes the max. One broken sensor or one mis-scaled feature becomes the entire answer. `explain_chebyshev` prints the runner-up gap so you can see how much the winning coordinate is carrying alone. The chessboard name is literal: it is the minimum number of king moves between two squares.
>
> It gets no `percentile` knob, and the reason is worth stating. `hausdorff` trims *nearest-neighbour* distances, and the trimmed form is a published metric (HD95); Chebyshev reduces *aligned coordinate* gaps, where trimming has no published meaning and stops being a metric - the 75th percentile of `[0,0,0,5]` vs `[0,0,0,0]` reports `0` for two vectors that plainly differ, breaking both identity-of-indiscernibles and the triangle inequality. When the outlier should be discounted rather than deleted, that is what `manhattan` and `euclidean` are for. Note also that `p=inf` is taken as a limit, not evaluated - `dist_minkowski(a, b, p=inf)` delegates to `dist_chebyshev`, because substituting infinity into the formula returns `1.0` for almost any input.

> **Tanimoto: binary vs continuous.** `tanimoto` is the binary/bit-vector coefficient `c / (a + b - c)` on aligned 0/1 vectors (or integer bitmasks with `binary=True`) - algebraically the *same* coefficient as `jaccard`, `c/(a+b-c) = |A∩B|/|A∪B|`, just computed positionally on bit vectors rather than on sets of element values. (So `jaccard` of the raw lists is not the same call - it compares value-sets - but the two coincide on the sets of "on" positions.) `tanimoto_continuous` is the real-valued generalization `A·B / (‖A‖² + ‖B‖² − A·B)` for count or weighted vectors, and it reduces exactly to the binary form on 0/1 input. Unlike `cosine`, Tanimoto is not scale-invariant.

If mathematical operations are not required, any data type is allowed.

| Canonical Name | Aliases | Default Output |
|---|---|---|
| `hamming` | - | dist |

### Probabilistic / Divergence Metrics

These metrics measure how much two probability distributions differ.

| Canonical Name | Aliases | Default Output |
|---|---|---|
| `bhattacharyya` | - | dist |
| `hellinger` | - | dist |
| `js_divergence` | `jensen_shannon` | dist |
| `kl_divergence` | `kullback_leibler` | dist |
| `wasserstein` | `earth_mover`, `emd` | dist |

### Two-Sample Statistics

These compare two **independent numeric samples** - how far apart they are, whether in central tendency (`welch_t`, `cohens_d`) or across the whole distribution (`energy`). Unlike the vector metrics, the inputs are *not* aligned and *need not be the same length*; order is irrelevant and duplicate values are significant (they are samples, not sets). `welch_t` and `cohens_d` each need at least 2 values (they estimate variance); `energy` needs at least 1. All return an unbounded `dist` (higher = more different) with a `sim = 1/(1+dist)` companion.

| Canonical Name | Aliases | Default Output |
|---|---|---|
| `welch_t` | `welch`, `two_sample_t`, `sed` | dist |
| `cohens_d` | `cohen_d`, `cohens` | dist |
| `energy` | `energy_distance`, `e_distance` | dist |

`welch_t` is the two-sample t-statistic - the mean difference divided by the standard error of the difference (`sqrt(s_A²/n_A + s_B²/n_B)`). `cohens_d` is the standardized effect size - the mean difference divided by the pooled standard deviation. (The standard error alone measures the *precision* of the difference, not its size; both metrics divide the mean difference by a spread term to measure the difference itself.)

`energy` is the **energy distance** (Székely & Rizzo): `sqrt(2·mean‖a−b‖ − mean‖a−a'‖ − mean‖b−b'‖)`, built from average pairwise Euclidean distances between and within the samples. Unlike the two t-style statistics - which only see a shift in the mean - it detects **any** difference in the distributions (mean, variance, or shape) and is `0` only when both samples come from the same distribution. It is a close cousin of `wasserstein` (in 1-D it integrates `(F−G)²` where Wasserstein integrates `|F−G|`), and the same formula generalizes unchanged to vector-valued observations. Example: two samples with equal means but different spread score `0` on `cohens_d`/`welch_t` yet nonzero on `energy`.

### Clustering / Partition Comparison

These compare **two clusterings of the same objects**, given as equal-length label sequences aligned by object index (object *i* has label `A[i]` and `B[i]`). They are **label-invariant** - relabelling the clusters does not change the score - because they count agreements over object *pairs* rather than matching labels. Each reduces the 2×2 pair-agreement table (`a` = a pair together in both, `b`/`c` = together in only one, `d` = apart in both) to a single similarity, and the three are the pair-counting form of existing coefficients (`rand_index` = `smc`, `fowlkes_mallows` = `cosine_set`, on pairs).

| Canonical Name | Aliases | Default Output |
|---|---|---|
| `rand_index` | `rand` | sim |
| `adjusted_rand` | `ari`, `adjusted_rand_index` | sim |
| `fowlkes_mallows` | `fm`, `fowlkes_mallows_index` | sim |

`rand_index` = `(a+d)/total` (fraction of pairs agreed on). `adjusted_rand` corrects that for chance and is the one to prefer - it can go **negative** (worse than random), so its `sim` is not bounded to [0,1]. `fowlkes_mallows` = `a/sqrt((a+b)(a+c))`. `explain_` prints the full `a/b/c/d` table.

---

## Output Types

Many metrics can return more than one type of output. The table below shows what each output type means:

| Output type | Meaning |
|---|---|
| `sim` | Similarity - higher is more similar (typically 0–1) |
| `dif` | Difference - higher is more different (typically 0–1) |
| `dist` | Distance - higher means further apart (range varies by metric) |
| `score` | Raw alignment score (Smith-Waterman, Needleman-Wunsch, Affine Gap, LCS, Prefix, Suffix) |
| `matrix` | Full dynamic programming matrix (Levenshtein, NW, SW, Affine Gap, LCS) |
| `trace` | Recovered structure from the DP matrix: the alignment path (NW, SW, Affine Gap) or the longest common subsequence itself (LCS) |

To request a specific output type:

```python
from simdif import simdif

# Get distance instead of the default similarity
simdif("cat", "car", ['levenshtein'], output='dist')

# Get the full DP matrix
simdif("cat", "car", ['levenshtein'], output='matrix')
```

---

## The Educational Interface

Every metric in simdif that has a `class` designation (`set`, `sequence`, or `vector`) exposes two educational functions.

### `explain_<metric>(a, b, ...)`

Runs the calculation and prints each step - what the inputs look like after preprocessing, what intermediate values are computed, and how the final score is assembled. Returns the score so it can be used programmatically.

### `info_<metric>()`

Prints a human-readable description of the metric: what it measures, its output range, and typical use cases. Takes no arguments.

```python
from simdif import explain_cosine, info_cosine

info_cosine()        # What is cosine similarity?
explain_cosine([1, 2, 3], [4, 5, 6])  # Show me the dot products, magnitudes, etc.
```

---

## Comparing Metrics Side by Side

One of simdif's most useful features for education is running many metrics over the same pair of inputs to see how they agree or disagree:

```python
from simdif import simdif

a = [1, 0, 1, 1, 0]
b = [1, 1, 0, 1, 0]

results = simdif(a, b, [
    'jaccard', 'dice', 'cosine_set',           # set metrics
    'hamming', 'kendall_tau',                  # sequence metrics
    'cosine', 'euclidean', 'manhattan'         # vector metrics
])

for name, score in results.items():
    print(f"{name:>20}: {score:.4f}")
```

This is especially useful for showing students that the "right" metric depends on what you care about - membership, order, magnitude, or distribution.

---

## Known Limitations

- **Not optimised for performance.** Pure Python implementations mean simdif is slow on large inputs. It is not a replacement for NumPy, SciPy, or specialised libraries like `rapidfuzz`.
- **Explanation output can be verbose on long inputs.** `explain_*` functions are designed for short, illustrative examples.
- **Some metrics require specific input types.** Passing a non-numeric input to a vector metric, or a non-sequence to a sequence metric, will raise an error.
- **Work in progress.** The library is functional but not fully polished. APIs may change.

---

## Contributing

Contributions, bug reports, and suggestions are welcome. If you add a new metric, please include both an `info_` and `explain_` function to keep the educational interface consistent.

---

## License

MIT
