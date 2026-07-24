# simdif

**simdif** is a pure-Python library for computing, comparing, and understanding similarity, difference, and distance metrics. It started as a collection of similarity and difference metrics, and has grown to include distance metrics and alignment scores (Smith-Waterman, Needleman-Wunsch). The design goal is **education first**: every metric ships with a step-by-step explanation function and a plain-English definition function so you can see exactly how a score is derived.

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
# Clone and install in editable mode (development)
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

## Metrics Reference

Metrics marked with an alias share their implementation with the canonical name. All aliases are fully supported in both `simdif()` calls and standalone `explain_` / `info_` functions.

### Set Metrics

These metrics treat inputs as unordered collections. Element frequency is ignored; only membership matters.

| Canonical Name | Aliases | Default Output |
|---|---|---|
| `braun_blanquet` | - | sim |
| `cosine_set` | `ochiai` | sim |
| `dice_sorensen` | `dice`, `sorensen`, `sorensen_dice` | sim |
| `jaccard` | `iou` | sim |
| `kulczynski` | `kulczynski_ii` | sim |
| `kulczynski_i` | - | sim |
| `mcconnaughey` | - | sim |
| `overlap` | `szymkiewicz_simpson`, `simpson` | sim |
| `phi` | `mcc`, `matthews` | sim |
| `rogers_tanimoto` | `sokal_ii`, `sokal_michener_ii`, `sokal_sneath_ii` | sim |
| `russel_rao` | `russell_rao`, `rr` | sim |
| `smc` | `sokal_michener` | sim |
| `sokal_sneath_i` | `ssi` | sim |
| `sokal_sneath_iii` | `ssiii` | sim |
| `tversky` | - | sim |
| `yule_q` | - | sim |

> **Universe-aware metrics.** `smc`, `rogers_tanimoto`, `phi`, and `yule_q` use "shared absences" (elements in neither set), so they take an optional `n_universe` parameter giving the size of the total element space. Without it, shared absences count as 0, which makes the association measures (`phi`, `yule_q`) degenerate — always pass `n_universe` for those. Example: `simdif(a, b, ['phi', 'yule_q'], n_universe=26)`.

### Sequence Metrics

These metrics treat inputs as ordered. The position of elements matters (e.g. `"abc" ≠ "bca"`).

| Canonical Name | Aliases | Default Output |
|---|---|---|
| `bm25` | `okapi`, `okapi_bm25` | score |
| `damerau_levenshtein` | `dl` | dist |
| `indel` | - | dist |
| `jaro` | - | sim |
| `jaro_winkler` | - | sim |
| `jukes_cantor` | `jc`, `jc69` | dist |
| `kendall_tau` | `kendall_tau_a`, `tau_a` | sim |
| `kendall_tau_b` | `tau_b` | sim |
| `kimura` | `k80`, `k2p` | dist |
| `lcs` | - | score |
| `levenshtein` | - | dist |
| `monge_elkan` | - | sim |
| `needleman_wunsch` | `needleman`, `wunsch` | score |
| `osa` | - | dist |
| `p_distance` | `p_dist` | dist |
| `ratcliff_obershelp` | `gestalt`, `ro`, `ratcliff`, `obershelp` | sim |
| `smith_waterman` | `smith`, `waterman` | score |
| `soundex` | - | sim |
| `spearman` | - | sim |

> **BM25 takes a corpus.** `bm25` scores a query A against a document B and is the one sequence metric that needs external context: an optional `corpus` keyword (a list of documents, each a list of terms) supplies the IDF and average document length — the same pattern as `n_universe` for the set metrics. Without it, A and B are used as a degenerate 2-document corpus. Tune `k1` (term-frequency saturation) and `b_norm` (length normalization; named to avoid clashing with the document argument B). Example: `simdif(query, doc, ['bm25'], corpus=[doc1, doc2, ...])`.

> **Evolutionary distances** (`p_distance`, `jukes_cantor`, `kimura`) estimate how far two aligned sequences have diverged — a progression from raw observation to biological correction. `p_distance` is the observed proportion of differing sites (generic, `==` only). `jukes_cantor` corrects it for multiple substitutions under a k-state model (`k=4` for DNA by default; `k=20` for protein). `kimura` (K80) additionally splits transitions from transversions, so it needs a symbol partition via the `groups` keyword (defaults to DNA purines `{A,G}` / pyrimidines `{C,T}`, case-insensitive); sites with out-of-group symbols are skipped. Both corrected distances saturate to `inf` when sequences are too diverged to estimate, and both assume inputs are already aligned (pair them with `needleman_wunsch` / `smith_waterman`).

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
| `index_of_dissimilarity` | `hoover`, `duncan` | dif |
| `lee` | - | dist |
| `mahalanobis` | - | dist |
| `manhattan` | `taxicab`, `cityblock` | dist |
| `minkowski` | - | dist |
| `pearson` | - | sim |

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
| `tanimoto` | - | sim |
| `wasserstein` | `earth_mover`, `emd` | dist |

### Two-Sample Statistics

These compare two **independent numeric samples** — how far apart their central tendencies are. Unlike the vector metrics, the inputs are *not* aligned and *need not be the same length*; order is irrelevant and duplicate values are significant (they are samples, not sets). Each sample needs at least 2 values. Both return an unbounded `dist` (higher = more different) with a `sim = 1/(1+dist)` companion.

| Canonical Name | Aliases | Default Output |
|---|---|---|
| `welch_t` | `welch`, `two_sample_t` | dist |
| `cohens_d` | `cohen_d`, `cohens` | dist |

`welch_t` is the two-sample t-statistic — the mean difference divided by the standard error of the difference (`sqrt(s_A²/n_A + s_B²/n_B)`). `cohens_d` is the standardized effect size — the mean difference divided by the pooled standard deviation. (The standard error alone measures the *precision* of the difference, not its size; both metrics divide the mean difference by a spread term to measure the difference itself.)

---

## Output Types

Many metrics can return more than one type of output. The table below shows what each output type means:

| Output type | Meaning |
|---|---|
| `sim` | Similarity - higher is more similar (typically 0–1) |
| `dif` | Difference - higher is more different (typically 0–1) |
| `dist` | Distance - higher means further apart (range varies by metric) |
| `score` | Raw alignment score (Smith-Waterman, Needleman-Wunsch, LCS) |
| `matrix` | Full dynamic programming matrix (Levenshtein, NW, SW, LCS) |
| `trace` | Recovered structure from the DP matrix: the alignment path (NW, SW) or the longest common subsequence itself (LCS) |

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
    'jaccard', 'dice', 'cosine_set',          # set metrics
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
