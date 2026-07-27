from ..simdif import Metric, METRICS, to_list
import bz2
import lzma
import zlib

# All three compressors are stdlib, so no optional-dependency gate is needed.
# zlib is the default: it has the smallest header, which matters most on the
# short classroom-length inputs simdif is built for (see info_ncd).
_COMPRESSORS = {
    'zlib': lambda data: zlib.compress(data, 9),
    'bz2': bz2.compress,
    'lzma': lzma.compress,
}

_TOKEN_SEPARATOR = b'\x00'


def info_ncd() -> str:
    return """
Normalized Compression Distance (NCD)
-------------------------------------
Measures similarity by asking a compressor a question: does knowing A help you
compress B? If the two inputs share structure, compressing them together costs
less than compressing them apart. Introduced by Cilibrasi & Vitanyi ("Clustering
by Compression", 2005) as a computable approximation of the Normalized
Information Distance built on Kolmogorov complexity.

Formula:
    NCD(A,B) = (C(AB) - min(C(A), C(B))) / max(C(A), C(B))

where C(x) is the byte length of x after compression and AB is the two inputs
concatenated.

What makes it unusual: NCD compares no elements. There is no alignment, no
token overlap, no vector arithmetic -- it never looks at A and B position by
position at all. It is the only metric here that needs zero knowledge of what
the data *is*, which is why the same call works on text, DNA, MIDI files, or
raw executables.

    compressor  'zlib' (default), 'bz2', 'lzma', or any callable
                bytes -> bytes

Roles:
    dist - the NCD itself. 0 = maximally similar, ~1 = nothing shared.
    sim  - 1 - dist, clamped to [0, 1].
    dif  - 1 - sim.

Range (dist): [0, ~1.1]. NOT a clean [0, 1]: the bound holds only for an ideal
compressor, and real ones exceed it slightly. `dist` is reported raw so the
leakage stays visible; only `sim`/`dif` are clamped.

WARNING -- NCD IS UNRELIABLE ON SHORT INPUTS. This is the metric's central
caveat and it bites hardest at exactly the input sizes used for teaching. Every
compressor emits a fixed header (~11 bytes for zlib, ~38 for bz2, ~60 for
lzma), and when the inputs are shorter than that header, the arithmetic is
measuring format overhead rather than shared information. Consequences:

    - Identical inputs do NOT score 0. dist_ncd("hello world", "hello world")
      is ~0.105 with zlib, not 0.0.
    - Rankings can invert outright. Under bz2, "hedge" vs "hog" scores 0.075
      while "hello world" vs itself scores 0.0625 -- two unrelated words look
      nearly as close as a string to its own copy.
    - The choice of compressor changes the answer, sometimes by more than the
      data does.

Use inputs of at least a few hundred bytes, and prefer 'bz2' or 'lzma' for
inputs beyond zlib's 32 KB window (zlib cannot see redundancy across a gap
wider than its window, so it silently reports two similar large files as
dissimilar). For short strings, use an edit-distance or token metric instead.

Note: Asymmetric. C(AB) and C(BA) can differ, so NCD(A,B) != NCD(B,A) in
general (same caveat as monge_elkan). The published definition uses C(AB) and
that is what is computed here; average the two directions yourself if you need
symmetry.

Note: Inputs are serialized to bytes before compression. str is encoded UTF-8;
bytes are used as-is; anything else is converted element by element and joined
with a NUL byte. This means "abc" and list("abc") do NOT give the same score --
the second is serialized as b"a\\x00b\\x00c". Pass strings as strings.

Aliases: normalized_compression_distance, compression, compression_distance
    """.strip()
info_normalized_compression_distance = info_ncd
info_compression = info_ncd
info_compression_distance = info_ncd


def _to_bytes(val) -> bytes:
    """Serialize an input to the byte string the compressor will see. Strings
    and bytes pass through directly so that the common case compresses exactly
    what the user typed; other sequences are joined with a separator so that
    ['ab', 'c'] and ['a', 'bc'] cannot collide."""
    if isinstance(val, (bytes, bytearray)):
        return bytes(val)
    if isinstance(val, str):
        return val.encode('utf-8')
    return _TOKEN_SEPARATOR.join(str(x).encode('utf-8') for x in to_list(val))


def _resolve_compressor(compressor):
    if callable(compressor):
        return compressor
    try:
        return _COMPRESSORS[compressor]
    except KeyError:
        raise ValueError(
            f"Unknown compressor {compressor!r}. Expected one of "
            f"{sorted(_COMPRESSORS)} or a callable bytes -> bytes."
        ) from None


def _compressed_sizes(a, b, compressor='zlib'):
    """The three measurements every role needs: C(A), C(B), C(AB)."""
    compress = _resolve_compressor(compressor)
    data_a, data_b = _to_bytes(a), _to_bytes(b)
    return (
        len(compress(data_a)),
        len(compress(data_b)),
        len(compress(data_a + data_b)),
    )


def explain_ncd(a, b, compressor='zlib', **kwargs) -> str:
    c_a, c_b, c_ab = _compressed_sizes(a, b, compressor)
    name = compressor if isinstance(compressor, str) else getattr(compressor, '__name__', 'custom')
    raw_a, raw_b = _to_bytes(a), _to_bytes(b)
    result = dist_ncd(a, b, compressor, **kwargs)
    caveat = ""
    if min(len(raw_a), len(raw_b)) < 100:
        caveat = ("\nNOTE: these inputs are shorter than ~100 bytes, where compressor header\n"
                  "overhead dominates the arithmetic -- treat this score as illustrative only.")
    return f"""
A: {len(raw_a)} bytes ({raw_a[:40]!r}{'...' if len(raw_a) > 40 else ''})
B: {len(raw_b)} bytes ({raw_b[:40]!r}{'...' if len(raw_b) > 40 else ''})
Compressor: {name}
Compressed separately: C(A) = {c_a}, C(B) = {c_b}
Compressed together:   C(AB) = {c_ab}
  (compressing B after A saved {c_a + c_b - c_ab} bytes versus compressing them apart)
NCD = (C(AB) - min(C(A), C(B))) / max(C(A), C(B))
    = ({c_ab} - {min(c_a, c_b)}) / {max(c_a, c_b)}
    = {c_ab - min(c_a, c_b)} / {max(c_a, c_b)}
Distance (dist): {result:.4f}
Similarity (sim): {sim_ncd(a, b, compressor, **kwargs):.4f}
Difference (dif): {dif_ncd(a, b, compressor, **kwargs):.4f}{caveat}
    """.strip()
explain_normalized_compression_distance = explain_ncd
explain_compression = explain_ncd
explain_compression_distance = explain_ncd


@Metric
def dist_ncd(a, b, compressor='zlib', **kwargs) -> float:
    c_a, c_b, c_ab = _compressed_sizes(a, b, compressor)
    denominator = max(c_a, c_b)
    if denominator == 0:
        # Only reachable from a custom compressor that emits nothing; the
        # stdlib three always produce a header. Two empty inputs are identical.
        return 0.0
    return (c_ab - min(c_a, c_b)) / denominator
dist_normalized_compression_distance = dist_ncd
dist_compression = dist_ncd
dist_compression_distance = dist_ncd


@Metric
def sim_ncd(a, b, compressor='zlib', **kwargs) -> float:
    # Clamped because NCD can drift slightly outside [0, 1] with a real
    # compressor, and a negative similarity would be more confusing than
    # useful. dist_ncd stays unclamped so the drift remains inspectable.
    return min(1.0, max(0.0, 1 - dist_ncd(a, b, compressor, **kwargs)))
sim_normalized_compression_distance = sim_ncd
sim_compression = sim_ncd
sim_compression_distance = sim_ncd


@Metric
def dif_ncd(a, b, compressor='zlib', **kwargs) -> float:
    return 1 - sim_ncd(a, b, compressor, **kwargs)
dif_normalized_compression_distance = dif_ncd
dif_compression = dif_ncd
dif_compression_distance = dif_ncd


METRICS['ncd'] = {
    'class': 'sequence',
    'default': 'dist',
    'dist': dist_ncd,
    'sim': sim_ncd,
    'dif': dif_ncd,
    'info': info_ncd,
    'explain': explain_ncd,
}
METRICS['normalized_compression_distance'] = METRICS['ncd']
METRICS['compression'] = METRICS['ncd']
METRICS['compression_distance'] = METRICS['ncd']
