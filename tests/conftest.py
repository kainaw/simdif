import sys
import importlib
import pytest

# Every optimized-library branch across the metrics gates on one of these
# names being in sys.modules (see the "Optimized library fallback" pattern:
# `if 'scipy' in sys.modules: ...`). Whether that gate is open depends on
# what some *other* import happened to pull in earlier in the process, which
# makes tests nondeterministic -- a test could exercise the reference path
# today and the optimized path tomorrow with no code change of its own.

OPTIONAL_LIBS = ('scipy', 'rapidfuzz', 'Levenshtein', 'ot')

# Each library is really imported at most once per process, the first time a
# test asks for it, and the same module object is reused after that. scipy in
# particular loads its submodules (scipy.stats, scipy._lib, ...) lazily and
# cross-references them against its own top-level package object; deleting
# sys.modules['scipy'] and then doing a SECOND real import (a fresh
# importlib.import_module) produces a new, different top-level object while
# old submodule entries from the first import are still cached under their
# dotted names, and the two generations do not agree with each other --
# surfaces as `AttributeError: module 'scipy' has no attribute '_lib'` deep in
# scipy.stats's own init. Only ever importing each library once, and toggling
# just its top-level sys.modules key afterward, avoids that entirely.
_lib_cache = {}


def _cached_import(name):
    if name not in _lib_cache:
        _lib_cache[name] = importlib.import_module(name)
    return _lib_cache[name]


@pytest.fixture(autouse=True)
def _hide_optimized_libs(monkeypatch):
    """Hide the optional libraries before every test, so all gates default to
    closed (the hand-written reference path) regardless of what is installed
    or already imported elsewhere in this process. Tests that want to
    exercise an optimized branch opt in explicitly via `optimized_lib`.
    """
    for name in OPTIONAL_LIBS:
        monkeypatch.delitem(sys.modules, name, raising=False)


@pytest.fixture
def optimized_lib(monkeypatch):
    """Opt a single test into one optimized-library branch. Skips the test
    (rather than failing it) if that library isn't installed here.
    """
    def _enable(name):
        try:
            module = _cached_import(name)
        except ImportError:
            pytest.skip(f"{name} is not installed")
        monkeypatch.setitem(sys.modules, name, module)
        return module
    return _enable
