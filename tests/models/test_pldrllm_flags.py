"""Regression test for the pldrllm wrapper's cache flags.

The flags must round-trip as plain booleans: an earlier version
assigned them with trailing commas, producing one-element tuples that
are truthy regardless of the value supplied, so cache-off requests were
silently ineffective in the generation branch.  This test needs no
model or tokenizer weights: lightweight stand-ins satisfy the
constructor, which only stores the objects and binds their methods.
"""
import pytest

torch = pytest.importorskip("torch")

from lm_eval.models.pldrllm import pldrllm


class _StubTokenizer:
    def encode(self, *a, **k):  # bound via functools.partial only
        raise NotImplementedError

    def decode(self, *a, **k):
        raise NotImplementedError


def _make(enable_kvcache, enable_Gcache):
    return pldrllm(model=object(), tokenizer=_StubTokenizer(),
                   enable_kvcache=enable_kvcache,
                   enable_Gcache=enable_Gcache, device="cpu")


@pytest.mark.parametrize("kv,g", [(True, True), (True, False),
                                  (False, True), (False, False)])
def test_cache_flags_round_trip_as_bools(kv, g):
    lm = _make(kv, g)
    assert lm.enable_kvcache is kv
    assert lm.enable_Gcache is g


def test_cache_off_is_falsy():
    # the historical failure mode: (False,) is truthy, so the
    # generation branch kept caching when asked not to
    lm = _make(False, False)
    assert not lm.enable_kvcache
    assert not lm.enable_Gcache
    assert not isinstance(lm.enable_kvcache, tuple)
    assert not isinstance(lm.enable_Gcache, tuple)
