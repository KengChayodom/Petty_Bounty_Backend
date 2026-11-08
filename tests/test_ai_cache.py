"""
Unit tests for app/services/ai_cache.py (AnalyzeCache).

The cache is the linchpin of the 2-step pipeline: /analyze writes the heavy
CLIP vector here and /sightings reads it back. We test the wrapper's own
contract (miss -> None, set/get round-trip, clear, class-level sharing) plus
the two guarantees the module docstring promises and the service relies on:
bounded memory (size eviction) and TTL expiry. Time is injected, never slept.
"""
from cachetools import TTLCache

from app.services.ai_cache import AnalyzeCache


def _payload(species="Cat", vec=None):
    # The service only ever reads feature_vector + species off the payload,
    # so a partial dict is enough here (TypedDict is not enforced at runtime).
    return {"species": species, "feature_vector": vec or [0.1, 0.2, 0.3]}


class TestAnalyzeCacheContract:
    def test_get_missing_key_returns_none(self):
        assert AnalyzeCache.get("https://nope/x.jpg") is None

    def test_set_then_get_round_trips_same_object(self):
        p = _payload(species="Dog")
        AnalyzeCache.set("https://img/1.jpg", p)
        assert AnalyzeCache.get("https://img/1.jpg") is p

    def test_set_overwrites_existing_key(self):
        AnalyzeCache.set("https://img/1.jpg", _payload(species="Cat"))
        AnalyzeCache.set("https://img/1.jpg", _payload(species="Bird"))
        assert AnalyzeCache.get("https://img/1.jpg")["species"] == "Bird"

    def test_clear_empties_the_store(self):
        AnalyzeCache.set("https://img/1.jpg", _payload())
        AnalyzeCache.clear()
        assert AnalyzeCache.get("https://img/1.jpg") is None

    def test_store_is_class_level_shared_state(self):
        # Two "instances" must see the same store — the pipeline depends on
        # /analyze and /sightings (different requests) sharing one cache.
        AnalyzeCache().set("https://img/shared.jpg", _payload(species="Dog"))
        assert AnalyzeCache().get("https://img/shared.jpg")["species"] == "Dog"


class TestAnalyzeCacheEviction:
    """Pins the bounded-memory + TTL guarantees by swapping in a small,
    clock-injectable store. We assert the wrapper honours whatever the
    configured TTLCache does — i.e. the service's memory safety is real."""

    def test_size_bound_evicts_least_recently_used(self, monkeypatch):
        monkeypatch.setattr(AnalyzeCache, "_store", TTLCache(maxsize=2, ttl=600))
        AnalyzeCache.set("a", _payload(species="A"))
        AnalyzeCache.set("b", _payload(species="B"))
        AnalyzeCache.set("c", _payload(species="C"))  # forces an eviction

        present = [k for k in ("a", "b", "c") if AnalyzeCache.get(k) is not None]
        assert len(present) == 2          # never exceeds maxsize
        assert "c" in present             # most recent survives

    def test_entry_expires_after_ttl(self, monkeypatch):
        clock = {"now": 1000.0}
        store = TTLCache(maxsize=8, ttl=10, timer=lambda: clock["now"])
        monkeypatch.setattr(AnalyzeCache, "_store", store)

        AnalyzeCache.set("k", _payload())
        assert AnalyzeCache.get("k") is not None   # fresh

        clock["now"] += 11                          # advance past the 10s TTL
        assert AnalyzeCache.get("k") is None        # expired, no real sleep
