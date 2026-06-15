import numpy as np

from app.cache import PredictionCache


def test_fingerprint_stable_and_quantized() -> None:
    cache = PredictionCache(capacity=20, decimals=2)
    a = np.full((100, 144), 0.123, dtype=np.float32)
    b = np.full((100, 144), 0.1234, dtype=np.float32)  # same to 2 decimals
    c = np.full((100, 144), 0.99, dtype=np.float32)

    assert cache.fingerprint(a) == cache.fingerprint(b)
    assert cache.fingerprint(a) != cache.fingerprint(c)


def test_get_put_and_hit_counting() -> None:
    cache = PredictionCache(capacity=20, decimals=2)
    key = "k"

    assert cache.get(key) is None  # miss
    cache.put(key, "hello", 0.9)
    assert cache.get(key) == ("hello", 0.9)  # hit
    assert cache.get(key) == ("hello", 0.9)  # hit

    stats = cache.stats()
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["size"] == 1


def test_lfu_eviction_keeps_hot_entries() -> None:
    cache = PredictionCache(capacity=2, decimals=2)
    cache.put("hot", "a", 0.5)
    cache.put("cold", "b", 0.5)

    # Make "hot" frequently used.
    for _ in range(5):
        cache.get("hot")

    # Inserting a third entry must evict the least frequently used ("cold").
    cache.put("new", "c", 0.5)

    assert cache.get("hot") is not None
    assert cache.get("new") is not None
    assert cache.get("cold") is None  # evicted


def test_capacity_zero_disables_storage() -> None:
    cache = PredictionCache(capacity=0)
    cache.put("k", "x", 1.0)
    assert cache.get("k") is None
