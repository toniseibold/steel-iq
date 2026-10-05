"""Distance indexing preserves directed values, cache reuse and radius membership."""

from dataclasses import dataclass

import pytest

from steelo.domain.models import Environment


@dataclass
class Center:
    name: str
    position: float

    def distance_to_other_processcenter(self, other):
        return abs(self.position - other.position)


def environment():
    env = Environment.__new__(Environment)
    env._distance_cache = {}
    env._distance_cache_stats = {"hits": 0, "misses": 0, "computations": 0}
    return env


def test_radius_matches_reference_and_reuses_cache():
    centers = [Center("a", 0), Center("b", 5), Center("c", 11)]
    env = environment()
    expected = {(a.name, b.name) for a in centers for b in centers if abs(a.position - b.position) <= 5}
    assert env.precompute_distances_for_hot_metal_check(centers, 5) == expected
    assert env._distance_cache_stats["computations"] == 9
    lookup = env.build_distance_function_for_trade_lp(centers)
    assert lookup("a", "b") == 5
    assert lookup("missing", "b") == float("inf")
    assert env._distance_cache_stats["computations"] == 9


def test_index_preserves_first_duplicate_and_new_year_centers():
    env = environment()
    lookup = env.build_distance_function_for_trade_lp([Center("a", 0), Center("a", 100), Center("b", 5)])
    assert lookup("a", "b") == 5
    next_year = env.build_distance_function_for_trade_lp([Center("a", 0), Center("c", 9)])
    assert next_year("a", "c") == 9


def test_bulk_lookup_does_not_rescan_center_list():
    class CountingCenter:
        reads = 0

        def __init__(self, name):
            self._name = name

        @property
        def name(self):
            type(self).reads += 1
            return self._name

        def distance_to_other_processcenter(self, other):
            return 10.0

    count = 40
    centers = [CountingCenter(str(i)) for i in range(count)]
    assert environment().precompute_distances_for_hot_metal_check(centers, 5) == set()
    assert CountingCenter.reads <= 3 * count**2


@pytest.mark.parametrize("indexed", [False, True])
def test_distance_direction_is_preserved(indexed):
    class DirectedCenter(Center):
        def distance_to_other_processcenter(self, other):
            return other.position - self.position

    centers = [DirectedCenter("a", 0), DirectedCenter("b", 5)]
    data = {c.name: c for c in centers} if indexed else centers
    env = environment()
    assert env.get_cached_distance("a", "b", data) == 5
    assert env.get_cached_distance("b", "a", data) == -5
