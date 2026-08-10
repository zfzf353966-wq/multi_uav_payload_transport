from __future__ import annotations

import itertools

from .grid_planner import AStar3D, PlannedPath
from .models import Point3D


class PathOracle:
    """Caches symmetric known-map A* paths between task locations."""

    def __init__(self, planner: AStar3D) -> None:
        self.planner = planner
        self._cache: dict[tuple[Point3D, Point3D], PlannedPath] = {}
        self.total_expanded_nodes = 0

    @staticmethod
    def _key(first: Point3D, second: Point3D) -> tuple[Point3D, Point3D]:
        return (first, second) if first <= second else (second, first)

    def preload(self, points: list[Point3D]) -> None:
        unique = sorted(set(points))
        for first, second in itertools.combinations(unique, 2):
            self.get(first, second)

    def get(self, start: Point3D, goal: Point3D) -> PlannedPath:
        if start == goal:
            return PlannedPath([start, goal], 0.0, 0)
        key = self._key(start, goal)
        if key not in self._cache:
            planned = self.planner.plan(key[0], key[1])
            self._cache[key] = planned
            self.total_expanded_nodes += planned.expanded_nodes
        planned = self._cache[key]
        if start == key[0]:
            return planned
        return PlannedPath(list(reversed(planned.points)), planned.distance, planned.expanded_nodes)

    @property
    def cached_pair_count(self) -> int:
        return len(self._cache)
