from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass

from .config import WorldConfig
from .models import Point3D

GridIndex = tuple[int, int, int]


@dataclass(frozen=True)
class PlannedPath:
    points: list[Point3D]
    distance: float
    expanded_nodes: int


class VoxelGrid:
    """Collision grid for the known Gazebo map, with safety inflation."""

    def __init__(self, config: WorldConfig) -> None:
        self.minimum = config.minimum
        self.maximum = config.maximum
        self.resolution = config.planning.resolution
        self.safety_margin = config.planning.safety_margin
        self.minimum_flight_altitude = config.planning.minimum_flight_altitude
        self.maximum_flight_altitude = config.planning.maximum_flight_altitude
        self.shape = tuple(
            int(round((high - low) / self.resolution)) + 1
            for low, high in zip(self.minimum, self.maximum)
        )
        self.occupied: set[GridIndex] = set()

        for obstacle in config.obstacles:
            half = tuple(length / 2.0 for length in obstacle.size)
            low = tuple(
                obstacle.center[axis] - half[axis] - self.safety_margin
                for axis in range(3)
            )
            high = tuple(
                obstacle.center[axis] + half[axis] + self.safety_margin
                for axis in range(3)
            )
            self._occupy_box(low, high)

        # The low-poly tree crown reaches roughly 5.55 m and has radius 1.55 m.
        tree_radius = 1.55 + self.safety_margin
        for x, y, _ in config.tree_positions:
            self._occupy_box(
                (x - tree_radius, y - tree_radius, 0.0),
                (x + tree_radius, y + tree_radius, 5.55 + self.safety_margin),
            )

    def _occupy_box(self, low: Point3D, high: Point3D) -> None:
        low_index = self.point_to_index(low, clamp=True)
        high_index = self.point_to_index(high, clamp=True)
        for ix in range(low_index[0], high_index[0] + 1):
            for iy in range(low_index[1], high_index[1] + 1):
                for iz in range(low_index[2], high_index[2] + 1):
                    self.occupied.add((ix, iy, iz))

    def point_to_index(self, point: Point3D, clamp: bool = False) -> GridIndex:
        index = tuple(
            int(round((coordinate - low) / self.resolution))
            for coordinate, low in zip(point, self.minimum)
        )
        if clamp:
            return tuple(
                max(0, min(value, self.shape[axis] - 1))
                for axis, value in enumerate(index)
            )
        if not self.in_bounds(index):
            raise ValueError(f"Point {point} is outside the planning bounds")
        return index

    def index_to_point(self, index: GridIndex) -> Point3D:
        return tuple(
            self.minimum[axis] + index[axis] * self.resolution
            for axis in range(3)
        )

    def in_bounds(self, index: GridIndex) -> bool:
        return all(0 <= index[axis] < self.shape[axis] for axis in range(3))

    def is_free(self, index: GridIndex) -> bool:
        if not self.in_bounds(index) or index in self.occupied:
            return False
        altitude = self.index_to_point(index)[2]
        return self.minimum_flight_altitude <= altitude <= self.maximum_flight_altitude

    def segment_is_free(self, start: Point3D, end: Point3D) -> bool:
        distance = math.dist(start, end)
        samples = max(1, int(math.ceil(distance / (self.resolution * 0.3))))
        for sample in range(samples + 1):
            ratio = sample / samples
            point = tuple(start[axis] + ratio * (end[axis] - start[axis]) for axis in range(3))
            if not self.is_free(self.point_to_index(point)):
                return False
        return True


class AStar3D:
    def __init__(self, grid: VoxelGrid) -> None:
        self.grid = grid
        self.neighbors = tuple(
            offset
            for offset in itertools.product((-1, 0, 1), repeat=3)
            if offset != (0, 0, 0)
        )

    def plan(self, start: Point3D, goal: Point3D) -> PlannedPath:
        start_index = self.grid.point_to_index(start)
        goal_index = self.grid.point_to_index(goal)
        if not self.grid.is_free(start_index):
            raise ValueError(f"Start point {start} is occupied")
        if not self.grid.is_free(goal_index):
            raise ValueError(f"Goal point {goal} is occupied")

        if start_index == goal_index:
            return PlannedPath([start, goal], math.dist(start, goal), 0)

        queue: list[tuple[float, float, GridIndex]] = []
        heapq.heappush(queue, (self._heuristic(start_index, goal_index), 0.0, start_index))
        came_from: dict[GridIndex, GridIndex] = {}
        cost_so_far: dict[GridIndex, float] = {start_index: 0.0}
        closed: set[GridIndex] = set()

        while queue:
            _, current_cost, current = heapq.heappop(queue)
            if current in closed:
                continue
            closed.add(current)
            if current == goal_index:
                raw = self._reconstruct(came_from, current)
                raw_points = [self.grid.index_to_point(index) for index in raw]
                raw_points[0] = start
                raw_points[-1] = goal
                smoothed = self._smooth(raw_points)
                return PlannedPath(smoothed, path_distance(smoothed), len(closed))

            for offset in self.neighbors:
                neighbor = tuple(current[axis] + offset[axis] for axis in range(3))
                if not self.grid.is_free(neighbor) or not self._diagonal_is_clear(current, offset):
                    continue
                step_cost = self.grid.resolution * math.sqrt(sum(value * value for value in offset))
                new_cost = current_cost + step_cost
                if new_cost >= cost_so_far.get(neighbor, math.inf):
                    continue
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current
                priority = new_cost + self._heuristic(neighbor, goal_index)
                heapq.heappush(queue, (priority, new_cost, neighbor))

        raise RuntimeError(f"No collision-free path from {start} to {goal}")

    def _diagonal_is_clear(self, current: GridIndex, offset: GridIndex) -> bool:
        changed_axes = [axis for axis, value in enumerate(offset) if value]
        if len(changed_axes) == 1:
            return True
        for axis in changed_axes:
            intermediate = list(current)
            intermediate[axis] += offset[axis]
            if not self.grid.is_free(tuple(intermediate)):
                return False
        return True

    def _heuristic(self, first: GridIndex, second: GridIndex) -> float:
        return self.grid.resolution * math.dist(first, second)

    @staticmethod
    def _reconstruct(came_from: dict[GridIndex, GridIndex], current: GridIndex) -> list[GridIndex]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def _smooth(self, points: list[Point3D]) -> list[Point3D]:
        if len(points) <= 2:
            return points
        result = [points[0]]
        anchor = 0
        while anchor < len(points) - 1:
            candidate = len(points) - 1
            while candidate > anchor + 1:
                if self.grid.segment_is_free(points[anchor], points[candidate]):
                    break
                candidate -= 1
            result.append(points[candidate])
            anchor = candidate
        return result


def path_distance(points: list[Point3D]) -> float:
    return sum(math.dist(first, second) for first, second in zip(points, points[1:]))
