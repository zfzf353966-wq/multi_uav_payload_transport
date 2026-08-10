from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

from .config import WorldConfig
from .grid_planner import AStar3D, VoxelGrid
from .models import Point3D, VehicleRoute


@dataclass(frozen=True)
class TrajectorySample:
    time: float
    position: Point3D
    velocity: Point3D
    acceleration: Point3D


@dataclass(frozen=True)
class ScheduledTrajectory:
    uav_id: str
    start_delay: float
    samples: tuple[TrajectorySample, ...]

    @property
    def duration(self) -> float:
        return self.samples[-1].time


def staging_position(spawn: Point3D, vehicle_index: int) -> Point3D:
    """Vertically separated holding points keep waiting UAVs out of corridors."""
    return (spawn[0], spawn[1], 3.0 + vehicle_index * 7.5)


def build_continuous_trajectory(
    path: list[Point3D],
    cruise_speed: float,
    grid: VoxelGrid,
    sample_period: float = 0.1,
    maximum_acceleration: float = 2.0,
    corner_radius: float = 1.0,
) -> tuple[TrajectorySample, ...]:
    """Round collision-free corners and time-parameterize the resulting path."""
    points = _remove_consecutive_duplicates(path)
    if len(points) < 2:
        raise ValueError("A trajectory requires at least two distinct points")
    geometry = _rounded_geometry(points, grid, corner_radius)
    geometry = _densify(geometry, spacing=0.25)
    distances = [0.0]
    for first, second in zip(geometry, geometry[1:]):
        distances.append(distances[-1] + math.dist(first, second))
    # Reserve margin for the combination of tangential and lateral demand and
    # for discrete numerical differentiation at curve transitions.
    tangential_acceleration = maximum_acceleration * 0.30
    lateral_acceleration = maximum_acceleration * 0.30
    speeds = [cruise_speed for _ in geometry]
    speeds[0] = 0.0
    speeds[-1] = 0.0
    # Curvature-aware speed limits keep rounded corners inside the requested
    # lateral acceleration envelope instead of traversing every bend at cruise.
    for index in range(1, len(geometry) - 1):
        first = _unit(_subtract(geometry[index], geometry[index - 1]))
        second = _unit(_subtract(geometry[index + 1], geometry[index]))
        angle = math.acos(max(-1.0, min(1.0, sum(a * b for a, b in zip(first, second)))))
        average_length = (
            math.dist(geometry[index - 1], geometry[index])
            + math.dist(geometry[index], geometry[index + 1])
        ) / 2.0
        curvature = angle / max(average_length, 1e-6)
        if curvature > 1e-6:
            speeds[index] = min(speeds[index], math.sqrt(lateral_acceleration / curvature))

    # Forward/backward passes enforce acceleration and braking limits.
    for index in range(1, len(speeds)):
        segment_length = distances[index] - distances[index - 1]
        speeds[index] = min(
            speeds[index], math.sqrt(speeds[index - 1] ** 2 + 2.0 * tangential_acceleration * segment_length)
        )
    for index in range(len(speeds) - 2, -1, -1):
        segment_length = distances[index + 1] - distances[index]
        speeds[index] = min(
            speeds[index], math.sqrt(speeds[index + 1] ** 2 + 2.0 * tangential_acceleration * segment_length)
        )
    times = [0.0]
    for index in range(1, len(geometry)):
        segment_length = distances[index] - distances[index - 1]
        average_speed = max(0.05, (speeds[index - 1] + speeds[index]) / 2.0)
        times.append(times[-1] + segment_length / average_speed)

    sample_count = max(1, int(math.ceil(times[-1] / sample_period)))
    sample_times = [min(index * sample_period, times[-1]) for index in range(sample_count + 1)]
    positions = [_interpolate_polyline(time_value, times, geometry) for time_value in sample_times]
    velocities = _finite_difference(positions, sample_times)
    velocities[0] = (0.0, 0.0, 0.0)
    velocities[-1] = (0.0, 0.0, 0.0)
    accelerations = _finite_difference(velocities, sample_times)
    accelerations[0] = (0.0, 0.0, 0.0)
    accelerations[-1] = (0.0, 0.0, 0.0)
    return tuple(
        TrajectorySample(round(time_value, 4), position, velocity, acceleration)
        for time_value, position, velocity, acceleration in zip(
            sample_times, positions, velocities, accelerations
        )
    )


def schedule_routes(
    config: WorldConfig,
    routes: list[VehicleRoute],
    grid: VoxelGrid,
    maximum_delay: float = 180.0,
) -> list[ScheduledTrajectory]:
    trajectories: list[ScheduledTrajectory] = []
    connector_planner = AStar3D(grid)
    for index, route in enumerate(routes):
        spawn = config.uav_spawn_points[index]
        staging = staging_position(spawn, index)
        if len(route.path) < 3:
            raise ValueError(f"Route for {route.uav.id} is too short")
        departure = connector_planner.plan(staging, route.path[1]).points
        arrival = connector_planner.plan(route.path[-2], staging).points
        route_path = [*departure, *route.path[2:-1], *arrival[1:]]
        samples = build_continuous_trajectory(
            route_path,
            route.uav.cruise_speed,
            grid,
            sample_period=config.planning.trajectory_sample_period,
            maximum_acceleration=config.planning.maximum_acceleration,
            corner_radius=config.planning.corner_radius,
        )

        delay = 0.0
        while any(
            trajectories_conflict(
                ScheduledTrajectory(route.uav.id, delay, samples),
                existing,
                config.planning.minimum_uav_separation,
            )
            for existing in trajectories
        ):
            delay += config.planning.schedule_step
            if delay > maximum_delay:
                raise RuntimeError(f"Could not find a conflict-free start time for {route.uav.id}")
        trajectories.append(ScheduledTrajectory(route.uav.id, delay, samples))
    return trajectories


def trajectories_conflict(
    first: ScheduledTrajectory,
    second: ScheduledTrajectory,
    minimum_separation: float,
    check_period: float = 0.1,
    landing_reservation: float = 12.0,
) -> bool:
    # Vehicles occupy their staging point before departure and reserve their
    # final staging / landing column after trajectory completion.
    overlap_start = 0.0
    overlap_end = min(
        first.start_delay + first.duration + landing_reservation,
        second.start_delay + second.duration + landing_reservation,
    )
    if overlap_end < overlap_start:
        return False
    count = int(math.ceil((overlap_end - overlap_start) / check_period))
    for index in range(count + 1):
        time_value = min(overlap_start + index * check_period, overlap_end)
        first_position = sample_at(first.samples, time_value - first.start_delay).position
        second_position = sample_at(second.samples, time_value - second.start_delay).position
        if math.dist(first_position, second_position) < minimum_separation:
            return True
    return False


def sample_at(samples: tuple[TrajectorySample, ...], time_value: float) -> TrajectorySample:
    if time_value <= 0.0:
        return samples[0]
    if time_value >= samples[-1].time:
        return samples[-1]
    times = [sample.time for sample in samples]
    upper = bisect.bisect_right(times, time_value)
    lower = upper - 1
    span = samples[upper].time - samples[lower].time
    ratio = (time_value - samples[lower].time) / span
    return TrajectorySample(
        time_value,
        _lerp(samples[lower].position, samples[upper].position, ratio),
        _lerp(samples[lower].velocity, samples[upper].velocity, ratio),
        _lerp(samples[lower].acceleration, samples[upper].acceleration, ratio),
    )


def _rounded_geometry(points: list[Point3D], grid: VoxelGrid, radius: float) -> list[Point3D]:
    result = [points[0]]
    for index in range(1, len(points) - 1):
        previous, corner, following = points[index - 1], points[index], points[index + 1]
        incoming_length = math.dist(previous, corner)
        outgoing_length = math.dist(corner, following)
        local_radius = min(radius, incoming_length * 0.25, outgoing_length * 0.25)
        if local_radius < 0.1:
            result.append(corner)
            continue
        incoming = _unit(_subtract(corner, previous))
        outgoing = _unit(_subtract(following, corner))
        entry = _subtract(corner, _scale(incoming, local_radius))
        exit_point = _add(corner, _scale(outgoing, local_radius))
        curve = [_quadratic_bezier(entry, corner, exit_point, step / 8.0) for step in range(9)]
        candidate = [result[-1], *curve, following]
        if all(grid.segment_is_free(a, b) for a, b in zip(candidate, candidate[1:])):
            result.extend(curve)
        else:
            result.append(corner)
    result.append(points[-1])
    return _remove_consecutive_duplicates(result)


def _densify(points: list[Point3D], spacing: float) -> list[Point3D]:
    result = [points[0]]
    for first, second in zip(points, points[1:]):
        count = max(1, int(math.ceil(math.dist(first, second) / spacing)))
        result.extend(_lerp(first, second, index / count) for index in range(1, count + 1))
    return result


def _interpolate_polyline(time_value: float, times: list[float], points: list[Point3D]) -> Point3D:
    if time_value >= times[-1]:
        return points[-1]
    upper = bisect.bisect_right(times, time_value)
    lower = max(0, upper - 1)
    span = times[upper] - times[lower]
    return _lerp(points[lower], points[upper], (time_value - times[lower]) / span)


def _finite_difference(values: list[Point3D], times: list[float]) -> list[Point3D]:
    result: list[Point3D] = []
    for index in range(len(values)):
        lower = max(0, index - 1)
        upper = min(len(values) - 1, index + 1)
        duration = max(1e-6, times[upper] - times[lower])
        result.append(tuple((values[upper][axis] - values[lower][axis]) / duration for axis in range(3)))
    return result


def _remove_consecutive_duplicates(points: list[Point3D]) -> list[Point3D]:
    result: list[Point3D] = []
    for point in points:
        if not result or math.dist(result[-1], point) > 1e-6:
            result.append(point)
    return result


def _quadratic_bezier(start: Point3D, control: Point3D, end: Point3D, ratio: float) -> Point3D:
    inverse = 1.0 - ratio
    return tuple(
        inverse * inverse * start[axis]
        + 2.0 * inverse * ratio * control[axis]
        + ratio * ratio * end[axis]
        for axis in range(3)
    )


def _lerp(first: Point3D, second: Point3D, ratio: float) -> Point3D:
    return tuple(first[axis] + ratio * (second[axis] - first[axis]) for axis in range(3))


def _subtract(first: Point3D, second: Point3D) -> Point3D:
    return tuple(first[axis] - second[axis] for axis in range(3))


def _add(first: Point3D, second: Point3D) -> Point3D:
    return tuple(first[axis] + second[axis] for axis in range(3))


def _scale(point: Point3D, factor: float) -> Point3D:
    return tuple(value * factor for value in point)


def _unit(point: Point3D) -> Point3D:
    length = math.sqrt(sum(value * value for value in point))
    return tuple(value / length for value in point)
