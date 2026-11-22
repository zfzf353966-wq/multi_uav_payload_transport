from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .models import Location, Obstacle, Point3D, UAV


def _point(value: list[float]) -> Point3D:
    if len(value) != 3:
        raise ValueError(f"Expected a 3D point, got {value!r}")
    return (float(value[0]), float(value[1]), float(value[2]))


@dataclass(frozen=True)
class PlanningSettings:
    resolution: float
    safety_margin: float
    minimum_flight_altitude: float
    maximum_flight_altitude: float
    makespan_weight: int
    solver_time_limit_seconds: int
    trajectory_sample_period: float
    maximum_acceleration: float
    corner_radius: float
    minimum_uav_separation: float
    schedule_step: float


@dataclass(frozen=True)
class TaskSettings:
    count: int
    seed: int
    package_quantity: int
    distribution: str = "random"
    exclusive_delivery_uav: bool = False


@dataclass(frozen=True)
class WorldConfig:
    name: str
    minimum: Point3D
    maximum: Point3D
    depot: Location
    uav_spawn_points: tuple[Point3D, ...]
    pickups: tuple[Location, ...]
    deliveries: tuple[Location, ...]
    obstacles: tuple[Obstacle, ...]
    tree_positions: tuple[Point3D, ...]
    uavs: tuple[UAV, ...]
    task_settings: TaskSettings
    planning: PlanningSettings


def load_world_config(path: str | Path) -> WorldConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw: dict[str, Any] = yaml.safe_load(stream)

    bounds = raw["world"]["bounds"]
    planning = raw["planning"]
    task_generation = raw["task_generation"]

    pickups = tuple(
        Location(item["id"], _point(item["approach_position"]))
        for item in raw["pickup_points"]
    )
    deliveries = tuple(
        Location(item["id"], _point(item["approach_position"]))
        for item in raw["delivery_points"]
    )
    obstacles = tuple(
        Obstacle(item["id"], _point(item["center"]), _point(item["size"]))
        for item in raw["static_obstacles"]
    )
    uavs = tuple(
        UAV(item["id"], int(item["capacity"]), float(item["cruise_speed"]))
        for item in raw["uavs"]
    )

    if len(uavs) != 3:
        raise ValueError(f"The baseline expects exactly three UAVs, got {len(uavs)}")
    if not pickups or not deliveries:
        raise ValueError("At least one pickup and delivery point are required")

    return WorldConfig(
        name=str(raw["world"]["name"]),
        minimum=_point(bounds["min"]),
        maximum=_point(bounds["max"]),
        depot=Location("depot", _point(raw["depot"]["approach_position"])),
        uav_spawn_points=tuple(_point(point) for point in raw["depot"]["uav_spawn_points"]),
        pickups=pickups,
        deliveries=deliveries,
        obstacles=obstacles,
        tree_positions=tuple(_point(point) for point in raw.get("tree_positions", [])),
        uavs=uavs,
        task_settings=TaskSettings(
            count=int(task_generation["count"]),
            seed=int(task_generation["seed"]),
            package_quantity=int(task_generation["package_quantity"]),
            distribution=str(task_generation.get("distribution", "random")),
            exclusive_delivery_uav=bool(task_generation.get("exclusive_delivery_uav", False)),
        ),
        planning=PlanningSettings(
            resolution=float(planning["resolution"]),
            safety_margin=float(planning["safety_margin"]),
            minimum_flight_altitude=float(planning["minimum_flight_altitude"]),
            maximum_flight_altitude=float(planning["maximum_flight_altitude"]),
            makespan_weight=int(planning["makespan_weight"]),
            solver_time_limit_seconds=int(planning["solver_time_limit_seconds"]),
            trajectory_sample_period=float(planning["trajectory_sample_period"]),
            maximum_acceleration=float(planning["maximum_acceleration"]),
            corner_radius=float(planning["corner_radius"]),
            minimum_uav_separation=float(planning["minimum_uav_separation"]),
            schedule_step=float(planning["schedule_step"]),
        ),
    )
