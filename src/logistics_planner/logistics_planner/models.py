from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

Point3D = Tuple[float, float, float]


@dataclass(frozen=True)
class Location:
    id: str
    position: Point3D


@dataclass(frozen=True)
class Obstacle:
    id: str
    center: Point3D
    size: Point3D


@dataclass(frozen=True)
class UAV:
    id: str
    capacity: int
    cruise_speed: float


@dataclass(frozen=True)
class Task:
    id: str
    pickup: Location
    delivery: Location
    quantity: int


@dataclass
class RouteStop:
    node_index: int
    label: str
    kind: str
    position: Point3D
    task_id: str | None
    load_after: int


@dataclass
class VehicleRoute:
    uav: UAV
    stops: list[RouteStop]
    path: list[Point3D]
    distance: float
    estimated_time: float
    peak_load: int
