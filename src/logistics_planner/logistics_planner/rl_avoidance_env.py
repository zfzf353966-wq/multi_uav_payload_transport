from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .shadow_avoidance import compute_shadow_avoidance, limit_vector


ELEVATIONS = np.radians(np.array([-12.0, 0.0, 12.0], dtype=np.float32))
AZIMUTHS = np.arange(12, dtype=np.float32) * (2.0 * np.pi / 12.0)


def sensor_directions() -> np.ndarray:
    """Return the fixed lower/middle/upper 3x12 FLU unit rays."""
    rays = []
    for elevation in ELEVATIONS:
        for azimuth in AZIMUTHS:
            rays.append(
                (
                    math.cos(float(elevation)) * math.cos(float(azimuth)),
                    math.cos(float(elevation)) * math.sin(float(azimuth)),
                    math.sin(float(elevation)),
                )
            )
    return np.asarray(rays, dtype=np.float32)


RAY_DIRECTIONS = sensor_directions()
FRAME_STACK = 4
RESIDUAL_SCALE = 0.8


@dataclass
class MovingSphere:
    position: np.ndarray
    velocity: np.ndarray
    radius: float


def ray_sphere_distances(
    origin: np.ndarray,
    directions: np.ndarray,
    spheres: list[MovingSphere],
    maximum_range: float,
) -> np.ndarray:
    """Nearest distance in each lidar sector for spherical obstacles.

    The real ROS observation stores the nearest point in a 30-degree sector,
    not only objects intersecting an infinitesimally thin centre ray. A
    15-degree acceptance cone plus the sphere's angular radius approximates
    that binning while retaining the fixed 3x12 layout.
    """
    distances = np.full(len(directions), maximum_range, dtype=np.float32)
    for sphere in spheres:
        relative = sphere.position - origin
        centre_distance = float(np.linalg.norm(relative))
        if centre_distance <= 1e-6:
            distances.fill(0.0)
            continue
        direction = relative / centre_distance
        angular_radius = math.asin(min(1.0, sphere.radius / centre_distance))
        angular_separation = np.arccos(np.clip(directions @ direction, -1.0, 1.0))
        visible = angular_separation <= angular_radius + math.radians(15.0)
        surface_distance = max(0.0, centre_distance - sphere.radius)
        candidate = np.where(visible, surface_distance, maximum_range).astype(np.float32)
        distances = np.minimum(distances, candidate)
    return np.clip(distances, 0.0, maximum_range)


class UAVLocalAvoidanceEnv(gym.Env[np.ndarray, np.ndarray]):
    """Fast kinematic environment matching the project's 36-sector lidar API."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        max_steps: int = 700,
        obstacle_count: int = 5,
        dt: float = 0.1,
        round_trip: bool = True,
    ) -> None:
        super().__init__()
        self.max_steps = int(max_steps)
        self.obstacle_count = int(obstacle_count)
        self.dt = float(dt)
        self.round_trip = bool(round_trip)
        self.sensor_range = 20.0
        self.agent_radius = 0.45
        self.cruise_speed = 1.5
        self.max_correction = 2.0
        self.max_speed = 3.0
        self.goal_radius = 0.8
        self.corridor_min = np.array([-15.0, -15.0, -5.0], dtype=np.float32)
        self.corridor_max = np.array([15.0, 15.0, 5.0], dtype=np.float32)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.concatenate((np.zeros(36 * FRAME_STACK), -np.ones(6))).astype(np.float32),
            high=np.ones(36 * FRAME_STACK + 6, dtype=np.float32),
            dtype=np.float32,
        )
        self.position = np.zeros(3, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.goal = np.zeros(3, dtype=np.float32)
        self.start = np.zeros(3, dtype=np.float32)
        self.outbound_goal = np.zeros(3, dtype=np.float32)
        self.heading = 0.0
        self.outbound_complete = False
        self.obstacles: list[MovingSphere] = []
        self.previous_action = np.zeros(3, dtype=np.float32)
        self.previous_goal_distance = 0.0
        self.step_count = 0
        self.scan_history: deque[np.ndarray] = deque(maxlen=FRAME_STACK)
        self.last_ttc = math.inf

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        route_angle = float(self.np_random.uniform(-math.pi, math.pi))
        route = np.array([math.cos(route_angle), math.sin(route_angle), 0.0], dtype=np.float32)
        self.start = -12.0 * route
        self.outbound_goal = 12.0 * route
        self.position = self.start.copy()
        self.goal = self.outbound_goal.copy()
        # Heading is deliberately independent of route direction so the goal
        # can appear anywhere in the body frame, including directly behind.
        self.heading = float(self.np_random.uniform(-math.pi, math.pi))
        self.outbound_complete = False
        self.velocity = np.zeros(3, dtype=np.float32)
        self.previous_action = np.zeros(3, dtype=np.float32)
        self.step_count = 0
        self.obstacles = self._spawn_obstacles()
        initial_scan = self._scan()
        self.scan_history.clear()
        for _ in range(FRAME_STACK):
            self.scan_history.append(initial_scan.copy())
        self.previous_goal_distance = self._goal_distance()
        observation = self._observation()
        return observation, self._info(collision=False, success=False, outside=False)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        desired = self._nominal_velocity()
        desired_body = self._world_to_body(desired)
        potential = compute_shadow_avoidance(
            self.scan_history[-1], desired_body,
            influence_distance=6.0, emergency_distance=2.0,
            max_correction=2.0, max_speed=self.max_speed,
        )
        residual = action * RESIDUAL_SCALE
        command_body = np.asarray(
            limit_vector(
                tuple(
                    potential.suggested_velocity[index] + float(residual[index])
                    for index in range(3)
                ),
                self.max_speed,
            ),
            dtype=np.float32,
        )
        command = self._body_to_world(command_body)
        speed = float(np.linalg.norm(command))
        if speed > self.max_speed:
            command *= self.max_speed / speed
        # First-order response approximates the PX4 velocity loop and prevents
        # unrealistically instantaneous action changes.
        alpha = 1.0 - math.exp(-self.dt / 0.35)
        self.velocity += np.float32(alpha) * (command - self.velocity)
        self.position += self.velocity * self.dt
        self._move_obstacles()
        self.scan_history.append(self._scan())
        self.step_count += 1

        goal_distance = self._goal_distance()
        progress = self.previous_goal_distance - goal_distance
        self.previous_goal_distance = goal_distance
        clearance = self._clearance()
        ttc = self._minimum_ttc()
        self.last_ttc = ttc
        collision = clearance <= 0.0
        outside = bool(
            np.any(self.position < self.corridor_min)
            or np.any(self.position > self.corridor_max)
        )
        waypoint_reached = goal_distance <= self.goal_radius
        success = waypoint_reached and (self.outbound_complete or not self.round_trip)
        if waypoint_reached and self.round_trip and not self.outbound_complete:
            self.outbound_complete = True
            self.goal = self.start.copy()
            goal_distance = self._goal_distance()
            self.previous_goal_distance = goal_distance
            reward_waypoint = 35.0
        else:
            reward_waypoint = 0.0
        terminated = collision or outside or success
        truncated = self.step_count >= self.max_steps and not terminated

        smoothness = float(np.linalg.norm(action - self.previous_action))
        reward = (
            10.0 * progress
            - 0.015
            - 0.008 * float(action @ action)
            - 0.015 * smoothness
            + reward_waypoint
        )
        if clearance < 3.5:
            reward -= 0.45 * (3.5 - max(clearance, 0.0)) ** 2
        # Penalize a closing collision course before clearance becomes
        # critical. TTC beyond six seconds is treated as non-urgent.
        if ttc < 6.0:
            reward -= 1.2 * (6.0 - max(ttc, 0.0)) ** 2 / 6.0
        # Continuous boundary and route penalties teach recovery before a
        # binary geofence termination is reached.
        boundary_margin = self._boundary_margin()
        if boundary_margin < 3.0:
            reward -= 0.65 * (3.0 - max(boundary_margin, 0.0)) ** 2
        route_deviation = self._route_deviation()
        if route_deviation > 3.0:
            reward -= 0.12 * (route_deviation - 3.0) ** 2
        if collision:
            reward -= 150.0
        elif outside:
            reward -= 60.0
        elif success:
            reward += 100.0
        self.previous_action = action.copy()

        observation = self._observation()
        return observation, float(reward), terminated, truncated, self._info(collision, success, outside)

    def _spawn_obstacles(self) -> list[MovingSphere]:
        obstacles = []
        # The primary obstacle is deliberately timed to be near the nominal
        # route when a straight-flying UAV arrives. This prevents an easy
        # dataset where doing nothing wins most episodes.
        route = self.outbound_goal / max(float(np.linalg.norm(self.outbound_goal)), 1e-6)
        lateral = np.array([-route[1], route[0], 0.0], dtype=np.float32)
        along_track = float(self.np_random.uniform(-2.5, 2.5))
        arrival_time = (along_track + 12.0) / self.cruise_speed
        crossing_speed = float(self.np_random.uniform(-0.55, 0.55))
        if abs(crossing_speed) < 0.18:
            crossing_speed = math.copysign(0.18, crossing_speed or 1.0)
        primary = MovingSphere(
            position=(
                route * along_track
                + lateral
                * np.clip(
                    -crossing_speed * arrival_time + self.np_random.uniform(-0.45, 0.45),
                    -5.5,
                    5.5,
                )
                + np.array([0.0, 0.0, self.np_random.uniform(-0.35, 0.35)], dtype=np.float32)
            ).astype(np.float32),
            velocity=(
                lateral * crossing_speed
                + route * self.np_random.uniform(-0.08, 0.08)
                + np.array([0.0, 0.0, self.np_random.uniform(-0.05, 0.05)], dtype=np.float32)
            ).astype(np.float32),
            radius=float(self.np_random.uniform(1.0, 1.35)),
        )
        obstacles.append(primary)
        for _ in range(max(0, self.obstacle_count - 1)):
            obstacles.append(
                MovingSphere(
                    position=np.array(
                        [self.np_random.uniform(-9.0, 9.0), self.np_random.uniform(-9.0, 9.0), self.np_random.uniform(-2.0, 2.0)],
                        dtype=np.float32,
                    ),
                    velocity=np.array(
                        [self.np_random.uniform(-0.6, 0.6), self.np_random.uniform(-0.6, 0.6), self.np_random.uniform(-0.25, 0.25)],
                        dtype=np.float32,
                    ),
                    radius=float(self.np_random.uniform(0.5, 0.9)),
                )
            )
        return obstacles

    def _move_obstacles(self) -> None:
        lower = np.array([-11.0, -11.0, -3.0], dtype=np.float32)
        upper = np.array([11.0, 11.0, 3.0], dtype=np.float32)
        for obstacle in self.obstacles:
            obstacle.position += obstacle.velocity * self.dt
            for axis in range(3):
                if obstacle.position[axis] < lower[axis]:
                    obstacle.position[axis] = lower[axis]
                    obstacle.velocity[axis] = abs(obstacle.velocity[axis])
                elif obstacle.position[axis] > upper[axis]:
                    obstacle.position[axis] = upper[axis]
                    obstacle.velocity[axis] = -abs(obstacle.velocity[axis])

    def _scan(self) -> np.ndarray:
        return ray_sphere_distances(
            self.position,
            np.asarray([self._body_to_world(ray) for ray in RAY_DIRECTIONS], dtype=np.float32),
            self.obstacles,
            self.sensor_range,
        )

    def _observation(self) -> np.ndarray:
        if not self.scan_history:
            scan = self._scan()
            for _ in range(FRAME_STACK):
                self.scan_history.append(scan.copy())
        relative_goal = np.clip(self._world_to_body(self.goal - self.position) / 25.0, -1.0, 1.0)
        normalized_velocity = np.clip(self._world_to_body(self.velocity) / self.max_speed, -1.0, 1.0)
        return np.concatenate(
            tuple(scan / self.sensor_range for scan in self.scan_history)
            + (relative_goal, normalized_velocity)
        ).astype(np.float32)

    def _nominal_velocity(self) -> np.ndarray:
        difference = self.goal - self.position
        distance = float(np.linalg.norm(difference))
        if distance < 1e-6:
            return np.zeros(3, dtype=np.float32)
        return difference * np.float32(min(self.cruise_speed, distance) / distance)

    def _goal_distance(self) -> float:
        return float(np.linalg.norm(self.goal - self.position))

    def _clearance(self) -> float:
        return min(
            float(np.linalg.norm(obstacle.position - self.position))
            - obstacle.radius
            - self.agent_radius
            for obstacle in self.obstacles
        )

    def _minimum_ttc(self) -> float:
        minimum = math.inf
        combined_radius = self.agent_radius
        for obstacle in self.obstacles:
            relative_position = obstacle.position - self.position
            relative_velocity = obstacle.velocity - self.velocity
            radius = combined_radius + obstacle.radius
            a = float(relative_velocity @ relative_velocity)
            b = 2.0 * float(relative_position @ relative_velocity)
            c = float(relative_position @ relative_position) - radius * radius
            if c <= 0.0:
                return 0.0
            if a < 1e-8 or b >= 0.0:
                continue
            discriminant = b * b - 4.0 * a * c
            if discriminant >= 0.0:
                impact = (-b - math.sqrt(discriminant)) / (2.0 * a)
                if impact >= 0.0:
                    minimum = min(minimum, impact)
        return minimum

    def _boundary_margin(self) -> float:
        return float(
            min(
                np.min(self.position - self.corridor_min),
                np.min(self.corridor_max - self.position),
            )
        )

    def _route_deviation(self) -> float:
        route = self.outbound_goal - self.start
        norm = float(np.linalg.norm(route))
        if norm < 1e-6:
            return 0.0
        direction = route / norm
        offset = self.position - self.start
        perpendicular = offset - direction * float(offset @ direction)
        return float(np.linalg.norm(perpendicular))

    def _body_to_world(self, vector: np.ndarray | Sequence[float]) -> np.ndarray:
        forward, left, up = map(float, vector)
        cosine, sine = math.cos(self.heading), math.sin(self.heading)
        return np.array(
            [forward * cosine - left * sine, forward * sine + left * cosine, up],
            dtype=np.float32,
        )

    def _world_to_body(self, vector: np.ndarray | Sequence[float]) -> np.ndarray:
        x, y, z = map(float, vector)
        cosine, sine = math.cos(self.heading), math.sin(self.heading)
        return np.array(
            [x * cosine + y * sine, -x * sine + y * cosine, z], dtype=np.float32
        )

    def _info(self, collision: bool, success: bool, outside: bool) -> dict[str, Any]:
        return {
            "goal_distance": self._goal_distance(),
            "minimum_clearance": self._clearance(),
            "collision": collision,
            "success": success,
            "outside": outside,
            "outbound_complete": self.outbound_complete,
            "minimum_ttc": self.last_ttc,
            "boundary_margin": self._boundary_margin(),
            "route_deviation": self._route_deviation(),
            "step_count": self.step_count,
        }


def make_env(seed: int = 0) -> UAVLocalAvoidanceEnv:
    env = UAVLocalAvoidanceEnv()
    env.reset(seed=seed)
    return env
