from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


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
        max_steps: int = 400,
        obstacle_count: int = 3,
        dt: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_steps = int(max_steps)
        self.obstacle_count = int(obstacle_count)
        self.dt = float(dt)
        self.sensor_range = 20.0
        self.agent_radius = 0.45
        self.cruise_speed = 1.5
        self.max_correction = 2.0
        self.max_speed = 3.0
        self.goal_radius = 0.8
        self.corridor_min = np.array([-14.0, -8.0, -4.0], dtype=np.float32)
        self.corridor_max = np.array([14.0, 8.0, 4.0], dtype=np.float32)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.concatenate((np.zeros(36), -np.ones(6))).astype(np.float32),
            high=np.ones(42, dtype=np.float32),
            dtype=np.float32,
        )
        self.position = np.zeros(3, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.goal = np.zeros(3, dtype=np.float32)
        self.obstacles: list[MovingSphere] = []
        self.previous_action = np.zeros(3, dtype=np.float32)
        self.previous_goal_distance = 0.0
        self.step_count = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.position = np.array([-12.0, 0.0, 0.0], dtype=np.float32)
        self.goal = np.array([12.0, 0.0, 0.0], dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.previous_action = np.zeros(3, dtype=np.float32)
        self.step_count = 0
        self.obstacles = self._spawn_obstacles()
        self.previous_goal_distance = self._goal_distance()
        observation = self._observation()
        return observation, self._info(collision=False, success=False, outside=False)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        desired = self._nominal_velocity()
        correction = action * self.max_correction
        command = desired + correction
        speed = float(np.linalg.norm(command))
        if speed > self.max_speed:
            command *= self.max_speed / speed
        # First-order response approximates the PX4 velocity loop and prevents
        # unrealistically instantaneous action changes.
        alpha = 1.0 - math.exp(-self.dt / 0.35)
        self.velocity += np.float32(alpha) * (command - self.velocity)
        self.position += self.velocity * self.dt
        self._move_obstacles()
        self.step_count += 1

        goal_distance = self._goal_distance()
        progress = self.previous_goal_distance - goal_distance
        self.previous_goal_distance = goal_distance
        clearance = self._clearance()
        collision = clearance <= 0.0
        outside = bool(
            np.any(self.position < self.corridor_min)
            or np.any(self.position > self.corridor_max)
        )
        success = goal_distance <= self.goal_radius
        terminated = collision or outside or success
        truncated = self.step_count >= self.max_steps and not terminated

        smoothness = float(np.linalg.norm(action - self.previous_action))
        reward = 10.0 * progress - 0.015 - 0.008 * float(action @ action) - 0.015 * smoothness
        if clearance < 3.5:
            reward -= 0.45 * (3.5 - max(clearance, 0.0)) ** 2
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
        primary_x = float(self.np_random.uniform(-2.5, 2.5))
        arrival_time = (primary_x + 12.0) / self.cruise_speed
        crossing_speed = float(self.np_random.uniform(-0.55, 0.55))
        if abs(crossing_speed) < 0.18:
            crossing_speed = math.copysign(0.18, crossing_speed or 1.0)
        primary = MovingSphere(
            position=np.array(
                [
                    primary_x,
                    np.clip(-crossing_speed * arrival_time + self.np_random.uniform(-0.45, 0.45), -5.5, 5.5),
                    self.np_random.uniform(-0.35, 0.35),
                ],
                dtype=np.float32,
            ),
            velocity=np.array(
                [self.np_random.uniform(-0.08, 0.08), crossing_speed, self.np_random.uniform(-0.05, 0.05)],
                dtype=np.float32,
            ),
            radius=float(self.np_random.uniform(1.0, 1.35)),
        )
        obstacles.append(primary)
        for _ in range(max(0, self.obstacle_count - 1)):
            obstacles.append(
                MovingSphere(
                    position=np.array(
                        [self.np_random.uniform(-7.0, 8.0), self.np_random.uniform(-5.0, 5.0), self.np_random.uniform(-2.0, 2.0)],
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
        lower = np.array([-9.0, -7.0, -3.0], dtype=np.float32)
        upper = np.array([9.0, 7.0, 3.0], dtype=np.float32)
        for obstacle in self.obstacles:
            obstacle.position += obstacle.velocity * self.dt
            for axis in range(3):
                if obstacle.position[axis] < lower[axis]:
                    obstacle.position[axis] = lower[axis]
                    obstacle.velocity[axis] = abs(obstacle.velocity[axis])
                elif obstacle.position[axis] > upper[axis]:
                    obstacle.position[axis] = upper[axis]
                    obstacle.velocity[axis] = -abs(obstacle.velocity[axis])

    def _observation(self) -> np.ndarray:
        scan = ray_sphere_distances(
            self.position, RAY_DIRECTIONS, self.obstacles, self.sensor_range
        )
        relative_goal = np.clip((self.goal - self.position) / 25.0, -1.0, 1.0)
        normalized_velocity = np.clip(self.velocity / self.max_speed, -1.0, 1.0)
        return np.concatenate(
            (scan / self.sensor_range, relative_goal, normalized_velocity)
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

    def _info(self, collision: bool, success: bool, outside: bool) -> dict[str, Any]:
        return {
            "goal_distance": self._goal_distance(),
            "minimum_clearance": self._clearance(),
            "collision": collision,
            "success": success,
            "outside": outside,
            "step_count": self.step_count,
        }


def make_env(seed: int = 0) -> UAVLocalAvoidanceEnv:
    env = UAVLocalAvoidanceEnv()
    env.reset(seed=seed)
    return env
