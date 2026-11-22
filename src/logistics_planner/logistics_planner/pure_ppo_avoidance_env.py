from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np
from gymnasium import spaces

from .rl_avoidance_env import FRAME_STACK, UAVLocalAvoidanceEnv


class PurePPOLocalAvoidanceEnv(UAVLocalAvoidanceEnv):
    """Horizontal direct-velocity PPO environment with randomized dynamics.

    The policy receives the same 150-value observation used by the residual
    policy, but directly commands body-frame forward and left velocity.  No
    artificial-potential-field output participates in :meth:`step`.
    """

    def __init__(
        self,
        *,
        max_steps: int = 700,
        obstacle_count: int = 5,
        dt: float = 0.1,
        round_trip: bool = True,
    ) -> None:
        super().__init__(
            max_steps=max_steps,
            obstacle_count=obstacle_count,
            dt=dt,
            round_trip=round_trip,
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.sensor_noise_std = 0.02
        self.sensor_dropout_probability = 0.005
        self.response_time_constant = 0.4
        self.maximum_acceleration = 2.5
        self.control_delay_steps = 0
        self.wind_acceleration = np.zeros(3, dtype=np.float32)
        self.command_queue: deque[np.ndarray] = deque()
        self.previous_action = np.zeros(2, dtype=np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        _, _ = super().reset(seed=seed, options=options)
        self.sensor_noise_std = float(self.np_random.uniform(0.005, 0.05))
        self.sensor_dropout_probability = float(self.np_random.uniform(0.0, 0.015))
        self.response_time_constant = float(self.np_random.uniform(0.25, 0.65))
        self.maximum_acceleration = float(self.np_random.uniform(1.8, 3.2))
        self.control_delay_steps = int(self.np_random.integers(0, 4))
        self.wind_acceleration = np.array(
            [
                self.np_random.uniform(-0.12, 0.12),
                self.np_random.uniform(-0.12, 0.12),
                0.0,
            ],
            dtype=np.float32,
        )
        self.command_queue = deque(
            (np.zeros(2, dtype=np.float32) for _ in range(self.control_delay_steps))
        )
        self.previous_action = np.zeros(2, dtype=np.float32)

        initial_scan = self._scan()
        self.scan_history.clear()
        for _ in range(FRAME_STACK):
            self.scan_history.append(initial_scan.copy())
        observation = self._observation()
        return observation, self._info(collision=False, success=False, outside=False)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if action.shape != (2,):
            raise ValueError("Pure PPO action must contain forward and left velocity")

        self.command_queue.append(action.copy())
        delayed_action = self.command_queue.popleft()
        target_body = np.array(
            [delayed_action[0] * self.max_speed, delayed_action[1] * self.max_speed, 0.0],
            dtype=np.float32,
        )
        target_velocity = self._body_to_world(target_body)

        acceleration = (target_velocity - self.velocity) / self.response_time_constant
        acceleration += self.wind_acceleration
        acceleration_norm = float(np.linalg.norm(acceleration))
        if acceleration_norm > self.maximum_acceleration:
            acceleration *= self.maximum_acceleration / acceleration_norm
        self.velocity += acceleration * self.dt
        self.velocity[2] = 0.0
        speed = float(np.linalg.norm(self.velocity[:2]))
        if speed > self.max_speed:
            self.velocity[:2] *= self.max_speed / speed
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
        reward_waypoint = 0.0
        if waypoint_reached and self.round_trip and not self.outbound_complete:
            self.outbound_complete = True
            self.goal = self.start.copy()
            self.previous_goal_distance = self._goal_distance()
            reward_waypoint = 40.0

        terminated = collision or outside or success
        truncated = self.step_count >= self.max_steps and not terminated
        smoothness = float(np.linalg.norm(action - self.previous_action))
        action_energy = float(action @ action)
        reward = 12.0 * progress - 0.02 - 0.006 * action_energy - 0.025 * smoothness
        reward += reward_waypoint
        if clearance < 3.5:
            reward -= 0.55 * (3.5 - max(clearance, 0.0)) ** 2
        if ttc < 6.0:
            reward -= 1.4 * (6.0 - max(ttc, 0.0)) ** 2 / 6.0
        boundary_margin = self._boundary_margin()
        if boundary_margin < 3.0:
            reward -= 0.65 * (3.0 - max(boundary_margin, 0.0)) ** 2
        route_deviation = self._route_deviation()
        if route_deviation > 4.0:
            reward -= 0.08 * (route_deviation - 4.0) ** 2
        if float(np.linalg.norm(self.velocity[:2])) < 0.15 and goal_distance > 1.5:
            reward -= 0.05
        if collision:
            reward -= 180.0
        elif outside:
            reward -= 80.0
        elif success:
            reward += 120.0
        self.previous_action = action.copy()

        return self._observation(), float(reward), terminated, truncated, self._info(
            collision, success, outside
        )

    def _scan(self) -> np.ndarray:
        distances = super()._scan()
        if self.sensor_noise_std > 0.0:
            distances = distances + self.np_random.normal(
                0.0, self.sensor_noise_std, size=distances.shape
            ).astype(np.float32)
        if self.sensor_dropout_probability > 0.0:
            dropout = self.np_random.random(distances.shape) < self.sensor_dropout_probability
            distances[dropout] = self.sensor_range
        return np.clip(distances, 0.0, self.sensor_range).astype(np.float32)

    def dynamics_info(self) -> dict[str, float | int]:
        return {
            "response_time_constant": self.response_time_constant,
            "maximum_acceleration": self.maximum_acceleration,
            "control_delay_steps": self.control_delay_steps,
            "sensor_noise_std": self.sensor_noise_std,
            "sensor_dropout_probability": self.sensor_dropout_probability,
        }


def make_env(seed: int = 0) -> PurePPOLocalAvoidanceEnv:
    env = PurePPOLocalAvoidanceEnv()
    env.reset(seed=seed)
    return env
