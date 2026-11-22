from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np

from .pure_ppo_avoidance_env import PurePPOLocalAvoidanceEnv
from .rl_avoidance_env import FRAME_STACK


class CurriculumStage1Env(PurePPOLocalAvoidanceEnv):
    """First curriculum lesson: direct-velocity flight to an unobstructed goal.

    The observation and two-dimensional action have exactly the same layout as
    the later pure-PPO avoidance environment.  Only the task difficulty is
    reduced: there are no obstacles or disturbances and the episode ends at
    the first goal.  This lets the policy learn the basic goal-seeking mapping
    before obstacle avoidance is introduced.
    """

    def __init__(self, *, max_steps: int = 300, dt: float = 0.1) -> None:
        super().__init__(
            max_steps=max_steps,
            obstacle_count=0,
            dt=dt,
            round_trip=False,
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        # Seed Gymnasium through the parent, then replace its difficult route
        # and randomized dynamics with this lesson's simple distribution.
        super().reset(seed=seed, options=options)

        self.start = np.zeros(3, dtype=np.float32)
        target_angle = float(self.np_random.uniform(-math.pi, math.pi))
        target_distance = float(self.np_random.uniform(6.0, 12.0))
        self.outbound_goal = np.array(
            [
                target_distance * math.cos(target_angle),
                target_distance * math.sin(target_angle),
                0.0,
            ],
            dtype=np.float32,
        )
        self.position = self.start.copy()
        self.goal = self.outbound_goal.copy()
        self.heading = float(self.np_random.uniform(-math.pi, math.pi))
        self.velocity = np.zeros(3, dtype=np.float32)
        self.previous_action = np.zeros(2, dtype=np.float32)
        self.previous_goal_distance = self._goal_distance()
        self.outbound_complete = False
        self.step_count = 0
        self.obstacles = []

        # Stage 1 deliberately excludes domain randomization.  These effects
        # are introduced only after basic navigation has converged.
        self.sensor_noise_std = 0.0
        self.sensor_dropout_probability = 0.0
        self.response_time_constant = 0.35
        self.maximum_acceleration = 3.0
        self.control_delay_steps = 0
        self.wind_acceleration = np.zeros(3, dtype=np.float32)
        self.command_queue = deque()

        initial_scan = self._scan()
        self.scan_history.clear()
        for _ in range(FRAME_STACK):
            self.scan_history.append(initial_scan.copy())
        observation = self._observation()
        return observation, self._info(collision=False, success=False, outside=False)

    def _clearance(self) -> float:
        """There is no collision object in the first curriculum lesson."""
        return math.inf

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply direct velocity and use dense goal-following lesson rewards."""
        clipped_action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        goal_body = self._world_to_body(self.goal - self.position)[:2]
        goal_distance_before = float(np.linalg.norm(goal_body))
        goal_direction = goal_body / max(goal_distance_before, 1e-6)
        previous_action = self.previous_action.copy()

        observation, _, terminated, truncated, info = super().step(clipped_action)
        progress = goal_distance_before - float(info["goal_distance"])
        alignment = float(clipped_action @ goal_direction)
        smoothness = float(np.linalg.norm(clipped_action - previous_action))
        reward = (
            12.0 * progress
            + 0.6 * alignment
            - 0.015
            - 0.015 * float(clipped_action @ clipped_action)
            - 0.02 * smoothness
        )
        if bool(info["success"]):
            reward += 120.0
        elif bool(info["outside"]):
            reward -= 100.0
        elif truncated:
            reward -= 20.0
        return observation, float(reward), terminated, truncated, info


def make_env(seed: int = 0) -> CurriculumStage1Env:
    env = CurriculumStage1Env()
    env.reset(seed=seed)
    return env
