from __future__ import annotations

import math
from typing import Any

import numpy as np

from .curriculum_stage1_env import CurriculumStage1Env
from .pure_ppo_avoidance_env import PurePPOLocalAvoidanceEnv
from .rl_avoidance_env import FRAME_STACK, MovingSphere, UAVLocalAvoidanceEnv


class CurriculumStage2Env(CurriculumStage1Env):
    """Second curriculum lesson: reach the goal around one static obstacle."""

    def __init__(self, *, max_steps: int = 400, dt: float = 0.1) -> None:
        super().__init__(max_steps=max_steps, dt=dt)
        self.obstacle_count = 1
        self.obstacle_passed = False
        self.post_obstacle_stagnation_steps = 0
        self.avoidance_active = False
        self.safe_recovery_ticks = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed, options=options)

        route = self.goal - self.start
        route_length = float(np.linalg.norm(route[:2]))
        route_direction = route / max(route_length, 1e-6)
        lateral = np.array(
            [-route_direction[1], route_direction[0], 0.0], dtype=np.float32
        )
        along_route = float(self.np_random.uniform(0.38, 0.62) * route_length)
        lateral_offset = float(self.np_random.uniform(-0.45, 0.45))
        radius = float(self.np_random.uniform(0.75, 1.15))
        self.obstacles = [
            MovingSphere(
                position=(
                    self.start
                    + route_direction * along_route
                    + lateral * lateral_offset
                ).astype(np.float32),
                velocity=np.zeros(3, dtype=np.float32),
                radius=radius,
            )
        ]
        self.obstacle_passed = False
        self.post_obstacle_stagnation_steps = 0
        self.avoidance_active = False
        self.safe_recovery_ticks = 0

        initial_scan = self._scan()
        self.scan_history.clear()
        for _ in range(FRAME_STACK):
            self.scan_history.append(initial_scan.copy())
        observation = self._observation()
        return observation, self._info(collision=False, success=False, outside=False)

    def _clearance(self) -> float:
        if not self.obstacles:
            return math.inf
        return UAVLocalAvoidanceEnv._clearance(self)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        clipped_action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        goal_body = self._world_to_body(self.goal - self.position)[:2]
        goal_distance_before = float(np.linalg.norm(goal_body))
        goal_direction = goal_body / max(goal_distance_before, 1e-6)
        previous_action = self.previous_action.copy()
        clearance_before = self._clearance()

        observation, _, terminated, truncated, info = PurePPOLocalAvoidanceEnv.step(
            self, clipped_action
        )
        progress = goal_distance_before - float(info["goal_distance"])
        alignment = float(clipped_action @ goal_direction)
        smoothness = float(np.linalg.norm(clipped_action - previous_action))
        clearance = float(info["minimum_clearance"])
        ttc = float(info["minimum_ttc"])

        route = self.goal - self.start
        route_direction = route / max(float(np.linalg.norm(route)), 1e-6)
        lateral_direction = np.array(
            [-route_direction[1], route_direction[0], 0.0], dtype=np.float32
        )
        agent_along = float((self.position - self.start) @ route_direction)
        obstacle_along = float(
            (self.obstacles[0].position - self.start) @ route_direction
        )
        just_passed = (
            not self.obstacle_passed
            and agent_along
            > obstacle_along + self.obstacles[0].radius + self.agent_radius
        )
        if just_passed:
            self.obstacle_passed = True
            self.post_obstacle_stagnation_steps = 0

        obstacle_relative = self.obstacles[0].position - self.position
        goal_world = self.goal - self.position
        goal_world_direction = goal_world / max(float(np.linalg.norm(goal_world)), 1e-6)
        obstacle_behind = float(obstacle_relative @ goal_world_direction) < 0.0

        # Hysteresis: risk activates avoidance immediately, while returning to
        # goal tracking requires eight consecutive safe ticks with the
        # obstacle behind.  This prevents rapid mode switching at the side of
        # the obstacle.
        if not self.avoidance_active and (clearance < 2.6 or ttc < 4.0):
            self.avoidance_active = True
            self.safe_recovery_ticks = 0
        elif self.avoidance_active:
            safe_to_recover = obstacle_behind and clearance > 1.6 and ttc > 3.0
            self.safe_recovery_ticks = (
                self.safe_recovery_ticks + 1 if safe_to_recover else 0
            )
            if self.safe_recovery_ticks >= 5:
                self.avoidance_active = False

        emergency = clearance < 1.0 or ttc < 1.5
        clearance_change = clearance - clearance_before
        lateral_speed = abs(float(self.velocity @ lateral_direction))
        closing_speed = float(self.velocity @ goal_world_direction)
        reward = (
            -0.015
            - 0.012 * float(clipped_action @ clipped_action)
            - 0.018 * smoothness
        )
        if emergency:
            # In immediate danger, making clearance is the only shaped
            # objective; moving closer to the obstacle is penalized strongly.
            reward += 4.0 * clearance_change + 0.2 * lateral_speed
            if clearance_change < 0.0:
                reward -= 7.0 * abs(clearance_change)
            guidance_mode = "emergency"
        elif self.avoidance_active:
            # Preserve weak goal pressure so detours still make useful
            # progress, without pulling the UAV back into the obstacle.
            reward += 6.0 * progress + 0.08 * alignment + 2.0 * clearance_change
            if not obstacle_behind and clearance < 2.5:
                reward += 0.12 * lateral_speed
            elif obstacle_behind:
                # Stop rewarding endless circling after the obstacle has
                # moved behind the vehicle, even while the exit timer runs.
                reward += 8.0 * progress + 0.25 * alignment + 0.2 * closing_speed
            guidance_mode = "avoidance"
        else:
            progress_weight = 24.0 if self.obstacle_passed else 12.0
            alignment_weight = 0.8 if self.obstacle_passed else 0.6
            reward += progress_weight * progress + alignment_weight * alignment
            if self.obstacle_passed:
                reward += 0.35 * closing_speed
            guidance_mode = "recovery" if self.obstacle_passed else "tracking"
        if just_passed:
            reward += 35.0
        if self.obstacle_passed and not self.avoidance_active:
            if progress < 0.002 and float(info["goal_distance"]) > 1.0:
                self.post_obstacle_stagnation_steps += 1
            else:
                self.post_obstacle_stagnation_steps = 0
            if self.post_obstacle_stagnation_steps > 10:
                reward -= min(
                    1.5, 0.04 * (self.post_obstacle_stagnation_steps - 10)
                )
            if progress < -0.005:
                reward -= 6.0 * abs(progress)
        if clearance < 1.5:
            reward -= 1.5 * (1.5 - max(clearance, 0.0)) ** 2
        if ttc < 3.0:
            reward -= 1.0 * (3.0 - max(ttc, 0.0)) ** 2 / 3.0
        if float(np.linalg.norm(self.velocity[:2])) < 0.2 and float(
            info["goal_distance"]
        ) > 1.2:
            reward -= 0.1
        if bool(info["collision"]):
            reward -= 250.0
        elif bool(info["outside"]):
            reward -= 100.0
        elif bool(info["success"]):
            reward += 120.0
        elif truncated:
            reward -= 80.0
        info["obstacle_passed"] = self.obstacle_passed
        info["post_obstacle_stagnation_steps"] = self.post_obstacle_stagnation_steps
        info["guidance_mode"] = guidance_mode
        info["safe_recovery_ticks"] = self.safe_recovery_ticks
        return observation, float(reward), terminated, truncated, info


def make_env(seed: int = 0) -> CurriculumStage2Env:
    env = CurriculumStage2Env()
    env.reset(seed=seed)
    return env
