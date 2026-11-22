"""Three-dimensional direct-velocity curriculum, with no potential-field calls."""
from __future__ import annotations

import math
from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .direct_ppo_contract import (
    ALTITUDE_MAX, ALTITUDE_MIN, MAX_SPEED, OBSERVATION_SIZE,
    action_to_velocity, build_observation, project_safe_velocity, project_directional_velocity,
)
from .rl_avoidance_env import FRAME_STACK, MovingSphere, UAVLocalAvoidanceEnv

STAGES = ("navigation", "static", "dynamic", "multi")


class DirectPPOEnv(UAVLocalAvoidanceEnv):
    def __init__(self, stage="navigation", *, max_steps=500, randomize=False,
                 round_trip=False, reward_version=2, safety_guards=False, history_warmup=False,
                 velocity_mapping="radial", hard_case_probability=0.0, direction_brake=False,
                 lidar_profile="legacy", obstacle_presence_probability=1.0):
        if stage not in STAGES:
            raise ValueError(f"unknown curriculum stage: {stage}")
        super().__init__(max_steps=max_steps, round_trip=round_trip)
        self.stage = stage
        self.randomize = randomize
        if reward_version not in (2, 3, 4):
            raise ValueError("reward version must be 2, 3 or 4")
        self.reward_version = reward_version
        self.safety_guards = safety_guards
        self.direction_brake = direction_brake
        if lidar_profile not in ("legacy", "gazebo") or not 0 <= obstacle_presence_probability <= 1:
            raise ValueError("invalid lidar profile or obstacle presence probability")
        self.lidar_profile = lidar_profile
        self.obstacle_presence_probability = obstacle_presence_probability
        self.history_warmup = history_warmup
        if velocity_mapping not in ("radial", "vertical_priority"):
            raise ValueError("unknown velocity mapping")
        self.velocity_mapping = velocity_mapping
        if not 0 <= hard_case_probability <= 1 or (stage == "navigation" and hard_case_probability):
            raise ValueError("hard-case probability must be in [0,1] and requires obstacles")
        self.hard_case_probability = float(hard_case_probability)
        self.passed_obstacles = set()
        self.encountered_obstacles = set()
        self.stagnation_steps = 0
        self.max_speed = MAX_SPEED
        self.corridor_min = np.array([-22.0, -22.0, ALTITUDE_MIN], dtype=np.float32)
        self.corridor_max = np.array([22.0, 22.0, ALTITUDE_MAX], dtype=np.float32)
        self.observation_space = spaces.Box(
            np.concatenate((np.zeros(144), -np.ones(6), np.zeros(3))).astype(np.float32),
            np.ones(OBSERVATION_SIZE, dtype=np.float32), dtype=np.float32,
        )
        self.orbits = []
        self.response_time = 0.35
        self.maximum_acceleration = 3.0
        self.delay = 0
        self.wind = np.zeros(3, dtype=np.float32)
        self.sensor_noise = 0.0
        self.queue = deque()

    def reset(self, *, seed=None, options=None):
        observation, info = self._reset_once(seed=seed, options=options)
        hard_case = False
        # Rejection sampling changes training frequency, not obstacle bounds or
        # evaluation scenes. Probability zero consumes no extra random numbers.
        if self.hard_case_probability and self.np_random.random() < self.hard_case_probability:
            for _ in range(64):
                if .5 <= self._clearance() <= 3.5:
                    hard_case = True
                    break
                observation, info = self._reset_once()
        info["hard_case"] = hard_case
        return observation, info

    def _reset_once(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        angle = float(self.np_random.uniform(-math.pi, math.pi))
        route = np.array([math.cos(angle), math.sin(angle), 0.0], dtype=np.float32)
        self.start = np.array([0.0, 0.0, 6.0], dtype=np.float32)
        distance = float(self.np_random.uniform(6.0, 12.0) if self.stage == "navigation"
                         else self.np_random.uniform(10.0, 16.0))
        self.outbound_goal = self.start + route * distance
        self.outbound_goal[2] += float(self.np_random.uniform(-1.5, 1.5))
        self.position = self.start.copy()
        self.goal = self.outbound_goal.copy()
        self.heading = float(self.np_random.uniform(-math.pi, math.pi))
        self.velocity = np.zeros(3, dtype=np.float32)
        self.previous_action = np.zeros(3, dtype=np.float32)
        self.step_count = 0
        self.outbound_complete = False
        self.passed_obstacles.clear()
        self.encountered_obstacles.clear()
        self.stagnation_steps = 0
        self.last_ttc = math.inf
        self.obstacles = []
        self.orbits = []
        count = 0 if self.stage == "navigation" else (3 if self.stage == "multi" else 1)
        if self.obstacle_presence_probability < 1 and self.np_random.random() >= self.obstacle_presence_probability:
            count = 0
        for index in range(count):
            fraction = float(self.np_random.uniform(0.4, 0.6)) if count == 1 else (index + 1) / 4
            centre = self.start + fraction * (self.outbound_goal - self.start)
            centre[2] = 6.0 + float(self.np_random.uniform(-0.3, 0.3))
            radius = float(self.np_random.uniform(.6 if self.lidar_profile == "gazebo" else .75, 1.15))
            moving = self.stage in ("dynamic", "multi")
            orbit_radius = float(self.np_random.uniform(.8, 3.1 if self.lidar_profile == "gazebo" else 2.5)) if moving else 0.0
            phase = float(self.np_random.uniform(-math.pi, math.pi))
            omega = float(self.np_random.uniform(0.25, 0.55)) * (-1 if self.np_random.random() < .5 else 1)
            self.orbits.append((centre.copy(), orbit_radius, phase, omega))
            self.obstacles.append(MovingSphere(centre.copy(), np.zeros(3, dtype=np.float32), radius))
        self._place_obstacles()
        self.response_time = float(self.np_random.uniform(.25, .55)) if self.randomize else .35
        self.maximum_acceleration = float(self.np_random.uniform(2.0, 3.2)) if self.randomize else 3.0
        self.delay = int(self.np_random.integers(0, 3)) if self.randomize else 0
        self.wind = self.np_random.uniform(-.08, .08, 3).astype(np.float32) if self.randomize else np.zeros(3, dtype=np.float32)
        self.sensor_noise = .02 if self.randomize else 0.0
        self.queue = deque(np.zeros(3, dtype=np.float32) for _ in range(self.delay))
        self.scan_history.clear()
        if self.history_warmup:
            # The live lidar is running before cruise starts. Sample genuine
            # past obstacle poses while the UAV remains at the initial pose.
            for index in range(FRAME_STACK):
                self._place_obstacles(time_seconds=(index - FRAME_STACK + 1) * self.dt)
                self.scan_history.append(self._scan())
        else:
            initial_scan = self._scan()
            for _ in range(FRAME_STACK):
                self.scan_history.append(initial_scan.copy())
        self.previous_goal_distance = self._goal_distance()
        return self._observation(), self._info(False, False, False)

    def _place_obstacles(self, time_seconds=None):
        for obstacle, (centre, radius, phase, omega) in zip(self.obstacles, self.orbits):
            angle = phase + omega * (self.step_count * self.dt if time_seconds is None else time_seconds)
            obstacle.position = centre + np.array(
                [radius * math.cos(angle), radius * math.sin(angle), 0.0], dtype=np.float32)
            obstacle.velocity = np.array(
                [-radius * omega * math.sin(angle), radius * omega * math.cos(angle), 0.0], dtype=np.float32)

    def _scan(self):
        if self.lidar_profile == "gazebo":
            from .gazebo_lidar_model import sector_scan
            distances = sector_scan(self.position, self.heading, self.obstacles)
        else:
            distances = super()._scan()
        if self.sensor_noise:
            distances += self.np_random.normal(0.0, self.sensor_noise, distances.shape).astype(np.float32)
        return np.clip(distances, 0.0, self.sensor_range)

    def _down_clearance(self):
        if self.lidar_profile == "gazebo":
            from .gazebo_lidar_model import downward_clearance
            return downward_clearance(self.position, self.heading, self.obstacles)
        nearest = float(self.position[2])  # ground plane at ENU z=0
        for obstacle in self.obstacles:
            relative = obstacle.position - self.position
            horizontal = float(np.linalg.norm(relative[:2]))
            # Approximate the live downward cone, including spherical surfaces.
            cone = max(0.0, -float(relative[2])) * math.tan(math.radians(15))
            if relative[2] < 0.0 and horizontal <= obstacle.radius + cone:
                nearest = min(nearest, max(0.0, float(np.linalg.norm(relative)) - obstacle.radius))
        return nearest

    def _clearance(self):
        return min((float(np.linalg.norm(o.position - self.position)) - o.radius - self.agent_radius
                    for o in self.obstacles), default=math.inf)

    def _observation(self):
        return build_observation(tuple(self.scan_history), self._world_to_body(self.goal - self.position),
                                 self._world_to_body(self.velocity), float(self.position[2]), self._down_clearance())

    def _reward_context(self):
        """Training-only geometry for reward; never supplies policy inputs/actions."""
        leg_start = self.outbound_goal if self.outbound_complete else self.start
        route = self.goal - leg_start
        direction = route / max(float(np.linalg.norm(route)), 1e-6)
        newly_passed = 0
        for index, obstacle in enumerate(self.obstacles):
            relative = obstacle.position - self.position
            along = float(relative @ direction)
            pass_margin = obstacle.radius + self.agent_radius + .75
            if along < -pass_margin and index not in self.passed_obstacles:
                self.passed_obstacles.add(index)
                newly_passed += 1
            if index not in self.passed_obstacles:
                clearance = float(np.linalg.norm(relative)) - obstacle.radius - self.agent_radius
                if clearance < 4.0:
                    self.encountered_obstacles.add(index)
        active = bool(self.encountered_obstacles - self.passed_obstacles)
        mode = "avoidance" if active else "recovery" if self.passed_obstacles else "tracking"
        return mode, newly_passed

    def _reward_v3(self, action, progress, clearance, old_clearance, distance):
        mode, newly_passed = self._reward_context()
        goal_direction = self._world_to_body(self.goal - self.position)
        goal_direction /= max(float(np.linalg.norm(goal_direction)), 1e-6)
        alignment = float(action @ goal_direction)
        reward = -.025 - .015 * float(action @ action)
        reward -= .02 * float(np.linalg.norm(action - self.previous_action))
        if mode == "avoidance":
            reward += 4.0 * progress + .05 * alignment
            reward += 1.5 * float(np.clip(clearance - old_clearance, -.5, .5))
            # Delay restoring goal altitude until the obstacle is safely behind.
            if clearance < 2.5:
                reward -= 1.2 * (2.5 - max(clearance, 0.0)) ** 2
            if self.last_ttc < 3.0:
                reward -= .8 * (3.0 - self.last_ttc) ** 2
        else:
            reward += (18.0 if mode == "recovery" else 12.0) * progress + .6 * alignment
            reward -= .08 * abs(float(self.goal[2] - self.position[2]))
            reward -= .04 * distance
            if distance > self.goal_radius and progress < .002:
                self.stagnation_steps += 1
            else:
                self.stagnation_steps = 0
            reward -= min(.8, .025 * self.stagnation_steps)
        if distance > self.goal_radius and float(np.linalg.norm(self.velocity)) < .2:
            reward -= .1
        reward += 20.0 * newly_passed
        return reward, mode

    def step(self, action):
        command_body = action_to_velocity(action, self.velocity_mapping)
        clipped = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.queue.append(command_body)
        delayed_body = self.queue.popleft()
        guard_flags = ()
        braked = False
        if self.direction_brake:
            delayed_body, braked = project_directional_velocity(delayed_body, self.scan_history[-1])
            if braked:
                guard_flags = ("direction",)
        if self.safety_guards:
            delayed_body, vertical_flags = project_safe_velocity(
                delayed_body, float(self.position[2]), self._down_clearance())
            guard_flags += vertical_flags
        target = self._body_to_world(delayed_body)
        acceleration = (target - self.velocity) / self.response_time + self.wind
        norm = float(np.linalg.norm(acceleration))
        if norm > self.maximum_acceleration:
            acceleration *= self.maximum_acceleration / norm
        old_position = self.position.copy()
        old_clearance = self._clearance()
        old_obstacles = [o.position.copy() for o in self.obstacles]
        self.velocity += acceleration * self.dt
        self.position += self.velocity * self.dt
        self.step_count += 1
        self._place_obstacles()
        # Swept relative segments catch collisions between simulation ticks.
        swept_clearance = math.inf
        for old_obstacle, obstacle in zip(old_obstacles, self.obstacles):
            relative = old_obstacle - old_position
            delta = (obstacle.position - self.position) - relative
            fraction = float(np.clip(-(relative @ delta) / max(float(delta @ delta), 1e-12), 0.0, 1.0))
            swept_clearance = min(swept_clearance, float(np.linalg.norm(relative + fraction * delta)) - obstacle.radius - self.agent_radius)
        self.scan_history.append(self._scan())
        distance = self._goal_distance()
        progress = self.previous_goal_distance - distance
        self.previous_goal_distance = distance
        clearance = min(self._clearance(), swept_clearance)
        self.last_ttc = self._minimum_ttc()
        outside = bool(np.any(self.position < self.corridor_min) or np.any(self.position > self.corridor_max))
        collision = clearance <= 0.0
        reached = distance <= self.goal_radius
        success = reached and (self.outbound_complete or not self.round_trip) and not collision and not outside
        reward = 12.0 * progress - .025 - .015 * float(clipped @ clipped)
        reward -= .02 * float(np.linalg.norm(clipped - self.previous_action))
        direction = self._world_to_body(self.goal - self.position)
        direction /= max(float(np.linalg.norm(direction)), 1e-6)
        reward += .6 * float(clipped @ direction)
        # A persistent height error must not become a stationary local optimum.
        # This remains a reward term; it never supplies a velocity command.
        reward -= .08 * abs(float(self.goal[2] - self.position[2]))
        if distance > self.goal_radius and float(np.linalg.norm(self.velocity)) < .2:
            reward -= .1
        if clearance < 1.5:
            reward -= .6 * (1.5 - max(clearance, 0.0)) ** 2
        if self.last_ttc < 2.0:
            reward -= .3 * (2.0 - self.last_ttc) ** 2
        reward_mode = "original"
        if self.reward_version in (3, 4):
            reward, reward_mode = self._reward_v3(clipped, progress, clearance, old_clearance, distance)
        if self.reward_version == 4:
            deviation = self._route_deviation()
            if reward_mode == "avoidance":
                # Allow a necessary local detour; discourage a generic large
                # excursion that bypasses every scene without observing it.
                reward -= .3 * max(0., deviation - 1.) ** 2
            else:
                reward -= .35 * deviation ** 2
                reward -= .15 * abs(float(self.goal[2] - self.position[2]))
        altitude_margin = min(float(self.position[2]) - ALTITUDE_MIN, ALTITUDE_MAX - float(self.position[2]))
        if altitude_margin < 1.0:
            reward -= (1.0 - max(altitude_margin, 0.0)) ** 2
        if reached and self.round_trip and not self.outbound_complete:
            self.outbound_complete = True
            self.goal = self.start.copy()
            self.passed_obstacles.clear()
            self.encountered_obstacles.clear()
            self.stagnation_steps = 0
            self.previous_goal_distance = self._goal_distance()
            reward += 40.0
        terminated = collision or outside or success
        truncated = self.step_count >= self.max_steps and not terminated
        if collision:
            reward -= 250.0
        elif outside:
            reward -= 150.0
        elif success:
            reward += 120.0
        elif truncated:
            reward -= 50.0
        self.previous_action = clipped.copy()
        info = self._info(collision, success, outside)
        info["minimum_clearance"] = clearance
        info["reward_mode"] = reward_mode
        info["safety_guarded"] = bool(guard_flags)
        info["safety_guard_flags"] = guard_flags
        info["direction_braked"] = braked
        return self._observation(), float(reward), terminated, truncated, info
