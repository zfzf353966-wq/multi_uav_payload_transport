"""City-corridor PPO training: original buildings/trees and orbiting spheres.

Geometry affects observations, reward and collision checks, never an avoidance
command. Tree trunks use conservative bounding boxes; crowns are exact spheres.
"""
from collections import deque
from pathlib import Path
import math
import gymnasium as gym
import numpy as np
import yaml

from .direct_ppo_env import DirectPPOEnv
from .gazebo_lidar_model import RAYS, DOWN_RAYS, SECTORS, rotate_rays, cast_ranges
from .rl_avoidance_env import MovingSphere
from .obstacle_scenario import scenario_parameters


def box_ranges(origin, rays, boxes, maximum=20., minimum=.6):
    lower, upper = boxes[:, 0], boxes[:, 1]
    parallel = np.abs(rays) < 1e-8
    inverse = np.divide(1., rays, out=np.zeros_like(rays), where=~parallel)
    a = (lower[None] - origin) * inverse[:, None]
    b = (upper[None] - origin) * inverse[:, None]
    near = np.where(parallel[:, None], -np.inf, np.minimum(a, b)).max(axis=2)
    far = np.where(parallel[:, None], np.inf, np.maximum(a, b)).min(axis=2)
    aligned = np.all(~parallel[:, None] | ((origin >= lower) & (origin <= upper))[None], axis=2)
    hit = np.where(near >= minimum, near, far)
    valid = aligned & (far >= near) & (hit >= minimum) & (hit <= maximum)
    return np.where(valid, hit, maximum).min(axis=1).astype(np.float32)


def box_clearance(position, boxes):
    centre = boxes.mean(axis=1)
    half = (boxes[:, 1] - boxes[:, 0]) / 2
    q = np.abs(np.asarray(position) - centre) - half
    signed = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)
    return float(signed.min())


class CityPPOEnv(DirectPPOEnv):
    def __init__(self, *, dynamic=True, randomize=True, round_trip=True, scenario=None,
                 max_steps=1200, runtime_selection=True, static_probability=0., reward_revision=1):
        super().__init__('multi', max_steps=max_steps, randomize=randomize,
                         round_trip=round_trip, reward_version=3, safety_guards=True,
                         direction_brake=True, history_warmup=True, lidar_profile='gazebo')
        self.dynamic = dynamic
        if not 0 <= static_probability <= 1 or reward_revision not in (1, 2):
            raise ValueError('invalid city mixture or reward revision')
        self.static_probability = static_probability
        self.reward_revision = reward_revision
        self.scenario = scenario
        self.runtime_selection = runtime_selection
        config = yaml.safe_load((Path(__file__).resolve().parents[3] / 'config/logistics_city.yaml').read_text())
        buildings = [(np.array(o['center']), np.array(o['size']) / 2) for o in config['static_obstacles']]
        trunks = [(np.array([x, y, z + 1.5]), np.array([.28, .28, 1.5])) for x, y, z in config['tree_positions']]
        self.boxes = np.array([[c - h, c + h] for c, h in buildings + trunks], dtype=np.float32)
        self.tree_centres = [np.array([x, y, z + 4], dtype=np.float32) for x, y, z in config['tree_positions']]
        self.corridor_min = np.array([-39., -29., 2.], dtype=np.float32)
        self.corridor_max = np.array([0., -16., 10.], dtype=np.float32)
        self.sweep_start = None

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        self.start = np.array([-34., -23., 6.], dtype=np.float32)
        self.outbound_goal = np.array([-6., -23., 6.], dtype=np.float32)
        self.position = self.start.copy()
        if self.randomize:
            self.position += self.np_random.uniform([-.2, -.3, -.4], [.2, .3, .4]).astype(np.float32)
        self.goal = self.outbound_goal.copy()
        self.heading = float(self.np_random.uniform(-.15, .15)) if self.randomize else 0.
        self.velocity = np.zeros(3, dtype=np.float32)
        self.previous_action = np.zeros(3, dtype=np.float32)
        self.step_count = 0
        self.outbound_complete = False
        self.passed_obstacles.clear()
        self.encountered_obstacles.clear()
        self.stagnation_steps = 0
        self.last_ttc = math.inf
        self.sweep_start = None
        self.obstacles = [MovingSphere(c.copy(), np.zeros(3, dtype=np.float32), 1.55) for c in self.tree_centres]
        self.orbits = [(c.copy(), 0., 0., 0.) for c in self.tree_centres]
        with_dynamic = self.dynamic and (not self.static_probability or self.np_random.random() >= self.static_probability)
        if with_dynamic:
            scenario = self.scenario if self.scenario is not None else int(self.np_random.integers(100000))
            offset = float(self.np_random.uniform(0, 20)) if self.randomize else 0.
            for item, radius in zip(scenario_parameters(scenario), [.75, .7, .65]):
                c = np.array([-20 + item['centre_x'], -23 + item['centre_y'], 6.], dtype=np.float32)
                self.obstacles.append(MovingSphere(c.copy(), np.zeros(3, dtype=np.float32), radius))
                self.orbits.append((c, item['radius'], item['phase'] + offset * item['speed'], item['speed']))
            c = np.array([10., 0., 8.], dtype=np.float32)
            self.obstacles.append(MovingSphere(c.copy(), np.zeros(3, dtype=np.float32), .45))
            self.orbits.append((c, 4., offset * .5, .5))
        self.response_time = float(self.np_random.uniform(.3, .55)) if self.randomize else .35
        self.maximum_acceleration = 3.
        self.delay = int(self.np_random.integers(0, 3)) if self.randomize else 0
        self.wind = np.zeros(3, dtype=np.float32)
        self.sensor_noise = 0.
        self.queue = deque(np.zeros(3, dtype=np.float32) for _ in range(self.delay))
        self.local_active = False
        self.local_clear_ticks = 0
        self.local_started = 0
        self.leg_steps = 0
        self.low_scan_streak = 0
        self.scan_history.clear()
        for index in range(4):
            self._place_obstacles((index - 3) * self.dt)
            self.scan_history.append(self._scan())
        self.previous_goal_distance = self._goal_distance()
        return self._observation(), self._info(False, False, False)

    def _ranges(self, rays, offset, maximum, minimum):
        origin = self.position + np.array([0., 0., offset], dtype=np.float32)
        world_rays = rotate_rays(rays, self.heading)
        return np.minimum(cast_ranges(origin, world_rays, self.obstacles, maximum, minimum),
                          box_ranges(origin, world_rays, self.boxes, maximum, minimum))

    def _scan(self):
        ranges = self._ranges(RAYS, .6, 20., .6).reshape(3, 120)
        sectors = np.full((3, 12), 20., dtype=np.float32)
        for layer in range(3):
            np.minimum.at(sectors[layer], SECTORS, ranges[layer])
        return sectors.reshape(-1)

    def _down_clearance(self):
        return float(self._ranges(DOWN_RAYS, -.4, 15., .3).min())

    def _clearance(self):
        positions = [self.position] if self.sweep_start is None else np.linspace(self.sweep_start, self.position, 5)
        return min(super()._clearance(), min(box_clearance(p, self.boxes) for p in positions) - self.agent_radius)

    def _reward_v3(self, action, progress, clearance, old_clearance, distance):
        # A stalled zero command must be worse than learning a safe detour.
        reward = 15. * progress - .04 - .025 * distance
        height_weight = .12 if self.reward_revision == 2 and clearance < 2. else .65
        reward -= height_weight * float(self.position[2] - 6.) ** 2
        reward -= .035 * float(self.position[1] + 23.) ** 2
        reward -= .01 * float(np.linalg.norm(action - self.previous_action))
        reward -= 3. * max(0., 1.2 - clearance) ** 2
        reward -= 4. * max(0., 1.1 - float(self.scan_history[-1].min())) ** 2
        if self.reward_revision == 2:
            reward -= .5 * max(0., 2. - self.last_ttc) ** 2
        if self._goal_distance() > 1 and float(np.linalg.norm(self.velocity)) < .15:
            reward -= .4
        return reward, 'city'

    def step(self, action):
        from .single_uav_avoidance_test import update_local_activation, desired_velocity_enu
        was_active = self.local_active
        self.local_active, self.local_clear_ticks = update_local_activation(
            self.local_active, self.local_clear_ticks, float(self.scan_history[-1].min()),
            (self.step_count - self.local_started) * self.dt)
        if self.local_active and not was_active:
            self.local_started = self.step_count
        approach = self.outbound_complete and np.linalg.norm(self.position - self.start) <= 5.
        controlled_by_ppo = self.local_active and not approach or not self.runtime_selection
        if not controlled_by_ppo:
            world = np.array(desired_velocity_enu(self.position, self.goal), dtype=np.float32)
            action = self._world_to_body(world) / np.array([2., 2., 1.], dtype=np.float32)
        self.sweep_start = self.position.copy()
        old_leg = self.outbound_complete
        obs, reward, terminated, truncated, info = super().step(action)
        self.sweep_start = None
        self.leg_steps = 0 if self.outbound_complete != old_leg else self.leg_steps + 1
        if controlled_by_ppo and info['direction_braked']:
            reward -= .6
        scan_bad = float(self.scan_history[-1].min()) < .8
        self.low_scan_streak = getattr(self, 'low_scan_streak', 0) + 1 if scan_bad else 0
        if not terminated and (self.low_scan_streak >= 2 or self.leg_steps >= 600):
            truncated = True
            reward -= 100.
        if self.reward_revision == 2:
            if info['collision']:
                reward -= 750.
            elif self.low_scan_streak >= 2:
                if info['success']:
                    reward -= 120.
                info['success'] = False
                terminated, truncated = False, True
                reward -= 400.
        info.update(ppo_control=controlled_by_ppo, scan_margin_violation=scan_bad)
        return obs, reward, terminated, truncated, info
