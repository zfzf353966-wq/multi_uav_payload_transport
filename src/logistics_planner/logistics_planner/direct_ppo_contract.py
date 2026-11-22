"""Shared training/live contract for direct body-frame FLU velocity PPO."""
from __future__ import annotations

import numpy as np
import math

MAX_SPEED = 2.0
MAX_VERTICAL_SPEED = 1.0
ALTITUDE_MIN = 2.0
ALTITUDE_MAX = 10.0
OBSERVATION_SIZE = 153


def sector_directional_clearance(sectors, velocity_body):
    if len(sectors) != 36:
        return 0.0
    forward, left, up = map(float, velocity_body)
    if math.sqrt(forward * forward + left * left + up * up) < .05:
        return math.inf
    sector = int(round(math.atan2(left, forward) / (2 * math.pi / 12))) % 12
    elevation = math.atan2(up, math.hypot(forward, left))
    layer = min(range(3), key=lambda index: abs(elevation - math.radians((-12., 0., 12.)[index])))
    candidates = [float(sectors[row * 12 + (sector + offset) % 12])
                  for row in range(max(0, layer - 1), min(3, layer + 2))
                  for offset in (-1, 0, 1)
                  if math.isfinite(sectors[row * 12 + (sector + offset) % 12])]
    return min(candidates, default=0.0)


def project_directional_velocity(velocity_body, sectors):
    values = tuple(float(v) for v in velocity_body)
    norm = math.sqrt(sum(v * v for v in values))
    if norm > MAX_SPEED:
        values = tuple(v * MAX_SPEED / norm for v in values)
    candidate = (values[0], values[1], max(-MAX_VERTICAL_SPEED, min(MAX_VERTICAL_SPEED, values[2])))
    speed = math.sqrt(sum(v * v for v in candidate))
    if speed > .05 and sector_directional_clearance(sectors, candidate) < .8 + .5 * speed + speed * speed / 4:
        return (0., 0., 0.), True
    return candidate, False


def project_safe_velocity(velocity_body, altitude, down_clearance, *, altitude_min=ALTITUDE_MIN, altitude_max=ALTITUDE_MAX):
    """Shared altitude/downward projection; returns command and intervention flags."""
    velocity = np.asarray(velocity_body, dtype=np.float32).copy()
    if velocity.shape != (3,) or not np.isfinite(velocity).all():
        raise ValueError("velocity must contain three finite values")
    if not np.isfinite(altitude) or not np.isfinite(down_clearance):
        raise ValueError("safety projection requires finite altitude and downward clearance")
    guarded = []
    if velocity[2] < 0 and down_clearance < 2.0:
        velocity[2] = 0.0
        guarded.append("downward")
    lower = (altitude_min + 1.0 - altitude) / 1.5
    upper = (altitude_max - 1.0 - altitude) / 1.5
    projected = float(np.clip(velocity[2], lower, upper))
    if abs(projected - float(velocity[2])) > 1e-6:
        velocity[2] = projected
        guarded.append("altitude")
    return velocity, tuple(guarded)


def action_to_velocity(action, mapping="radial") -> np.ndarray:
    values = np.asarray(action, dtype=np.float32)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("direct PPO action must contain three finite FLU values")
    velocity = np.clip(values, -1.0, 1.0) * np.array(
        [MAX_SPEED, MAX_SPEED, MAX_VERTICAL_SPEED], dtype=np.float32
    )
    if mapping == "vertical_priority":
        horizontal = float(np.linalg.norm(velocity[:2]))
        budget = float(np.sqrt(max(0.0, MAX_SPEED ** 2 - float(velocity[2]) ** 2)))
        if horizontal > budget:
            velocity[:2] *= budget / horizontal
        return velocity
    if mapping != "radial":
        raise ValueError(f"unknown velocity mapping: {mapping}")
    speed = float(np.linalg.norm(velocity))
    if speed > MAX_SPEED:
        velocity *= MAX_SPEED / speed
    return velocity


def build_observation(frames, goal_body, velocity_body, altitude, down_clearance, *, altitude_min=ALTITUDE_MIN, altitude_max=ALTITUDE_MAX):
    scan = np.asarray(frames, dtype=np.float32)
    goal = np.asarray(goal_body, dtype=np.float32)
    velocity = np.asarray(velocity_body, dtype=np.float32)
    if scan.shape != (4, 36) or goal.shape != (3,) or velocity.shape != (3,):
        raise ValueError("expected four 36-sector frames and three-vector goal/velocity")
    if not all(np.isfinite(v).all() for v in (scan, goal, velocity)):
        raise ValueError("observation inputs must be finite")
    if not np.isfinite(altitude) or not np.isfinite(down_clearance):
        raise ValueError("altitude and downward clearance must be finite")
    height = altitude_max - altitude_min
    if height <= 0:
        raise ValueError('altitude limits must be ordered')
    return np.concatenate((
        np.clip(scan.reshape(-1) / 20.0, 0.0, 1.0),
        np.clip(goal / 25.0, -1.0, 1.0),
        np.clip(velocity / MAX_SPEED, -1.0, 1.0),
        np.clip([(altitude - altitude_min) / height,
                 (altitude_max - altitude) / height, down_clearance / 20.0], 0.0, 1.0),
    )).astype(np.float32)
