import numpy as np
from gymnasium.utils.env_checker import check_env

from logistics_planner.rl_avoidance_env import (
    MovingSphere,
    RAY_DIRECTIONS,
    UAVLocalAvoidanceEnv,
    ray_sphere_distances,
)


def test_environment_conforms_to_gymnasium_api() -> None:
    check_env(UAVLocalAvoidanceEnv(max_steps=20), skip_render_check=True)


def test_reset_is_reproducible_and_observation_is_normalized() -> None:
    env = UAVLocalAvoidanceEnv()
    first, _ = env.reset(seed=7)
    first_obstacles = np.array([obstacle.position for obstacle in env.obstacles])
    second, _ = env.reset(seed=7)
    second_obstacles = np.array([obstacle.position for obstacle in env.obstacles])
    assert np.array_equal(first, second)
    assert np.array_equal(first_obstacles, second_obstacles)
    assert env.observation_space.contains(first)


def test_forward_ray_detects_sphere_surface() -> None:
    sphere = MovingSphere(
        position=np.array([5.0, 0.0, 0.0], dtype=np.float32),
        velocity=np.zeros(3, dtype=np.float32),
        radius=1.0,
    )
    distances = ray_sphere_distances(
        np.zeros(3, dtype=np.float32), RAY_DIRECTIONS, [sphere], 20.0
    )
    assert distances[12] == np.float32(4.0)
    assert distances[18] == np.float32(20.0)


def test_zero_action_uses_nominal_goal_controller() -> None:
    env = UAVLocalAvoidanceEnv(obstacle_count=1)
    env.reset(seed=2)
    old_x = float(env.position[0])
    observation, reward, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
    assert float(env.position[0]) > old_x
    assert env.observation_space.contains(observation)
    assert np.isfinite(reward)
    assert not truncated
    assert info["goal_distance"] < 24.0
