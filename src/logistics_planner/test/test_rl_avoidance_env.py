import numpy as np
from gymnasium.utils.env_checker import check_env

from logistics_planner.rl_avoidance_env import (
    FRAME_STACK,
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
    old_distance = float(np.linalg.norm(env.goal - env.position))
    observation, reward, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
    assert float(np.linalg.norm(env.goal - env.position)) < old_distance
    assert env.observation_space.contains(observation)
    assert np.isfinite(reward)
    assert not truncated
    assert info["goal_distance"] < old_distance


def test_random_heading_can_place_goal_behind_vehicle() -> None:
    env = UAVLocalAvoidanceEnv()
    found_behind = False
    for seed in range(20):
        observation, _ = env.reset(seed=seed)
        if observation[36 * FRAME_STACK] < 0.0:
            found_behind = True
            break
    assert found_behind


def test_outbound_waypoint_switches_goal_back_to_start() -> None:
    env = UAVLocalAvoidanceEnv()
    env.reset(seed=3)
    env.position = env.outbound_goal.copy()
    env.velocity.fill(0.0)
    # Keep obstacles far from the forced waypoint transition.
    for obstacle in env.obstacles:
        obstacle.position = np.array([0.0, 14.0, 4.0], dtype=np.float32)
        obstacle.velocity.fill(0.0)
    _, _, terminated, _, info = env.step(np.zeros(3, dtype=np.float32))
    assert not terminated
    assert info["outbound_complete"]
    assert np.array_equal(env.goal, env.start)


def test_four_scan_frames_are_present_and_advance() -> None:
    env = UAVLocalAvoidanceEnv(obstacle_count=1)
    observation, _ = env.reset(seed=4)
    assert observation.shape == (36 * FRAME_STACK + 6,)
    assert np.array_equal(observation[:36], observation[108:144])
    env.obstacles[0].velocity = np.array([0.0, 2.0, 0.0], dtype=np.float32)
    next_observation, *_ = env.step(np.zeros(3, dtype=np.float32))
    assert np.array_equal(next_observation[:36], observation[36:72])


def test_ttc_detects_closing_obstacle_and_ignores_receding_one() -> None:
    env = UAVLocalAvoidanceEnv(obstacle_count=1)
    env.reset(seed=1)
    env.position.fill(0.0)
    env.velocity = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    env.obstacles[0] = MovingSphere(
        position=np.array([5.0, 0.0, 0.0], dtype=np.float32),
        velocity=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        radius=0.55,
    )
    assert env._minimum_ttc() == np.float32(2.0)
    env.obstacles[0].velocity = np.array([2.0, 0.0, 0.0], dtype=np.float32)
    assert np.isinf(env._minimum_ttc())


def test_boundary_margin_becomes_small_before_geofence_exit() -> None:
    env = UAVLocalAvoidanceEnv()
    env.reset(seed=1)
    env.position = env.corridor_max - np.array([0.5, 1.0, 2.0], dtype=np.float32)
    assert env._boundary_margin() == np.float32(0.5)
