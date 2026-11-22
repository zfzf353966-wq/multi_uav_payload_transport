import numpy as np
from gymnasium.utils.env_checker import check_env

from logistics_planner.pure_ppo_avoidance_env import PurePPOLocalAvoidanceEnv


def test_pure_ppo_environment_conforms_to_gymnasium_api() -> None:
    check_env(PurePPOLocalAvoidanceEnv(max_steps=20), skip_render_check=True)


def test_action_directly_commands_horizontal_velocity_without_potential_field() -> None:
    env = PurePPOLocalAvoidanceEnv(obstacle_count=1)
    env.reset(seed=5)
    env.obstacles[0].position = np.array([0.0, 14.0, 4.0], dtype=np.float32)
    env.obstacles[0].velocity.fill(0.0)
    env.control_delay_steps = 0
    env.command_queue.clear()
    env.wind_acceleration.fill(0.0)
    initial_velocity = env.velocity.copy()
    env.step(np.array([1.0, 0.0], dtype=np.float32))
    assert np.linalg.norm(env.velocity[:2]) > np.linalg.norm(initial_velocity[:2])
    assert env.velocity[2] == 0.0


def test_reset_randomizes_bounded_px4_approximations_reproducibly() -> None:
    env = PurePPOLocalAvoidanceEnv()
    first_observation, _ = env.reset(seed=9)
    first_dynamics = env.dynamics_info()
    second_observation, _ = env.reset(seed=9)
    assert np.array_equal(first_observation, second_observation)
    assert first_dynamics == env.dynamics_info()
    assert 0 <= first_dynamics["control_delay_steps"] <= 3
