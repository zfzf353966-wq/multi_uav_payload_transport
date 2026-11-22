import numpy as np
from gymnasium.utils.env_checker import check_env

from logistics_planner.curriculum_stage1_env import CurriculumStage1Env


def test_stage1_environment_conforms_to_gymnasium_api() -> None:
    check_env(CurriculumStage1Env(max_steps=20), skip_render_check=True)


def test_stage1_contains_no_obstacles_or_disturbances() -> None:
    env = CurriculumStage1Env()
    observation, _ = env.reset(seed=7)
    assert env.obstacles == []
    assert env.control_delay_steps == 0
    assert env.sensor_noise_std == 0.0
    assert np.all(env.wind_acceleration == 0.0)
    assert np.all(observation[: 36 * 4] == 1.0)


def test_forward_action_reduces_distance_when_goal_is_ahead() -> None:
    env = CurriculumStage1Env()
    env.reset(seed=11)
    env.heading = float(np.arctan2(env.goal[1], env.goal[0]))
    initial_distance = env._goal_distance()
    for _ in range(10):
        env.step(np.array([1.0, 0.0], dtype=np.float32))
    assert env._goal_distance() < initial_distance
