import numpy as np
from gymnasium.utils.env_checker import check_env

from logistics_planner.curriculum_stage2_env import CurriculumStage2Env


def test_stage2_environment_conforms_to_gymnasium_api() -> None:
    check_env(CurriculumStage2Env(max_steps=20), skip_render_check=True)


def test_stage2_has_one_static_obstacle_blocking_the_route() -> None:
    env = CurriculumStage2Env()
    env.reset(seed=17)
    assert len(env.obstacles) == 1
    obstacle = env.obstacles[0]
    assert np.all(obstacle.velocity == 0.0)
    route = env.goal - env.start
    along_fraction = float((obstacle.position - env.start) @ route / (route @ route))
    assert 0.38 <= along_fraction <= 0.62
    assert 0.75 <= obstacle.radius <= 1.15
    assert np.min(env.scan_history[-1]) < env.sensor_range


def test_stage2_reports_collision_when_agent_enters_obstacle() -> None:
    env = CurriculumStage2Env()
    env.reset(seed=23)
    env.position = env.obstacles[0].position.copy()
    assert env._clearance() < 0.0


def test_post_obstacle_reward_prefers_motion_toward_goal() -> None:
    toward_env = CurriculumStage2Env()
    away_env = CurriculumStage2Env()
    toward_env.reset(seed=31)
    away_env.reset(seed=31)
    for env in (toward_env, away_env):
        route = env.goal - env.start
        route_direction = route / np.linalg.norm(route)
        env.position = env.obstacles[0].position + route_direction * 2.0
        env.velocity.fill(0.0)
        env.previous_goal_distance = env._goal_distance()
        env.obstacle_passed = True
    goal_body = toward_env._world_to_body(toward_env.goal - toward_env.position)[:2]
    toward_action = goal_body / np.linalg.norm(goal_body)
    _, toward_reward, _, _, _ = toward_env.step(toward_action)
    _, away_reward, _, _, _ = away_env.step(-toward_action)
    assert toward_reward > away_reward


def test_avoidance_hysteresis_requires_sustained_safe_ticks() -> None:
    env = CurriculumStage2Env()
    env.reset(seed=41)
    route = env.goal - env.start
    route_direction = route / np.linalg.norm(route)
    env.position = env.obstacles[0].position - route_direction * 2.0
    env.step(np.zeros(2, dtype=np.float32))
    assert env.avoidance_active

    env.position = env.obstacles[0].position + route_direction * 4.0
    env.velocity.fill(0.0)
    env.previous_goal_distance = env._goal_distance()
    for _ in range(4):
        _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
        assert info["guidance_mode"] == "avoidance"
    _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert info["guidance_mode"] == "recovery"
