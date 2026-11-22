import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from logistics_planner.direct_ppo_contract import action_to_velocity, build_observation
from logistics_planner.direct_ppo_env import DirectPPOEnv, STAGES


@pytest.mark.parametrize("stage", STAGES)
def test_environment_contract(stage):
    check_env(DirectPPOEnv(stage, max_steps=20), skip_render_check=True)


def test_vertical_action_changes_height_without_potential_field(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("potential field called")
    monkeypatch.setattr("logistics_planner.rl_avoidance_env.compute_shadow_avoidance", fail)
    env = DirectPPOEnv()
    env.reset(seed=5)
    height = float(env.position[2])
    for _ in range(5):
        env.step([0.0, 0.0, 1.0])
    assert env.position[2] > height
    assert env.velocity[2] > 0.0


def test_live_and_training_observation_are_identical():
    env = DirectPPOEnv("dynamic", randomize=True)
    observation, _ = env.reset(seed=11)
    live = build_observation(tuple(env.scan_history), env._world_to_body(env.goal - env.position),
                             env._world_to_body(env.velocity), env.position[2], env._down_clearance())
    np.testing.assert_array_equal(observation, live)
    assert observation.shape == (153,)


def test_speed_is_bounded_and_invalid_actions_are_rejected():
    velocity = action_to_velocity([1, 1, 1])
    assert np.linalg.norm(velocity) <= 2.000001
    assert 0 < velocity[2] <= 1.0
    for action in ([0, 0], [0, 0, np.nan], [0, 0, np.inf]):
        with pytest.raises(ValueError):
            action_to_velocity(action)


def test_dynamic_spheres_move_and_collision_is_terminal():
    env = DirectPPOEnv("dynamic")
    env.reset(seed=8)
    before = env.obstacles[0].position.copy()
    env.step([0, 0, 0])
    assert not np.array_equal(before, env.obstacles[0].position)
    env.position = env.obstacles[0].position.copy()
    _, _, terminated, _, info = env.step([0, 0, 0])
    assert terminated and info["collision"]


def test_altitude_violation_is_terminal():
    env = DirectPPOEnv()
    env.reset(seed=10)
    env.position[2] = 10.1
    _, _, terminated, _, info = env.step([0, 0, 1])
    assert terminated and info["outside"]


def test_direct_arbiter_preserves_vertical_velocity_and_brakes_on_stale_data():
    from logistics_planner.single_uav_avoidance_test import direct_velocity_decision
    clear = [20.0] * 36
    result = direct_velocity_decision((.4, .2, .8), .1, clear, .1)
    assert result.source == "rl"
    assert result.velocity_body == pytest.approx((.4, .2, .8))
    for velocity, age, scan_age in ((None, .1, .1), ((1, 0, 1), .6, .1), ((1, 0, 1), .1, .6)):
        result = direct_velocity_decision(velocity, age, clear, scan_age)
        assert result.source == "fallback"
        assert result.velocity_body == (0, 0, 0)


def test_direct_arbiter_brakes_instead_of_using_potential_guidance():
    from logistics_planner.single_uav_avoidance_test import direct_velocity_decision
    result = direct_velocity_decision((2, 0, 0), .1, [1.0] * 36, .1)
    assert result.source == "fallback"
    assert result.reason == "safety_brake"
    assert result.velocity_body == (0, 0, 0)


def test_px4_direct_control_does_not_replace_policy_vertical_velocity():
    import time
    from types import SimpleNamespace
    from logistics_planner.single_uav_avoidance_test import SingleUAVAvoidanceTest
    commands = []
    node = SimpleNamespace(
        phase="outbound", phase_started=time.monotonic(),
        _distance_to=lambda target: 28.0,
        sectors_time=time.monotonic(), down_clearance_time=time.monotonic(),
        minimum_down_clearance=float("inf"), down_clearance=6.0,
        rl_suggested_body=(.4, .2, .8), rl_suggestion_time=time.monotonic(),
        latest_sectors=(20.0,) * 36, downward_guard_ticks=0, geofence_guard_ticks=0,
        rl_control_ticks=0, approach_ticks=0, fallback_ticks=0, avoidance_ticks=0,
        contact_risk_ticks=0, maximum_contact_risk_ticks=0,
        maximum_lateral_deviation=0.0, maximum_vertical_deviation=0.0,
        arbiter_publisher=SimpleNamespace(publish=lambda message: None),
        _record_control_sample=lambda **kwargs: None,
        _publish_velocity_target=commands.append,
        _abort=lambda reason: pytest.fail(reason),
    )
    SingleUAVAvoidanceTest._direct_control_step(node, (-34, -23, 6), (1.4, 0, 0), 0.0)
    assert commands == [pytest.approx((.4, -.2, -.8))]
    assert node.rl_control_ticks == 1


def test_single_obstacle_scenario_has_no_dangling_joints(tmp_path):
    import xml.etree.ElementTree as ET
    from pathlib import Path
    from logistics_planner.obstacle_scenario import generate_scenario
    root = Path(__file__).resolve().parents[3]
    output = tmp_path / "single.sdf"
    generate_scenario(root / "simulation/tests/avoidance_test_complex_moving_spheres.sdf", output, 1, count=1)
    model = ET.parse(output).getroot().find("model")
    assert [link.get("name") for link in model.findall("link") if link.get("name", "").endswith("_sphere")] == ["orange_sphere"]
    names = {link.get("name") for link in model.findall("link")}
    for joint in model.findall("joint"):
        assert joint.findtext("child") in names
