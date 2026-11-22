import numpy as np
import pytest
import torch
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import PPO

from logistics_planner.direct_ppo_env import DirectPPOEnv
from logistics_planner.direct_ppo_features import DirectPPOFeatures
from logistics_planner.direct_ppo_policy import BoundedMeanPolicy


def test_reward_v3_conforms_to_gym_and_has_no_oracle_observation():
    env = DirectPPOEnv("static", reward_version=3, max_steps=20)
    check_env(env, skip_render_check=True)
    original = DirectPPOEnv("static", reward_version=2)
    changed = DirectPPOEnv("static", reward_version=3)
    first, _ = original.reset(seed=9)
    second, _ = changed.reset(seed=9)
    np.testing.assert_array_equal(first, second)
    for _ in range(5):
        first, _, d1, t1, _ = original.step([.2, -.1, .3])
        second, _, d2, t2, _ = changed.step([.2, -.1, .3])
        np.testing.assert_array_equal(first, second)
        assert (d1, t1) == (d2, t2)


def test_avoidance_reward_remains_active_until_safely_past_obstacle():
    env = DirectPPOEnv("static", reward_version=3)
    env.reset(seed=13)
    direction = env.goal - env.start
    direction /= np.linalg.norm(direction)
    obstacle = env.obstacles[0]
    env.position = obstacle.position - 2.0 * direction
    assert env._reward_context()[0] == "avoidance"
    # Losing proximity alone must not restore target-height tracking.
    env.position[2] += 6.0
    assert env._reward_context()[0] == "avoidance"
    env.position = obstacle.position + 3.0 * direction
    assert env._reward_context() == ("recovery", 1)
    assert env._reward_context() == ("recovery", 0)


def test_near_obstacle_reward_prefers_escaping_to_approaching():
    rewards = []
    for sign in (1, -1):
        env = DirectPPOEnv("static", reward_version=3)
        env.reset(seed=21)
        direction = env.goal - env.start
        direction /= np.linalg.norm(direction)
        obstacle = env.obstacles[0]
        env.position = obstacle.position - direction * (obstacle.radius + env.agent_radius + .3)
        env.velocity.fill(0)
        env.previous_goal_distance = env._goal_distance()
        _, reward, _, _, info = env.step(sign * env._world_to_body(direction))
        assert info["reward_mode"] == "avoidance"
        rewards.append(reward)
    assert rewards[1] > rewards[0]


def test_recovery_reward_prefers_progress_over_stopping():
    rewards = []
    for move in (True, False):
        env = DirectPPOEnv("static", reward_version=3)
        env.reset(seed=31)
        env.position = env.goal - np.array([1.2, 0, 0], dtype=np.float32)
        env.velocity.fill(0)
        env.previous_goal_distance = env._goal_distance()
        action = env._world_to_body(np.array([.4, 0, 0])) if move else np.zeros(3)
        _, reward, _, _, _ = env.step(action)
        rewards.append(reward)
    assert rewards[0] > rewards[1]


def test_bounded_mean_loads_original_weights_and_survives_training_and_save(tmp_path):
    torch.set_num_threads(1)
    env = DirectPPOEnv("navigation", max_steps=10)
    original = PPO("MlpPolicy", env, n_steps=16, batch_size=16, n_epochs=1,
                   policy_kwargs={"net_arch": [16, 16], "features_extractor_class": DirectPPOFeatures}, device="cpu")
    with torch.no_grad():
        original.policy.action_net.weight.fill_(0)
        original.policy.action_net.bias.fill_(3)
    original.save(tmp_path / "original")
    model = PPO.load(tmp_path / "original.zip", env=env,
                     custom_objects={"policy_class": BoundedMeanPolicy}, device="cpu")
    obs, _ = env.reset(seed=3)
    tensor, _ = model.policy.obs_to_tensor(obs)
    distribution = model.policy.get_distribution(tensor)
    mean = distribution.distribution.mean
    assert torch.all(mean > 0) and torch.all(mean < 1)
    mean.sum().backward()
    assert torch.all(model.policy.action_net.bias.grad > 0)
    model.learn(total_timesteps=16)
    model.save(tmp_path / "bounded")
    loaded = PPO.load(tmp_path / "bounded.zip", device="cpu")
    assert isinstance(loaded.policy, BoundedMeanPolicy)
    np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0], loaded.predict(obs, deterministic=True)[0])


def test_checkpoint_selection_prefers_passing_safety_gate_to_higher_unsafe_success():
    from logistics_planner.train_direct_ppo import checkpoint_score
    safe = dict(success_rate=.96, collision_rate=0, outside_rate=0, mean_return=200)
    unsafe = dict(success_rate=.99, collision_rate=.01, outside_rate=0.01, mean_return=300)
    assert checkpoint_score(safe) > checkpoint_score(unsafe)


@pytest.mark.parametrize("vertical_action", [-1, 1])
def test_shared_guard_prevents_sustained_vertical_command_from_leaving_height_range(vertical_action):
    env = DirectPPOEnv("navigation", safety_guards=True)
    env.reset(seed=17)
    interventions = 0
    for _ in range(200):
        _, _, _, _, info = env.step([0, 0, vertical_action])
        interventions += info["safety_guarded"]
        assert not info["outside"]
        assert 2 <= env.position[2] <= 10
    assert interventions > 0


def test_shared_projection_matches_existing_runtime_altitude_filter():
    from logistics_planner.direct_ppo_contract import project_safe_velocity
    from logistics_planner.single_uav_avoidance_test import guard_altitude_velocity
    for height in (2.5, 3.0, 6.0, 8.8, 9.5):
        for vertical in (-1, 0, 1):
            command, _ = project_safe_velocity([.4, .2, vertical], height, 20)
            reference = guard_altitude_velocity([0, 0, height], [.4, .2, vertical])
            np.testing.assert_allclose(command, reference.velocity_enu, atol=1e-6)
    command, flags = project_safe_velocity([.4, .2, -1], 6, 1)
    assert command[2] == 0 and flags == ("downward",)
