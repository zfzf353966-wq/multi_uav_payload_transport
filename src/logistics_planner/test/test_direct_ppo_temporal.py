import numpy as np
import torch
from stable_baselines3 import PPO

from logistics_planner.direct_ppo_contract import build_observation
from logistics_planner.direct_ppo_env import DirectPPOEnv
from logistics_planner.direct_ppo_features import DirectPPOFeatures, TemporalDirectPPOFeatures
from logistics_planner.direct_ppo_policy import BoundedMeanPolicy, transfer_policy_weights


def test_temporal_features_encode_approach_and_mask_visibility_transitions():
    env = DirectPPOEnv()
    extractor = TemporalDirectPPOFeatures(env.observation_space)
    frames = np.array([[10.] * 36, [9.8] * 36, [9.6] * 36, [9.4] * 36], dtype=np.float32)
    obs = build_observation(frames, [10, 0, 0], [0, 0, 0], 6, 6)
    features = extractor(torch.as_tensor(obs[None]))
    assert features.shape == (1, 261)
    np.testing.assert_allclose(features.detach().numpy()[0, 153:], .5, atol=3e-6)
    frames[0].fill(20)
    frames[2].fill(20)
    obs = build_observation(frames, [10, 0, 0], [0, 0, 0], 6, 6)
    features = extractor(torch.as_tensor(obs[None]))
    assert torch.count_nonzero(features[..., 153:]) == 0


def test_feature_expansion_preserves_policy_and_value_and_can_train_and_reload(tmp_path):
    torch.set_num_threads(1)
    env = DirectPPOEnv("dynamic", reward_version=3, safety_guards=True, max_steps=20)
    common = dict(n_steps=16, batch_size=16, n_epochs=1, device="cpu", seed=11)
    source = PPO(BoundedMeanPolicy, env, **common,
                 policy_kwargs={"net_arch": [16, 16], "features_extractor_class": DirectPPOFeatures})
    target = PPO(BoundedMeanPolicy, env, **common,
                 policy_kwargs={"net_arch": [16, 16], "features_extractor_class": TemporalDirectPPOFeatures})
    transfer_policy_weights(source.policy, target.policy)
    obs, _ = env.reset(seed=13)
    for _ in range(5):
        obs, _, _, _, _ = env.step([.2, .1, .1])
    np.testing.assert_allclose(source.predict(obs, deterministic=True)[0], target.predict(obs, deterministic=True)[0], atol=1e-7)
    tensor, _ = source.policy.obs_to_tensor(obs)
    with torch.no_grad():
        np.testing.assert_allclose(source.policy.predict_values(tensor), target.policy.predict_values(tensor), atol=1e-6)
    target.learn(total_timesteps=32)
    target.save(tmp_path / "temporal")
    loaded = PPO.load(tmp_path / "temporal.zip", device="cpu")
    assert isinstance(loaded.policy.features_extractor, TemporalDirectPPOFeatures)
    np.testing.assert_array_equal(target.predict(obs, deterministic=True)[0], loaded.predict(obs, deterministic=True)[0])


def test_preflight_history_preserves_initial_geometry_and_has_real_motion():
    repeated = DirectPPOEnv("dynamic")
    warm = DirectPPOEnv("dynamic", history_warmup=True)
    first, _ = repeated.reset(seed=19)
    second, _ = warm.reset(seed=19)
    np.testing.assert_array_equal(repeated.position, warm.position)
    np.testing.assert_array_equal(repeated.goal, warm.goal)
    np.testing.assert_array_equal(repeated.obstacles[0].position, warm.obstacles[0].position)
    np.testing.assert_array_equal(repeated.obstacles[0].velocity, warm.obstacles[0].velocity)
    np.testing.assert_array_equal(first[108:], second[108:])
    assert not np.array_equal(warm.scan_history[0], warm.scan_history[-1])
    assert warm.step_count == 0


def test_vertical_priority_preserves_requested_climb_and_total_speed():
    from logistics_planner.direct_ppo_contract import action_to_velocity
    legacy = action_to_velocity([1, 1, 1])
    priority = action_to_velocity([1, 1, 1], "vertical_priority")
    assert legacy[2] < .7
    assert priority[2] == 1
    assert np.linalg.norm(priority) <= 2.000001
    np.testing.assert_allclose(priority[:2] / np.linalg.norm(priority[:2]), legacy[:2] / np.linalg.norm(legacy[:2]))
    env = DirectPPOEnv("dynamic", velocity_mapping="vertical_priority")
    env.reset(seed=7)
    _, _, _, _, info = env.step([0, 0, 1])
    assert env.velocity[2] > 0 and not info["outside"]


def test_hard_case_sampling_is_reproducible_and_does_not_change_default_distribution():
    regular = DirectPPOEnv("dynamic")
    default = DirectPPOEnv("dynamic", hard_case_probability=0)
    first, _ = regular.reset(seed=8)
    second, _ = default.reset(seed=8)
    np.testing.assert_array_equal(first, second)
    hard = DirectPPOEnv("dynamic", hard_case_probability=1)
    for seed in range(10):
        first, info = hard.reset(seed=seed)
        assert info["hard_case"] and .5 <= hard._clearance() <= 3.5
        second, _ = hard.reset(seed=seed)
        np.testing.assert_array_equal(first, second)


def test_correlated_exploration_preserves_mean_and_trains_and_serializes(tmp_path):
    torch.set_num_threads(1)
    env = DirectPPOEnv("dynamic", reward_version=3, safety_guards=True, max_steps=20)
    kwargs = dict(n_steps=16, batch_size=16, n_epochs=1, device="cpu", seed=17,
                  policy_kwargs={"net_arch": [16, 16], "features_extractor_class": DirectPPOFeatures})
    source = PPO(BoundedMeanPolicy, env, **kwargs)
    target = PPO(BoundedMeanPolicy, env, use_sde=True, sde_sample_freq=8, **kwargs)
    transfer_policy_weights(source.policy, target.policy)
    obs, _ = env.reset(seed=19)
    np.testing.assert_allclose(source.predict(obs, deterministic=True)[0], target.predict(obs, deterministic=True)[0], atol=1e-7)
    target.learn(total_timesteps=32)
    target.velocity_mapping = "radial"
    target.save(tmp_path / "sde")
    loaded = PPO.load(tmp_path / "sde.zip", device="cpu")
    assert loaded.use_sde and loaded.velocity_mapping == "radial"
    np.testing.assert_array_equal(target.predict(obs, deterministic=True)[0], loaded.predict(obs, deterministic=True)[0])


def test_environment_direction_brake_stops_blocked_command_and_releases_in_clear_space():
    env = DirectPPOEnv("dynamic", safety_guards=True, direction_brake=True)
    env.reset(seed=21)
    env.scan_history[-1] = np.ones(36, dtype=np.float32)
    before = env.position.copy()
    _, _, _, _, info = env.step([1, 0, 0])
    assert info["direction_braked"] and "direction" in info["safety_guard_flags"]
    np.testing.assert_array_equal(before, env.position)
    env.scan_history[-1] = np.full(36, 20, dtype=np.float32)
    _, _, _, _, info = env.step([1, 0, 0])
    assert not info["direction_braked"]
    assert np.linalg.norm(env.velocity) > 0


def test_local_activation_requires_clearance_hysteresis_and_minimum_hold_time():
    from logistics_planner.single_uav_avoidance_test import update_local_activation
    active, ticks = update_local_activation(False, 0, 5.9, 0)
    assert active and ticks == 0
    active, ticks = update_local_activation(active, ticks, 20, 2)
    assert active and ticks == 0
    for _ in range(14):
        active, ticks = update_local_activation(active, ticks, 20, 4)
        assert active
    active, ticks = update_local_activation(active, ticks, 20, 4)
    assert not active and ticks == 0
