import numpy as np
import pytest
from logistics_planner.city_ppo_env import CityPPOEnv, box_ranges, box_clearance


def test_box_rays_include_parallel_misses_and_inside_exits():
    box = np.array([[[5., -1, -1], [7., 1, 1]]], dtype=np.float32)
    rays = np.array([[1., 0, 0], [0, 1, 0], [-1., 0, 0]], dtype=np.float32)
    assert box_ranges(np.zeros(3), rays, box) == pytest.approx([5., 20., 20.])
    assert box_ranges(np.array([6., 0, 0]), rays, box) == pytest.approx([1., 1., 1.])
    assert box_clearance([4., 0, 0], box) == pytest.approx(1.)
    assert box_clearance([6., 0, 0], box) == pytest.approx(-1.)


def test_city_reset_replays_geometry_and_clears_failure_state():
    env = CityPPOEnv(scenario=1, randomize=False)
    obs, _ = env.reset(seed=1)
    assert obs.shape == (153,) and env.observation_space.contains(obs)
    assert len(env.boxes) == 12 and len(env.obstacles) == 10
    assert env.outbound_goal == pytest.approx([-6, -23, 6])
    env.low_scan_streak = 4
    again, _ = env.reset(seed=1)
    np.testing.assert_array_equal(obs, again)
    assert env.low_scan_streak == 0


def test_city_tree_clearance_and_safe_height_reward():
    env = CityPPOEnv(dynamic=False, randomize=False)
    env.reset()
    env.position = np.array([-12.65, -23.19, 3.57], dtype=np.float32)
    low = env._scan().min()
    low_reward = env._reward_v3(np.zeros(3), 0, 2., 2., 7.)[0]
    env.position[2] = 6.
    assert env._scan().min() > low
    assert env._reward_v3(np.zeros(3), 0, 2., 2., 7.)[0] > low_reward


def test_city_training_mixture_does_not_apply_to_default_evaluation():
    mixed = CityPPOEnv(static_probability=1.)
    mixed.reset(seed=3)
    full = CityPPOEnv()
    full.reset(seed=3)
    assert len(mixed.obstacles) == 6
    assert len(full.obstacles) == 10


def test_revision_two_penalizes_collisions_more_strongly():
    rewards = []
    for revision in (1, 2):
        env = CityPPOEnv(dynamic=False, randomize=False, reward_revision=revision,
                         runtime_selection=False)
        env.reset(seed=1)
        env.position = np.array([-10., -25., 4.], dtype=np.float32)
        _, reward, terminated, _, info = env.step(np.zeros(3, dtype=np.float32))
        assert terminated and info['collision'] and not info['success']
        rewards.append(reward)
    assert rewards[1] < rewards[0] - 700


def test_city_checkpoint_does_not_reward_blind_spot_collisions():
    from logistics_planner.train_city_ppo import city_checkpoint_score
    safe = dict(collision_rate=0., outside_rate=0., safe_success_rate=2/3,
                success_rate=2/3, full_scan_margin_rate=.75, mean_return=16.)
    blind_collision = dict(safe, collision_rate=1/6, full_scan_margin_rate=5/6,
                           mean_return=213.)
    assert city_checkpoint_score(safe) > city_checkpoint_score(blind_collision)
