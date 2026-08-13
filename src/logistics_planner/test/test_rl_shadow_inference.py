import numpy as np
import pytest

from logistics_planner.rl_shadow_inference import (
    build_policy_observation,
    policy_action_to_velocity,
)


def test_live_observation_matches_training_shape_and_scaling() -> None:
    observation = build_policy_observation(
        [[10.0] * 36, [12.0] * 36, [14.0] * 36, [16.0] * 36],
        (25.0, -12.5, 0.0),
        (3.0, -1.5, 0.0),
    )
    assert observation.shape == (150,)
    assert observation.dtype == np.float32
    assert observation[:36] == pytest.approx([0.5] * 36)
    assert observation[108:144] == pytest.approx([0.8] * 36)
    assert observation[144:147] == pytest.approx((1.0, -0.5, 0.0))
    assert observation[147:150] == pytest.approx((1.0, -0.5, 0.0))


def test_policy_action_is_scaled_and_speed_limited() -> None:
    correction, suggested = policy_action_to_velocity(
        (1.0, 0.5, 0.0), (2.0, 0.0, 0.0), maximum_speed=3.0
    )
    assert correction == pytest.approx((0.8, 0.4, 0.0))
    assert suggested == pytest.approx((2.8, 0.4, 0.0))


def test_invalid_policy_observation_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_policy_observation([[20.0] * 36] * 3, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0))
