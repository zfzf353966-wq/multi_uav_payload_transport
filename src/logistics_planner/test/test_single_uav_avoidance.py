import math

import pytest

from logistics_planner.single_uav_avoidance_test import (
    arbitrate_velocity,
    body_flu_to_ned,
    desired_velocity_enu,
    directional_clearance,
    limit_vector,
    use_rl_for_phase,
)


def test_body_flu_to_ned_at_north_heading() -> None:
    assert body_flu_to_ned((2.0, 1.0, 0.5), 0.0) == pytest.approx((2.0, -1.0, -0.5))


def test_body_flu_to_ned_at_east_heading() -> None:
    assert body_flu_to_ned((2.0, 1.0, 0.5), math.pi / 2.0) == pytest.approx((1.0, 2.0, -0.5))


def test_desired_velocity_is_bounded_and_stabilizes_altitude() -> None:
    velocity = desired_velocity_enu((0.0, 0.0, 3.0), (10.0, 0.0, 5.0))
    assert velocity == pytest.approx((1.4, 0.0, 0.8))


def test_limit_vector_preserves_direction() -> None:
    assert limit_vector((3.0, 4.0, 0.0), 2.0) == pytest.approx((1.2, 1.6, 0.0))


def test_rl_is_accepted_with_fresh_clear_data() -> None:
    result = arbitrate_velocity(
        requested_source="rl",
        rl_velocity=(1.0, 0.5, 0.0),
        rl_age=0.1,
        fallback_velocity=(1.0, -0.5, 0.0),
        sectors=[20.0] * 36,
    )
    assert result.source == "rl"
    assert result.velocity_body == pytest.approx((1.0, 0.5, 0.0))


def test_stale_rl_falls_back_to_potential_field() -> None:
    result = arbitrate_velocity(
        requested_source="rl",
        rl_velocity=(1.0, 0.5, 0.0),
        rl_age=0.6,
        fallback_velocity=(0.2, -1.0, 0.0),
        sectors=[20.0] * 36,
    )
    assert result.source == "fallback"
    assert result.reason == "rl_stale"
    assert result.velocity_body == pytest.approx((0.2, -1.0, 0.0))


def test_emergency_scan_forces_fallback() -> None:
    sectors = [20.0] * 36
    sectors[12] = 1.8
    result = arbitrate_velocity(
        requested_source="rl",
        rl_velocity=(1.0, 0.0, 0.0),
        rl_age=0.1,
        fallback_velocity=(-0.5, 1.0, 0.0),
        sectors=sectors,
    )
    assert result.source == "fallback"
    assert result.reason == "emergency_clearance"


def test_directional_clearance_tracks_forward_sector() -> None:
    sectors = [20.0] * 36
    sectors[12] = 3.0
    assert directional_clearance(sectors, (1.0, 0.0, 0.0)) == pytest.approx(3.0)


def test_rl_round_trip_hands_precision_approach_to_traditional_control() -> None:
    assert use_rl_for_phase("outbound", True, 2.0)
    assert use_rl_for_phase("return", True, 8.0)
    assert not use_rl_for_phase("return", True, 4.9)
    assert not use_rl_for_phase("return", False, 8.0)
