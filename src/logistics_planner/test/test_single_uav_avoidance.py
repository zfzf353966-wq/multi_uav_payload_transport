import math

import pytest

from logistics_planner.single_uav_avoidance_test import (
    arbitrate_velocity,
    body_flu_to_ned,
    choose_emergency_tangent_sign,
    desired_velocity_enu,
    directional_clearance,
    emergency_escape_velocity,
    guard_altitude_velocity,
    guard_geofence_velocity,
    limit_vector,
    update_emergency_latch,
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


def test_emergency_scan_uses_predictive_distance() -> None:
    sectors = [20.0] * 36
    sectors[12] = 3.0
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


def test_geofence_guard_leaves_safe_velocity_unchanged() -> None:
    result = guard_geofence_velocity(
        (-20.0, -23.0, 6.0), (1.2, -0.5, 0.1)
    )
    assert result.velocity_enu == pytest.approx((1.2, -0.5, 0.1))
    assert result.guarded_axes == ()


def test_geofence_guard_reverses_outward_velocity_inside_soft_margin() -> None:
    result = guard_geofence_velocity(
        (-20.1, -28.0, 6.0), (1.3, -1.8, 0.0)
    )
    assert result.velocity_enu == pytest.approx((1.3, 2.0 / 3.0, 0.0))
    assert result.guarded_axes == ("y",)


def test_geofence_guard_slows_velocity_before_soft_boundary() -> None:
    result = guard_geofence_velocity(
        (-20.0, -26.5, 6.0), (1.0, -2.0, 0.0)
    )
    assert result.velocity_enu[1] == pytest.approx(-1.0 / 3.0)
    assert result.guarded_axes == ("y",)


def test_altitude_guard_does_not_restrict_horizontal_map_exit() -> None:
    result = guard_altitude_velocity((-50.0, -40.0, 6.0), (-3.0, -2.0, 0.0))
    assert result.velocity_enu == pytest.approx((-3.0, -2.0, 0.0))
    assert result.guarded_axes == ()


def test_altitude_guard_still_limits_emergency_climb() -> None:
    result = guard_altitude_velocity((-50.0, -40.0, 8.5), (-3.0, -2.0, 1.5))
    assert result.velocity_enu == pytest.approx((-3.0, -2.0, 1.0 / 3.0))
    assert result.guarded_axes == ("z",)


def test_emergency_latch_requires_stable_clearance_before_release() -> None:
    active, clear_ticks = update_emergency_latch(False, 0, 3.0)
    assert active and clear_ticks == 0
    for expected in range(1, 5):
        active, clear_ticks = update_emergency_latch(active, clear_ticks, 5.0)
        assert active and clear_ticks == expected
    active, clear_ticks = update_emergency_latch(active, clear_ticks, 5.0)
    assert not active and clear_ticks == 0


def test_emergency_latch_resets_release_counter_if_obstacle_returns() -> None:
    active, clear_ticks = update_emergency_latch(True, 3, 4.0)
    assert active and clear_ticks == 0


def test_emergency_escape_combines_away_and_locked_sideways_motion() -> None:
    sectors = [20.0] * 36
    sectors[12] = 2.0
    forward, left, up = emergency_escape_velocity(sectors, tangent_sign=-1)
    assert forward < 0.0
    assert left > 0.0
    assert math.hypot(forward, left) == pytest.approx(3.0)
    assert up == 0.0


def test_emergency_escape_does_not_ignore_obstacle_behind() -> None:
    sectors = [20.0] * 36
    sectors[18] = 2.0
    forward, left, up = emergency_escape_velocity(sectors)
    assert forward > 0.0
    assert math.hypot(forward, left) == pytest.approx(3.0)
    assert up == 0.0


def test_emergency_escape_uses_nearest_obstacle_across_lidar_layers() -> None:
    sectors = [20.0] * 36
    sectors[3] = 3.0
    sectors[27] = 1.0
    forward, left, up = emergency_escape_velocity(sectors)
    assert forward > 0.0
    assert left < 0.0
    assert math.hypot(forward, left) == pytest.approx(3.0)
    assert up == 0.0


def test_tangent_side_prefers_clearer_left_for_forward_obstacle() -> None:
    sectors = [20.0] * 36
    sectors[12] = 2.0
    sectors[21] = 2.5  # right side is more constrained, but front stays nearest
    assert choose_emergency_tangent_sign(sectors, (1.0, 0.0, 0.0)) == -1
