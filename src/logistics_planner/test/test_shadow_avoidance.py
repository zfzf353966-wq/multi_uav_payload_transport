import math

import pytest

from logistics_planner.shadow_avoidance import (
    compute_shadow_avoidance,
    ned_to_body_flu,
)


def clear_sectors():
    return [20.0] * 36


def test_ned_to_body_flu_at_east_heading():
    assert ned_to_body_flu((0.0, 2.0, -1.0), math.pi / 2.0) == pytest.approx(
        (2.0, 0.0, 1.0), abs=1e-7
    )


def test_clear_path_preserves_desired_velocity():
    result = compute_shadow_avoidance(clear_sectors(), (3.0, 0.0, 0.0))
    assert result.action == "CLEAR"
    assert result.suggested_velocity == (3.0, 0.0, 0.0)
    assert result.correction == (0.0, 0.0, 0.0)


def test_centred_forward_obstacle_selects_clearer_left_side():
    sectors = clear_sectors()
    sectors[12] = 3.0  # middle layer, directly forward
    sectors[12 + 10] = 2.5  # make the right side less attractive
    result = compute_shadow_avoidance(sectors, (3.0, 0.0, 0.0))
    assert result.active
    assert result.correction[0] < 0.0
    assert result.correction[1] > 0.0
    assert result.action == "AVOID_LEFT"


def test_obstacle_above_requests_downward_correction():
    sectors = clear_sectors()
    sectors[24] = 3.0  # upper layer, directly forward
    result = compute_shadow_avoidance(sectors, (3.0, 0.0, 0.0))
    assert result.correction[2] < 0.0
    assert result.action in ("AVOID_DOWN", "AVOID_LEFT")


def test_obstacle_behind_is_ignored_while_flying_forward():
    sectors = clear_sectors()
    sectors[12 + 6] = 1.0
    result = compute_shadow_avoidance(sectors, (3.0, 0.0, 0.0))
    assert result.action == "CLEAR"


def test_emergency_distance_marks_close_obstacle():
    sectors = clear_sectors()
    sectors[12] = 1.5
    result = compute_shadow_avoidance(sectors, (3.0, 0.0, 0.0))
    assert result.active
    assert result.action.startswith("EMERGENCY_")
