import math

import pytest

from logistics_planner.single_uav_avoidance_test import (
    body_flu_to_ned,
    desired_velocity_enu,
    limit_vector,
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
