import math

import pytest

from logistics_planner.sector_observation import minimum_point_distance, sectorize_points


def test_sectorize_uses_layer_major_order_and_nearest_distance():
    nan = float("nan")
    points = [(nan, nan, nan)] * 12
    points = list(points)
    # Three layers, four organized points per layer.
    points[0] = (5.0, 0.0, 0.0)  # lower, forward
    points[1] = (2.0, 0.0, 0.0)  # same sector: nearest wins
    points[4] = (0.0, 3.0, 0.0)  # middle, left
    points[8] = (-4.0, 0.0, 1.0)  # upper, rear

    values = sectorize_points(points, width=4, height=3, sector_count=12)

    assert len(values) == 36
    assert values[0] == pytest.approx(2.0)
    assert values[12 + 3] == pytest.approx(3.0)
    assert values[24 + 6] == pytest.approx(math.sqrt(17.0))
    assert values[1] == pytest.approx(20.0)


def test_sector_zero_wraps_around_forward_axis():
    angle = math.radians(-5.0)
    point = (4.0 * math.cos(angle), 4.0 * math.sin(angle), 0.0)
    values = sectorize_points([point], width=1, height=1, sector_count=12)
    assert values[0] == pytest.approx(4.0)


def test_downward_cone_reports_nearest_finite_point():
    points = [
        (float("nan"), 0.0, 0.0),
        (0.0, 0.0, 4.0),
        (0.3, 0.4, 0.0),
    ]
    assert minimum_point_distance(points) == pytest.approx(0.5)


def test_downward_cone_defaults_to_sensor_range_when_clear():
    assert minimum_point_distance([(float("nan"),) * 3]) == pytest.approx(15.0)
