from pathlib import Path
import xml.etree.ElementTree as ET

import pytest


def test_dynamic_sphere_has_an_offset_orbit_joint() -> None:
    project_root = Path(__file__).resolve().parents[3]
    model_path = project_root / "simulation/tests/avoidance_test_moving_sphere.sdf"
    root = ET.parse(model_path).getroot()

    sphere_pose = root.find(".//link[@name='sphere_link']/pose")
    joint_pose = root.find(".//joint[@name='orbit_joint']/pose")
    velocity = root.find(".//plugin/joint_name/../initial_velocity")

    assert sphere_pose is not None
    assert [float(value) for value in sphere_pose.text.split()[:3]] == pytest.approx(
        [2.5, 0.0, 0.0]
    )
    assert joint_pose is not None
    assert joint_pose.attrib.get("relative_to") == "__model__"
    assert velocity is not None
    assert float(velocity.text) == pytest.approx(0.35)
