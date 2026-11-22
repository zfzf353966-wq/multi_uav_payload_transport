from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from gymnasium.utils.env_checker import check_env

from logistics_planner.rl_avoidance_env import MovingSphere
from logistics_planner.gazebo_lidar_model import RAYS, sector_scan, downward_clearance
from logistics_planner.direct_ppo_env import DirectPPOEnv


def test_discrete_elevation_rays_do_not_invent_vertical_coverage():
    position = np.array([0, 0, 6], dtype=np.float32)
    sphere = MovingSphere(np.array([10, 0, 6.6], dtype=np.float32), np.zeros(3), 1)
    sectors = sector_scan(position, 0, [sphere]).reshape(3, 12)
    assert np.all(sectors[0] == 20) and np.all(sectors[2] == 20)
    assert 9 < sectors[1, 0] < 10
    assert len(RAYS) == 360


def test_downward_mount_offset_and_surface_intersection():
    position = np.array([0, 0, 6], dtype=np.float32)
    np.testing.assert_allclose(downward_clearance(position, 0, []), 5.6, atol=1e-5)
    sphere = MovingSphere(np.array([0, 0, 3], dtype=np.float32), np.zeros(3), .5)
    np.testing.assert_allclose(downward_clearance(position, 0, [sphere]), 2.1, atol=1e-5)


def test_gazebo_profile_gym_and_empty_scene_sampling():
    check_env(DirectPPOEnv("dynamic", lidar_profile="gazebo", history_warmup=True,
                          safety_guards=True, direction_brake=True, max_steps=20), skip_render_check=True)
    empty = DirectPPOEnv("dynamic", lidar_profile="gazebo", obstacle_presence_probability=0)
    empty.reset(seed=3)
    assert empty.obstacles == []
    assert empty.observation_space.shape == (153,)


def test_lab_world_contains_only_ground_depot_and_sensor_models():
    root = Path(__file__).resolve().parents[3]
    world = ET.parse(root / "simulation/worlds/ppo_avoidance_lab.sdf").getroot().find("world")
    assert not world.findall("include")
    assert {m.get("name") for m in world.findall("model")} == {
        "ground_plane", "roads", "uav_depot", "uav_1_lidar", "uav_2_lidar", "uav_3_lidar"}


def test_leveled_sensor_pose_matches_training_mount():
    # Gazebo's distro protobuf bindings require their Python backend selected
    # before TensorBoard/PyTorch can initialize protobuf in the parent process.
    import json, os, subprocess, sys
    code = """
import math,json
from logistics_planner.lidar_tf_bridge import LidarPoseFollower,Pose
pose=Pose()
pose.position.x,pose.position.y,pose.position.z=1,2,3
pose.orientation.x=math.sin(.2);pose.orientation.w=math.cos(.2)
level=LidarPoseFollower._sensor_pose(pose,'uav_1_lidar',True)
print(json.dumps([level.position.x,level.position.y,level.position.z,
                 level.orientation.x,level.orientation.y,level.orientation.w]))
"""
    result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True,
                            env={**os.environ, "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python"})
    np.testing.assert_allclose(json.loads(result.stdout), [1, 2, 3.6, 0, 0, 1])
