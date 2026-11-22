"""Live direct PPO velocity inference; no nominal/APF velocity input."""
from __future__ import annotations

import argparse
import math
import time
from collections import deque
from pathlib import Path

import rclpy
import torch
from geometry_msgs.msg import Vector3Stamped
from px4_msgs.msg import VehicleLocalPosition
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from stable_baselines3 import PPO
from std_msgs.msg import Float32, Float32MultiArray, String

from .direct_ppo_contract import action_to_velocity, build_observation
from .single_uav_avoidance_test import SPAWN_ENU, ned_vector_to_body_flu


class DirectPPOInference(Node):
    def __init__(self, model_path, vehicle_index=1):
        super().__init__(f"direct_ppo_inference_{vehicle_index}")
        self.vehicle_index = vehicle_index
        from .config import load_world_config
        self.spawn_enu = load_world_config(Path(__file__).resolve().parents[3] / 'config/logistics_city.yaml').uav_spawn_points[vehicle_index - 1]
        topic = f'/mupt/uav_{vehicle_index}'
        torch.set_num_threads(1)
        self.model = PPO.load(model_path, device="cpu")
        if self.model.observation_space.shape != (153,) or self.model.action_space.shape != (3,):
            raise ValueError("requires a direct 3-D model with 153 observations and 3 actions")
        self.frames = deque(maxlen=4)
        self.goal = self.position = self.down = None
        self.times = {}
        self.publisher = self.create_publisher(Vector3Stamped, topic + "/ppo_direct/velocity_body", qos_profile_sensor_data)
        self.status = self.create_publisher(String, topic + "/ppo_direct/status", qos_profile_sensor_data)
        self.create_subscription(Float32MultiArray, topic + "/observation/sectors", self._scan, qos_profile_sensor_data)
        self.create_subscription(Float32, topic + "/observation/down_clearance", self._down, qos_profile_sensor_data)
        self.create_subscription(Vector3Stamped, topic + "/planner/relative_goal_body", self._goal, qos_profile_sensor_data)
        self.create_subscription(VehicleLocalPosition, f"/mupt_px4_{vehicle_index}/fmu/out/vehicle_local_position_v1", self._position, qos_profile_sensor_data)
        self.create_timer(.1, self._update)
        self.get_logger().info(f"Loaded direct 3-D PPO: {model_path}; no potential-field input")

    def _scan(self, message):
        if len(message.data) == 36 and all(math.isfinite(v) for v in message.data):
            if time.monotonic()-self.times.get('scan',time.monotonic())>.5:
                self.frames.clear()
            self.frames.append(tuple(message.data))
            self.times["scan"] = time.monotonic()

    def _down(self, message):
        if math.isfinite(message.data) and message.data >= 0:
            self.down = float(message.data)
            self.times["down"] = time.monotonic()

    def _goal(self, message):
        values = (message.vector.x, message.vector.y, message.vector.z)
        if all(math.isfinite(v) for v in values):
            self.goal = values
            self.times["goal"] = time.monotonic()

    def _position(self, message):
        if all(math.isfinite(v) for v in (message.z, message.vx, message.vy, message.vz, message.heading)):
            self.position = message
            self.times["position"] = time.monotonic()

    def _update(self):
        now = time.monotonic()
        missing = [key for key in ("scan", "down", "goal", "position") if now - self.times.get(key, -math.inf) > .5]
        status = String()
        if missing or len(self.frames) != 4:
            status.data = "WAITING " + ",".join(missing or ["four_scan_frames"])
        else:
            position = self.position
            velocity = ned_vector_to_body_flu((position.vx, position.vy, position.vz), position.heading)
            observation = build_observation(tuple(self.frames), self.goal, velocity,
                                            self.spawn_enu[2] - position.z, self.down,
                                            altitude_min=getattr(self.model, 'altitude_min', 2.),
                                            altitude_max=getattr(self.model, 'altitude_max', 10.))
            action, _ = self.model.predict(observation, deterministic=True)
            command = action_to_velocity(action, getattr(self.model, "velocity_mapping", "radial"))
            message = Vector3Stamped()
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = f"uav_{self.vehicle_index}/lidar_link"
            message.vector.x, message.vector.y, message.vector.z = map(float, command)
            self.publisher.publish(message)
            status.data = f"DIRECT_3D velocity_flu={command.tolist()} potential_field_used=false"
        self.status.publish(status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument('--vehicle-index', type=int, choices=(1, 2, 3), default=1)
    args = parser.parse_args()
    rclpy.init()
    node = DirectPPOInference(args.model, args.vehicle_index)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
