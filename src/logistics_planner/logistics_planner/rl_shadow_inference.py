from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import rclpy
from geometry_msgs.msg import Vector3Stamped
from px4_msgs.msg import VehicleLocalPosition
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from stable_baselines3 import PPO
from std_msgs.msg import Float32MultiArray, String

from .shadow_avoidance import limit_vector, ned_to_body_flu


def build_policy_observation(
    sectors: Sequence[float],
    relative_goal_body: Sequence[float],
    velocity_body: Sequence[float],
    *,
    sensor_range: float = 20.0,
    goal_scale: float = 25.0,
    velocity_scale: float = 3.0,
) -> np.ndarray:
    """Build the exact 42-value observation used by the training environment."""
    scan = np.asarray(sectors, dtype=np.float32)
    goal = np.asarray(relative_goal_body, dtype=np.float32)
    velocity = np.asarray(velocity_body, dtype=np.float32)
    if scan.shape != (36,) or goal.shape != (3,) or velocity.shape != (3,):
        raise ValueError("expected 36 sectors, 3 goal values, and 3 velocity values")
    if not all(np.isfinite(values).all() for values in (scan, goal, velocity)):
        raise ValueError("policy observation inputs must be finite")
    return np.concatenate(
        (
            np.clip(scan / sensor_range, 0.0, 1.0),
            np.clip(goal / goal_scale, -1.0, 1.0),
            np.clip(velocity / velocity_scale, -1.0, 1.0),
        )
    ).astype(np.float32)


def policy_action_to_velocity(
    action: Sequence[float],
    desired_velocity_body: Sequence[float],
    *,
    correction_scale: float = 2.0,
    maximum_speed: float = 3.0,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    normalized = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    desired = np.asarray(desired_velocity_body, dtype=np.float32)
    if normalized.shape != (3,) or desired.shape != (3,):
        raise ValueError("action and desired velocity must contain three values")
    correction = tuple(float(value) for value in normalized * correction_scale)
    suggested = limit_vector(
        tuple(float(desired[index]) + correction[index] for index in range(3)),
        maximum_speed,
    )
    return correction, suggested


class RLShadowInference(Node):
    """Run a PPO policy on live Gazebo data without commanding PX4."""

    def __init__(self, model_path: Path, namespace: str = "/mupt_px4_1") -> None:
        super().__init__("rl_shadow_inference")
        self.model = PPO.load(model_path, device="cpu")
        self.namespace = namespace.rstrip("/")
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.action_publisher = self.create_publisher(
            Vector3Stamped, "/mupt/uav_1/rl_shadow/action_normalized", sensor_qos
        )
        self.correction_publisher = self.create_publisher(
            Vector3Stamped, "/mupt/uav_1/rl_shadow/correction_body", sensor_qos
        )
        self.suggested_publisher = self.create_publisher(
            Vector3Stamped, "/mupt/uav_1/rl_shadow/suggested_velocity_body", sensor_qos
        )
        self.status_publisher = self.create_publisher(
            String, "/mupt/uav_1/rl_shadow/status", sensor_qos
        )
        self.create_subscription(
            Float32MultiArray,
            "/mupt/uav_1/observation/sectors",
            self._sectors_callback,
            sensor_qos,
        )
        self.create_subscription(
            Vector3Stamped,
            "/mupt/uav_1/planner/relative_goal_body",
            self._goal_callback,
            sensor_qos,
        )
        self.create_subscription(
            Vector3Stamped,
            "/mupt/uav_1/planner/desired_velocity_body",
            self._desired_callback,
            sensor_qos,
        )
        self.create_subscription(
            VehicleLocalPosition,
            self.namespace + "/fmu/out/vehicle_local_position_v1",
            self._position_callback,
            sensor_qos,
        )
        self.sectors: tuple[float, ...] | None = None
        self.relative_goal: tuple[float, float, float] | None = None
        self.desired_velocity: tuple[float, float, float] | None = None
        self.position: VehicleLocalPosition | None = None
        self.update_times: dict[str, float] = {}
        self.inference_count = 0
        self.started = time.monotonic()
        self.last_wait_log = 0.0
        self.timer = self.create_timer(0.1, self._update)
        self.get_logger().info(
            f"PPO shadow inference loaded {model_path}; advisory outputs only, PX4 is not controlled"
        )

    def _sectors_callback(self, message: Float32MultiArray) -> None:
        if len(message.data) == 36 and all(math.isfinite(value) for value in message.data):
            self.sectors = tuple(float(value) for value in message.data)
            self.update_times["sectors"] = time.monotonic()

    def _goal_callback(self, message: Vector3Stamped) -> None:
        self.relative_goal = self._vector(message)
        self.update_times["goal"] = time.monotonic()

    def _desired_callback(self, message: Vector3Stamped) -> None:
        self.desired_velocity = self._vector(message)
        self.update_times["desired"] = time.monotonic()

    def _position_callback(self, message: VehicleLocalPosition) -> None:
        if all(math.isfinite(value) for value in (message.vx, message.vy, message.vz, message.heading)):
            self.position = message
            self.update_times["position"] = time.monotonic()

    @staticmethod
    def _vector(message: Vector3Stamped) -> tuple[float, float, float]:
        return (message.vector.x, message.vector.y, message.vector.z)

    def _update(self) -> None:
        now = time.monotonic()
        required = ("sectors", "goal", "desired", "position")
        if any(name not in self.update_times or now - self.update_times[name] > 0.5 for name in required):
            if now - self.last_wait_log > 5.0:
                missing = [
                    name
                    for name in required
                    if name not in self.update_times or now - self.update_times[name] > 0.5
                ]
                self.get_logger().info(f"waiting for fresh inputs: {', '.join(missing)}")
                self.last_wait_log = now
            return
        velocity_body = ned_to_body_flu(
            (self.position.vx, self.position.vy, self.position.vz), self.position.heading
        )
        observation = build_policy_observation(
            self.sectors, self.relative_goal, velocity_body
        )
        started = time.perf_counter()
        action, _ = self.model.predict(observation, deterministic=True)
        latency_ms = (time.perf_counter() - started) * 1000.0
        correction, suggested = policy_action_to_velocity(action, self.desired_velocity)
        self._publish_vector(action, self.action_publisher)
        self._publish_vector(correction, self.correction_publisher)
        self._publish_vector(suggested, self.suggested_publisher)
        status = String()
        status.data = (
            f"SHADOW count={self.inference_count} latency={latency_ms:.2f}ms "
            f"min_scan={min(self.sectors):.2f}m"
        )
        self.status_publisher.publish(status)
        self.inference_count += 1

    def _publish_vector(self, values: Sequence[float], publisher) -> None:
        message = Vector3Stamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "uav_1/lidar_link"
        message.vector.x, message.vector.y, message.vector.z = map(float, values)
        publisher.publish(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PPO advisory inference on live Gazebo data")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--namespace", default="/mupt_px4_1")
    arguments = parser.parse_args()
    if not arguments.model.exists():
        raise SystemExit(f"model not found: {arguments.model}")
    rclpy.init()
    node = RLShadowInference(arguments.model, arguments.namespace)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
