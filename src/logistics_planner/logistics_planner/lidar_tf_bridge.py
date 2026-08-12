from __future__ import annotations

import math
import os
import threading

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from geometry_msgs.msg import TransformStamped
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node as GazeboNode, SubscribeOptions
import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


class LidarPoseFollower(Node):
    """Keep lightweight Gazebo lidar pods on the three X500 models and publish TF."""

    def __init__(self) -> None:
        super().__init__("lidar_pose_follower")
        self.declare_parameter("world", "logistics_city")
        self.world = str(self.get_parameter("world").value)
        self.gz_subscriber = GazeboNode()
        self.gz_commands = GazeboNode()
        self.pose_service = f"/world/{self.world}/set_pose_vector"
        self.model_poses: dict[str, Pose] = {}
        self.sensor_poses: dict[int, Pose] = {}
        self.lock = threading.Lock()
        self.consecutive_command_failures = 0
        self.tf_broadcaster = TransformBroadcaster(self)
        pose_options = SubscribeOptions()
        pose_options.msgs_per_sec = 60
        self.gz_subscriber.subscribe(
            Pose_V,
            f"/world/{self.world}/pose/info",
            self._pose_callback,
            pose_options,
        )
        self.pose_timer = self.create_timer(0.02, self._update_sensor_poses)
        self.tf_timer = self.create_timer(1.0 / 30.0, self._publish_transforms)

    def _pose_callback(self, message: Pose_V) -> None:
        with self.lock:
            for pose in message.pose:
                if pose.name in ("x500_1", "x500_2", "x500_3"):
                    stored = Pose()
                    stored.CopyFrom(pose)
                    self.model_poses[pose.name] = stored

    def _update_sensor_poses(self) -> None:
        with self.lock:
            vehicle_poses = {name: pose for name, pose in self.model_poses.items()}
        sensor_poses: list[tuple[int, Pose]] = []
        request = Pose_V()
        for index in range(1, 4):
            vehicle_pose = vehicle_poses.get(f"x500_{index}")
            if vehicle_pose is None:
                continue
            sensor_pose = self._sensor_pose(vehicle_pose, f"uav_{index}_lidar")
            request.pose.add().CopyFrom(sensor_pose)
            sensor_poses.append((index, sensor_pose))

        if not sensor_poses:
            return
        success, response = self.gz_commands.request(
            self.pose_service, request, Pose_V, Boolean, 100
        )
        if not success or not response.data:
            self.consecutive_command_failures += 1
            if self.consecutive_command_failures == 10:
                self.get_logger().warning(
                    "Gazebo rejected ten consecutive lidar pose updates"
                )
            return
        self.consecutive_command_failures = 0
        with self.lock:
            for index, sensor_pose in sensor_poses:
                stored = Pose()
                stored.CopyFrom(sensor_pose)
                self.sensor_poses[index] = stored

    def _publish_transforms(self) -> None:
        with self.lock:
            sensor_poses = {index: pose for index, pose in self.sensor_poses.items()}
        stamp = self.get_clock().now().to_msg()
        for index, sensor_pose in sensor_poses.items():
            self._publish_transform(index, sensor_pose, stamp)

    @staticmethod
    def _sensor_pose(vehicle_pose: Pose, sensor_name: str) -> Pose:
        result = Pose()
        result.name = sensor_name
        result.orientation.CopyFrom(vehicle_pose.orientation)
        # Keep the virtual origin above the rotor disk. Otherwise the -12 deg
        # layer sees the X500's own arms as false obstacles at minimum range.
        offset = rotate_vector((0.0, 0.0, 0.60), vehicle_pose.orientation)
        result.position.x = vehicle_pose.position.x + offset[0]
        result.position.y = vehicle_pose.position.y + offset[1]
        result.position.z = vehicle_pose.position.z + offset[2]
        return result

    def _publish_transform(self, index: int, pose: Pose, stamp) -> None:
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = "map"
        transform.child_frame_id = f"uav_{index}/lidar_link"
        transform.transform.translation.x = pose.position.x
        transform.transform.translation.y = pose.position.y
        transform.transform.translation.z = pose.position.z
        transform.transform.rotation.x = pose.orientation.x
        transform.transform.rotation.y = pose.orientation.y
        transform.transform.rotation.z = pose.orientation.z
        transform.transform.rotation.w = pose.orientation.w
        self.tf_broadcaster.sendTransform(transform)


def rotate_vector(vector: tuple[float, float, float], quaternion) -> tuple[float, float, float]:
    """Rotate a vector by an xyzw quaternion without external dependencies."""
    x, y, z, w = quaternion.x, quaternion.y, quaternion.z, quaternion.w
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-9:
        return vector
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    vx, vy, vz = vector
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def main() -> None:
    rclpy.init()
    node = LidarPoseFollower()
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
