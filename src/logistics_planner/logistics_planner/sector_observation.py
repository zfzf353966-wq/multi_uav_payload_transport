from __future__ import annotations

import math
from typing import Iterable, Sequence

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float32MultiArray, MultiArrayDimension


LAYER_NAMES = ("lower", "middle", "upper")


def sectorize_points(
    points: Iterable[Sequence[float]],
    *,
    width: int,
    height: int,
    sector_count: int = 12,
    max_range: float = 20.0,
) -> list[float]:
    """Return layer-major nearest ranges for an organized xyz point cloud.

    Layer order follows the Gazebo lidar rows: lower, middle, upper. Sector 0
    is centred on the vehicle's +X (forward) axis and sectors increase
    counter-clockwise around +Z.
    """
    if width <= 0 or height <= 0 or sector_count <= 0:
        raise ValueError("width, height, and sector_count must be positive")

    result = [float(max_range)] * (height * sector_count)
    for index, point in enumerate(points):
        if index >= width * height:
            break
        x, y, z = float(point[0]), float(point[1]), float(point[2])
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            continue
        distance = math.sqrt(x * x + y * y + z * z)
        if not math.isfinite(distance) or distance < 0.0 or distance > max_range:
            continue

        layer = index // width
        angle = math.atan2(y, x)
        # Half a sector offset makes sector 0 symmetric around the +X axis.
        sector_width = 2.0 * math.pi / sector_count
        sector = int(math.floor((angle + 0.5 * sector_width) / sector_width))
        sector %= sector_count
        output_index = layer * sector_count + sector
        result[output_index] = min(result[output_index], distance)
    return result


def make_observation_message(values: Sequence[float], sector_count: int) -> Float32MultiArray:
    message = Float32MultiArray()
    layer_dimension = MultiArrayDimension()
    layer_dimension.label = "lower_middle_upper"
    layer_dimension.size = len(values) // sector_count
    layer_dimension.stride = len(values)
    sector_dimension = MultiArrayDimension()
    sector_dimension.label = "counterclockwise_from_forward"
    sector_dimension.size = sector_count
    sector_dimension.stride = sector_count
    message.layout.dim = [layer_dimension, sector_dimension]
    message.data = [float(value) for value in values]
    return message


class SectorObservationNode(Node):
    """Compress three organized lidar point clouds into 36 distance values."""

    def __init__(self) -> None:
        super().__init__("sector_observation")
        self.declare_parameter("sector_count", 12)
        self.declare_parameter("max_range", 20.0)
        self.sector_count = int(self.get_parameter("sector_count").value)
        self.max_range = float(self.get_parameter("max_range").value)
        if self.sector_count <= 0 or self.max_range <= 0.0:
            raise ValueError("sector_count and max_range must be positive")

        self.output_publishers = {}
        self.input_subscriptions = []
        for index in range(1, 4):
            output_topic = f"/mupt/uav_{index}/observation/sectors"
            self.output_publishers[index] = self.create_publisher(
                Float32MultiArray, output_topic, qos_profile_sensor_data
            )
            self.input_subscriptions.append(
                self.create_subscription(
                    PointCloud2,
                    f"/mupt/uav_{index}/scan/points",
                    lambda message, index=index: self._point_cloud_callback(index, message),
                    qos_profile_sensor_data,
                )
            )
        self.get_logger().info(
            f"Publishing 3 x {self.sector_count} sector distances for three UAVs"
        )

    def _point_cloud_callback(self, index: int, message: PointCloud2) -> None:
        if message.height != len(LAYER_NAMES):
            self.get_logger().warning(
                f"UAV {index} cloud has {message.height} layers; expected 3"
            )
            return
        points = point_cloud2.read_points(
            message, field_names=["x", "y", "z"], skip_nans=False
        )
        values = sectorize_points(
            points,
            width=message.width,
            height=message.height,
            sector_count=self.sector_count,
            max_range=self.max_range,
        )
        self.output_publishers[index].publish(
            make_observation_message(values, self.sector_count)
        )


def main() -> None:
    rclpy.init()
    node = SectorObservationNode()
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
