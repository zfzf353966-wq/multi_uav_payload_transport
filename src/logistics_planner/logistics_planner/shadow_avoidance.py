from __future__ import annotations

import math
import statistics
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Sequence

import rclpy
from geometry_msgs.msg import Vector3Stamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32MultiArray, String


ELEVATIONS = (math.radians(-12.0), 0.0, math.radians(12.0))


@dataclass(frozen=True)
class AvoidanceResult:
    suggested_velocity: tuple[float, float, float]
    correction: tuple[float, float, float]
    action: str
    active: bool
    minimum_distance: float


def ned_to_body_flu(
    vector_ned: Sequence[float], heading: float
) -> tuple[float, float, float]:
    """Convert a NED vector to forward-left-up using PX4 NED heading."""
    north, east, down = map(float, vector_ned)
    cosine, sine = math.cos(heading), math.sin(heading)
    return (
        north * cosine + east * sine,
        north * sine - east * cosine,
        -down,
    )


def compute_shadow_avoidance(
    sectors: Sequence[float],
    desired_velocity: Sequence[float],
    *,
    influence_distance: float = 6.0,
    emergency_distance: float = 2.0,
    max_correction: float = 2.0,
    max_speed: float = 4.0,
) -> AvoidanceResult:
    """Potential-field safety suggestion from 3x12 nearest distances.

    Inputs and outputs use body FLU. Obstacles outside the intended direction
    of travel are ignored, while nearly stationary vehicles consider their
    forward hemisphere. The result is advisory and never commands PX4.
    """
    if len(sectors) != 36:
        raise ValueError("expected exactly 36 sector distances")
    desired = tuple(float(value) for value in desired_velocity)
    if len(desired) != 3 or not all(math.isfinite(value) for value in desired):
        raise ValueError("desired_velocity must contain three finite values")
    if not 0.0 < emergency_distance < influence_distance:
        raise ValueError("distance thresholds must satisfy 0 < emergency < influence")

    desired_norm = vector_norm(desired)
    travel_direction = (
        tuple(value / desired_norm for value in desired)
        if desired_norm > 0.2
        else (1.0, 0.0, 0.0)
    )
    repulsion = [0.0, 0.0, 0.0]
    minimum = influence_distance
    strongest = 0.0

    for layer, elevation in enumerate(ELEVATIONS):
        cos_elevation = math.cos(elevation)
        for sector in range(12):
            distance = float(sectors[layer * 12 + sector])
            if not math.isfinite(distance) or distance >= influence_distance:
                continue
            azimuth = sector * 2.0 * math.pi / 12.0
            direction = (
                cos_elevation * math.cos(azimuth),
                cos_elevation * math.sin(azimuth),
                math.sin(elevation),
            )
            relevance = sum(a * b for a, b in zip(direction, travel_direction))
            if relevance <= 0.0:
                continue
            minimum = min(minimum, max(0.0, distance))
            proximity = (influence_distance - max(0.0, distance)) / (
                influence_distance - emergency_distance
            )
            weight = min(1.5, max(0.0, proximity)) * relevance
            strongest = max(strongest, weight)
            for axis in range(3):
                repulsion[axis] -= direction[axis] * weight

    if strongest <= 0.0:
        return AvoidanceResult(desired, (0.0, 0.0, 0.0), "CLEAR", False, influence_distance)

    repulsion_norm = vector_norm(repulsion)
    if repulsion_norm > 1e-6:
        correction = tuple(
            max_correction * strongest * value / repulsion_norm for value in repulsion
        )
    else:
        correction = (0.0, 0.0, 0.0)

    # A perfectly centred obstacle produces only braking. Pick the clearer
    # horizontal side deterministically so the advisory output can go around.
    if correction[0] < -0.05 and abs(correction[1]) < 0.15:
        left_clearance = min(float(sectors[layer * 12 + sector]) for layer in range(3) for sector in (1, 2, 3))
        right_clearance = min(float(sectors[layer * 12 + sector]) for layer in range(3) for sector in (9, 10, 11))
        lateral = max_correction * max(0.35, strongest)
        correction = (correction[0], lateral if left_clearance >= right_clearance else -lateral, correction[2])

    correction = limit_vector(correction, max_correction)
    suggested = limit_vector(
        tuple(desired[axis] + correction[axis] for axis in range(3)), max_speed
    )
    emergency = minimum <= emergency_distance
    action = action_from_correction(correction, emergency)
    return AvoidanceResult(suggested, correction, action, True, minimum)


def action_from_correction(correction: Sequence[float], emergency: bool) -> str:
    prefix = "EMERGENCY_" if emergency else "AVOID_"
    _, lateral, vertical = correction
    if abs(vertical) > abs(lateral) and abs(vertical) > 0.1:
        return prefix + ("UP" if vertical > 0.0 else "DOWN")
    if abs(lateral) > 0.1:
        return prefix + ("LEFT" if lateral > 0.0 else "RIGHT")
    return prefix + "BRAKE"


def limit_vector(vector: Sequence[float], maximum: float) -> tuple[float, float, float]:
    values = tuple(float(value) for value in vector)
    norm = vector_norm(values)
    if norm <= maximum or norm < 1e-9:
        return values
    scale = maximum / norm
    return tuple(value * scale for value in values)


def vector_norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in vector))


class ShadowAvoidanceNode(Node):
    """Publish advisory avoidance velocities without writing to PX4 inputs."""

    def __init__(self) -> None:
        super().__init__("shadow_avoidance")
        self.declare_parameter("influence_distance", 6.0)
        self.declare_parameter("emergency_distance", 2.0)
        self.declare_parameter("max_correction", 2.0)
        self.declare_parameter("max_speed", 4.0)
        self.influence_distance = float(self.get_parameter("influence_distance").value)
        self.emergency_distance = float(self.get_parameter("emergency_distance").value)
        self.max_correction = float(self.get_parameter("max_correction").value)
        self.max_speed = float(self.get_parameter("max_speed").value)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.sectors: dict[int, list[float]] = {}
        self.sector_history = {index: deque(maxlen=3) for index in range(1, 4)}
        self.desired_velocities: dict[int, tuple[float, float, float]] = {}
        self.desired_update_times: dict[int, float] = {}
        self.output_publishers = {}
        self.correction_publishers = {}
        self.action_publishers = {}
        self.input_subscriptions = []
        self.last_action = {index: "" for index in range(1, 4)}
        self.avoidance_side = {index: 0 for index in range(1, 4)}

        for index in range(1, 4):
            self.output_publishers[index] = self.create_publisher(
                Vector3Stamped, f"/mupt/uav_{index}/avoidance/suggested_velocity_body", sensor_qos
            )
            self.correction_publishers[index] = self.create_publisher(
                Vector3Stamped, f"/mupt/uav_{index}/avoidance/correction_body", sensor_qos
            )
            self.action_publishers[index] = self.create_publisher(
                String, f"/mupt/uav_{index}/avoidance/action", sensor_qos
            )
            self.input_subscriptions.extend(
                [
                    self.create_subscription(
                        Float32MultiArray,
                        f"/mupt/uav_{index}/observation/sectors",
                        lambda message, index=index: self._set_sectors(index, message),
                        sensor_qos,
                    ),
                    self.create_subscription(
                        Vector3Stamped,
                        f"/mupt/uav_{index}/planner/desired_velocity_body",
                        lambda message, index=index: self._set_desired(index, message),
                        sensor_qos,
                    ),
                ]
            )
        self.timer = self.create_timer(0.1, self._update)
        self.get_logger().info("Shadow avoidance active: advisory outputs only; PX4 is not controlled")

    def _set_sectors(self, index: int, message: Float32MultiArray) -> None:
        if len(message.data) == 36:
            history = self.sector_history[index]
            history.append(list(message.data))
            if len(history) < history.maxlen:
                return
            self.sectors[index] = [
                statistics.median(frame[sector] for frame in history)
                for sector in range(36)
            ]

    def _set_desired(self, index: int, message: Vector3Stamped) -> None:
        self.desired_velocities[index] = (
            message.vector.x,
            message.vector.y,
            message.vector.z,
        )
        self.desired_update_times[index] = time.monotonic()

    def _update(self) -> None:
        for index in range(1, 4):
            if index not in self.sectors or index not in self.desired_velocities:
                continue
            if time.monotonic() - self.desired_update_times[index] > 0.5:
                self.last_action[index] = ""
                self.avoidance_side[index] = 0
                continue
            desired = limit_vector(self.desired_velocities[index], self.max_speed)
            result = compute_shadow_avoidance(
                self.sectors[index],
                desired,
                influence_distance=self.influence_distance,
                emergency_distance=self.emergency_distance,
                max_correction=self.max_correction,
                max_speed=self.max_speed,
            )
            if result.action == "CLEAR":
                self.avoidance_side[index] = 0
            elif result.action.endswith("LEFT") or result.action.endswith("RIGHT"):
                proposed_side = 1 if result.action.endswith("LEFT") else -1
                if self.avoidance_side[index] == 0:
                    self.avoidance_side[index] = proposed_side
                elif proposed_side != self.avoidance_side[index]:
                    side = self.avoidance_side[index]
                    correction = (
                        result.correction[0],
                        abs(result.correction[1]) * side,
                        result.correction[2],
                    )
                    suggested = limit_vector(
                        tuple(desired[axis] + correction[axis] for axis in range(3)),
                        self.max_speed,
                    )
                    prefix = "EMERGENCY_" if result.action.startswith("EMERGENCY_") else "AVOID_"
                    result = replace(
                        result,
                        correction=correction,
                        suggested_velocity=suggested,
                        action=prefix + ("LEFT" if side > 0 else "RIGHT"),
                    )
            self._publish_vector(index, result.suggested_velocity, self.output_publishers[index])
            self._publish_vector(index, result.correction, self.correction_publishers[index])
            action = String()
            action.data = f"{result.action} min={result.minimum_distance:.2f}m"
            self.action_publishers[index].publish(action)
            if result.action != self.last_action[index]:
                self.get_logger().info(f"uav_{index}: {action.data}")
                self.last_action[index] = result.action

    def _publish_vector(self, index: int, values: Sequence[float], publisher) -> None:
        message = Vector3Stamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = f"uav_{index}/lidar_link"
        message.vector.x, message.vector.y, message.vector.z = map(float, values)
        publisher.publish(message)


def main() -> None:
    rclpy.init()
    node = ShadowAvoidanceNode()
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
