from __future__ import annotations

import argparse
import math
import time
from typing import Sequence

import rclpy
from geometry_msgs.msg import Vector3Stamped
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleAttitude,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
)
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32MultiArray, String

Point3D = tuple[float, float, float]
SPAWN_ENU: Point3D = (-34.0, -23.0, 0.13)
TAKEOFF_ENU: Point3D = (-34.0, -23.0, 6.0)
OUTBOUND_GOAL_ENU: Point3D = (-6.0, -23.0, 6.0)


def enu_to_local_ned(target: Point3D, origin: Point3D) -> Point3D:
    return (target[1] - origin[1], target[0] - origin[0], -(target[2] - origin[2]))


def local_ned_to_enu(position: Point3D, origin: Point3D) -> Point3D:
    return (origin[0] + position[1], origin[1] + position[0], origin[2] - position[2])


def ned_vector_to_body_flu(vector: Point3D, heading: float) -> Point3D:
    north, east, down = vector
    cosine, sine = math.cos(heading), math.sin(heading)
    return (north * cosine + east * sine, north * sine - east * cosine, -down)


def quaternion_roll_pitch(quaternion: tuple[float, float, float, float]) -> tuple[float, float]:
    w, x, y, z = quaternion
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_sine = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    return roll, math.asin(pitch_sine)


def limit_vector(vector: Sequence[float], maximum: float) -> Point3D:
    values = tuple(float(value) for value in vector)
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= maximum or norm < 1e-9:
        return (values[0], values[1], values[2])
    scale = maximum / norm
    return (values[0] * scale, values[1] * scale, values[2] * scale)


def desired_velocity_enu(
    current: Point3D,
    target: Point3D,
    *,
    cruise_speed: float = 1.4,
    vertical_gain: float = 0.8,
) -> Point3D:
    """Bounded goal-seeking velocity with independent altitude stabilization."""
    dx, dy, dz = (target[index] - current[index] for index in range(3))
    horizontal_distance = math.hypot(dx, dy)
    if horizontal_distance > 1e-6:
        horizontal_speed = min(cruise_speed, horizontal_distance)
        vx = dx * horizontal_speed / horizontal_distance
        vy = dy * horizontal_speed / horizontal_distance
    else:
        vx = vy = 0.0
    vz = max(-0.8, min(0.8, vertical_gain * dz))
    return (vx, vy, vz)


def body_flu_to_ned(vector: Sequence[float], heading: float) -> Point3D:
    """Convert forward-left-up velocity to PX4 north-east-down."""
    forward, left, up = map(float, vector)
    cosine, sine = math.cos(heading), math.sin(heading)
    return (
        forward * cosine + left * sine,
        forward * sine - left * cosine,
        -up,
    )


class SingleUAVAvoidanceTest(Node):
    """Closed-loop lidar avoidance test that commands only PX4 vehicle one."""

    def __init__(self, namespace: str = "/mupt_px4_1") -> None:
        super().__init__("single_uav_closed_loop_avoidance_test")
        self.namespace = namespace.rstrip("/")
        self.system_id = 2

        command_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        telemetry_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.mode_publisher = self.create_publisher(
            OffboardControlMode, self.namespace + "/fmu/in/offboard_control_mode", command_qos
        )
        self.setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, self.namespace + "/fmu/in/trajectory_setpoint", command_qos
        )
        self.command_publisher = self.create_publisher(
            VehicleCommand, self.namespace + "/fmu/in/vehicle_command", command_qos
        )
        self.desired_publisher = self.create_publisher(
            Vector3Stamped, "/mupt/uav_1/planner/desired_velocity_body", sensor_qos
        )
        self.create_subscription(
            VehicleLocalPosition,
            self.namespace + "/fmu/out/vehicle_local_position_v1",
            self._position_callback,
            telemetry_qos,
        )
        self.create_subscription(
            VehicleAttitude,
            self.namespace + "/fmu/out/vehicle_attitude",
            self._attitude_callback,
            telemetry_qos,
        )
        self.create_subscription(
            VehicleStatus,
            self.namespace + "/fmu/out/vehicle_status_v1",
            self._status_callback,
            telemetry_qos,
        )
        self.create_subscription(
            Vector3Stamped,
            "/mupt/uav_1/avoidance/suggested_velocity_body",
            self._suggestion_callback,
            sensor_qos,
        )
        self.create_subscription(
            String,
            "/mupt/uav_1/avoidance/action",
            self._action_callback,
            sensor_qos,
        )
        self.create_subscription(
            Float32MultiArray,
            "/mupt/uav_1/observation/sectors",
            self._sectors_callback,
            sensor_qos,
        )

        self.position: VehicleLocalPosition | None = None
        self.attitude: VehicleAttitude | None = None
        self.status: VehicleStatus | None = None
        self.suggested_body: Point3D | None = None
        self.suggestion_time = 0.0
        self.action = ""
        self.action_time = 0.0
        self.phase = "prestream"
        self.phase_started = time.monotonic()
        self.started = self.phase_started
        self.stable_ticks = 0
        self.warmup_ticks = 0
        self.mode_requested_tick = 0
        self.mode_requested = False
        self.arm_requested = False
        self.landing_commanded = False
        self.command_ticks = 0
        self.outbound_complete = False
        self.avoidance_ticks = 0
        self.minimum_distance = math.inf
        self.maximum_lateral_deviation = 0.0
        self.abort_reason: str | None = None
        self.complete = False
        self.success = False
        self.timer = self.create_timer(0.1, self._tick)
        self.get_logger().info(
            f"closed-loop test bound exclusively to {self.namespace}; UAV 2 and UAV 3 remain disarmed"
        )

    def _position_callback(self, message: VehicleLocalPosition) -> None:
        self.position = message

    def _attitude_callback(self, message: VehicleAttitude) -> None:
        self.attitude = message

    def _status_callback(self, message: VehicleStatus) -> None:
        self.status = message

    def _suggestion_callback(self, message: Vector3Stamped) -> None:
        values = (message.vector.x, message.vector.y, message.vector.z)
        if all(math.isfinite(value) for value in values):
            self.suggested_body = values
            self.suggestion_time = time.monotonic()

    def _action_callback(self, message: String) -> None:
        self.action = message.data
        self.action_time = time.monotonic()

    def _sectors_callback(self, message: Float32MultiArray) -> None:
        finite = [float(value) for value in message.data if math.isfinite(value) and value > 0.05]
        if finite:
            self.minimum_distance = min(self.minimum_distance, min(finite))

    def _tick(self) -> None:
        now = time.monotonic()
        if self.complete:
            return
        if now - self.started > 180.0:
            self._abort("test exceeded 180 s")
        if self.phase == "landing":
            self._handle_landing()
            return

        self._publish_heartbeat(position_control=self.phase in ("prestream", "takeoff"))
        if self.position is None or self.attitude is None or self.status is None:
            if now - self.started > 20.0:
                self._abort("PX4 telemetry was not received within 20 s")
            return
        if self.status.failure_detector_status != VehicleStatus.FAILURE_NONE:
            self._abort(f"PX4 failure detector status={self.status.failure_detector_status}")
            return
        if self.phase not in ("prestream", "landing") and (
            self.status.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD
        ):
            self._abort("vehicle left Offboard mode")
            return

        if self.phase == "prestream":
            self._publish_position_target(TAKEOFF_ENU)
            self._handle_offboard_entry(now)
        elif self.phase == "takeoff":
            self._publish_position_target(TAKEOFF_ENU)
            if self._settled_at(TAKEOFF_ENU, radius=0.5, maximum_speed=0.5):
                self._set_phase("outbound")
                self.get_logger().info("takeoff complete; closed-loop avoidance enabled")
            elif now - self.phase_started > 35.0:
                self._abort("takeoff did not stabilize within 35 s")
        elif self.phase == "outbound":
            self._closed_loop_step(OUTBOUND_GOAL_ENU)
            if self._distance_to(OUTBOUND_GOAL_ENU) <= 1.0:
                self.outbound_complete = True
                self._set_phase("return")
                self.get_logger().info("outbound goal reached beyond the test obstacle; returning")
            elif now - self.phase_started > 60.0:
                self._abort("closed-loop controller did not pass the obstacle within 60 s")
        elif self.phase == "return":
            self._closed_loop_step(TAKEOFF_ENU)
            if self._settled_at(TAKEOFF_ENU, radius=0.8, maximum_speed=0.7):
                self._finish_flight()
            elif now - self.phase_started > 70.0:
                self._abort("vehicle did not return to the launch area within 70 s")

    def _handle_offboard_entry(self, now: float) -> None:
        # Arming invalidates the disarmed-only preflight predicate. Consume the
        # accepted state transition before evaluating that predicate again.
        if (
            self.status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
            and self.status.arming_state == VehicleStatus.ARMING_STATE_ARMED
        ):
            self._set_phase("takeoff")
            return
        if not self._preflight_stable():
            self.stable_ticks = 0
            self.warmup_ticks = 0
            if now - self.started > 60.0:
                self._abort("preflight estimates did not stabilize within 60 s")
            return
        self.stable_ticks += 1
        if self.stable_ticks < 30:
            return
        self.warmup_ticks += 1
        offboard = self.status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
        if self.warmup_ticks >= 20 and (
            not self.mode_requested or (not offboard and self.warmup_ticks % 20 == 0)
        ):
            self._vehicle_command(
                VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0
            )
            if not self.mode_requested:
                self.mode_requested = True
                self.mode_requested_tick = self.warmup_ticks
                self.get_logger().info("Offboard mode requested")
        if (
            self.mode_requested
            and self.warmup_ticks >= self.mode_requested_tick + 5
            and offboard
            and self.status.arming_state == VehicleStatus.ARMING_STATE_DISARMED
            and (not self.arm_requested or self.warmup_ticks % 20 == 0)
        ):
            self._vehicle_command(
                VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
                param1=1.0,
                param2=21196.0,
                from_external=False,
            )
            if not self.arm_requested:
                self.arm_requested = True
                self.get_logger().info("Offboard confirmed; UAV 1 arm requested")
        if offboard and self.status.arming_state == VehicleStatus.ARMING_STATE_ARMED:
            self._set_phase("takeoff")

    def _closed_loop_step(self, target: Point3D) -> None:
        current = self._current_enu()
        desired_enu = desired_velocity_enu(current, target)
        desired_ned = (desired_enu[1], desired_enu[0], -desired_enu[2])
        heading = self.position.heading if math.isfinite(self.position.heading) else 0.0
        desired_body = ned_vector_to_body_flu(desired_ned, heading)
        self._publish_desired_body(desired_body)

        now = time.monotonic()
        if self.suggested_body is None or now - self.suggestion_time > 0.5:
            if now - self.phase_started > 1.5:
                self._abort("avoidance suggestion became stale for more than 0.5 s")
                return
            command_ned = desired_ned
        else:
            command_ned = body_flu_to_ned(self.suggested_body, heading)
            if now - self.action_time <= 0.5 and self.action and not self.action.startswith("CLEAR"):
                self.avoidance_ticks += 1

        command_ned = limit_vector(command_ned, 2.0)
        # Never allow an avoidance correction to turn into an uncontrolled
        # climb or descent. The test corridor is deliberately narrow and low.
        altitude_error = target[2] - current[2]
        command_ned = (command_ned[0], command_ned[1], -max(-0.8, min(0.8, altitude_error)))
        self.maximum_lateral_deviation = max(
            self.maximum_lateral_deviation, abs(current[1] - TAKEOFF_ENU[1])
        )
        if not (-37.0 <= current[0] <= 1.0 and -29.0 <= current[1] <= -16.0 and 2.0 <= current[2] <= 10.0):
            self._abort(f"geofence exceeded at ENU {tuple(round(value, 2) for value in current)}")
            return
        self._publish_velocity_target(command_ned)

    def _finish_flight(self) -> None:
        failures = []
        if not self.outbound_complete:
            failures.append("outbound goal was not reached")
        if self.avoidance_ticks < 5:
            failures.append("avoidance did not activate for at least five control ticks")
        if self.maximum_lateral_deviation < 0.25:
            failures.append("no measurable lateral detour was produced")
        if failures:
            self.abort_reason = "; ".join(failures)
        self._command_land()

    def _abort(self, reason: str) -> None:
        if self.complete or self.phase == "landing":
            return
        self.abort_reason = reason
        self.get_logger().error(f"closed-loop test abort: {reason}")
        if self.status and self.status.arming_state == VehicleStatus.ARMING_STATE_ARMED:
            self._command_land()
        else:
            self.complete = True

    def _command_land(self) -> None:
        self._set_phase("landing")
        self.landing_commanded = True
        self.command_ticks = 0
        self._vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("landing commanded")

    def _handle_landing(self) -> None:
        if self.status and self.status.arming_state == VehicleStatus.ARMING_STATE_DISARMED:
            self.success = self.abort_reason is None
            self.complete = True
            if self.success:
                self.get_logger().info(
                    "PASS: UAV 1 closed-loop avoidance completed; "
                    f"avoidance_ticks={self.avoidance_ticks}, "
                    f"minimum_scan={self.minimum_distance:.2f} m, "
                    f"lateral_detour={self.maximum_lateral_deviation:.2f} m"
                )
            else:
                self.get_logger().error(f"FAIL: {self.abort_reason}")
            return
        self._publish_heartbeat(position_control=False)
        self._publish_velocity_target((0.0, 0.0, 0.0))
        if self.command_ticks % 20 == 0:
            self._vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        if (
            time.monotonic() - self.phase_started > 25.0
            and self.position is not None
            and abs(self.position.z) < 0.5
            and math.sqrt(self.position.vx**2 + self.position.vy**2 + self.position.vz**2) < 0.3
        ):
            # SITL can remain armed on the pad after an abort before takeoff.
            self._vehicle_command(
                VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
                param1=0.0,
                param2=21196.0,
                from_external=False,
            )
        self.command_ticks += 1

    def _preflight_stable(self) -> bool:
        if self.position is None or self.attitude is None or self.status is None:
            return False
        if (
            self.status.arming_state != VehicleStatus.ARMING_STATE_DISARMED
            or self.status.failsafe
            or self.status.failure_detector_status != VehicleStatus.FAILURE_NONE
            or not all(
                (
                    self.position.xy_valid,
                    self.position.z_valid,
                    self.position.v_xy_valid,
                    self.position.v_z_valid,
                )
            )
        ):
            return False
        values = (
            self.position.x,
            self.position.y,
            self.position.z,
            self.position.vx,
            self.position.vy,
            self.position.vz,
        )
        if not all(math.isfinite(value) for value in values):
            return False
        speed = math.sqrt(self.position.vx**2 + self.position.vy**2 + self.position.vz**2)
        roll, pitch = quaternion_roll_pitch(tuple(self.attitude.q))
        return speed <= 0.2 and abs(roll) <= math.radians(5) and abs(pitch) <= math.radians(5)

    def _publish_heartbeat(self, *, position_control: bool) -> None:
        message = OffboardControlMode()
        message.timestamp = self._timestamp()
        message.position = position_control
        message.velocity = not position_control
        self.mode_publisher.publish(message)

    def _publish_position_target(self, target_enu: Point3D) -> None:
        target_ned = enu_to_local_ned(target_enu, SPAWN_ENU)
        message = TrajectorySetpoint()
        message.timestamp = self._timestamp()
        message.position = list(target_ned)
        message.velocity = [math.nan, math.nan, math.nan]
        message.acceleration = [math.nan, math.nan, math.nan]
        message.yaw = math.nan
        message.yawspeed = math.nan
        self.setpoint_publisher.publish(message)
        if self.position is not None:
            error = tuple(target_ned[index] - value for index, value in enumerate((self.position.x, self.position.y, self.position.z)))
            self._publish_desired_body(
                ned_vector_to_body_flu(limit_vector(error, 1.5), self.position.heading)
            )

    def _publish_velocity_target(self, velocity_ned: Point3D) -> None:
        message = TrajectorySetpoint()
        message.timestamp = self._timestamp()
        message.position = [math.nan, math.nan, math.nan]
        message.velocity = list(velocity_ned)
        message.acceleration = [math.nan, math.nan, math.nan]
        message.yaw = math.nan
        message.yawspeed = math.nan
        self.setpoint_publisher.publish(message)

    def _publish_desired_body(self, velocity_body: Point3D) -> None:
        message = Vector3Stamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "uav_1/lidar_link"
        message.vector.x, message.vector.y, message.vector.z = velocity_body
        self.desired_publisher.publish(message)

    def _vehicle_command(self, command: int, **parameters: float) -> None:
        message = VehicleCommand()
        message.timestamp = self._timestamp()
        message.command = command
        for number in range(1, 8):
            setattr(message, f"param{number}", float(parameters.get(f"param{number}", 0.0)))
        message.target_system = self.system_id
        message.target_component = 1
        message.source_system = 1
        message.source_component = 1
        message.from_external = bool(parameters.get("from_external", True))
        self.command_publisher.publish(message)

    def _current_enu(self) -> Point3D:
        return local_ned_to_enu(
            (self.position.x, self.position.y, self.position.z), SPAWN_ENU
        )

    def _distance_to(self, target: Point3D) -> float:
        return math.dist(self._current_enu(), target)

    def _settled_at(self, target: Point3D, *, radius: float, maximum_speed: float) -> bool:
        speed = math.sqrt(self.position.vx**2 + self.position.vy**2 + self.position.vz**2)
        return self._distance_to(target) <= radius and speed <= maximum_speed

    def _set_phase(self, phase: str) -> None:
        self.phase = phase
        self.phase_started = time.monotonic()

    def _timestamp(self) -> int:
        return int(self.get_clock().now().nanoseconds / 1000)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a closed-loop lidar avoidance test on UAV 1")
    parser.add_argument("--namespace", default="/mupt_px4_1")
    parser.add_argument("--timeout", type=float, default=190.0)
    arguments = parser.parse_args()
    rclpy.init()
    node = SingleUAVAvoidanceTest(arguments.namespace)
    deadline = time.monotonic() + arguments.timeout
    try:
        while rclpy.ok() and not node.complete and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if not node.complete:
            node._abort("runner timeout")
            end = time.monotonic() + 20.0
            while rclpy.ok() and not node.complete and time.monotonic() < end:
                rclpy.spin_once(node, timeout_sec=0.1)
        success = node.success
    except KeyboardInterrupt:
        node._abort("operator interrupted the test")
        success = False
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
