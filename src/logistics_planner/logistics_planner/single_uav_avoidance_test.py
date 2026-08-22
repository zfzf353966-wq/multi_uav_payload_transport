from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
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
from std_msgs.msg import Float32, Float32MultiArray, String

Point3D = tuple[float, float, float]
SPAWN_ENU: Point3D = (-34.0, -23.0, 0.13)
TAKEOFF_ENU: Point3D = (-34.0, -23.0, 6.0)
OUTBOUND_GOAL_ENU: Point3D = (-6.0, -23.0, 6.0)
RL_RETURN_APPROACH_DISTANCE = 5.0
GEOFENCE_MIN_ENU: Point3D = (-37.0, -29.0, 2.0)
GEOFENCE_MAX_ENU: Point3D = (1.0, -16.0, 10.0)
GEOFENCE_SOFT_MARGIN: Point3D = (2.0, 2.0, 1.0)
GEOFENCE_LOOKAHEAD_SECONDS = 1.5
ALTITUDE_MIN_ENU = 2.0
ALTITUDE_MAX_ENU = 10.0
ALTITUDE_SOFT_MARGIN = 1.0
EMERGENCY_ENTER_DISTANCE = 3.8
EMERGENCY_EXIT_DISTANCE = 4.2
EMERGENCY_RELEASE_TICKS = 5
EMERGENCY_HORIZONTAL_ESCAPE_SPEED = 3.0
EMERGENCY_ESCAPE_SPEED = 3.0
MINIMUM_PASS_CLEARANCE = 0.8
DOWNWARD_STOP_CLEARANCE = 2.0


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


def use_rl_for_phase(
    phase: str, rl_return: bool, distance_to_landing: float
) -> bool:
    """Keep learned avoidance away from the precision landing approach."""
    return phase == "outbound" or (
        phase == "return"
        and rl_return
        and distance_to_landing > RL_RETURN_APPROACH_DISTANCE
    )


def body_flu_to_ned(vector: Sequence[float], heading: float) -> Point3D:
    """Convert forward-left-up velocity to PX4 north-east-down."""
    forward, left, up = map(float, vector)
    cosine, sine = math.cos(heading), math.sin(heading)
    return (
        forward * cosine + left * sine,
        forward * sine - left * cosine,
        -up,
    )


def ned_vector_to_enu(vector: Sequence[float]) -> Point3D:
    north, east, down = map(float, vector)
    return (east, north, -down)


def enu_vector_to_ned(vector: Sequence[float]) -> Point3D:
    east, north, up = map(float, vector)
    return (north, east, -up)


@dataclass(frozen=True)
class ArbitrationResult:
    velocity_body: Point3D
    source: str
    reason: str


@dataclass(frozen=True)
class GeofenceGuardResult:
    velocity_enu: Point3D
    guarded_axes: tuple[str, ...]


def guard_geofence_velocity(
    position_enu: Sequence[float],
    velocity_enu: Sequence[float],
    *,
    lower: Point3D = GEOFENCE_MIN_ENU,
    upper: Point3D = GEOFENCE_MAX_ENU,
    soft_margin: Point3D = GEOFENCE_SOFT_MARGIN,
    lookahead_seconds: float = GEOFENCE_LOOKAHEAD_SECONDS,
) -> GeofenceGuardResult:
    """Project velocity so its lookahead point remains inside a soft fence.

    This is a deterministic safety filter, not a replacement controller. It
    changes only velocity components that would continue toward a boundary;
    all other PPO residual components remain untouched.
    """
    position = tuple(float(value) for value in position_enu)
    velocity = tuple(float(value) for value in velocity_enu)
    if len(position) != 3 or len(velocity) != 3:
        raise ValueError("position and velocity must contain three values")
    if lookahead_seconds <= 0.0:
        raise ValueError("lookahead_seconds must be positive")

    guarded = list(velocity)
    guarded_axes = []
    axis_names = ("x", "y", "z")
    for axis, name in enumerate(axis_names):
        soft_lower = lower[axis] + soft_margin[axis]
        soft_upper = upper[axis] - soft_margin[axis]
        minimum_velocity = (soft_lower - position[axis]) / lookahead_seconds
        maximum_velocity = (soft_upper - position[axis]) / lookahead_seconds
        projected = min(max(guarded[axis], minimum_velocity), maximum_velocity)
        if not math.isclose(projected, guarded[axis], abs_tol=1e-6):
            guarded[axis] = projected
            guarded_axes.append(name)
    return GeofenceGuardResult(
        (guarded[0], guarded[1], guarded[2]), tuple(guarded_axes)
    )


def guard_altitude_velocity(
    position_enu: Sequence[float], velocity_enu: Sequence[float]
) -> GeofenceGuardResult:
    """Apply the predictive safety filter only to altitude.

    Horizontal map bounds are deliberately unrestricted during the current
    algorithm-validation phase so avoidance can leave the nominal map area.
    """
    return guard_geofence_velocity(
        position_enu,
        velocity_enu,
        lower=(-math.inf, -math.inf, ALTITUDE_MIN_ENU),
        upper=(math.inf, math.inf, ALTITUDE_MAX_ENU),
        soft_margin=(0.0, 0.0, ALTITUDE_SOFT_MARGIN),
    )


def update_emergency_latch(
    active: bool,
    clear_ticks: int,
    minimum_scan: float,
    *,
    enter_distance: float = EMERGENCY_ENTER_DISTANCE,
    exit_distance: float = EMERGENCY_EXIT_DISTANCE,
    release_ticks: int = EMERGENCY_RELEASE_TICKS,
) -> tuple[bool, int]:
    """Latch conservative avoidance until clearance is stably restored."""
    if not math.isfinite(minimum_scan) or minimum_scan <= enter_distance:
        return True, 0
    if not active:
        return False, 0
    if minimum_scan < exit_distance:
        return True, 0
    clear_ticks += 1
    if clear_ticks >= release_ticks:
        return False, 0
    return True, clear_ticks


def emergency_escape_velocity(
    sectors: Sequence[float],
    *,
    tangent_sign: int = 1,
    influence_distance: float = EMERGENCY_EXIT_DISTANCE,
    horizontal_speed: float = EMERGENCY_HORIZONTAL_ESCAPE_SPEED,
) -> Point3D:
    """Return a locked-side horizontal escape velocity.

    Unlike the route-following potential field, emergency escape considers all
    azimuths. This matters for moving obstacles that can overtake the vehicle
    from behind. The tangential component prevents pure backward flight, while
    a caller-held tangent sign prevents left/right command reversal.
    """
    if len(sectors) != 36:
        return (0.0, 0.0, 0.0)
    repulsion_forward = 0.0
    repulsion_left = 0.0
    nearest: tuple[float, int] | None = None
    for sector in range(12):
        distances = [
            float(sectors[layer * 12 + sector])
            for layer in range(3)
            if math.isfinite(float(sectors[layer * 12 + sector]))
        ]
        if not distances:
            continue
        distance = min(distances)
        if nearest is None or distance < nearest[0]:
            nearest = (distance, sector)
        if distance >= influence_distance:
            continue
        azimuth = sector * 2.0 * math.pi / 12.0
        proximity = (influence_distance - max(0.05, distance)) / influence_distance
        weight = proximity * proximity / max(0.2, distance)
        repulsion_forward -= math.cos(azimuth) * weight
        repulsion_left -= math.sin(azimuth) * weight

    magnitude = math.hypot(repulsion_forward, repulsion_left)
    if magnitude < 1e-6 and nearest is not None and nearest[0] < influence_distance:
        azimuth = nearest[1] * 2.0 * math.pi / 12.0
        repulsion_forward = -math.cos(azimuth)
        repulsion_left = -math.sin(azimuth)
        magnitude = 1.0
    if magnitude < 1e-6:
        return (0.0, 0.0, 0.0)
    away_forward = repulsion_forward / magnitude
    away_left = repulsion_left / magnitude
    tangent_forward = -away_left * (1 if tangent_sign >= 0 else -1)
    tangent_left = away_forward * (1 if tangent_sign >= 0 else -1)
    combined_forward = 0.55 * away_forward + 0.85 * tangent_forward
    combined_left = 0.55 * away_left + 0.85 * tangent_left
    combined_norm = math.hypot(combined_forward, combined_left)
    return (
        horizontal_speed * combined_forward / combined_norm,
        horizontal_speed * combined_left / combined_norm,
        0.0,
    )


def choose_emergency_tangent_sign(
    sectors: Sequence[float], desired_velocity: Sequence[float]
) -> int:
    """Choose one side at emergency entry and keep it for the whole episode."""
    if len(sectors) != 36:
        return 1
    sector_distances = [
        min(float(sectors[layer * 12 + sector]) for layer in range(3))
        for sector in range(12)
    ]
    nearest_sector = min(range(12), key=sector_distances.__getitem__)
    azimuth = nearest_sector * 2.0 * math.pi / 12.0
    away = (-math.cos(azimuth), -math.sin(azimuth))
    positive_tangent = (-away[1], away[0])
    desired_forward, desired_left = map(float, desired_velocity[:2])
    alignment = (
        positive_tangent[0] * desired_forward + positive_tangent[1] * desired_left
    )
    if abs(alignment) > 0.1:
        return 1 if alignment >= 0.0 else -1
    left_clearance = min(
        sector_distances[sector] for sector in (1, 2, 3, 4, 5)
    )
    right_clearance = min(
        sector_distances[sector] for sector in (7, 8, 9, 10, 11)
    )
    # For a forward obstacle, negative follows the body-left tangent.
    return -1 if left_clearance >= right_clearance else 1


def directional_clearance(sectors: Sequence[float], velocity_body: Sequence[float]) -> float:
    """Return the closest 3x12 sector distance along a body-FLU velocity."""
    if len(sectors) != 36:
        return 0.0
    forward, left, up = map(float, velocity_body)
    speed = math.sqrt(forward * forward + left * left + up * up)
    if speed < 0.05:
        return math.inf
    azimuth = math.atan2(left, forward)
    sector = int(round(azimuth / (2.0 * math.pi / 12.0))) % 12
    elevation = math.atan2(up, math.hypot(forward, left))
    layer = min(range(3), key=lambda index: abs(elevation - math.radians((-12.0, 0.0, 12.0)[index])))
    candidates = []
    for layer_index in range(max(0, layer - 1), min(3, layer + 2)):
        for offset in (-1, 0, 1):
            value = float(sectors[layer_index * 12 + (sector + offset) % 12])
            if math.isfinite(value):
                candidates.append(value)
    return min(candidates, default=0.0)


def arbitrate_velocity(
    *,
    requested_source: str,
    rl_velocity: Sequence[float] | None,
    rl_age: float,
    fallback_velocity: Sequence[float],
    sectors: Sequence[float],
    maximum_speed: float = 2.0,
    emergency_distance: float = EMERGENCY_ENTER_DISTANCE,
) -> ArbitrationResult:
    """Select PPO only inside a conservative envelope, otherwise fall back."""
    fallback = limit_vector(fallback_velocity, maximum_speed)
    if requested_source != "rl":
        return ArbitrationResult(fallback, "potential", "configured")
    if rl_velocity is None or rl_age > 0.5:
        return ArbitrationResult(fallback, "fallback", "rl_stale")
    values = tuple(float(value) for value in rl_velocity)
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        return ArbitrationResult(fallback, "fallback", "rl_invalid")
    finite_sectors = [float(value) for value in sectors if math.isfinite(value)]
    if len(sectors) != 36 or not finite_sectors:
        return ArbitrationResult(fallback, "fallback", "scan_invalid")
    if min(finite_sectors) <= emergency_distance:
        return ArbitrationResult(fallback, "fallback", "emergency_clearance")
    candidate = limit_vector(values, maximum_speed)
    speed = math.sqrt(sum(value * value for value in candidate))
    required_clearance = 1.2 + speed * speed / 3.0
    if directional_clearance(sectors, candidate) < required_clearance:
        return ArbitrationResult(fallback, "fallback", "direction_blocked")
    return ArbitrationResult(candidate, "rl", "accepted")


class SingleUAVAvoidanceTest(Node):
    """Closed-loop lidar avoidance test that commands only PX4 vehicle one."""

    def __init__(
        self,
        namespace: str = "/mupt_px4_1",
        test_label: str = "static",
        avoidance_source: str = "potential",
        rl_return: bool = False,
        output_dir: str | None = None,
    ) -> None:
        super().__init__("single_uav_closed_loop_avoidance_test")
        self.namespace = namespace.rstrip("/")
        self.test_label = test_label
        self.avoidance_source = avoidance_source
        self.rl_return = rl_return
        self.system_id = 2

        self.output_dir = Path(output_dir).resolve() if output_dir else None
        self.trace_file = None
        self.trace_writer = None
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.trace_file = (self.output_dir / "control_trace.csv").open(
                "w", encoding="utf-8", newline=""
            )
            self.trace_writer = csv.DictWriter(
                self.trace_file,
                fieldnames=(
                    "elapsed_s", "phase", "x_enu", "y_enu", "z_enu",
                    "vx_ned", "vy_ned", "vz_ned", "minimum_scan_m",
                    "down_clearance_m",
                    "desired_forward", "desired_left", "desired_up",
                    "potential_forward", "potential_left", "potential_up",
                    "rl_forward", "rl_left", "rl_up",
                    "command_forward", "command_left", "command_up",
                    "source", "reason",
                ),
            )
            self.trace_writer.writeheader()

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
        self.relative_goal_publisher = self.create_publisher(
            Vector3Stamped, "/mupt/uav_1/planner/relative_goal_body", sensor_qos
        )
        self.arbiter_publisher = self.create_publisher(
            String, "/mupt/uav_1/controller/arbiter", sensor_qos
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
        self.create_subscription(
            Float32,
            "/mupt/uav_1/observation/down_clearance",
            self._down_clearance_callback,
            sensor_qos,
        )
        self.create_subscription(
            Vector3Stamped,
            "/mupt/uav_1/rl_shadow/suggested_velocity_body",
            self._rl_suggestion_callback,
            sensor_qos,
        )

        self.position: VehicleLocalPosition | None = None
        self.attitude: VehicleAttitude | None = None
        self.status: VehicleStatus | None = None
        self.suggested_body: Point3D | None = None
        self.suggestion_time = 0.0
        self.rl_suggested_body: Point3D | None = None
        self.rl_suggestion_time = 0.0
        self.latest_sectors: tuple[float, ...] = tuple([20.0] * 36)
        self.down_clearance = math.inf
        self.down_clearance_time = 0.0
        self.minimum_down_clearance = math.inf
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
        self.rl_control_ticks = 0
        self.fallback_ticks = 0
        self.traditional_ticks = 0
        self.geofence_guard_ticks = 0
        self.emergency_latched = False
        self.emergency_clear_ticks = 0
        self.emergency_latch_ticks = 0
        self.emergency_tangent_sign = 0
        self.downward_guard_ticks = 0
        self.last_arbiter_state = ""
        self.minimum_distance = math.inf
        self.contact_risk_ticks = 0
        self.maximum_contact_risk_ticks = 0
        self.path_length = 0.0
        self.previous_trace_position: Point3D | None = None
        self.near_ticks = {"rl": 0, "fallback": 0, "potential": 0}
        self.critical_ticks = {"rl": 0, "fallback": 0, "potential": 0}
        self.control_samples = 0
        self.summary_written = False
        self.maximum_lateral_deviation = 0.0
        self.abort_reason: str | None = None
        self.complete = False
        self.success = False
        self.timer = self.create_timer(0.1, self._tick)
        self.get_logger().info(
            f"closed-loop test bound exclusively to {self.namespace}; "
            f"source={self.avoidance_source}; UAV 2 and UAV 3 remain disarmed"
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

    def _rl_suggestion_callback(self, message: Vector3Stamped) -> None:
        values = (message.vector.x, message.vector.y, message.vector.z)
        if all(math.isfinite(value) for value in values):
            self.rl_suggested_body = values
            self.rl_suggestion_time = time.monotonic()

    def _sectors_callback(self, message: Float32MultiArray) -> None:
        if len(message.data) == 36 and all(math.isfinite(value) for value in message.data):
            self.latest_sectors = tuple(float(value) for value in message.data)
        finite = [float(value) for value in message.data if math.isfinite(value) and value > 0.05]
        if finite:
            self.minimum_distance = min(self.minimum_distance, min(finite))

    def _down_clearance_callback(self, message: Float32) -> None:
        value = float(message.data)
        if math.isfinite(value) and value > 0.0:
            self.down_clearance = value
            self.down_clearance_time = time.monotonic()

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
        relative_enu = tuple(target[index] - current[index] for index in range(3))
        relative_ned = (relative_enu[1], relative_enu[0], -relative_enu[2])
        self._publish_relative_goal_body(ned_vector_to_body_flu(relative_ned, heading))

        now = time.monotonic()
        if now - self.down_clearance_time <= 0.5:
            self.minimum_down_clearance = min(
                self.minimum_down_clearance, self.down_clearance
            )
        if self.suggested_body is None or now - self.suggestion_time > 0.5:
            if now - self.phase_started > 1.5:
                self._abort("avoidance suggestion became stale for more than 0.5 s")
                return
            potential_body = desired_body
        else:
            potential_body = self.suggested_body
            if now - self.action_time <= 0.5 and self.action and not self.action.startswith("CLEAR"):
                self.avoidance_ticks += 1

        finite_scan = [value for value in self.latest_sectors if math.isfinite(value)]
        minimum_scan = min(finite_scan, default=math.inf)
        if minimum_scan < MINIMUM_PASS_CLEARANCE:
            self.contact_risk_ticks += 1
            self.maximum_contact_risk_ticks = max(
                self.maximum_contact_risk_ticks, self.contact_risk_ticks
            )
        else:
            self.contact_risk_ticks = 0
        was_emergency_latched = self.emergency_latched
        self.emergency_latched, self.emergency_clear_ticks = update_emergency_latch(
            self.emergency_latched, self.emergency_clear_ticks, minimum_scan
        )
        if self.emergency_latched and not was_emergency_latched:
            self.emergency_tangent_sign = choose_emergency_tangent_sign(
                self.latest_sectors, desired_body
            )
        elif not self.emergency_latched:
            self.emergency_tangent_sign = 0
        if self.emergency_latched:
            self.emergency_latch_ticks += 1
            decision = ArbitrationResult(
                emergency_escape_velocity(
                    self.latest_sectors,
                    tangent_sign=self.emergency_tangent_sign,
                ),
                "fallback",
                "emergency_horizontal_escape",
            )
        else:
            decision = arbitrate_velocity(
                requested_source=(
                    self.avoidance_source
                    if use_rl_for_phase(
                        self.phase, self.rl_return, self._distance_to(TAKEOFF_ENU)
                    )
                    else "potential"
                ),
                rl_velocity=self.rl_suggested_body,
                rl_age=now - self.rl_suggestion_time,
                fallback_velocity=potential_body,
                sectors=self.latest_sectors,
            )
        speed_limit = EMERGENCY_ESCAPE_SPEED if self.emergency_latched else 2.0
        command_ned = limit_vector(
            body_flu_to_ned(decision.velocity_body, heading), speed_limit
        )
        # Avoidance remains horizontal; altitude is independently stabilized.
        altitude_error = target[2] - current[2]
        command_ned = (
            command_ned[0],
            command_ned[1],
            -max(-0.8, min(0.8, altitude_error)),
        )
        decision_reason = decision.reason
        down_data_fresh = now - self.down_clearance_time <= 0.5
        if command_ned[2] > 0.0 and (
            not down_data_fresh or self.down_clearance < DOWNWARD_STOP_CLEARANCE
        ):
            command_ned = (command_ned[0], command_ned[1], 0.0)
            self.downward_guard_ticks += 1
            decision_reason = "downward_guarded"
        guard = guard_altitude_velocity(current, ned_vector_to_enu(command_ned))
        command_ned = enu_vector_to_ned(guard.velocity_enu)
        if guard.guarded_axes:
            self.geofence_guard_ticks += 1
            decision_reason = "geofence_guarded:" + ",".join(guard.guarded_axes)
        decision = ArbitrationResult(
            ned_vector_to_body_flu(command_ned, heading),
            decision.source,
            decision_reason,
        )

        if decision.source == "rl":
            self.rl_control_ticks += 1
        elif decision.source == "fallback":
            self.fallback_ticks += 1
        elif decision.source == "potential":
            self.traditional_ticks += 1
        arbiter_state = f"{decision.source}:{decision.reason}"
        arbiter_message = String()
        arbiter_message.data = (
            f"source={decision.source} reason={decision.reason} "
            f"rl_ticks={self.rl_control_ticks} fallback_ticks={self.fallback_ticks} "
            f"traditional_ticks={self.traditional_ticks} "
            f"geofence_guard_ticks={self.geofence_guard_ticks} "
            f"emergency_latch_ticks={self.emergency_latch_ticks} "
            f"downward_guard_ticks={self.downward_guard_ticks}"
        )
        self.arbiter_publisher.publish(arbiter_message)
        if arbiter_state != self.last_arbiter_state:
            self.get_logger().info(f"arbiter {arbiter_message.data}")
            self.last_arbiter_state = arbiter_state

        self._record_control_sample(
            current=current,
            desired_body=desired_body,
            potential_body=potential_body,
            rl_body=self.rl_suggested_body,
            decision=decision,
        )
        self.maximum_lateral_deviation = max(
            self.maximum_lateral_deviation, abs(current[1] - TAKEOFF_ENU[1])
        )
        if not ALTITUDE_MIN_ENU <= current[2] <= ALTITUDE_MAX_ENU:
            self._abort(
                f"altitude safety range exceeded at ENU "
                f"{tuple(round(value, 2) for value in current)}"
            )
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
        if self.avoidance_source == "rl" and self.rl_control_ticks < 20:
            failures.append("PPO was not accepted for at least twenty control ticks")
        if self.maximum_contact_risk_ticks >= 2:
            failures.append(
                f"lidar clearance stayed below the {MINIMUM_PASS_CLEARANCE:.2f} m "
                f"no-contact gate for {self.maximum_contact_risk_ticks} consecutive ticks"
            )
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
            self._write_summary()
            if self.success:
                self.get_logger().info(
                    f"PASS: UAV 1 {self.test_label} closed-loop avoidance completed; "
                    f"avoidance_ticks={self.avoidance_ticks}, "
                    f"minimum_scan={self.minimum_distance:.2f} m, "
                    f"lateral_detour={self.maximum_lateral_deviation:.2f} m, "
                    f"rl_ticks={self.rl_control_ticks}, fallback_ticks={self.fallback_ticks}, "
                    f"traditional_ticks={self.traditional_ticks}, "
                    f"geofence_guard_ticks={self.geofence_guard_ticks}, "
                    f"emergency_latch_ticks={self.emergency_latch_ticks}, "
                    f"downward_guard_ticks={self.downward_guard_ticks}"
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

    def _record_control_sample(
        self,
        *,
        current: Point3D,
        desired_body: Point3D,
        potential_body: Point3D,
        rl_body: Point3D | None,
        decision: ArbitrationResult,
    ) -> None:
        finite_scan = [value for value in self.latest_sectors if math.isfinite(value)]
        minimum_scan = min(finite_scan, default=math.inf)
        self.control_samples += 1
        if minimum_scan <= 6.0:
            self.near_ticks[decision.source] += 1
        if minimum_scan <= 4.0:
            self.critical_ticks[decision.source] += 1
        if self.previous_trace_position is not None:
            self.path_length += math.dist(current, self.previous_trace_position)
        self.previous_trace_position = current
        if self.trace_writer is None:
            return
        rl = rl_body if rl_body is not None else (math.nan, math.nan, math.nan)
        self.trace_writer.writerow(
            {
                "elapsed_s": f"{time.monotonic() - self.started:.3f}",
                "phase": self.phase,
                "x_enu": f"{current[0]:.4f}", "y_enu": f"{current[1]:.4f}",
                "z_enu": f"{current[2]:.4f}", "vx_ned": f"{self.position.vx:.4f}",
                "vy_ned": f"{self.position.vy:.4f}", "vz_ned": f"{self.position.vz:.4f}",
                "minimum_scan_m": f"{minimum_scan:.4f}",
                "down_clearance_m": f"{self.down_clearance:.4f}",
                "desired_forward": f"{desired_body[0]:.4f}",
                "desired_left": f"{desired_body[1]:.4f}", "desired_up": f"{desired_body[2]:.4f}",
                "potential_forward": f"{potential_body[0]:.4f}",
                "potential_left": f"{potential_body[1]:.4f}", "potential_up": f"{potential_body[2]:.4f}",
                "rl_forward": f"{rl[0]:.4f}", "rl_left": f"{rl[1]:.4f}",
                "rl_up": f"{rl[2]:.4f}", "command_forward": f"{decision.velocity_body[0]:.4f}",
                "command_left": f"{decision.velocity_body[1]:.4f}",
                "command_up": f"{decision.velocity_body[2]:.4f}",
                "source": decision.source, "reason": decision.reason,
            }
        )

    def _write_summary(self) -> None:
        if self.summary_written or self.output_dir is None:
            return
        self.summary_written = True
        near_total = sum(self.near_ticks.values())
        critical_total = sum(self.critical_ticks.values())
        summary = {
            "test_label": self.test_label,
            "configured_source": self.avoidance_source,
            "success": self.success,
            "abort_reason": self.abort_reason,
            "duration_s": round(time.monotonic() - self.started, 3),
            "path_length_m": round(self.path_length, 3),
            "minimum_scan_m": round(self.minimum_distance, 3),
            "minimum_down_clearance_m": round(self.minimum_down_clearance, 3),
            "maximum_contact_risk_ticks": self.maximum_contact_risk_ticks,
            "maximum_lateral_deviation_m": round(self.maximum_lateral_deviation, 3),
            "control_samples": self.control_samples,
            "all_ticks": {
                "rl": self.rl_control_ticks,
                "fallback": self.fallback_ticks,
                "potential": self.traditional_ticks,
            },
            "geofence_guard_ticks": self.geofence_guard_ticks,
            "emergency_latch_ticks": self.emergency_latch_ticks,
            "downward_guard_ticks": self.downward_guard_ticks,
            "within_6m_ticks": self.near_ticks,
            "within_4m_ticks": self.critical_ticks,
            "rl_share_within_6m": round(self.near_ticks["rl"] / near_total, 4) if near_total else 0.0,
            "rl_share_within_4m": round(self.critical_ticks["rl"] / critical_total, 4) if critical_total else 0.0,
        }
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        if self.trace_file is not None:
            self.trace_file.flush()
        self.get_logger().info(f"evaluation files written to {self.output_dir}")

    def close_evaluation(self) -> None:
        if self.trace_file is not None and not self.trace_file.closed:
            self.trace_file.close()

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

    def _publish_relative_goal_body(self, relative_goal_body: Point3D) -> None:
        message = Vector3Stamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "uav_1/lidar_link"
        message.vector.x, message.vector.y, message.vector.z = relative_goal_body
        self.relative_goal_publisher.publish(message)

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
    parser.add_argument("--test-label", default="static")
    parser.add_argument("--avoidance-source", default="potential", choices=("potential", "rl"))
    parser.add_argument(
        "--rl-return",
        action="store_true",
        help="Use the RL policy on the return leg as well as the outbound leg",
    )
    parser.add_argument("--timeout", type=float, default=190.0)
    parser.add_argument("--output-dir", default="")
    arguments = parser.parse_args()
    rclpy.init()
    node = SingleUAVAvoidanceTest(
        arguments.namespace,
        arguments.test_label,
        arguments.avoidance_source,
        arguments.rl_return,
        arguments.output_dir or None,
    )
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
        node.close_evaluation()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
