from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

# Ubuntu's Gazebo Python messages are generated for protobuf 3.x, while the
# training virtual environment may contain a newer protobuf runtime.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.pose_pb2 import Pose
from gz.transport13 import Node as GazeboTransportNode
import rclpy
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
)
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from .config import load_world_config
from .models import Point3D
from .trajectory import TrajectorySample, sample_at, staging_position


@dataclass
class VehicleMission:
    namespace: str
    system_id: int
    spawn_enu: Point3D
    staging_enu: Point3D
    trajectory: tuple[TrajectorySample, ...]
    trajectory_start_delay: float
    stops: list[dict]
    position: VehicleLocalPosition | None = None
    status: VehicleStatus | None = None
    phase: str = "prestream"
    stop_index: int = 0
    onboard_tasks: list[str] = field(default_factory=list)
    payload_update_ticks: int = 0
    warmup_ticks: int = 0
    dwell_ticks: int = 0
    command_retries: int = 0
    prestream_announced: bool = False
    mode_command_sent: bool = False
    arm_command_sent: bool = False
    hold_enu: Point3D | None = None
    landed: bool = False
    landing_commanded: bool = False


class GazeboPackageManager:
    """Moves visual package models without adding payload dynamics yet."""

    def __init__(self, world: str, plan: dict, config_path: Path) -> None:
        self._service = f"/world/{world}/set_pose"
        self._node = GazeboTransportNode()
        self._pickup_positions, self._delivery_positions = payload_locations(config_path)
        self._tasks = {task["id"]: task for task in plan["tasks"]}
        self._reset_packages()

    def _set_pose(self, task_id: str, point: Point3D) -> bool:
        request = Pose()
        request.name = f"package_{task_id.rsplit('_', 1)[-1]}"
        request.position.x, request.position.y, request.position.z = point
        success, response = self._node.request(self._service, request, Pose, Boolean, 100)
        return bool(success and response.data)

    def _reset_packages(self) -> None:
        grouped: dict[str, list[str]] = {}
        for task in self._tasks.values():
            grouped.setdefault(task["pickup"], []).append(task["id"])
        for pickup_id, task_ids in grouped.items():
            for index, task_id in enumerate(sorted(task_ids)):
                self._set_pose(task_id, offset_point(self._pickup_positions[pickup_id], index, len(task_ids)))

    def carry(self, task_ids: list[str], vehicle_position: Point3D) -> None:
        for index, task_id in enumerate(task_ids):
            x_offset = (index - (len(task_ids) - 1) / 2.0) * 0.45
            self._set_pose(task_id, (vehicle_position[0] + x_offset, vehicle_position[1], vehicle_position[2] - 0.65))

    def deliver(self, task_id: str) -> None:
        task = self._tasks[task_id]
        same_destination = sorted(
            candidate["id"] for candidate in self._tasks.values() if candidate["delivery"] == task["delivery"]
        )
        index = same_destination.index(task_id)
        target = offset_point(self._delivery_positions[task["delivery"]], index, len(same_destination))
        self._set_pose(task_id, target)


class MultiUAVOffboard(Node):
    def __init__(self, plan_path: Path, config_path: Path) -> None:
        super().__init__("multi_uav_offboard_mission")
        config = load_world_config(config_path)
        with plan_path.open("r", encoding="utf-8") as stream:
            plan = json.load(stream)
        self.package_manager = GazeboPackageManager(plan["world"], plan, config_path)
        route_by_uav = {route["uav"]: route for route in plan["routes"]}

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.missions: list[VehicleMission] = []
        self.publishers_by_namespace: dict[str, tuple] = {}
        for index, (uav, spawn) in enumerate(zip(config.uavs, config_raw_spawns(config_path))):
            namespace = f"/px4_{index + 1}"
            staging = staging_position(spawn, index)
            route_data = route_by_uav[uav.id]
            if not route_data.get("trajectory"):
                raise ValueError(
                    "Plan has no continuous trajectory. Run traditional_baseline again before offboard_mission."
                )
            trajectory = tuple(
                TrajectorySample(
                    float(sample["t"]),
                    tuple(float(value) for value in sample["position"]),
                    tuple(float(value) for value in sample["velocity"]),
                    tuple(float(value) for value in sample["acceleration"]),
                )
                for sample in route_data["trajectory"]
            )
            stops = [stop for stop in route_data["stops"] if stop["kind"] != "depot"]
            mission = VehicleMission(
                namespace,
                index + 2,
                spawn,
                staging,
                trajectory,
                float(route_data["trajectory_start_delay_s"]),
                stops,
            )
            self.missions.append(mission)
            mode_pub = self.create_publisher(OffboardControlMode, namespace + "/fmu/in/offboard_control_mode", qos)
            setpoint_pub = self.create_publisher(TrajectorySetpoint, namespace + "/fmu/in/trajectory_setpoint", qos)
            command_pub = self.create_publisher(VehicleCommand, namespace + "/fmu/in/vehicle_command", qos)
            self.publishers_by_namespace[namespace] = (mode_pub, setpoint_pub, command_pub)
            self.create_subscription(
                VehicleLocalPosition,
                namespace + "/fmu/out/vehicle_local_position_v1",
                lambda message, vehicle=mission: setattr(vehicle, "position", message),
                qos,
            )
            self.create_subscription(
                VehicleStatus,
                namespace + "/fmu/out/vehicle_status_v1",
                lambda message, vehicle=mission: setattr(vehicle, "status", message),
                qos,
            )

        self.schedule_epoch: float | None = None
        self.schedule_time_shift = 0.0
        self.schedule_last_tick = time.monotonic()
        self.minimum_actual_separation = math.inf
        self.maximum_concurrent_airborne = 0
        self.safety_hold_ticks = 0
        self.timer = self.create_timer(0.1, self._tick)

    @property
    def complete(self) -> bool:
        return all(mission.landed for mission in self.missions)

    @property
    def ready(self) -> bool:
        return all(mission.position is not None and mission.status is not None for mission in self.missions)

    def _tick(self) -> None:
        now = time.monotonic()
        schedule_tick_duration = min(0.2, max(0.0, now - self.schedule_last_tick))
        self.schedule_last_tick = now
        pause_schedule = False
        safety_holds = self._runtime_safety_holds()
        self.safety_hold_ticks += len(safety_holds)
        for mission in self.missions:
            if mission.landed:
                continue
            if mission.landing_commanded:
                if mission.status and mission.status.arming_state == VehicleStatus.ARMING_STATE_DISARMED:
                    mission.landed = True
                    self.get_logger().info(f"{mission.namespace}: landed and disarmed")
                elif mission.command_retries % 20 == 0:
                    self._vehicle_command(mission, VehicleCommand.VEHICLE_CMD_NAV_LAND)
                mission.command_retries += 1
                continue
            self._publish_heartbeat(mission)
            if mission.position is None or mission.status is None:
                continue
            self._update_package_state(mission)
            if mission.phase == "prestream":
                self._publish_position_target(mission, mission.staging_enu)
                self._handle_offboard_entry(mission)
                if (
                    mission.status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
                    and mission.status.arming_state == VehicleStatus.ARMING_STATE_ARMED
                ):
                    mission.phase = "takeoff"
                continue

            if mission.phase == "takeoff":
                self._publish_position_target(mission, mission.staging_enu)
                if self._settled_at(mission, mission.staging_enu, radius=0.6, maximum_speed=0.6):
                    mission.phase = "staged"
                    mission.dwell_ticks = 0
                    self.get_logger().info(f"{mission.namespace}: staging point reached; waiting for schedule")
                continue

            if mission.phase == "staged":
                self._publish_position_target(mission, mission.staging_enu)
                schedule_time = (
                    now - self.schedule_epoch - self.schedule_time_shift
                    if self.schedule_epoch is not None
                    else -math.inf
                )
                if schedule_time >= mission.trajectory_start_delay:
                    mission.phase = "tracking"
                    self.get_logger().info(
                        f"{mission.namespace}: continuous trajectory started "
                        f"(delay={mission.trajectory_start_delay:.1f} s)"
                    )
                continue

            if mission.phase != "tracking" or self.schedule_epoch is None:
                continue

            elapsed = max(
                0.0,
                now - self.schedule_epoch - self.schedule_time_shift - mission.trajectory_start_delay,
            )
            desired = sample_at(mission.trajectory, elapsed)
            actual_enu = local_ned_to_enu(
                (mission.position.x, mission.position.y, mission.position.z), mission.spawn_enu
            )
            tracking_error = math.dist(actual_enu, desired.position)
            must_hold = mission.namespace in safety_holds
            # Pause trajectory time if tracking error grows, allowing PX4 to
            # catch the moving reference instead of letting it run away.
            if must_hold:
                pause_schedule = True
                mission.hold_enu = mission.hold_enu or actual_enu
                self._publish_position_target(mission, mission.hold_enu)
                continue
            mission.hold_enu = None
            if tracking_error > 2.0:
                pause_schedule = True
            self._publish_trajectory_target(mission, desired)

            if elapsed < mission.trajectory[-1].time:
                continue
            if not self._settled_at(mission, mission.staging_enu, radius=0.25, maximum_speed=0.25):
                continue
            mission.dwell_ticks += 1
            if mission.dwell_ticks < 15:
                continue
            mission.landing_commanded = True
            mission.phase = "landing"
            mission.command_retries = 0
            self._vehicle_command(mission, VehicleCommand.VEHICLE_CMD_NAV_LAND)
            self.get_logger().info(f"{mission.namespace}: trajectory complete, landing")

        if self.schedule_epoch is None and all(
            mission.phase in ("staged", "tracking", "landing") or mission.landed
            for mission in self.missions
        ):
            self.schedule_epoch = now + 1.0
            delays = ", ".join(
                f"{mission.namespace}={mission.trajectory_start_delay:.1f}s" for mission in self.missions
            )
            self.get_logger().info(f"all UAVs staged; parallel schedule armed: {delays}")
        elif pause_schedule:
            # A shared virtual clock preserves all offline separation guarantees
            # when any vehicle needs extra time to catch its reference.
            self.schedule_time_shift += schedule_tick_duration

    def _handle_offboard_entry(self, mission: VehicleMission) -> None:
        mission.warmup_ticks += 1
        if not mission.prestream_announced:
            mission.prestream_announced = True
            self.get_logger().info(
                f"{mission.namespace}: pre-streaming Offboard setpoints for 2.0 s before arming"
            )
        # Enter Offboard first, then arm on a later tick. Sending both
        # commands together can briefly arm in the previous mode and let
        # the no-RC failsafe select RTL before Offboard becomes active.
        offboard_active = mission.status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
        should_request_mode = mission.warmup_ticks >= 20 and not offboard_active and (
            not mission.mode_command_sent or mission.warmup_ticks % 20 == 0
        )
        if should_request_mode:
            self._vehicle_command(
                mission,
                VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
                param1=1.0,
                param2=6.0,
            )
            if not mission.mode_command_sent:
                mission.mode_command_sent = True
                self.get_logger().info(f"{mission.namespace}: Offboard mode requested")

        should_arm = offboard_active and mission.status.arming_state == VehicleStatus.ARMING_STATE_DISARMED and (
            not mission.arm_command_sent or mission.warmup_ticks % 20 == 0
        )
        if should_arm:
            self._vehicle_command(
                mission,
                VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
                param1=1.0,
                # PX4 force-arm magic, allowed here only for SITL without a GCS heartbeat.
                param2=21196.0,
                from_external=False,
            )
            if not mission.arm_command_sent:
                mission.arm_command_sent = True
                self.get_logger().info(f"{mission.namespace}: Offboard confirmed; arm requested")


    def _settled_at(
        self, mission: VehicleMission, target_enu: Point3D, radius: float, maximum_speed: float
    ) -> bool:
        if mission.position is None:
            return False
        target_ned = enu_to_local_ned(target_enu, mission.spawn_enu)
        error = math.dist((mission.position.x, mission.position.y, mission.position.z), target_ned)
        velocity = (mission.position.vx, mission.position.vy, mission.position.vz)
        speed = math.sqrt(sum(value**2 for value in velocity)) if all(map(math.isfinite, velocity)) else math.inf
        return error <= radius and speed <= maximum_speed

    def _runtime_safety_holds(self) -> set[str]:
        holds: set[str] = set()
        armed = [
            mission
            for mission in self.missions
            if mission.position is not None
            and mission.status is not None
            and mission.status.arming_state == VehicleStatus.ARMING_STATE_ARMED
        ]
        self.maximum_concurrent_airborne = max(self.maximum_concurrent_airborne, len(armed))
        for first_index, first in enumerate(armed):
            first_position = local_ned_to_enu(
                (first.position.x, first.position.y, first.position.z), first.spawn_enu
            )
            for second in armed[first_index + 1 :]:
                second_position = local_ned_to_enu(
                    (second.position.x, second.position.y, second.position.z), second.spawn_enu
                )
                self.minimum_actual_separation = min(
                    self.minimum_actual_separation, math.dist(first_position, second_position)
                )

        airborne = [
            mission
            for mission in self.missions
            if mission.position is not None and mission.phase in ("takeoff", "staged", "tracking")
        ]
        for first_index, first in enumerate(airborne):
            first_position = local_ned_to_enu(
                (first.position.x, first.position.y, first.position.z), first.spawn_enu
            )
            first_velocity = ned_vector_to_enu((first.position.vx, first.position.vy, first.position.vz))
            for second in airborne[first_index + 1 :]:
                second_position = local_ned_to_enu(
                    (second.position.x, second.position.y, second.position.z), second.spawn_enu
                )
                second_velocity = ned_vector_to_enu((second.position.vx, second.position.vy, second.position.vz))
                relative_position = tuple(b - a for a, b in zip(first_position, second_position))
                relative_velocity = tuple(b - a for a, b in zip(first_velocity, second_velocity))
                velocity_squared = sum(value * value for value in relative_velocity)
                closest_time = 0.0
                if velocity_squared > 1e-6:
                    closest_time = max(
                        0.0,
                        min(
                            1.5,
                            -sum(a * b for a, b in zip(relative_position, relative_velocity))
                            / velocity_squared,
                        ),
                    )
                closest_vector = tuple(
                    relative_position[axis] + closest_time * relative_velocity[axis]
                    for axis in range(3)
                )
                closest_distance = math.sqrt(sum(value * value for value in closest_vector))
                if closest_distance < 2.5:
                    lower_priority = max(
                        (first, second),
                        key=lambda item: (item.trajectory_start_delay, item.system_id),
                    )
                    holds.add(lower_priority.namespace)
        return holds

    def _update_package_state(self, mission: VehicleMission) -> None:
        world_position = local_ned_to_enu(
            (mission.position.x, mission.position.y, mission.position.z), mission.spawn_enu
        )
        # Several pickups or deliveries can share one roof, so consume every
        # consecutive stop that is currently within the service radius.
        while mission.stop_index < len(mission.stops):
            stop = mission.stops[mission.stop_index]
            if math.dist(world_position, tuple(stop["position"])) > 1.5:
                break
            task_id = stop["task_id"]
            if stop["kind"] == "pickup":
                if task_id not in mission.onboard_tasks:
                    mission.onboard_tasks.append(task_id)
                self.get_logger().info(f"{mission.namespace}: picked up {task_id}")
            else:
                if task_id in mission.onboard_tasks:
                    mission.onboard_tasks.remove(task_id)
                self.package_manager.deliver(task_id)
                self.get_logger().info(f"{mission.namespace}: delivered {task_id}")
            mission.stop_index += 1

        mission.payload_update_ticks += 1
        if mission.onboard_tasks and mission.payload_update_ticks % 2 == 0:
            self.package_manager.carry(mission.onboard_tasks, world_position)

    def _publish_heartbeat(self, mission: VehicleMission) -> None:
        message = OffboardControlMode()
        message.timestamp = self._timestamp()
        message.position = True
        self.publishers_by_namespace[mission.namespace][0].publish(message)

    def _publish_position_target(self, mission: VehicleMission, target_enu: Point3D) -> None:
        self._publish_setpoint(
            mission,
            enu_to_local_ned(target_enu, mission.spawn_enu),
            (math.nan, math.nan, math.nan),
            (math.nan, math.nan, math.nan),
        )

    def _publish_trajectory_target(self, mission: VehicleMission, sample: TrajectorySample) -> None:
        self._publish_setpoint(
            mission,
            enu_to_local_ned(sample.position, mission.spawn_enu),
            enu_vector_to_ned(sample.velocity),
            enu_vector_to_ned(sample.acceleration),
        )

    def _publish_setpoint(
        self, mission: VehicleMission, target_ned: Point3D, velocity_ned: Point3D, acceleration_ned: Point3D
    ) -> None:
        message = TrajectorySetpoint()
        message.timestamp = self._timestamp()
        message.position = list(target_ned)
        message.velocity = list(velocity_ned)
        message.acceleration = list(acceleration_ned)
        message.yaw = math.nan
        message.yawspeed = math.nan
        self.publishers_by_namespace[mission.namespace][1].publish(message)

    def _vehicle_command(self, mission: VehicleMission, command: int, **parameters: float) -> None:
        message = VehicleCommand()
        message.timestamp = self._timestamp()
        message.command = command
        for number in range(1, 8):
            setattr(message, f"param{number}", float(parameters.get(f"param{number}", 0.0)))
        message.target_system = mission.system_id
        message.target_component = 1
        message.source_system = 1
        message.source_component = 1
        message.from_external = bool(parameters.get("from_external", True))
        self.publishers_by_namespace[mission.namespace][2].publish(message)

    def _timestamp(self) -> int:
        return int(self.get_clock().now().nanoseconds / 1000)


def config_raw_spawns(config_path: Path) -> list[Point3D]:
    import yaml

    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    return [tuple(float(value) for value in point) for point in raw["depot"]["uav_spawn_points"]]


def payload_locations(config_path: Path) -> tuple[dict[str, Point3D], dict[str, Point3D]]:
    import yaml

    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    pickups = {
        item["id"]: tuple(float(value) for value in item["package_position"])
        for item in raw["pickup_points"]
    }
    deliveries = {
        item["id"]: tuple(float(value) for value in item["marker_position"])
        for item in raw["delivery_points"]
    }
    return pickups, deliveries


def offset_point(point: Point3D, index: int, count: int) -> Point3D:
    spacing = 0.6
    return (point[0] + (index - (count - 1) / 2.0) * spacing, point[1], point[2] + 0.13)


def enu_to_local_ned(target: Point3D, origin: Point3D) -> Point3D:
    return (
        target[1] - origin[1],
        target[0] - origin[0],
        -(target[2] - origin[2]),
    )


def local_ned_to_enu(position: Point3D, origin: Point3D) -> Point3D:
    return (
        origin[0] + position[1],
        origin[1] + position[0],
        origin[2] - position[2],
    )


def enu_vector_to_ned(vector: Point3D) -> Point3D:
    return (vector[1], vector[0], -vector[2])


def ned_vector_to_enu(vector: Point3D) -> Point3D:
    return (vector[1], vector[0], -vector[2])


def main() -> None:
    root = Path(os.environ.get("MUPT_ROOT", Path.cwd()))
    parser = argparse.ArgumentParser(description="Execute a traditional plan on three PX4 SITL vehicles")
    parser.add_argument("--plan", type=Path, default=root / "runs/traditional_baseline_ros/plan.json")
    parser.add_argument("--config", type=Path, default=root / "config/logistics_city.yaml")
    parser.add_argument("--timeout", type=float, default=900.0)
    arguments = parser.parse_args()

    rclpy.init()
    node = MultiUAVOffboard(arguments.plan, arguments.config)
    try:
        discovery_deadline = time.monotonic() + 180.0
        while rclpy.ok() and not node.ready and time.monotonic() < discovery_deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
        if not node.ready:
            raise TimeoutError("Did not receive all three PX4 status and position topics")
        print("All three PX4 vehicles discovered; continuous parallel mission preparing", flush=True)
        deadline = time.monotonic() + arguments.timeout
        while rclpy.ok() and not node.complete and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
        if not node.complete:
            raise TimeoutError("Offboard mission did not complete before the timeout")
        print(
            "PASS: all three PX4 missions completed and landed; "
            f"max_concurrent={node.maximum_concurrent_airborne}, "
            f"minimum_separation={node.minimum_actual_separation:.2f} m, "
            f"safety_hold_ticks={node.safety_hold_ticks}"
        )
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
