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


@dataclass
class VehicleMission:
    namespace: str
    system_id: int
    spawn_enu: Point3D
    waypoints_enu: list[Point3D]
    stops: list[dict]
    position: VehicleLocalPosition | None = None
    status: VehicleStatus | None = None
    active: bool = False
    waypoint_index: int = 0
    stop_index: int = 0
    onboard_tasks: list[str] = field(default_factory=list)
    payload_update_ticks: int = 0
    warmup_ticks: int = 0
    dwell_ticks: int = 0
    command_retries: int = 0
    prestream_announced: bool = False
    mode_command_sent: bool = False
    arm_command_sent: bool = False
    landed: bool = False
    landing_commanded: bool = False
    started_at: float = field(default_factory=time.monotonic)


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
            raw_path = [tuple(point) for point in route_by_uav[uav.id]["path"]]
            staging = (spawn[0], spawn[1], 3.0)
            if len(raw_path) >= 2:
                raw_path[0] = staging
                raw_path[-1] = staging
            else:
                raw_path = [staging]
            stops = [stop for stop in route_by_uav[uav.id]["stops"] if stop["kind"] != "depot"]
            mission = VehicleMission(namespace, index + 2, spawn, raw_path, stops)
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

        self.active_mission_index = 0
        self.missions[0].active = True
        self.timer = self.create_timer(0.1, self._tick)

    @property
    def complete(self) -> bool:
        return all(mission.landed for mission in self.missions)

    @property
    def ready(self) -> bool:
        return all(mission.position is not None and mission.status is not None for mission in self.missions)

    def _tick(self) -> None:
        for mission in self.missions:
            if mission.landed:
                continue
            if mission.landing_commanded:
                if mission.status and mission.status.arming_state == VehicleStatus.ARMING_STATE_DISARMED:
                    mission.landed = True
                    self.get_logger().info(f"{mission.namespace}: landed and disarmed")
                    self._activate_next_mission()
                elif mission.command_retries % 20 == 0:
                    self._vehicle_command(mission, VehicleCommand.VEHICLE_CMD_NAV_LAND)
                mission.command_retries += 1
                continue

            # Safe baseline scheduling: later vehicles remain disarmed until the
            # active route has landed. This prevents shared A* corridors from
            # becoming physical mid-air collisions.
            if not mission.active:
                continue

            target_enu = mission.waypoints_enu[mission.waypoint_index]
            target_ned = enu_to_local_ned(target_enu, mission.spawn_enu)
            self._publish_heartbeat(mission)
            self._publish_setpoint(mission, target_ned)
            if mission.position is None or mission.status is None:
                continue
            self._update_package_state(mission)
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

            if mission.warmup_ticks < 20:
                continue
            error = math.dist(
                (mission.position.x, mission.position.y, mission.position.z),
                target_ned,
            )
            is_final_waypoint = mission.waypoint_index == len(mission.waypoints_enu) - 1
            acceptance_radius = 0.25 if is_final_waypoint else 0.6
            velocity = (mission.position.vx, mission.position.vy, mission.position.vz)
            speed = math.sqrt(sum(value**2 for value in velocity)) if all(map(math.isfinite, velocity)) else math.inf
            if error > acceptance_radius or (is_final_waypoint and speed > 0.25):
                mission.dwell_ticks = 0
                continue
            mission.dwell_ticks += 1
            # Intermediate points are fly-through points. Only the final point
            # must settle, which removes stop-and-go motion while making the
            # subsequent land-at-current-position command accurate.
            required_dwell_ticks = 15 if is_final_waypoint else 1
            if mission.dwell_ticks < required_dwell_ticks:
                continue
            mission.dwell_ticks = 0
            self.get_logger().info(
                f"{mission.namespace}: waypoint {mission.waypoint_index + 1}/"
                f"{len(mission.waypoints_enu)} reached, error={error:.2f} m"
            )
            mission.waypoint_index += 1
            if mission.waypoint_index >= len(mission.waypoints_enu):
                mission.waypoint_index = len(mission.waypoints_enu) - 1
                mission.landing_commanded = True
                mission.command_retries = 0
                self._vehicle_command(mission, VehicleCommand.VEHICLE_CMD_NAV_LAND)
                self.get_logger().info(f"{mission.namespace}: route complete, landing")

    def _activate_next_mission(self) -> None:
        self.active_mission_index += 1
        if self.active_mission_index >= len(self.missions):
            return
        mission = self.missions[self.active_mission_index]
        mission.active = True
        mission.started_at = time.monotonic()
        self.get_logger().info(
            f"{mission.namespace}: previous route is clear; activating safe sequential mission"
        )

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

    def _publish_setpoint(self, mission: VehicleMission, target_ned: Point3D) -> None:
        message = TrajectorySetpoint()
        message.timestamp = self._timestamp()
        message.position = list(target_ned)
        message.velocity = [math.nan, math.nan, math.nan]
        message.acceleration = [math.nan, math.nan, math.nan]
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
        print("All three PX4 vehicles discovered; safe sequential mission started", flush=True)
        deadline = time.monotonic() + arguments.timeout
        while rclpy.ok() and not node.complete and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
        if not node.complete:
            raise TimeoutError("Offboard mission did not complete before the timeout")
        print("PASS: all three PX4 missions completed and landed")
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
