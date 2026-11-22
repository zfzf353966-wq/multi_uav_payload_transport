#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup_env.bash"

RUN_DIR="${MUPT_ROOT}/log/lidar_rviz"
RVIZ_ENABLED=1

case "${1:-}" in
  "") ;;
  --no-rviz) RVIZ_ENABLED=0 ;;
  *)
    echo "Usage: $0 [--no-rviz]" >&2
    exit 2
    ;;
esac

if (( RVIZ_ENABLED == 1 )) && [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  echo "No desktop display is available. Use --no-rviz for ROS 2 data only." >&2
  exit 1
fi

if ! gz topic -l 2>/dev/null | grep -Fx "/mupt/uav_1/scan" >/dev/null; then
  echo "Gazebo lidar topics were not found." >&2
  echo "Start the simulation first: scripts/run_three_px4.bash --gui" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}"
declare -a PROCESS_GROUPS=()

cleanup() {
  trap - EXIT INT TERM
  echo "Stopping lidar bridge and RViz..."
  for ((i=${#PROCESS_GROUPS[@]}-1; i>=0; i--)); do
    kill -TERM -- "-${PROCESS_GROUPS[i]}" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

start_group() {
  local log_file="$1"
  shift
  setsid "$@" </dev/null >"${log_file}" 2>&1 &
  PROCESS_GROUPS+=("$!")
}

BRIDGES=(
  "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"
  "/mupt/uav_1/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan"
  "/mupt/uav_2/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan"
  "/mupt/uav_3/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan"
  "/mupt/uav_1/scan/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked"
  "/mupt/uav_2/scan/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked"
  "/mupt/uav_3/scan/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked"
  "/mupt/uav_1/down_scan/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked"
  "/mupt/uav_2/down_scan/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked"
  "/mupt/uav_3/down_scan/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked"
)

echo "Starting Gazebo-to-ROS 2 lidar bridge..."
start_group "${RUN_DIR}/ros_gz_bridge.log" \
  ros2 run ros_gz_bridge parameter_bridge "${BRIDGES[@]}"

echo "Starting lidar pose follower and TF publisher..."
start_group "${RUN_DIR}/lidar_tf_bridge.log" \
  ros2 run logistics_planner lidar_tf_bridge --ros-args -p use_sim_time:=true \
    -p "world:=${PX4_GZ_WORLD:-logistics_city}" -p "level_sensors:=${MUPT_LEVEL_LIDAR:-false}" \
    -p "native_follow:=${MUPT_NATIVE_LIDAR_FOLLOW:-false}"

echo "Starting 36-value sector observation node..."
start_group "${RUN_DIR}/sector_observation.log" \
  ros2 run logistics_planner sector_observation --ros-args -p use_sim_time:=true

if [[ "${MUPT_AVOIDANCE_MODE:-potential}" != direct ]]; then
  echo "Starting advisory-only shadow avoidance node..."
  start_group "${RUN_DIR}/shadow_avoidance.log" \
    ros2 run logistics_planner shadow_avoidance --ros-args -p use_sim_time:=true
fi

deadline=$((SECONDS + 20))
until ros2 topic list 2>/dev/null | grep -Fx "/mupt/uav_3/observation/sectors" >/dev/null; do
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for bridged lidar topics. See ${RUN_DIR}." >&2
    exit 1
  fi
  sleep 1
done

if (( RVIZ_ENABLED == 1 )); then
  echo "Starting RViz..."
  start_group "${RUN_DIR}/rviz.log" \
    env -u GIO_MODULE_DIR -u GTK_PATH \
    rviz2 -d "${MUPT_ROOT}/config/lidar_three_uav.rviz" --ros-args -p use_sim_time:=true
fi

echo "Three lidar topics are available:"
echo "  /mupt/uav_1/scan"
echo "  /mupt/uav_2/scan"
echo "  /mupt/uav_3/scan"
echo "Three-layer point clouds are available below each scan topic as /points."
echo "36-value observations are available as /mupt/uav_N/observation/sectors."
echo "Downward clearance is available as /mupt/uav_N/observation/down_clearance."
if [[ "${MUPT_AVOIDANCE_MODE:-potential}" != direct ]]; then
  echo "Shadow suggestions are available as /mupt/uav_N/avoidance/*."
fi
echo "Logs: ${RUN_DIR}"
echo "Press Ctrl-C to stop this bridge (the Gazebo simulation keeps running)."

while :; do
  sleep 2
  for pid in "${PROCESS_GROUPS[@]}"; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "A lidar visualization process exited. Check ${RUN_DIR}." >&2
      exit 1
    fi
  done
done
