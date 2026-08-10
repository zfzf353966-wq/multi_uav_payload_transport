#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup_env.bash"

PX4_BUILD="${PX4_AUTOPILOT_DIR}/build/px4_sitl_default"
PX4_BIN="${PX4_BUILD}/bin/px4"
PX4_ETC="${PX4_BUILD}/etc"
GZ_ENV="${PX4_BUILD}/rootfs/gz_env.sh"
RUN_DIR="${MUPT_ROOT}/log/three_px4"
HEADLESS_MODE="${HEADLESS:-0}"
WORLD_NAME="${PX4_GZ_WORLD:-logistics_city}"
VEHICLE_MODEL="${PX4_SIM_MODEL:-gz_x500}"
SELF_TEST=0

case "${1:-}" in
  "") ;;
  --gui) HEADLESS_MODE=0 ;;
  --headless) HEADLESS_MODE=1 ;;
  --check) SELF_TEST=1; HEADLESS_MODE=1 ;;
  *)
    echo "Usage: $0 [--gui|--headless|--check]" >&2
    exit 2
    ;;
esac

if [[ "${HEADLESS_MODE}" == "0" && -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  echo "No desktop display is available. Use --headless or run from a desktop terminal." >&2
  exit 1
fi

for required in "${PX4_BIN}" "${PX4_ETC}" "${GZ_ENV}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing PX4 build artifact: ${required}" >&2
    echo "Build it with: cd ${PX4_AUTOPILOT_DIR} && make px4_sitl" >&2
    exit 1
  fi
done

mkdir -p "${RUN_DIR}"
set +u
# shellcheck disable=SC1090
source "${GZ_ENV}"
set -u

declare -a PROCESS_GROUPS=()

cleanup() {
  trap - EXIT INT TERM
  echo "Stopping three-UAV simulation..."
  for ((i=${#PROCESS_GROUPS[@]}-1; i>=0; i--)); do
    kill -TERM -- "-${PROCESS_GROUPS[i]}" 2>/dev/null || true
  done
  sleep 1
  for ((i=${#PROCESS_GROUPS[@]}-1; i>=0; i--)); do
    kill -KILL -- "-${PROCESS_GROUPS[i]}" 2>/dev/null || true
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

wait_for_gazebo() {
  local deadline=$((SECONDS + 45))
  until gz service -l 2>/dev/null | grep -q "^/world/${WORLD_NAME}/scene/info$"; do
    if (( SECONDS >= deadline )); then
      echo "Gazebo did not become ready. See ${RUN_DIR}/px4_1.log" >&2
      exit 1
    fi
    sleep 1
  done
}

start_px4() {
  local instance="$1"
  local pose="$2"
  local standalone="$3"
  local work_dir="${RUN_DIR}/instance_${instance}"
  mkdir -p "${work_dir}"

  local -a px4_env=(env)
  if [[ "${HEADLESS_MODE}" == "0" ]]; then
    # Do not inherit an old HEADLESS=1 from the user's shell.
    px4_env+=(-u HEADLESS)
  fi
  px4_env+=(
    "PX4_SYS_AUTOSTART=4013"
    "PX4_PARAM_SDLOG_MODE=0"
    "PX4_PARAM_SDLOG_BACKEND=0"
    "PX4_PARAM_NAV_DLL_ACT=0"
    # This SITL workflow intentionally has no RC transmitter. Ignore RC loss
    # only in Offboard (bit 2); Offboard-link-loss protection remains active.
    "PX4_PARAM_COM_RCL_EXCEPT=4"
    "PX4_PARAM_COM_RC_IN_MODE=4"
    "PX4_SIM_MODEL=${VEHICLE_MODEL}"
    "PX4_GZ_MODEL_POSE=${pose}"
    "PX4_GZ_WORLD=${WORLD_NAME}"
    "PX4_GZ_WORLDS=${MUPT_SIMULATION_DIR}/worlds"
  )
  if [[ "${standalone}" == "1" ]]; then
    px4_env+=("PX4_GZ_STANDALONE=1")
  fi
  if [[ "${HEADLESS_MODE}" == "1" ]]; then
    px4_env+=("HEADLESS=1")
  fi

  start_group "${RUN_DIR}/px4_${instance}.log" \
    "${px4_env[@]}" "${PX4_BIN}" -d -i "${instance}" -w "${work_dir}" "${PX4_ETC}"
}

check_ros_topics() {
  local deadline=$((SECONDS + 60))
  local topic
  local all_ready

  echo "Waiting for all three PX4 ROS 2 namespaces..."
  while (( SECONDS < deadline )); do
    all_ready=1
    for instance in 1 2 3; do
      topic="/px4_${instance}/fmu/out/vehicle_status_v1"
      if ! ros2 topic list 2>/dev/null | grep -Fxq "${topic}"; then
        all_ready=0
        break
      fi
    done
    (( all_ready == 1 )) && break
    sleep 2
  done

  if (( all_ready != 1 )); then
    echo "Timed out waiting for all three PX4 ROS 2 topics." >&2
    return 1
  fi

  for instance in 1 2 3; do
    topic="/px4_${instance}/fmu/out/vehicle_status_v1"
    echo "Checking ${topic}..."
    timeout 12s ros2 topic echo --once "${topic}" px4_msgs/msg/VehicleStatus \
      >/dev/null
  done
  echo "PASS: all three PX4 instances publish VehicleStatus over ROS 2."
}

echo "Starting Micro XRCE-DDS Agent..."
start_group "${RUN_DIR}/micro_xrce_agent.log" MicroXRCEAgent udp4 -p 8888
sleep 1

echo "Starting PX4 instance 1 and Gazebo Harmonic..."
start_px4 1 "-32,-23,0.13" 0
wait_for_gazebo

echo "Starting PX4 instances 2 and 3..."
start_px4 2 "-30,-23,0.13" 1
start_px4 3 "-28,-23,0.13" 1

echo "Three-UAV simulation is running."
echo "Gazebo world: ${WORLD_NAME}"
echo "PX4 model: ${VEHICLE_MODEL}"
if [[ "${HEADLESS_MODE}" == "0" ]]; then
  echo "Gazebo GUI: enabled"
else
  echo "Gazebo GUI: disabled (headless)"
fi
echo "ROS namespaces: /px4_1, /px4_2, /px4_3"
echo "Logs: ${RUN_DIR}"

if (( SELF_TEST == 1 )); then
  check_ros_topics
  exit 0
fi

echo "Press Ctrl-C to stop all simulation processes."

while :; do
  sleep 2
  for pid in "${PROCESS_GROUPS[@]}"; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "A simulation process exited unexpectedly. Check ${RUN_DIR}." >&2
      exit 1
    fi
  done
done
