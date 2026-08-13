#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup_env.bash"

WORLD_NAME="${PX4_GZ_WORLD:-logistics_city}"
OBSTACLE_NAME="mupt_complex_moving_obstacles"
OBSTACLE_FILE="${MUPT_ROOT}/simulation/tests/avoidance_test_complex_moving_spheres.sdf"
CONTROLLER="rl"
SCENARIO="1"
OUTPUT_DIR=""
while (( $# )); do
  case "$1" in
    --controller) CONTROLLER="${2:-}"; shift 2 ;;
    --scenario) SCENARIO="${2:-}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:-}"; shift 2 ;;
    *) echo "Usage: $0 [--controller rl|potential] [--scenario N] [--output-dir DIR]" >&2; exit 2 ;;
  esac
done
if [[ "${CONTROLLER}" != "rl" && "${CONTROLLER}" != "potential" ]]; then
  echo "Controller must be 'rl' or 'potential'." >&2
  exit 2
fi
if ! [[ "${SCENARIO}" =~ ^[0-9]+$ ]]; then
  echo "Scenario must be a non-negative integer." >&2
  exit 2
fi
if [[ -z "${OUTPUT_DIR}" ]]; then
  RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
  OUTPUT_DIR="${MUPT_ROOT}/runs/avoidance_ab/${CONTROLLER}_${RUN_STAMP}"
fi
mkdir -p "${OUTPUT_DIR}"
OBSTACLE_FILE="${OUTPUT_DIR}/obstacles.sdf"
python -m logistics_planner.obstacle_scenario \
  --template "${MUPT_ROOT}/simulation/tests/avoidance_test_complex_moving_spheres.sdf" \
  --output "${OBSTACLE_FILE}" --scenario "${SCENARIO}"

if ! gz service -l 2>/dev/null | grep -Fx "/world/${WORLD_NAME}/create" >/dev/null; then
  echo "Gazebo world is not running. Start scripts/run_three_px4.bash first." >&2
  exit 1
fi
REQUIRED_TOPICS=(/mupt/uav_1/avoidance/suggested_velocity_body)
if [[ "${CONTROLLER}" == "rl" ]]; then
  REQUIRED_TOPICS+=(/mupt/uav_1/rl_shadow/suggested_velocity_body)
fi
for topic in "${REQUIRED_TOPICS[@]}"; do
  deadline=$((SECONDS + 15))
  until ros2 topic list 2>/dev/null | grep -Fx "${topic}" >/dev/null; do
    if (( SECONDS >= deadline )); then break; fi
    sleep 1
  done
  if ! ros2 topic list 2>/dev/null | grep -Fx "${topic}" >/dev/null; then
    echo "Required topic is missing: ${topic}" >&2
    echo "Start the lidar stack, and for RL also start scripts/run_rl_shadow.bash." >&2
    exit 1
  fi
done
if ros2 node list 2>/dev/null | grep -E '/multi_uav_offboard_mission$|/single_uav_closed_loop_avoidance_test$' >/dev/null; then
  echo "Another PX4 mission controller is already running. Stop it first." >&2
  exit 1
fi

remove_obstacles() {
  gz service -s "/world/${WORLD_NAME}/remove" \
    --reqtype gz.msgs.Entity --reptype gz.msgs.Boolean --timeout 1000 \
    --req "name: '${OBSTACLE_NAME}' type: MODEL" >/dev/null 2>&1 || true
}
trap remove_obstacles EXIT INT TERM

remove_obstacles
create_output="$(gz service -s "/world/${WORLD_NAME}/create" \
  --reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean --timeout 1000 \
  --req "sdf_filename: '${OBSTACLE_FILE}' name: '${OBSTACLE_NAME}' pose: {position: {x: -20, y: -23, z: 6}}")"
if ! grep -q 'data: true' <<<"${create_output}"; then
  echo "Gazebo did not create the complex obstacle course: ${create_output}" >&2
  exit 1
fi

echo "Single-UAV complex round-trip test starting on UAV 1."
echo "Scenario: ${SCENARIO}; controller: ${CONTROLLER}; evaluation output: ${OUTPUT_DIR}"
echo "UAV 2 and UAV 3 remain idle. The temporary obstacles are removed automatically."

COMMAND=(ros2 run logistics_planner single_uav_avoidance_test
  --test-label "${CONTROLLER}_complex_round_trip"
  --avoidance-source "${CONTROLLER}"
  --output-dir "${OUTPUT_DIR}"
  --timeout 240)
if [[ "${CONTROLLER}" == "rl" ]]; then
  COMMAND+=(--rl-return)
fi
"${COMMAND[@]}"
