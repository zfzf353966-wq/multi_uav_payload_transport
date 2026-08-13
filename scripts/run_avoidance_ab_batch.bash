#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup_env.bash"

SCENARIOS="1,2,3,4,5"
if [[ "${1:-}" == "--scenarios" ]]; then
  SCENARIOS="${2:-}"
elif [[ -n "${1:-}" ]]; then
  echo "Usage: $0 [--scenarios 1,2,3,4,5]" >&2
  exit 2
fi
IFS=',' read -r -a SCENARIO_IDS <<<"${SCENARIOS}"
BATCH_ROOT="${MUPT_ROOT}/runs/avoidance_ab/batch_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${BATCH_ROOT}"

SIM_PID=""
LIDAR_PID=""
RL_PID=""
cleanup_stack() {
  for pid in "${RL_PID}" "${LIDAR_PID}" "${SIM_PID}"; do
    if [[ -n "${pid}" ]]; then kill -TERM "${pid}" 2>/dev/null || true; fi
  done
  for pid in "${RL_PID}" "${LIDAR_PID}" "${SIM_PID}"; do
    if [[ -n "${pid}" ]]; then wait "${pid}" 2>/dev/null || true; fi
  done
  SIM_PID=""; LIDAR_PID=""; RL_PID=""
}
trap cleanup_stack EXIT INT TERM

wait_for() {
  local description="$1"
  local command="$2"
  local deadline=$((SECONDS + 75))
  until bash -c "${command}"; do
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting for ${description}." >&2
      return 1
    fi
    sleep 1
  done
}

run_one() {
  local scenario="$1"
  local controller="$2"
  local output_dir
  output_dir="${BATCH_ROOT}/scenario_$(printf '%02d' "${scenario}")/${controller}"
  local log_dir="${output_dir}/process_logs"
  mkdir -p "${log_dir}"
  echo "=== Scenario ${scenario}, controller ${controller} ==="

  HEADLESS=1 "${SCRIPT_DIR}/run_three_px4.bash" --headless >"${log_dir}/simulation.log" 2>&1 &
  SIM_PID=$!
  wait_for "Gazebo" "gz service -l 2>/dev/null | grep -q '^/world/logistics_city/create$'"
  wait_for "Gazebo lidar sensor" "gz topic -l 2>/dev/null | grep -Fxq '/mupt/uav_1/scan'"

  "${SCRIPT_DIR}/run_lidar_rviz.bash" --no-rviz >"${log_dir}/lidar.log" 2>&1 &
  LIDAR_PID=$!
  wait_for "lidar observations" "ros2 topic list 2>/dev/null | grep -Fxq '/mupt/uav_1/observation/sectors'"

  if [[ "${controller}" == "rl" ]]; then
    "${SCRIPT_DIR}/run_rl_shadow.bash" >"${log_dir}/rl.log" 2>&1 &
    RL_PID=$!
    wait_for "PPO output" "ros2 topic list 2>/dev/null | grep -Fxq '/mupt/uav_1/rl_shadow/suggested_velocity_body'"
  fi

  set +e
  "${SCRIPT_DIR}/run_single_complex_round_trip.bash" \
    --controller "${controller}" --scenario "${scenario}" --output-dir "${output_dir}" \
    >"${log_dir}/mission.log" 2>&1
  local status=$?
  set -e
  cleanup_stack
  sleep 2
  if (( status != 0 )); then
    echo "Scenario ${scenario} ${controller} failed; see ${log_dir}/mission.log" >&2
  fi
}

for scenario in "${SCENARIO_IDS[@]}"; do
  if ! [[ "${scenario}" =~ ^[0-9]+$ ]]; then
    echo "Invalid scenario id: ${scenario}" >&2
    exit 2
  fi
  run_one "${scenario}" potential
  run_one "${scenario}" rl
done

python -m logistics_planner.avoidance_batch_report "${BATCH_ROOT}"
echo "A/B batch complete: ${BATCH_ROOT}"
