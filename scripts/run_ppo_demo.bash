#!/usr/bin/env bash
# PX4 + lidar + direct PPO + return/land, in the lab or original logistics city.
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/setup_env.bash"
MODEL=""
MODE=--gui
SCENARIO=1
ENVIRONMENT=lab
OBSTACLES=""
CORRIDOR_LENGTH=""
OUTPUT_DIR="${MUPT_ROOT}/runs/direct_ppo_3d/demo_$(date +%Y%m%d_%H%M%S)"
while (( $# )); do
  case "$1" in
    --headless|--gui) MODE="$1"; shift ;;
    --model) MODEL="${2:?missing model}"; shift 2 ;;
    --scenario) SCENARIO="${2:?missing scenario}"; shift 2 ;;
    --environment) ENVIRONMENT="${2:?missing environment}"; shift 2 ;;
    --obstacles) OBSTACLES="${2:?missing obstacle count}"; shift 2 ;;
    --corridor-length) CORRIDOR_LENGTH="${2:?missing corridor length}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:?missing output directory}"; shift 2 ;;
    --help) echo "Usage: $0 [--gui|--headless] [--environment lab|city] [--obstacles 1|3] [--corridor-length METRES] [--model MODEL.zip] [--scenario N] [--output-dir DIR]"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "${ENVIRONMENT}" in
  lab) WORLD=ppo_avoidance_lab; OBSTACLES="${OBSTACLES:-1}"; CORRIDOR_LENGTH="${CORRIDOR_LENGTH:-12}" ;;
  city) WORLD=logistics_city_ppo; OBSTACLES="${OBSTACLES:-3}"; CORRIDOR_LENGTH="${CORRIDOR_LENGTH:-28}" ;;
  *) echo "Environment must be lab or city" >&2; exit 2 ;;
esac
if [[ -z "${MODEL}" ]]; then
  if [[ "${ENVIRONMENT}" == city && -f "${MUPT_ROOT}/runs/direct_ppo_3d/city_training/city_demo_model.zip" ]]; then
    MODEL="${MUPT_ROOT}/runs/direct_ppo_3d/city_training/city_demo_model.zip"
  else
    MODEL="${MUPT_ROOT}/runs/direct_ppo_3d/dynamic_gazebo_lidar_route_v1/best_model.zip"
  fi
fi
if [[ "${OBSTACLES}" != 1 && "${OBSTACLES}" != 3 ]]; then echo "Obstacle count must be 1 or 3" >&2; exit 2; fi
python - "${CORRIDOR_LENGTH}" "${OBSTACLES}" <<'PY'
import sys
length = float(sys.argv[1])
if not 6 <= length <= 30:
    raise SystemExit('Corridor length must be between 6 and 30 m')
if sys.argv[2] == '3' and length < 24:
    raise SystemExit('Three moving obstacles require a corridor of at least 24 m')
PY
if [[ ! -f "${MODEL}" ]]; then echo "Model not found: ${MODEL}" >&2; exit 1; fi
if [[ "${MODE}" == --gui && -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  echo "A desktop display is required for --gui; use --headless for recorded diagnostics." >&2
  exit 1
fi
if ! [[ "${SCENARIO}" =~ ^[0-9]+$ ]]; then echo "Scenario must be a non-negative integer" >&2; exit 2; fi
if [[ -e "${OUTPUT_DIR}/summary.json" ]]; then echo "Choose a new output directory for this run." >&2; exit 2; fi
if pgrep -f "${PX4_AUTOPILOT_DIR}/build/px4_sitl_default/bin/px4 -d" >/dev/null; then
  echo "A PX4 simulation is already running. Finish that session before starting this demo." >&2
  exit 1
fi
mkdir -p "${OUTPUT_DIR}"
cp -- "${MODEL}" "${OUTPUT_DIR}/policy.zip"
MODEL="${OUTPUT_DIR}/policy.zip"
python - "${OUTPUT_DIR}" "${MODEL}" "${SCENARIO}" "${WORLD}" "${OBSTACLES}" "${CORRIDOR_LENGTH}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root,model=Path(sys.argv[1]),Path(sys.argv[2])
(root/'demo_config.json').write_text(json.dumps(dict(model=str(model.resolve()),
    model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),world=sys.argv[4],
    corridor_length_m=float(sys.argv[6]),obstacles=int(sys.argv[5]),scenario=int(sys.argv[3]),level_lidar=True,native_lidar_follow=True,
    controller='direct_ppo',local_activation=True,advisory_potential_field_started=False),indent=2)+'\n')
PY
export PX4_GZ_WORLD="${WORLD}"
export MUPT_LEVEL_LIDAR=true
export MUPT_AVOIDANCE_MODE=direct
export MUPT_NATIVE_LIDAR_FOLLOW=true
export GZ_SIM_SYSTEM_PLUGIN_PATH="${MUPT_ROOT}/build/mupt_sensor_follower${GZ_SIM_SYSTEM_PLUGIN_PATH:+:${GZ_SIM_SYSTEM_PLUGIN_PATH}}"
cmake -S "${MUPT_ROOT}/simulation/plugins" -B "${MUPT_ROOT}/build/mupt_sensor_follower" >"${OUTPUT_DIR}/plugin_build.log" 2>&1
cmake --build "${MUPT_ROOT}/build/mupt_sensor_follower" -j2 >>"${OUTPUT_DIR}/plugin_build.log" 2>&1
if [[ "${ENVIRONMENT}" == city ]]; then
  python "${SCRIPT_DIR}/generate_ppo_city_world.py" >"${OUTPUT_DIR}/world_generation.log"
fi
cp -- "${MUPT_ROOT}/simulation/worlds/${WORLD}.sdf" "${OUTPUT_DIR}/world.sdf"
declare -a GROUPS_PIDS=()
cleanup() {
  trap - EXIT INT TERM
  for ((i=${#GROUPS_PIDS[@]}-1; i>=0; i--)); do
    kill -TERM -- "-${GROUPS_PIDS[i]}" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM
start_group() {
  local label="$1"; shift
  setsid "$@" >"${OUTPUT_DIR}/${label}.log" 2>&1 &
  GROUPS_PIDS+=("$!")
}
echo "PPO prototype demo: ${ENVIRONMENT}, ${OBSTACLES} added moving spheres, ${CORRIDOR_LENGTH} m corridor, direct FLU velocity."
echo "Output: ${OUTPUT_DIR}"
start_group simulation "${SCRIPT_DIR}/run_three_px4.bash" "${MODE}"
deadline=$((SECONDS + 60))
until gz service -l 2>/dev/null | grep -Fx "/world/${PX4_GZ_WORLD}/create" >/dev/null; do
  if (( SECONDS > deadline )); then echo "Gazebo startup timed out; see simulation.log" >&2; exit 1; fi
  sleep 1
done
# The world create service appears before GPU lidar finishes initializing.
deadline=$((SECONDS + 60))
until gz topic -l 2>/dev/null | grep -Fx "/mupt/uav_1/scan" >/dev/null; do
  if ! kill -0 "${GROUPS_PIDS[0]}" 2>/dev/null; then echo "Simulation exited; see simulation.log" >&2; exit 1; fi
  if (( SECONDS > deadline )); then echo "Gazebo lidar startup timed out; see simulation.log" >&2; exit 1; fi
  sleep 1
done
start_group lidar "${SCRIPT_DIR}/run_lidar_rviz.bash" --no-rviz
deadline=$((SECONDS + 30))
until ros2 topic list 2>/dev/null | grep -Fx "/mupt/uav_1/observation/down_clearance" >/dev/null; do
  if ! kill -0 "${GROUPS_PIDS[1]}" 2>/dev/null; then echo "Lidar stack exited; see lidar.log" >&2; exit 1; fi
  if (( SECONDS > deadline )); then echo "Lidar startup timed out; see lidar.log" >&2; exit 1; fi
  sleep 1
done
start_group inference "${SCRIPT_DIR}/run_direct_ppo_inference.bash" --model "${MODEL}"
deadline=$((SECONDS + 20))
until ros2 topic list 2>/dev/null | grep -Fx "/mupt/uav_1/ppo_direct/velocity_body" >/dev/null; do
  if ! kill -0 "${GROUPS_PIDS[2]}" 2>/dev/null; then echo "Inference exited; see inference.log" >&2; exit 1; fi
  if (( SECONDS > deadline )); then echo "Inference startup timed out; see inference.log" >&2; exit 1; fi
  sleep 1
done
RESULT=0
start_group poses env "PYTHONPATH=${MUPT_ROOT}/src/logistics_planner${PYTHONPATH:+:${PYTHONPATH}}" \
  python -m logistics_planner.gazebo_pose_recorder --world "${PX4_GZ_WORLD}" \
  --output "${OUTPUT_DIR}/gazebo_poses.jsonl"
"${SCRIPT_DIR}/run_single_complex_round_trip.bash" --controller direct --scenario "${SCENARIO}" \
  --obstacles "${OBSTACLES}" --corridor-length "${CORRIDOR_LENGTH}" --local-activation --output-dir "${OUTPUT_DIR}" >"${OUTPUT_DIR}/mission.log" 2>&1 || RESULT=$?
cat "${OUTPUT_DIR}/mission.log"
if [[ -f "${OUTPUT_DIR}/summary.json" ]]; then
  PLOT_ARGS=()
  if [[ "${ENVIRONMENT}" == lab ]]; then PLOT_ARGS+=(--lab); fi
  python "${SCRIPT_DIR}/plot_ppo_flight.py" --run-dir "${OUTPUT_DIR}" "${PLOT_ARGS[@]}"
fi
exit "${RESULT}"
