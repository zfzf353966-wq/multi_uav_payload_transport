#!/usr/bin/env bash
# Full original city, seeded 3D obstacles, concurrent A* + PPO delivery fleet.
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/setup_env.bash"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"
export GZ_IP="${GZ_IP:-127.0.0.1}"
export PYTHONPATH="${MUPT_ROOT}/src/logistics_planner${PYTHONPATH:+:${PYTHONPATH}}"
MODE=--gui
SEED=1
RANDOM_COUNT=8
DEPARTURE_COUNT=2
PREPARE_ONLY=0
# 冻结策略属于可交付资产；每次运行的数据仍独立写入 runs/。
MODEL="${MUPT_ROOT}/models/ppo_delivery/delivery_model.zip"
OUTPUT="${MUPT_ROOT}/runs/delivery_$(date +%Y%m%d_%H%M%S)"
while (( $# )); do
  case "$1" in
    --gui|--headless) MODE="$1"; shift ;;
    --seed) SEED="${2:?missing seed}"; shift 2 ;;
    --model) MODEL="${2:?missing model}"; shift 2 ;;
    --output-dir) OUTPUT="${2:?missing output}"; shift 2 ;;
    --random-obstacles) RANDOM_COUNT="${2:?missing count}"; shift 2 ;;
    --departure-obstacles) DEPARTURE_COUNT="${2:?missing count}"; shift 2 ;;
    --prepare-only) PREPARE_ONLY=1; shift ;;
    --help) echo "Usage: $0 [--gui|--headless] [--seed N] [--random-obstacles N] [--departure-obstacles 1|2] [--model ZIP] [--output-dir DIR] [--prepare-only]"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if pgrep -f "${PX4_AUTOPILOT_DIR}/build/px4_sitl_default/bin/px4 -d" >/dev/null; then
  echo 'Finish the existing PX4 simulation before starting delivery.' >&2; exit 1
fi
if [[ -e "${OUTPUT}" ]]; then echo 'Choose a fresh output directory.' >&2; exit 2; fi
if (( PREPARE_ONLY==0 )); then
  if [[ "$MODE" == --gui && -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
    echo 'Use --headless without a desktop display.' >&2; exit 1
  fi
  python - "${MODEL}" <<'PY'
import sys,torch
from stable_baselines3 import PPO
torch.set_num_threads(1)
m=PPO.load(sys.argv[1],device='cpu')
if m.observation_space.shape!=(153,) or m.action_space.shape!=(3,) or getattr(m,'altitude_max',10.)!=32.:
    raise SystemExit('Delivery requires a direct 153/3 PPO model trained with altitude_max=32; old 2–10 m models cannot be substituted.')
PY
fi
mkdir -p "${OUTPUT}"
OUTPUT="$(cd "${OUTPUT}" && pwd)"
CONFIG="${MUPT_ROOT}/config/logistics_city.yaml"
python -m logistics_planner.baseline --config "${CONFIG}" --output-dir "${OUTPUT}/planning" >"${OUTPUT}/planning.log" 2>&1
python -m logistics_planner.delivery_plan --plan "${OUTPUT}/planning/plan.json" --config "${CONFIG}" --output "${OUTPUT}/plan.json"
python -m logistics_planner.city_dynamic_obstacles --config "${CONFIG}" --plan "${OUTPUT}/plan.json" \
  --source "${MUPT_ROOT}/simulation/worlds/logistics_city.sdf" \
  --output "${MUPT_ROOT}/simulation/worlds/logistics_delivery_ppo.sdf" \
  --manifest "${OUTPUT}/obstacles.json" --seed "${SEED}" --random-count "${RANDOM_COUNT}" --departure-count "${DEPARTURE_COUNT}"
cp "${MUPT_ROOT}/simulation/worlds/logistics_delivery_ppo.sdf" "${OUTPUT}/world.sdf"
cp "${CONFIG}" "${OUTPUT}/city_config.yaml"
if (( PREPARE_ONLY==1 )); then echo "Prepared A* delivery plan and 3D city: ${OUTPUT}"; exit 0; fi
cp "${MODEL}" "${OUTPUT}/policy.zip"
MODEL="${OUTPUT}/policy.zip"
python - "${OUTPUT}" <<'PY'
import sys,json,hashlib
from pathlib import Path
p=Path(sys.argv[1])
(p/'run_config.json').write_text(json.dumps(dict(world='logistics_delivery_ppo',vehicles=3,
    concurrent=True,global_planner='3D A*',local_planner='PPO direct FLU velocity',
    altitude_limits=[2,32],potential_field_used=False,
    waypoint_lookahead_m=8,observation_freshness_s=.5,communication_hold_timeout_s=2,
    service_dwell_s=2,service_radius_m=.8,service_speed_mps=.45,
    fleet_early_yield_m=3.5,fleet_acceptance_separation_m=2.5,
    fleet_velocity_reserve_m=3.2,peer_astar_reservation_radius_m=3.6,
    fleet_velocity_constraints_apply_to='both aircraft, including the priority winner',
    local_takeover='planned-direction novelty, predicted collision risk, or unknown wide-cloud approach',
    sensed_replanning='temporary A* occupancy from real safety-cloud surface hits; no dynamic truth poses',
    safety_layer='requested velocity-component projection with radial, physical-map and wide-cloud constraints',
    safety_cloud=dict(vertical_layers=11,vertical_degrees=[-60,60],range_m=8,policy_input=False),
    task_distribution='balanced: two orders at each delivery point; exclusive UAV owner per destination',
    return_cruise_altitudes_m=[26,28,30],landing_home_xy_tolerance_m=1.5,
    payload_backend='Gazebo PackageFollower; state transitions acknowledged by native client',
    visual_parcels_excluded_from_lidar=True,
    model_sha256=hashlib.sha256((p/'policy.zip').read_bytes()).hexdigest(),
    payload_mode='visual models; no mass or cable dynamics'),indent=2)+'\n')
PY
export PX4_GZ_WORLD=logistics_delivery_ppo
export MUPT_LEVEL_LIDAR=true MUPT_NATIVE_LIDAR_FOLLOW=true MUPT_AVOIDANCE_MODE=direct
export GZ_SIM_SYSTEM_PLUGIN_PATH="${MUPT_ROOT}/build/mupt_sensor_follower${GZ_SIM_SYSTEM_PLUGIN_PATH:+:${GZ_SIM_SYSTEM_PLUGIN_PATH}}"
cmake -S "${MUPT_ROOT}/simulation/plugins" -B "${MUPT_ROOT}/build/mupt_sensor_follower" >"${OUTPUT}/plugin_build.log" 2>&1
cmake --build "${MUPT_ROOT}/build/mupt_sensor_follower" -j2 >>"${OUTPUT}/plugin_build.log" 2>&1
declare -a OWNED=()
cleanup() {
  trap - EXIT INT TERM
  for ((i=${#OWNED[@]}-1;i>=0;i--)); do kill -TERM -- "-${OWNED[i]}" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM
start() { local label="$1"; shift; setsid "$@" >"${OUTPUT}/${label}.log" 2>&1 & OWNED+=("$!"); }
echo "Three concurrent A* + PPO deliveries; output: ${OUTPUT}"
start simulation "${SCRIPT_DIR}/run_three_px4.bash" "${MODE}"
deadline=$((SECONDS+90))
until gz topic -l 2>/dev/null | grep -Fx /mupt/uav_1/scan >/dev/null; do
  if (( SECONDS>deadline )); then echo 'Gazebo/lidar startup timed out.' >&2; exit 1; fi
  sleep 1
done
deadline=$((SECONDS+20))
until gz service -l 2>/dev/null | grep -Fx "/world/${PX4_GZ_WORLD}/package_state" >/dev/null; do
  if (( SECONDS>deadline )); then echo 'Native package-state service is unavailable; inspect simulation.log.' >&2; exit 1; fi
  sleep 1
done
start lidar "${SCRIPT_DIR}/run_lidar_rviz.bash" --no-rviz
start safety_bridge ros2 run ros_gz_bridge parameter_bridge \
  '/mupt/uav_1/safety_scan/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked' \
  '/mupt/uav_2/safety_scan/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked' \
  '/mupt/uav_3/safety_scan/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked'
for index in 1 2 3; do
  start "inference_${index}" "${SCRIPT_DIR}/run_direct_ppo_inference.bash" --model "${MODEL}" --vehicle-index "${index}"
done
deadline=$((SECONDS+40))
until ros2 topic list 2>/dev/null | grep -Fx /mupt/uav_3/ppo_direct/velocity_body >/dev/null; do
  if (( SECONDS>deadline )); then echo 'PPO startup timed out; inspect inference logs.' >&2; exit 1; fi
  sleep 1
done
start poses python -m logistics_planner.gazebo_pose_recorder --world "$PX4_GZ_WORLD" --output "${OUTPUT}/gazebo_poses.jsonl"
RESULT=0
python -m logistics_planner.delivery_ppo_mission --plan "${OUTPUT}/plan.json" --config "${CONFIG}" \
  --output-dir "${OUTPUT}" --timeout 900 >"${OUTPUT}/mission.log" 2>&1 || RESULT=$?
cat "${OUTPUT}/mission.log"
if [[ -f "${OUTPUT}/summary.json" ]]; then
  python "${SCRIPT_DIR}/plot_ppo_delivery.py" --run-dir "${OUTPUT}"
fi
exit "${RESULT}"
