#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup_env.bash"

WORLD_NAME="${PX4_GZ_WORLD:-logistics_city}"
OBSTACLE_NAME="mupt_closed_loop_moving_sphere"
OBSTACLE_FILE="${MUPT_ROOT}/simulation/tests/avoidance_test_moving_sphere.sdf"

if ! gz service -l 2>/dev/null | grep -Fx "/world/${WORLD_NAME}/create" >/dev/null; then
  echo "Gazebo world is not running. Start scripts/run_three_px4.bash first." >&2
  exit 1
fi

deadline=$((SECONDS + 15))
until ros2 topic list 2>/dev/null | grep -Fx "/mupt/uav_1/avoidance/suggested_velocity_body" >/dev/null; do
  if (( SECONDS >= deadline )); then break; fi
  sleep 1
done
if ! ros2 topic list 2>/dev/null | grep -Fx "/mupt/uav_1/avoidance/suggested_velocity_body" >/dev/null; then
  echo "Avoidance topics are missing. Start scripts/run_lidar_rviz.bash first." >&2
  exit 1
fi
if ros2 node list 2>/dev/null | grep -E '/multi_uav_offboard_mission$|/single_uav_closed_loop_avoidance_test$' >/dev/null; then
  echo "Another PX4 mission controller is already running. Stop it before this isolated test." >&2
  exit 1
fi

remove_obstacle() {
  gz service -s "/world/${WORLD_NAME}/remove" \
    --reqtype gz.msgs.Entity \
    --reptype gz.msgs.Boolean \
    --timeout 1000 \
    --req "name: '${OBSTACLE_NAME}' type: MODEL" >/dev/null 2>&1 || true
}
trap remove_obstacle EXIT INT TERM

remove_obstacle
create_output="$(gz service -s "/world/${WORLD_NAME}/create" \
  --reqtype gz.msgs.EntityFactory \
  --reptype gz.msgs.Boolean \
  --timeout 1000 \
  --req "sdf_filename: '${OBSTACLE_FILE}' name: '${OBSTACLE_NAME}' pose: {position: {x: -20, y: -23, z: 6}}")"
if ! grep -q 'data: true' <<<"${create_output}"; then
  echo "Gazebo did not create the moving test obstacle: ${create_output}" >&2
  exit 1
fi

echo "Moving red sphere created: 2.5 m orbit centred at ENU (-20, -23, 6)."
echo "Tangential speed is about 0.88 m/s; it repeatedly crosses UAV 1's route."
echo "Only UAV 1 will arm; UAV 2 and UAV 3 remain idle."
echo "The moving sphere is removed automatically when this command exits."

ros2 run logistics_planner single_uav_avoidance_test --test-label dynamic
