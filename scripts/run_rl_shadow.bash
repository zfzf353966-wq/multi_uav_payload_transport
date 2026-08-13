#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup_env.bash"

MODEL_PATH="${MUPT_ROOT}/runs/rl_residual_stack4/ppo_local_avoidance.zip"
if [[ "${1:-}" == "--model" ]]; then
  MODEL_PATH="${2:-}"
elif [[ -n "${1:-}" ]]; then
  echo "Usage: $0 [--model MODEL.zip]" >&2
  exit 2
fi
if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "PPO model not found: ${MODEL_PATH}" >&2
  echo "Train it first with scripts/train_rl_avoidance.bash." >&2
  exit 1
fi
if ! ros2 topic list 2>/dev/null | grep -Fx "/mupt/uav_1/observation/sectors" >/dev/null; then
  echo "Lidar sectors are missing. Start scripts/run_lidar_rviz.bash first." >&2
  exit 1
fi

echo "Starting PPO shadow inference with ${MODEL_PATH}"
echo "This process publishes /mupt/uav_1/rl_shadow/* and never commands PX4."
exec python -m logistics_planner.rl_shadow_inference --model "${MODEL_PATH}"
