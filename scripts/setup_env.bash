#!/usr/bin/env bash
# Source this file before building or running the project:
#   source scripts/setup_env.bash

_MUPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Ignore packages installed in ~/.local. They currently contain NumPy 2.x and
# are ABI-incompatible with the Ubuntu 22.04 ROS/OpenCV packages.
export PYTHONNOUSERSITE=1

# ROS 2 generated setup files reference optional variables and are not safe
# under `set -u`. Preserve the caller's nounset setting around the sources.
_MUPT_HAD_NOUNSET=0
case "$-" in
  *u*) _MUPT_HAD_NOUNSET=1; set +u ;;
esac

# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source "${_MUPT_ROOT}/.venv/bin/activate"

if [[ -f "${_MUPT_ROOT}/install/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "${_MUPT_ROOT}/install/setup.bash"
fi

if [[ "${_MUPT_HAD_NOUNSET}" == "1" ]]; then
  set -u
fi

export MUPT_ROOT="${_MUPT_ROOT}"
export PX4_AUTOPILOT_DIR="${PX4_AUTOPILOT_DIR:-$(dirname "${MUPT_ROOT}")/PX4-Autopilot}"
export MUPT_SIMULATION_DIR="${MUPT_ROOT}/simulation"
export GZ_SIM_RESOURCE_PATH="${MUPT_SIMULATION_DIR}/models:${MUPT_SIMULATION_DIR}/worlds:${GZ_SIM_RESOURCE_PATH:-}"
unset _MUPT_ROOT _MUPT_HAD_NOUNSET
