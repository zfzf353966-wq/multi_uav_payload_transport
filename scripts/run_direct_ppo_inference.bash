#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/setup_env.bash"
export PYTHONPATH="${MUPT_ROOT}/src/logistics_planner${PYTHONPATH:+:${PYTHONPATH}}"
exec python -m logistics_planner.direct_ppo_inference "$@"
