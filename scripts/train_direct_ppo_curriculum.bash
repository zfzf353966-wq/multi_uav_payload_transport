#!/usr/bin/env bash
# Stops at the first failed holdout gate. Never promotes a failing checkpoint.
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/setup_env.bash"
OUTPUT_ROOT="${MUPT_ROOT}/runs/direct_ppo_3d/curriculum"
STAGE_STEPS="${MUPT_STAGE_STEPS:-200000}"
PREVIOUS_MODEL=""
START_STAGE="navigation"
while (( $# )); do
  case "$1" in
    --output-root) OUTPUT_ROOT="${2:?missing output root}"; shift 2 ;;
    --start-stage) START_STAGE="${2:?missing stage}"; shift 2 ;;
    --initial-model) PREVIOUS_MODEL="${2:?missing initial model}"; shift 2 ;;
    *) echo "Usage: $0 [--output-root DIR] [--start-stage navigation|static|dynamic|multi] [--initial-model MODEL.zip]" >&2; exit 2 ;;
  esac
done
case "${START_STAGE}" in navigation|static|dynamic|multi) ;; *) echo "Unknown start stage" >&2; exit 2 ;; esac
if [[ "${START_STAGE}" != navigation && ! -f "${PREVIOUS_MODEL}" ]]; then
  echo "Resuming the curriculum requires an existing --initial-model." >&2
  exit 2
fi
ACTIVE=0
for stage in navigation static dynamic multi; do
  if [[ "${stage}" == "${START_STAGE}" ]]; then ACTIVE=1; fi
  if (( ACTIVE == 0 )); then continue; fi
  stage_dir="${OUTPUT_ROOT}/${stage}"
  mkdir -p "${stage_dir}"
  args=(--stage "${stage}" --timesteps "${STAGE_STEPS}" --eval-episodes 200 --output-dir "${stage_dir}")
  if [[ -n "${PREVIOUS_MODEL}" ]]; then args+=(--initial-model "${PREVIOUS_MODEL}"); fi
  if [[ "${stage}" == multi ]]; then args+=(--randomize --round-trip); fi
  "${SCRIPT_DIR}/train_direct_ppo.bash" "${args[@]}" | tee "${stage_dir}/training.log"
  python - "${stage_dir}/metrics.json" <<'PY'
import json, sys
metrics = json.load(open(sys.argv[1]))
if not metrics["gate_passed"]:
    raise SystemExit("Holdout gate failed; inspect metrics before continuing curriculum.")
PY
  PREVIOUS_MODEL="$(python - "${stage_dir}/metrics.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["selected_model"])
PY
)"
done
