#!/usr/bin/env bash
# Equal-budget controls: unchanged continuation, reward-only, bounded-mean-only.
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/setup_env.bash"
OUTPUT_ROOT="${1:-${MUPT_ROOT}/runs/direct_ppo_3d/ablation_v1}"
INITIAL_MODEL="${MUPT_ROOT}/runs/direct_ppo_3d/curriculum_v2/static/best_model.zip"
if [[ -e "${OUTPUT_ROOT}/experiment.json" ]]; then
  echo "Output already contains an experiment. Choose a new output directory." >&2
  exit 2
fi
mkdir -p "${OUTPUT_ROOT}"
python - "${OUTPUT_ROOT}" "${INITIAL_MODEL}" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
root, model = Path(sys.argv[1]), Path(sys.argv[2])
manifest = dict(status="running", started_utc=datetime.now(timezone.utc).isoformat(),
                initial_model=str(model), model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                arms={"control": [2,"gaussian"], "reward": [3,"gaussian"], "bounded": [2,"bounded"]},
                timesteps_per_arm=200000, training_seed=42,
                validation_seed=60000, validation_episodes=100,
                evaluation_seed=700000, evaluation_episodes=200)
(root / "experiment.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY
declare -a PIDS=()
cleanup() {
  for pid in "${PIDS[@]}"; do kill -TERM -- "-${pid}" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM
COMMON=(--stage static --timesteps 200000 --initial-model "${INITIAL_MODEL}"
  --seed 42 --learning-rate 0.0001 --n-envs 8
  --validation-episodes 100 --validation-seed 60000 --eval-episodes 200 --eval-seed 700000)
for arm in control reward bounded; do
  reward=2
  policy=gaussian
  if [[ "${arm}" == reward ]]; then reward=3; fi
  if [[ "${arm}" == bounded ]]; then policy=bounded; fi
  mkdir -p "${OUTPUT_ROOT}/${arm}"
  setsid "${SCRIPT_DIR}/train_direct_ppo.bash" "${COMMON[@]}" \
    --reward-version "${reward}" --policy "${policy}" --output-dir "${OUTPUT_ROOT}/${arm}" \
    >"${OUTPUT_ROOT}/${arm}/training.log" 2>&1 &
  PIDS+=("$!")
done
FAILED=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then FAILED=1; fi
done
python - "${OUTPUT_ROOT}" "${FAILED}" <<'PY'
import csv, json, sys
from datetime import datetime, timezone
from pathlib import Path
root = Path(sys.argv[1])
manifest = json.loads((root / "experiment.json").read_text())
manifest.update(status="failed" if int(sys.argv[2]) else "completed", finished_utc=datetime.now(timezone.utc).isoformat())
rows = []
for arm in manifest["arms"]:
    path = root / arm / "metrics.json"
    if path.exists():
        metrics = json.loads(path.read_text())
        rows.append(dict(arm=arm, gate_passed=metrics["gate_passed"], selected_training_steps=metrics["selected_training_steps"],
                         **{key: value for key, value in metrics["evaluation"].items() if key != "mean_return"}))
(root / "experiment.json").write_text(json.dumps(manifest, indent=2) + "\n")
(root / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
if rows:
    with (root / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
print(json.dumps(rows, indent=2))
PY
exit "${FAILED}"
