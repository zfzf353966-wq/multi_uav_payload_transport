#!/usr/bin/env bash
# Preserve the full original city; add three dynamic obstacles by default.
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_ppo_demo.bash" --environment city "$@"
