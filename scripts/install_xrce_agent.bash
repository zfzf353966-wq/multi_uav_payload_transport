#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup_env.bash"

SOURCE_DIR="${MUPT_ROOT}/.deps/micro_xrce_agent_v2.4.2/source"
BUILD_DIR="${MUPT_ROOT}/.deps/micro_xrce_agent_v2.4.2/build_humble"
INSTALL_DIR="${MUPT_ROOT}/.deps/micro_xrce_agent_v2.4.2/install"
REPOSITORY="https://github.com/eProsima/Micro-XRCE-DDS-Agent.git"

if [[ -x "${INSTALL_DIR}/bin/MicroXRCEAgent" && \
      -e "${INSTALL_DIR}/lib/libmicroxrcedds_agent.so.2.4" ]]; then
  echo "Micro XRCE-DDS Agent v2.4.2 is already installed at ${INSTALL_DIR}."
  exit 0
fi

mkdir -p "$(dirname "${SOURCE_DIR}")"
if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
  git clone --branch v2.4.2 --depth 1 "${REPOSITORY}" "${SOURCE_DIR}"
fi

cmake -S "${SOURCE_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
  -DUAGENT_USE_SYSTEM_FASTCDR=ON \
  -DUAGENT_USE_SYSTEM_FASTDDS=ON \
  -DUAGENT_BUILD_EXAMPLES=OFF \
  -DUAGENT_BUILD_TESTS=OFF
cmake --build "${BUILD_DIR}" --parallel "$(nproc)"
cmake --install "${BUILD_DIR}"

echo "Installed Humble-compatible Micro XRCE-DDS Agent v2.4.2 at ${INSTALL_DIR}."
