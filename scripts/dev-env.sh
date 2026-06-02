#!/usr/bin/env bash
set -euo pipefail

DOWNLOAD_ORT="${DOWNLOAD_ORT:-1}"
RUN_CHECK="${RUN_CHECK:-1}"
RUN_BUILD="${RUN_BUILD:-1}"
RUN_SMOKE="${RUN_SMOKE:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ORT_VERSION="1.24.4"
ORT_DIR="${REPO_ROOT}/vendor/onnxruntime"
ORT_ARCHIVE="${ORT_DIR}/onnxruntime-linux-x64-${ORT_VERSION}.tgz"
ORT_EXTRACT_ROOT="${ORT_DIR}/onnxruntime-linux-x64-${ORT_VERSION}"
ORT_SO="${ORT_EXTRACT_ROOT}/onnxruntime-linux-x64-${ORT_VERSION}/lib/libonnxruntime.so"

if [[ "${DOWNLOAD_ORT}" == "1" && ! -f "${ORT_SO}" ]]; then
  mkdir -p "${ORT_DIR}"

  if [[ ! -f "${ORT_ARCHIVE}" ]]; then
    URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz"
    echo "Downloading ONNX Runtime from ${URL}"
    curl -fsSL -o "${ORT_ARCHIVE}" "${URL}"
  fi

  rm -rf "${ORT_EXTRACT_ROOT}"
  mkdir -p "${ORT_EXTRACT_ROOT}"
  tar -xzf "${ORT_ARCHIVE}" -C "${ORT_EXTRACT_ROOT}"
fi

if [[ ! -f "${ORT_SO}" ]]; then
  echo "ONNX Runtime shared library not found at: ${ORT_SO}" >&2
  exit 1
fi

export ORT_DYLIB_PATH="${ORT_SO}"
echo "ORT_DYLIB_PATH=${ORT_DYLIB_PATH}"

cd "${REPO_ROOT}"

if [[ "${RUN_CHECK}" == "1" ]]; then
  cargo check --locked
fi

if [[ "${RUN_BUILD}" == "1" ]]; then
  cargo build --locked
fi

if [[ "${RUN_SMOKE}" == "1" ]]; then
  EXTENSION_SO=""
  for search_dir in "${REPO_ROOT}/target/debug" "${REPO_ROOT}/target/debug/deps"; do
    EXTENSION_SO="$(find "${search_dir}" -maxdepth 1 -type f -name '*deepface*.so' | sort | head -n 1 || true)"
    if [[ -n "${EXTENSION_SO}" ]]; then
      break
    fi
  done
  if [[ -z "${EXTENSION_SO}" || ! -f "${EXTENSION_SO}" ]]; then
    echo "Extension .so not found in target/debug or target/debug/deps" >&2
    exit 1
  fi

  php -n -d "extension=${EXTENSION_SO}" "${REPO_ROOT}/scripts/smoke_extension.php"
fi
