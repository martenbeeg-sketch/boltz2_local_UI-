#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/home/user/programs/git-projects/boltz2-app-local"
CONDA_BIN="/home/user/mambaforge/bin/conda"
ENV_NAME="${BOLTZ_APP_ENV:-boltz2-ui}"

cd "${APP_DIR}"
exec "${CONDA_BIN}" run -n "${ENV_NAME}" bash run.sh "$@"
