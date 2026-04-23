#!/usr/bin/env bash
set -euo pipefail

PORT="${STREAMLIT_PORT_START:-8501}"
MAX_PORT="${STREAMLIT_PORT_END:-8510}"

find_free_port() {
  local port
  for ((port=PORT; port<=MAX_PORT; port++)); do
    if ! ss -ltn "( sport = :${port} )" | tail -n +2 | grep -q .; then
      echo "${port}"
      return 0
    fi
  done
  return 1
}

if [[ -n "${CONDA_DEFAULT_ENV:-}" || -n "${MAMBA_DEFAULT_ENV:-}" ]]; then
  echo "Using active environment: ${CONDA_DEFAULT_ENV:-${MAMBA_DEFAULT_ENV}}"
else
  if [[ ! -d .venv ]]; then
    python3 -m venv .venv
  fi
  source .venv/bin/activate
fi

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

PORT="$(find_free_port)" || {
  echo "No free port found between ${STREAMLIT_PORT_START:-8501} and ${STREAMLIT_PORT_END:-8510}" >&2
  exit 1
}

export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_SERVER_HEADLESS=true

URL="http://localhost:${PORT}"

open_browser() {
  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "${URL}" >/dev/null 2>&1 || true
    return 0
  fi
  python -c "import webbrowser; webbrowser.open('${URL}')" >/dev/null 2>&1 || true
}

cleanup() {
  if [[ -n "${STREAMLIT_PID:-}" ]] && kill -0 "${STREAMLIT_PID}" >/dev/null 2>&1; then
    kill "${STREAMLIT_PID}" >/dev/null 2>&1 || true
    wait "${STREAMLIT_PID}" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT INT TERM

echo "Starting Streamlit on ${URL}"
streamlit run app.py --server.port "${PORT}" --server.address 0.0.0.0 &
STREAMLIT_PID=$!

sleep 2
open_browser

wait "${STREAMLIT_PID}"
