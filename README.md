# Boltz-2 Local UI

Host-side Streamlit app that launches `boltz predict` inside Docker (`ovoex-boltz2`).

## Repository layout

- `app.py`: Streamlit interface
- `boltz_runner.py`: YAML generation, MSA cache handling, and docker command execution
- `visualization.py`: structure + confidence plots
- `run.sh`: launcher (auto-port + auto-browser)
- `Dockerfile`: Boltz-2 container image
- `requirements.txt`: runtime Python deps
- `environment.yml`: optional conda environment
- `.streamlit/config.toml`: streamlit defaults

## Quick start

### 1) Build Docker image

```bash
docker build -t ovoex-boltz2 .
```

### 2) Create conda environment (recommended)

```bash
conda env create -f environment.yml
conda activate boltz2-ui
```

### 3) Run UI

```bash
bash run.sh
```

The launcher selects the first free port between `8501` and `8510` and opens the browser automatically.

## One-command launcher

If `/home/user/.local/bin` is on your `PATH`, you can run:

```bash
app
```

This uses `/home/user/mambaforge/bin/conda run -n boltz2-ui` and starts the app from this project directory.

## Runtime defaults

```bash
export BOLTZ_DOCKER_IMAGE=ovoex-boltz2
export BOLTZ_CACHE_DIR=/mnt/db/reference_files/boltz_models
export BOLTZ_MSA_REPOSITORY_DIR=/mnt/db/reference_files/boltz_models/msa_repository
export BOLTZ_DOCKER_ARGS="--ipc=host --shm-size=48G"
export BOLTZ_TIMEOUT_SECONDS=1800
export BOLTZ_RESULTS_DIR=./results
export STREAMLIT_PORT_START=8501
export STREAMLIT_PORT_END=8510
```

You can also override these inside the app sidebar.

## Implemented behavior

- Required `Job name`; each run writes to a dedicated job folder.
- `New prediction` clears the form/session state.
- GPU selection (`0`, `1`, `all`) controls Docker `--gpus`.
- Affinity/property support for ligand runs.
- Physics toggle (`--use_potentials`).
- YAML preview popover before execution.
- Local MSA repository by sequence hash.
- MSA cache hardening:
  - validates cached `.a3m` before use
  - strips trailing NUL bytes when possible
  - quarantines invalid files if still broken
  - writes cache atomically to avoid partial file corruption

## Git workflow (suggested)

```bash
git init
git add .
git commit -m "Initial Boltz-2 local UI"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```
