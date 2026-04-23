# Installation

## Prerequisites

- NVIDIA driver + `nvidia-container-toolkit`
- Docker with GPU support
- Conda or Mamba

## Steps

```bash
cd /home/user/programs/git-projects/boltz2-app-local
conda env create -f environment.yml
conda activate boltz2-ui
docker build -t ovoex-boltz2 .
bash run.sh
```

Or with the global launcher command:

```bash
app
```

## Optional runtime variables

```bash
export BOLTZ_DOCKER_IMAGE=ovoex-boltz2
export BOLTZ_CACHE_DIR=/mnt/db/reference_files/boltz_models
export BOLTZ_MSA_REPOSITORY_DIR=/mnt/db/reference_files/boltz_models/msa_repository
export BOLTZ_DOCKER_ARGS="--ipc=host --shm-size=48G"
export BOLTZ_RESULTS_DIR=/home/user/programs/git-projects/boltz2-app-local/results
```
