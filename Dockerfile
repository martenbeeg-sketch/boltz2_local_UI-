FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=UTC \
    CONDA_DIR=/opt/conda \
    CONDA_ENV_NAME=boltz2 \
    TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
    PATH=/opt/conda/bin:$PATH \
    CUDA_HOME=/usr/local/cuda \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        curl \
        git \
        libgfortran5 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        wget && \
    rm -rf /var/lib/apt/lists/*

RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p "${CONDA_DIR}" && \
    rm -f /tmp/miniforge.sh

SHELL ["/bin/bash", "-lc"]

RUN mamba create -y -n "${CONDA_ENV_NAME}" -c conda-forge python=3.10 pip && \
    mamba run -n "${CONDA_ENV_NAME}" pip install --no-cache-dir --upgrade pip setuptools wheel && \
    mamba run -n "${CONDA_ENV_NAME}" pip install --no-cache-dir \
        --index-url "${TORCH_INDEX_URL}" \
        torch torchvision torchaudio && \
    mamba run -n "${CONDA_ENV_NAME}" pip install --no-cache-dir \
        boltz \
        cuequivariance-torch \
        cuequivariance-ops-torch-cu12 && \
    mamba run -n "${CONDA_ENV_NAME}" python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda)" && \
    mamba clean -a -y

RUN echo "source ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV_NAME}" >> /root/.bashrc

WORKDIR /work

ENTRYPOINT ["/bin/bash", "-lc", "source /opt/conda/etc/profile.d/conda.sh && conda activate boltz2 && exec boltz \"$@\"", "--"]
CMD ["--help"]
