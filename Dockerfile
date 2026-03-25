ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:24.08-py3
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_PREFER_BINARY=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace/camplusplus

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    gfortran \
    libsndfile1 \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

# DGX Spark is ARM64 (aarch64). Prefer the PyTorch stack provided by the NGC
# base image instead of forcing x86_64 CUDA wheels from download.pytorch.org.
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install -r /tmp/requirements.txt

COPY . /workspace/camplusplus

ENV PYTHONPATH=/workspace/camplusplus

CMD ["/bin/bash"]
