ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.12-py3
FROM ${BASE_IMAGE}

ARG FFMPEG_VERSION=7.1.1

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_PREFER_BINARY=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace/camplusplus

RUN apt-get update && apt-get install -y --no-install-recommends \
    autoconf \
    automake \
    build-essential \
    ca-certificates \
    cmake \
    gfortran \
    git \
    liblapack-dev \
    libopenblas-dev \
    libopus-dev \
    libsndfile1 \
    libtool \
    libvo-amrwbenc-dev \
    nasm \
    pkg-config \
    python3-dev \
    wget \
    yasm \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu's packaged ffmpeg in the container image lacks libvo_amrwbenc, but the
# project's AMR-WB path depends on it. Build ffmpeg once with the required
# encoder support so AMR-WB generation works on DGX Spark as well.
RUN set -eux; \
    cd /tmp; \
    wget -O ffmpeg.tar.xz "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz"; \
    tar -xf ffmpeg.tar.xz; \
    cd "ffmpeg-${FFMPEG_VERSION}"; \
    ./configure \
      --prefix=/usr/local \
      --pkg-config-flags="--static" \
      --extra-cflags="-I/usr/local/include" \
      --extra-ldflags="-L/usr/local/lib" \
      --extra-libs="-lpthread -lm" \
      --bindir=/usr/local/bin \
      --enable-gpl \
      --enable-libopus \
      --enable-libvo-amrwbenc \
      --enable-shared \
      --disable-debug \
      --disable-doc; \
    make -j"$(nproc)"; \
    make install; \
    ldconfig; \
    /usr/local/bin/ffmpeg -hide_banner -encoders | grep -q 'libvo_amrwbenc'; \
    rm -rf /tmp/ffmpeg*

COPY requirements.txt /tmp/requirements.txt
COPY requirements.docker.txt /tmp/requirements.docker.txt

# DGX Spark is ARM64 (aarch64). Prefer the PyTorch stack bundled in the NGC
# image and install a Python 3.12-compatible dependency set for Docker builds.
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install -r /tmp/requirements.docker.txt && \
    python3 - <<'PY'
import re
import subprocess
import sys
import torch

version = torch.__version__
match = re.match(r"(\d+\.\d+\.\d+)", version)
torchaudio_version = match.group(1) if match else version.split("+")[0]
cmd = [sys.executable, "-m", "pip", "install", "--no-deps", f"torchaudio=={torchaudio_version}"]
print("Installing torchaudio with:", " ".join(cmd))
subprocess.run(cmd, check=True)
PY

COPY . /workspace/camplusplus

ENV PYTHONPATH=/workspace/camplusplus

CMD ["/bin/bash"]
