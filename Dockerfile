ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.02-py3
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace/camplusplus

# Build an ffmpeg that definitely exposes the encoders needed by this repo.
# We keep torch from the NGC base image, but build torchaudio separately
# because recent NGC images may not ship it by default.
RUN apt-get update && apt-get install -y --no-install-recommends \
    autoconf \
    automake \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    libopus-dev \
    libsndfile1 \
    libsndfile1-dev \
    libtool \
    libvo-amrwbenc-dev \
    nasm \
    ninja-build \
    pkg-config \
    wget \
    xz-utils \
    yasm \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    cd /tmp; \
    wget -O ffmpeg.tar.xz https://ffmpeg.org/releases/ffmpeg-7.1.1.tar.xz; \
    tar -xf ffmpeg.tar.xz; \
    cd ffmpeg-7.1.1; \
    ./configure \
      --prefix=/usr/local \
      --disable-debug \
      --disable-doc \
      --enable-gpl \
      --enable-version3 \
      --enable-libopus \
      --enable-libvo-amrwbenc; \
    make -j"$(nproc)"; \
    make install; \
    hash -r; \
    ffmpeg -hide_banner -encoders | grep -E 'libopus|aac|libvo_amrwbenc|pcm_alaw|pcm_mulaw'; \
    rm -rf /tmp/ffmpeg-7.1.1 /tmp/ffmpeg.tar.xz

COPY req_no_torch.txt /tmp/req_no_torch.txt

RUN python -m pip install -r /tmp/req_no_torch.txt && \
    set -eux; \
    TORCH_MM=$(python -c "import torch; print('.'.join(torch.__version__.split('+')[0].split('.')[:2]))"); \
    AUDIO_REF="release/${TORCH_MM}"; \
    cd /tmp; \
    wget -O audio.tar.gz "https://github.com/pytorch/audio/archive/refs/heads/${AUDIO_REF}.tar.gz"; \
    tar -xzf audio.tar.gz; \
    BUILD_SOX=0 USE_FFMPEG=1 python -m pip install --no-build-isolation "/tmp/audio-${AUDIO_REF//\//-}" && \
    python - <<'PY'
import torch
import torchaudio
print("torch:", torch.__version__)
print("torchaudio:", torchaudio.__version__)
print("cuda:", torch.version.cuda)
PY

CMD ["/bin/bash"]
