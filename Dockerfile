FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

WORKDIR /app

RUN python -m pip install --no-cache-dir --upgrade pip uv

COPY pyproject.toml setup.py README.md /app/
RUN uv sync

# Install flash-attn for GPU optimization (skip on ARM64)
RUN if [ "$(uname -m)" = "x86_64" ]; then \
    uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl; \
    fi

COPY . /app
RUN uv pip install --no-deps /app/src/fishfarm

ENV PYTHONPATH=/app/src:/app/src/fishfarm
ENV HF_HOME=/huggingface
ENV HF_HUB_ENABLE_HF_TRANSFER=1

CMD ["uv", "run", "python", "watcher.py"]