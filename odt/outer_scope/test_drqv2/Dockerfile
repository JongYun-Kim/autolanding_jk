# docker build --build-arg TARGET=server -t rl-agent:server .
# docker build --build-arg TARGET=jetson --build-arg TORCH_WHL_URL=... -t rl-agent:jetson .

ARG TARGET=server
ARG TORCH_WHL_URL=""
ARG VISION_WHL_URL=""
ARG EXTRA_PIP_ARGS=""

FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime AS server_base
FROM nvcr.io/nvidia/l4t-cuda:12.6.11-runtime-ubuntu22.04 AS jetson_base
FROM dustynv/l4t-pytorch:r36.4.0 AS jetson_torch_base

FROM server_base AS choose_server
FROM jetson_base  AS choose_jetson
FROM jetson_torch_base AS choose_jetson_torch
# Pick the base matching TARGET
FROM choose_${TARGET} AS base

# Basic OS deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-setuptools ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Jetson requires installing torch/vision wheels (aarch64)
RUN if [ "${TARGET}" = "jetson" ]; then \
        if [ -z "${TORCH_WHL_URL}" ]; then \
            echo "ERROR: Set TORCH_WHL_URL to Jetson aarch64 torch wheel" && exit 1; \
        fi; \
        python3 -m pip install --no-cache-dir ${EXTRA_PIP_ARGS} "${TORCH_WHL_URL}"; \
        if [ -n "${VISION_WHL_URL}" ]; then \
            python3 -m pip install --no-cache-dir ${EXTRA_PIP_ARGS} "${VISION_WHL_URL}"; \
        fi; \
    else \
        echo "Server target: using preinstalled PyTorch"; \
    fi

WORKDIR /app

# Project layout: put the agent trio under /app/rl
COPY rl ./rl
# Optional pinned deps
COPY requirements.txt .
RUN if [ -f requirements.txt ]; then \
        python3 -m pip install --no-cache-dir -r requirements.txt; \
    fi

# Default runtime envs (can be overridden by docker run -e ...)
ENV WEIGHTS=/app/weights/policy.pt \
    FRAME_STACK=3 H=84 W=84 DRONE_STATE_DIM=7 FEATURE_DIM=50 HIDDEN_DIM=1024 \
    DEVICE="" FREQ_HZ=20.0 BATCH=1 SEED=0

# Prepare weights dir; leave file absent (run_agent.py will random-init if not found)
RUN mkdir -p /app/weights && echo "ready" > /app/weights/.keep

CMD ["python3", "rl/run_agent.py"]
