# docker build --build-arg TARGET=server -t rl-policy:server .
# docker build --build-arg TARGET=jetson --build-arg TORCH_WHL_URL=... -t rl-policy:jetson .

ARG TARGET=server
ARG TORCH_WHL_URL=""
ARG VISION_WHL_URL=""
ARG EXTRA_PIP_ARGS=""

FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime AS server_base
FROM nvcr.io/nvidia/l4t-cuda:12.6.11-runtime-ubuntu22.04 AS jetson_base

FROM server_base AS choose_server
FROM jetson_base  AS choose_jetson
FROM choose_${TARGET} AS base

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-setuptools ca-certificates && \
    rm -rf /var/lib/apt/lists/*

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
COPY rl ./rl
COPY requirements.txt .
RUN if [ -f requirements.txt ]; then \
        python3 -m pip install --no-cache-dir -r requirements.txt; \
    fi

ENV DISCRETE=0 DET=1 WEIGHTS=/app/weights/policy.pt
ARG OBS_DIM=16
ARG ACTION_DIM=3

ENV OBS_DIM=${OBS_DIM}
ENV ACTION_DIM=${ACTION_DIM}

RUN python3 - <<PY
from rl.model import MinimalMLP, PolicyConfig
import torch, os

obs_dim = int(os.environ["OBS_DIM"])
action_dim = int(os.environ["ACTION_DIM"])

os.makedirs("weights", exist_ok=True)
policy = MinimalMLP(PolicyConfig(obs_dim=obs_dim, action_dim=action_dim))
torch.save(policy.state_dict(), "weights/policy.pt")
print("Dummy policy created in container")
PY

CMD ["python3", "rl/run_policy.py"]