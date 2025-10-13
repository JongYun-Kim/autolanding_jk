#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere; script resolves to its folder.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build & run the agent container on server target; also do a CUDA check.
python3 "${SCRIPT_DIR}/container_agent.py" \
  --target server \
  --image rl-agent:server \
  --frame-stack 3 \
  --height 84 \
  --width 84 \
  --drone-state-dim 7 \
  --feature-dim 128 \
  --hidden-dim 1024 \
  --freq-hz 32 \
  --batch 1 \
  --seed 42 \
  --cuda-test