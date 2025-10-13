#!/usr/bin/env bash
set -euo pipefail

# Run container.py from the dir of the script, making it safe to be run anywhere
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "${SCRIPT_DIR}/container.py" \
  --target server \
  --image rl-policy:server \
  --obs-dim 16 \
  --action-dim 4 \
  --det \
  --cuda-test