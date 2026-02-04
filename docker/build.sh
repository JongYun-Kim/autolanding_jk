#!/usr/bin/env bash
set -euo pipefail

# =========================
# Config
IMAGE_NAME="rl-landing"
IMAGE_TAG="latest"
DOCKERFILE="Dockerfile"
CONTEXT_DIR="."

# User config (defaults, can be overridden via environment)
BUILD_UID="${BUILD_UID:-$(id -u)}"
BUILD_GID="${BUILD_GID:-$(id -g)}"
BUILD_USERNAME="${BUILD_USERNAME:-agent}"
BUILD_PASSWORD="${BUILD_PASSWORD:-rl}"

# Sanity checks
if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker not found"
  exit 1
fi
if [ ! -f "${DOCKERFILE}" ]; then
  echo "Error: ${DOCKERFILE} not found in current directory"
  exit 1
fi

# Build
echo "Building image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "  UID: ${BUILD_UID}, GID: ${BUILD_GID}, USERNAME: ${BUILD_USERNAME}"

docker build \
  --pull \
  --build-arg UID="${BUILD_UID}" \
  --build-arg GID="${BUILD_GID}" \
  --build-arg USERNAME="${BUILD_USERNAME}" \
  --build-arg PASSWORD="${BUILD_PASSWORD}" \
  -t "${IMAGE_NAME}:${IMAGE_TAG}" \
  -f "${DOCKERFILE}" \
  "${CONTEXT_DIR}"

echo "Build complete."
docker images "${IMAGE_NAME}:${IMAGE_TAG}"
