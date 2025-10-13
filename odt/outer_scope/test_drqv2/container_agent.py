#!/usr/bin/env python3
"""
Host-side Python orchestrator for building & running the RL *agent* container.

- TARGET=server  : base=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime  (x86_64 server)
- TARGET=jetson  : base=nvcr.io/nvidia/l4t-cuda:12.6.11-runtime-ubuntu22.04 (Jetson Orin / JetPack 6.2)

Jetson requires TORCH_WHL_URL (and optional VISION_WHL_URL) for aarch64 wheels.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

def sh(cmd, check=True, capture_output=False, env=None):
    print(">>", " ".join(cmd))
    return subprocess.run(cmd, check=check, capture_output=capture_output, text=True, env=env)

def ensure_tooling():
    if shutil.which("docker") is None:
        print("ERROR: docker: NOT INSTALLED", file=sys.stderr)
        sys.exit(1)
    try:
        sh(["docker", "--version"])
        sh(["docker", "info"], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("ERROR: CANNOT access docker daemon. Check service/permissions.", file=sys.stderr)
        sys.exit(1)

def build_image(args):
    build_args = [
        "docker", "build",
        "-t", args.image,
        ".",
        "--build-arg", f"TARGET={args.target}",
    ]
    if args.target == "jetson":
        if not args.torch_whl:
            print("ERROR: Jetson build must get --torch-whl URL !!", file=sys.stderr)
            sys.exit(1)
        build_args += ["--build-arg", f"TORCH_WHL_URL={args.torch_whl}"]
        if args.vision_whl:
            build_args += ["--build-arg", f"VISION_WHL_URL={args.vision_whl}"]
        if args.extra_pip_args:
            build_args += ["--build-arg", f"EXTRA_PIP_ARGS={args.extra_pip_args}"]
    if args.target == "jetson_torch":
        if args.torch_whl or args.vision_whl:
            print("TORCH_WHL_URL and VISION_WHL_URL are ignored for target=jetson_torch")

    if args.no_cache:
        build_args.append("--no-cache")

    sh(build_args)

def run_container(args):
    cwd = str(HERE)
    Path(args.weights).parent.mkdir(parents=True, exist_ok=True)

    # Environment variables to the container (consumed by rl/run_agent.py)
    envs = {
        "FRAME_STACK": str(args.frame_stack),
        "H":           str(args.height),
        "W":           str(args.width),
        "DRONE_STATE_DIM": str(args.drone_state_dim),
        "FEATURE_DIM": str(args.feature_dim),
        "HIDDEN_DIM":  str(args.hidden_dim),
        "DEVICE":      args.device,
        "FREQ_HZ":     str(args.freq_hz),
        "BATCH":       str(args.batch),
        "SEED":        str(args.seed),
        "WEIGHTS":     f"/app/{args.weights}",
    }

    env_flags = []
    for k, v in envs.items():
        env_flags += ["-e", f"{k}={v}"]

    run_cmd = [
        "docker", "run", "--rm", "-it",
        "--gpus", "all",
        # If you prefer live-source mounting during dev, uncomment the next line:
        # "-v", f"{cwd}:/app",
        args.image,
        "python3", "rl/run_agent.py"
    ]
    run_cmd = run_cmd[:-3] + env_flags + run_cmd[-3:]
    sh(run_cmd)

def quick_cuda_test(args):
    """Run torch.cuda.is_available() inside container."""
    code = 'import torch;print("cuda_available", torch.cuda.is_available())'
    run_cmd = [
        "docker", "run", "--rm",
        "--gpus", "all",
        args.image,
        "python3", "-c", code,
    ]
    sh(run_cmd)

def parse_args():
    ap = argparse.ArgumentParser(description="Build & Run RL agent container (host Python)")
    ap.add_argument("--target", choices=["server", "jetson"], default="server")
    ap.add_argument("--image", default="rl-agent:server", help="Docker image tag to build/run")
    ap.add_argument("--torch-whl", default="", help="[jetson only] PyTorch aarch64 wheel URL")
    ap.add_argument("--vision-whl", default="", help="[jetson only] torchvision aarch64 wheel URL (optional)")
    ap.add_argument("--extra-pip-args", default="", help="[jetson only] pip extra args")
    ap.add_argument("--skip-build", action="store_true", help="Skip docker build step if already built")
    ap.add_argument("--no-cache", action="store_true", help="Build without cache")
    ap.add_argument("--cuda-test", action="store_true", help="Run a small CUDA availability test")

    # Agent runtime knobs
    ap.add_argument("--frame-stack", type=int, default=3)
    ap.add_argument("--height", type=int, default=84)
    ap.add_argument("--width", type=int, default=84)
    ap.add_argument("--drone-state-dim", type=int, default=11)
    ap.add_argument("--feature-dim", type=int, default=128)
    ap.add_argument("--hidden-dim", type=int, default=1024)
    ap.add_argument("--device", default="", help='Override device (e.g., "cpu" or "cuda")')
    ap.add_argument("--freq-hz", type=float, default=32.0)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weights", default="weights/policy.pt", help="Path under project root; mounted as /app/<path> in container")

    return ap.parse_args()

def main():
    args = parse_args()
    ensure_tooling()

    # Tag defaulting similar to the example
    if args.image == "rl-agent:server" and args.target == "jetson":
        args.image = "rl-agent:jetson"

    if not args.skip_build:
        build_image(args)

    if args.cuda_test:
        quick_cuda_test(args)

    run_container(args)

if __name__ == "__main__":
    main()
