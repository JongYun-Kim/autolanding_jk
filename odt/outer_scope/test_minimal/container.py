#!/usr/bin/env python3
"""
Host-side Python orchestrator for building & running the RL policy container.

- TARGET=server  : base=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime (x86_64 서버 테스트용)
- TARGET=jetson  : base=nvcr.io/nvidia/l4t-cuda:12.6.11-runtime-ubuntu22.04 (Jetson Orin Nano / JetPack 6.2)

Jetson에서는 TORCH_WHL_URL(필수), VISION_WHL_URL(선택)로 aarch64 전용 휠을 설치해야 GPU 가속이 됨.
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
    # 기본 동작 점검
    try:
        sh(["docker", "--version"])
        sh(["docker", "info"], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print("ERROR: CANNOT access docker daemon. Check your service or the permissions.", file=sys.stderr)
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

    if args.obs_dim:
        build_args += ["--build-arg", f"OBS_DIM={args.obs_dim}"]
    if args.action_dim:
        build_args += ["--build-arg", f"ACTION_DIM={args.action_dim}"]

    if args.no_cache:
        build_args.append("--no-cache")

    sh(build_args)

def run_container(args):
    cwd = str(HERE)
    Path(args.weights).parent.mkdir(parents=True, exist_ok=True)

    # 컨테이너로 넘길 환경변수
    envs = {
        "OBS_DIM": str(args.obs_dim),
        "ACTION_DIM": str(args.action_dim),
        "DISCRETE": "1" if args.discrete else "0",
        "DET": "1" if args.det else "0",
        "WEIGHTS": f"/app/{args.weights}",
    }
    env_flags = []
    for k, v in envs.items():
        env_flags += ["-e", f"{k}={v}"]

    # GPU 플래그: 서버/Jetson 모두 --gpus all 사용(Orin은 nvidia-container-runtime이 기본이어야 함)
    run_cmd = [
        "docker", "run", "--rm", "-it",
        "--gpus", "all",
        # "-v", f"{cwd}:/app",
        args.image,
        "python3", args.script
    ]
    # Unpack env flags before the image name & script
    run_cmd = run_cmd[:-3] + env_flags + run_cmd[-3:]
    sh(run_cmd)

def quick_cuda_test(args):
    """컨테이너 내부에서 torch.cuda.is_available() 간단 체크."""
    code = 'import torch;print("cuda_available", torch.cuda.is_available())'
    run_cmd = [
        "docker", "run", "--rm",
        "--gpus", "all",
        args.image,
        "python3", "-c", code,
    ]
    sh(run_cmd)

def parse_args():
    ap = argparse.ArgumentParser(description="Build & Run RL policy container (host Python)")
    ap.add_argument("--target", choices=["server", "jetson"], default="server",
                    help="Build target: server(amd64) OR jetson(arm64, JetPack 6.2)")
    ap.add_argument("--image", default="rl-policy:server", help="Docker image tag")
    ap.add_argument("--torch-whl", default="", help="[jetson only] PyTorch aarch64 wheel URL")
    ap.add_argument("--vision-whl", default="", help="[jetson only] torchvision aarch64 wheel URL (optional)")
    ap.add_argument("--extra-pip-args", default="", help="[jetson only] pip extra args")
    ap.add_argument("--skip-build", action="store_true", help="Skips build if you have one already.")
    ap.add_argument("--obs-dim", type=int, default=16)
    ap.add_argument("--action-dim", type=int, default=2)
    ap.add_argument("--discrete", action="store_true")
    ap.add_argument("--det", action="store_true", help="deterministic action")
    ap.add_argument("--weights", default="weights/policy.pt", help="Weights dir on the host machine")
    ap.add_argument("--script", default="rl/run_policy.py",
                    help="policy script dir in the container")
    ap.add_argument("--cuda-test", action="store_true", help="Runs torch.cuda.is_available() after build")
    ap.add_argument("--no-cache", action="store_true", help="No cache in docker build")
    return ap.parse_args()

def main():
    args = parse_args()
    ensure_tooling()

    # target에 따라 기본 이미지 태그 바꿔주기
    if args.image == "rl-policy:server" and args.target == "jetson":
        args.image = "rl-policy:jetson"

    if not args.skip_build:
        build_image(args)

    if args.cuda_test:
        quick_cuda_test(args)

    run_container(args)

if __name__ == "__main__":
    main()
