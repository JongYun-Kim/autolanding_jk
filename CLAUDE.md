# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Deep reinforcement learning system for autonomous drone landing with gimbal-mounted camera control, built on PyBullet physics simulation and the DRQv2 algorithm. The agent learns to land a drone on a moving ground vehicle using RGB camera observations, with curriculum learning to progressively enable gimbal control.

## Running Training

```bash
# Gimbal + curriculum (default)
python train.py

# Gimbal, no curriculum
python train.py --config-name=config_gimbal

# No gimbal
python train.py --config-name=config

# Override config options via Hydra
python train.py seed=42 lr=3e-4 batch_size=256

# Select curriculum preset
python train.py curriculum_preset=aggressive
```

Training outputs go to `./exp_local/<date>/<time>_<overrides>/` (configured in Hydra).

## Docker

```bash
# Build
cd docker && bash build.sh

# PyTorch installed separately with CUDA 12.4 index
# gym==0.23.1 is pinned; requires numpy<2.0
```

## Architecture

### Core Training Loop (root-level files)

- **`train.py`** — Unified training script. `Workspace` class manages environment creation (gimbal/non-gimbal via `gimbal_mode` config flag), curriculum stage transitions, checkpointing (per-stage top-8), and evaluation.
- **`drqv2.py`** — DRQv2 agent with `EncoderBaseline` (CNN + state concat), `Actor`, `Critic`, optional `AuxiliaryGimbalHead`, and per-network LR scheduling.
- **`dmc.py`** — Environment factory (`make()`, `make_with_gimbal()`). Wraps gym environments with `ActionRepeatWrapper`, `FrameStackWrapper`, and produces `ExtendedTimeStep` namedtuples that carry `oracle_gimbal` for auxiliary tasks.
- **`replay_buffer.py`** — Disk-backed experience replay with multi-worker data loading.
- **`utils.py`** — Seeds, soft update, weight init, `TruncatedNormal`, scheduling helpers (`Until`, `Every`).
- **`lr_scheduler.py`** — Per-network learning rate schedules (step, exponential, linear decay).
- **`logger.py`** / **`video.py`** — CSV/TensorBoard logging and video recording with action overlays.

### Simulation Environment (`gym_pybullet_drones/`)

- **`envs/BaseAviary.py`** — Core PyBullet simulation engine. Defines `DroneModel`, `Physics`, `ImageType` enums. Handles physics stepping, rendering, drone dynamics.
- **`envs/single_agent_rl/BaseSingleAgentAviary.py`** — Single-agent RL interface. Defines `ActionType` (VEL, RPM, etc.) and `ObservationType` (RGB, KIN) enums.
- **`envs/single_agent_rl/LandingAviary.py`** — Three environment classes of increasing complexity:
  - `LandingAviary` — Base landing task
  - `LandingGimbalAviary` — Adds gimbal control (position/velocity/acceleration modes)
  - `LandingGimbalCurriculumAviary` — Adds curriculum stage management via `CurriculumStageSpec`
- **`control/`** — PID controllers (`DSLPIDControl`, `SimplePIDControl`).

### Configuration System (`cfgs/`)

Hydra-based composable configs. Entry points:
- **`config_gimbal_curriculum.yaml`** — Default: gimbal + curriculum learning
- **`config_gimbal.yaml`** — Gimbal without curriculum
- **`config.yaml`** — No gimbal (drone-only)
- **`test_config.yaml`** — Evaluation config

Sub-configs:
- **`task/`** — Environment parameters (landing task variants, gimbal control modes)
- **`curriculum_preset/`** — Stage progression rules (balanced/conservative/aggressive/full_gimbal_only)
- **`lr_schedule/`** — Learning rate schedule profiles

### Curriculum Stages

Stages S0-S4 progressively enable gimbal control:
- **S0**: Gimbal locked (drone-only learning)
- **S1-S4**: Increasing gimbal freedom (action dimensions, angle ranges)

Stage advancement requires sustained success rate over a sliding window of episodes, with cooldown and consecutive-pass requirements to prevent premature promotion.

## Key Conventions

- The `gimbal_mode` config flag controls whether gimbal environments are used. Curriculum code paths are further gated by `curr_enabled`.
- The agent `_target_` field in all configs points to `drqv2.DrQV2Agent`.
- Old checkpoints referencing `drqv2-w-new_nets` are loaded via a `sys.modules` compatibility shim in `train.py`.
- The `ExtendedTimeStep` namedtuple (defined in `dmc.py`) is the core data structure flowing through the training pipeline.
- `state_dim_per_frame` must be 7 for non-gimbal configs (vel(3)+quat(4)) and 11 for gimbal configs (vel(3)+quat(4)+gimbal_quat(4)). The constructor defaults to 11.
- Environment rendering requires `PYBULLET_RENDER=1` environment variable for GUI mode.
- Physics runs at 240Hz, aggregated by 10 (24Hz control), with 4 step repeats in `LandingAviary` giving ~6Hz effective decision rate.
