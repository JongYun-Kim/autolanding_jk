# Autonomous Drone Landing with Reinforcement Learning

## Overview

This project implements an autonomous drone landing system using Deep Reinforcement Learning with curriculum learning. The drone uses visual feedback from a gimbal-mounted camera to land on a target helipad. The system progressively trains the agent from basic drone control to full gimbal control through a structured curriculum.

## Key Features

- **Curriculum Learning**: Progressive training from simple drone control to complex gimbal-camera control
- **Vision-based Landing**: RGB camera observations for target tracking and landing
- **Gimbal Control**: 2-DOF gimbal (pitch + yaw) for active camera pointing
- **Time-aware Encoders**: Advanced encoder architectures that process temporal image sequences
- **Flexible Configuration**: Hydra-based configuration system for easy experimentation

## Project Structure

```
.
├── train_gimbal_curriculum.py      # Main training script (ENTRY POINT)
├── drqv2-w-new_nets.py            # RL agent implementation (DrQv2 with custom encoders)
├── gym_pybullet_drones/           # Custom gym environment
│   └── envs/single_agent_rl/
│       └── LandingAviary.py       # Landing environment with gimbal support
├── cfgs/                          # Hydra configuration files
│   ├── config_gimbal_curriculum.yaml
│   ├── curriculum_preset/         # Curriculum stage definitions
│   ├── encoder_bundle/            # Encoder configurations
│   └── task/                      # Task/environment configs
├── utils.py                       # Utility functions
├── replay_buffer.py               # Experience replay buffer
├── logger.py                      # Logging utilities
└── video.py                       # Video recording utilities
```

## Main Training Script: `train_gimbal_curriculum.py`

### Purpose
The main executable for training the drone landing agent with curriculum learning.

### Key Components

#### 1. **Workspace Class**
Central training orchestrator that manages:
- Environment setup (train/eval)
- Agent instantiation
- Replay buffer management
- Curriculum progression logic
- Checkpoint saving/loading
- Training loop execution

#### 2. **Curriculum Learning System**
Automatically advances through training stages based on success rate:

**Configuration Parameters** (from `curriculum_preset`):
- `window`: Rolling window size for success rate calculation
- `min_episodes`: Minimum episodes before evaluation
- `success_rate`: Target success rate to advance (e.g., 0.75)
- `max_stage`: Maximum curriculum stage
- `cooldown_episodes`: Episodes to wait after advancing
- `min_stage_episodes`: Minimum episodes per stage
- `require_consecutive_windows`: Consecutive passing windows required

**Progression Logic**:
1. Records episode outcomes (success/failure via landing detection)
2. Calculates rolling success rate over recent episodes
3. Tracks consecutive successful windows
4. Advances to next stage when criteria met
5. Resets statistics and applies new stage configuration

#### 3. **Checkpoint Management**
Multiple checkpoint strategies:
- **Rolling snapshot**: `snapshot.pt` (latest state)
- **Stage checkpoints**: Saved at curriculum stage transitions
- **Best checkpoints**: Top-8 by success rate (≥80%)
- **Auto-resume**: Automatically resumes from best available checkpoint

### Usage

Basic training:
```bash
python train_gimbal_curriculum.py
```

With custom configuration:
```bash
python train_gimbal_curriculum.py curriculum_preset=aggressive encoder_bundle.agent=enc_A
```

Resume from checkpoint:
```bash
python train_gimbal_curriculum.py checkpoint_dir=path/to/snapshot.pt
```

### Curriculum Stages (Default: Balanced Preset)

The curriculum typically progresses through these stages:

0. **S0_lock**: Gimbal locked, learn basic drone control only
1. **S1_yaw_small**: Yaw-only gimbal, narrow range (~15%)
2. **S2_yaw_pitch_small**: Pitch+Yaw, small range (~25-35%)
3. **S3_mid**: Medium range (~60%), stronger alignment rewards
4. **S4_full**: Full gimbal range, complete task

Each stage configures:
- Gimbal angle ranges (pitch, roll, yaw scales)
- Visibility reward inclusion/weight
- Alignment penalty/reward coefficients
- Smoothness penalties for gimbal motion

## Reinforcement Learning Agent: `drqv2-w-new_nets.py`

### Algorithm
DrQv2 (Data-Regularized Q-learning v2) - off-policy actor-critic with:
- Twin Q-networks (double Q-learning)
- Soft target updates
- Data augmentation (random shifts + brightness)

### Architecture Selection

The agent now supports **multiple encoder, actor, and critic architectures** for ablation studies:

- **Encoders**: 3 types (Baseline, Encoder A, Encoder B)
- **Actors**: 4 types (Original, 2 autoregressive variants, Multi-head attention)
- **Critics**: 2 types (Original, Factored)

All combinations are configurable via Hydra. See `ACTOR_CRITIC_ARCHITECTURES_GUIDE.md` for details.

### Encoder Architectures

Three encoder options for processing observations:

#### **Baseline Encoder** (`enc='org'`)
- Traditional CNN + state concatenation
- Input: [B, T, H, W] stacked frames + [B, T×11] drone states
- Output: Flattened features [B, 32×35×35 + T×11]

#### **Encoder A** (`enc='A'`) - Time-aware Lightweight
- Per-frame global image tokens via CNN+GAP
- State tokenization: [velocity, drone_quat, gimbal_quat]
- Frame-wise cross-attention (state ← image)
- Optional temporal self-attention across frames
- Configurable: `d_model`, `nhead`, `cross_depth`, `self_depth`

#### **Encoder B** (`enc='B'`) - Time-aware High-Capacity
- Patch-based image processing (ViT-style)
- Frame-local encoder: [state_tokens(3), image_patches(Np)]
- Temporal encoder: aggregated tokens across time
- Two-stage hierarchical processing
- Configurable: `d_model`, `nhead`, `patch`, `depth_local`, `depth_temp`

### Actor Architectures

#### 1. **Original Actor** (`actor_type='original'`)
- Baseline joint action generation
- All 5 actions sampled simultaneously
```
repr [B, repr_dim]
  → trunk (Linear + BN + Tanh) [B, feature_dim]
  → policy (MLP) [B, 5]
  → tanh → TruncatedNormal(mu, std)
```

#### 2. **Autoregressive Gimbal-First** (`actor_type='autoregressive_gimbal_first'`)
- Point-then-move strategy
- Gimbal action → Drone action (conditioned)
```
repr → trunk → gimbal_policy → [pitch, yaw]
              ↓
         concat with repr → drone_policy → [vx, vy, vz]
```

#### 3. **Autoregressive Drone-First** (`actor_type='autoregressive_drone_first'`)
- Move-then-point strategy
- Drone action → Gimbal action (conditioned)
```
repr → trunk → drone_policy → [vx, vy, vz]
              ↓
         concat with repr → gimbal_policy → [pitch, yaw]
```

#### 4. **Multi-Head Attention Actor** (`actor_type='multihead_attention'`) ⭐ RECOMMENDED
- Parallel but contextually-aware actions
- Bidirectional coordination via attention
```
repr → trunk → [drone_query, gimbal_query]
              ↓
         cross-attention with obs
              ↓
         self-attention (drone ↔ gimbal)
              ↓
         [drone_policy, gimbal_policy]
```

### Critic Architectures

#### 1. **Original Critic** (`critic_type='original'`)
- Baseline joint Q-value estimation
```
repr [B, repr_dim] + action [B, 5]
  → trunk (Linear + BN + Tanh) [B, feature_dim]
  → Q1/Q2 (MLP) [B, 1]
```

#### 2. **Factored Critic** (`critic_type='factored'`)
- Decomposes Q-value: `Q(s,a) = Q_drone(s,a_d) + Q_joint(s,a_full)`
- Better for autoregressive actors
```
Q1 = Q1_drone(repr + drone_action) + Q1_joint(repr + full_action)
Q2 = Q2_drone(repr + drone_action) + Q2_joint(repr + full_action)
```

**Action Space**: `[vx, vy, vz, gimbal_pitch, gimbal_yaw]` (5D, shared across all architectures)

## Environment: `LandingAviary.py`

### Class Hierarchy
1. **LandingAviary**: Basic landing task without gimbal
2. **LandingGimbalAviary**: Adds 2-DOF gimbal control
3. **LandingGimbalCurriculumAviary**: Adds curriculum stage management

### LandingGimbalCurriculumAviary

**Observation Space**:
- RGB images: [84×84×4] (grayscale with alpha)
- Frame stacking: 3 frames
- Drone state: velocity(3) + drone_quat(4) + gimbal_quat(4) = 11D per frame

**Action Space**: `[-1, 1]^5`
- Drone velocity: [vx, vy, vz]
- Gimbal: [pitch_norm, yaw_norm]

**Reward Components**:
1. **Horizontal-Vertical (HV) Reward**:
   - XY distance to target (exponential shaping)
   - Z velocity matching (optimal: -0.5 m/s)
   - Landing bonus: +140 for successful landing

2. **Visibility Reward** (curriculum-dependent):
   - Alignment: `cos^5(angle_to_oracle)` × `viz_weight`
   - Not visible: `not_visible_penalty`

3. **Smoothness Penalty**:
   - Penalizes large gimbal changes: `λ × ||Δgimbal_norm||^2`

**Curriculum Stage Configuration**:
Each stage (CurriculumStageSpec) defines:
- `gimbal_enabled`: Enable/disable gimbal control
- `lock_down`: Force gimbal to initial position
- `scale`: (pitch, roll, yaw) range scales ∈ [0,1]
- `include_viz_reward`: Toggle visibility reward
- `viz_weight`: Alignment reward coefficient
- `not_visible_penalty`: Penalty when target not visible
- `smooth_penalty`: Gimbal smoothness coefficient
- `yaw_only` / `pitch_only`: Axis-specific training flags

## Configuration System (Hydra)

### Main Config: `cfgs/config_gimbal_curriculum.yaml`

Key parameters:
```yaml
frame_stack: 3                    # Temporal observation depth
action_repeat: 1                  # Action repeat factor
num_seed_frames: 4800            # Random exploration frames
eval_every_frames: 30000         # Evaluation frequency
num_train_frames: 3000000        # Total training frames

# Agent
lr: 1.0e-4                       # Learning rate
feature_dim: 128                 # Feature dimension
hidden_dim: 512                  # MLP hidden size
batch_size: 512                  # Replay batch size

# Architecture selection
encoder_bundle@agent: enc_B      # original | enc_A | enc_B
actor_critic_bundle@agent: original  # original | autoregressive_gimbal_first |
                                     # autoregressive_drone_first | multihead_attention |
                                     # multihead_attention_factored
```

### Example Configurations

**Baseline (original everything)**:
```bash
python train_gimbal_curriculum.py \
  encoder_bundle@agent=original \
  actor_critic_bundle@agent=original
```

**Recommended (Encoder B + Multi-head Attention)**:
```bash
python train_gimbal_curriculum.py \
  encoder_bundle@agent=enc_B \
  actor_critic_bundle@agent=multihead_attention
```

**Autoregressive (Encoder A + Gimbal-First + Factored Critic)**:
```bash
python train_gimbal_curriculum.py \
  encoder_bundle@agent=enc_A \
  actor_critic_bundle@agent=autoregressive_gimbal_first
```

### Curriculum Presets

**Conservative** (`curriculum_preset: conservative`):
- Slow progression, high success rate requirement
- More stages, smaller increments
- Suitable for stable convergence

**Balanced** (`curriculum_preset: balanced`):
- Default preset
- 5 stages with moderate progression
- Good trade-off between speed and stability

**Aggressive** (`curriculum_preset: aggressive`):
- Fast progression, fewer stages
- Lower success rate thresholds
- Faster training but less stable

## Key Algorithms & Methods

### Curriculum Advancement Logic (train_gimbal_curriculum.py:218-221)
```python
def _should_advance_curriculum(self, seed_until_step):
    if not self._eligible_to_evaluate(seed_until_step):
        return False
    return self.consecutive_passes >= self._c_require_consec
```

Checks:
1. Not in seed phase
2. Cooldown expired
3. Minimum stage episodes satisfied
4. Success rate threshold met for required consecutive windows

### Oracle Gimbal Computation (LandingAviary.py:488-592)
Computes ideal gimbal orientation to point camera at helipad:
1. Calculate target direction in world frame
2. Transform to drone body frame
3. Extract yaw/pitch via atan2 (numerically stable)
4. Clip to gimbal angle ranges
5. Return quaternion + angles (rad + normalized)

### Visibility Checking (LandingAviary.py:594-624)
Uses `CameraVisibilityChecker` to determine if helipad is in FOV:
- Projects helipad corners to camera frame
- Checks visibility ratio > threshold (0.14%)
- Accounts for FOV margins (2° safety margin)

## Training Workflow

1. **Initialization**:
   - Setup environments (train/eval)
   - Create agent with selected encoder
   - Initialize replay buffer
   - Load checkpoint if available

2. **Episode Loop**:
   - Collect experience with current policy
   - Store transitions in replay buffer
   - Update agent (actor + critic)
   - Record success/failure for curriculum

3. **Curriculum Progression**:
   - Track rolling success rate
   - Advance stage when criteria met
   - Save stage checkpoint
   - Update environment configuration

4. **Evaluation**:
   - Periodic evaluation episodes
   - Video recording (first episode)
   - Log metrics (reward, success rate, stage)

5. **Checkpointing**:
   - Rolling snapshot every episode
   - Stage transitions
   - Best models (SR ≥ 0.8)

## Important Files & Locations

- **Training logs**: `exp_local/YYYY.MM.DD/HHMMSS_<config>/`
- **Checkpoints**: Same directory as logs
  - `snapshot.pt`: Latest checkpoint
  - `snapshot_stage*_*.pt`: Stage checkpoints
  - `snapshot_best_sr*_*.pt`: Best checkpoints
- **Videos**: Saved in experiment directory if `save_video: true`
- **Tensorboard**: Launch with `tensorboard --logdir exp_local/`

## Common Issues & Notes

### Outdated Files (DO NOT USE)
- `odt/` folders - Outdated experiments
- `odt_envs.py` - Old environment implementation
- `train.py`, `train_gimbal.py` - Superseded by `train_gimbal_curriculum.py`

### GPU Memory
- Default batch size: 512 (requires ~8GB GPU)
- Reduce `batch_size` if OOM errors occur
- Encoder B uses more memory than A/baseline

### Curriculum Tuning
If agent struggles to advance:
- Increase `window` size for smoother metrics
- Decrease `success_rate` threshold
- Adjust `cooldown_episodes` / `min_stage_episodes`
- Modify stage scales in curriculum preset

## Actor/Critic Architecture Improvements ✅ IMPLEMENTED

The agent now supports multiple actor and critic architectures (previously Task #2):

### Implemented Architectures

1. **Autoregressive Actor (Gimbal → Drone)** ✅
   - `actor_type='autoregressive_gimbal_first'`
   - Point-then-move strategy
   - Gimbal action sampled first, drone action conditioned on it

2. **Autoregressive Actor (Drone → Gimbal)** ✅
   - `actor_type='autoregressive_drone_first'`
   - Move-then-point strategy
   - Drone action sampled first, gimbal action conditioned on it

3. **Multi-Head Attention Actor** ✅ RECOMMENDED
   - `actor_type='multihead_attention'`
   - Parallel but contextually-aware actions
   - Cross-attention between observation and action queries
   - Self-attention for drone-gimbal coordination

4. **Factored Critic** ✅
   - `critic_type='factored'`
   - Decomposes Q-value into drone base value + coordination value
   - Recommended for use with autoregressive actors

### Usage

See `ACTOR_CRITIC_ARCHITECTURES_GUIDE.md` for detailed usage instructions and ablation study guidelines.

## Citation & Credits

Based on DrQv2 algorithm and PyBullet drone simulation framework.

## License

See repository license file.
