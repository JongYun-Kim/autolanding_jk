# Repository Guide: Autonomous Drone Landing with RL

**Purpose**: AI-oriented reference for development. Optimized for future sessions adding features (networks, schedulers, eval scripts) and refactoring (save logic, gimbal control, backward compatibility).

**Last Updated**: 2025-11-20
**Maintenance**: Update relevant sections when modifying architecture, data flow, or curriculum system.

---

## 1. Project Overview

**Goal**: Autonomous drone landing using DrQ-v2 reinforcement learning with camera and gimbal control.

**Key Technical Details**:
- **Algorithm**: DrQ-v2 (DDPG backbone, twin Q-networks, n-step returns, random shift augmentation)
- **Environment**: `gym_pybullet_drones` (PyBullet physics)
- **Control**: RL outputs velocity commands; environment handles dynamics and low-level control
- **Main Challenge**: 5D action space (3D velocity + 2D gimbal) solved via curriculum learning

**System Evolution**: Built on prior work with downward-facing camera (no gimbal). Current system adds gimbal pan/tilt control.

---

## 2. Quick Start

### Training Commands

```bash
# Curriculum learning (RECOMMENDED for gimbal control)
python train_gimbal_curriculum.py \
    curriculum_preset=balanced \
    actor_critic_bundle=original \
    encoder_bundle=original \
    lr_schedule=constant

# Gimbal control without curriculum
python train_gimbal.py task_name=gimbal-landing-aviary-v0 seed=1

# Simple 3D velocity control (no gimbal)
python train.py task_name=landing-aviary-v0 seed=42
```

### Configuration Override Examples

```bash
# Experiment with architecture
python train_gimbal_curriculum.py \
    actor_critic_bundle=multihead_attention \
    encoder_bundle=enc_A

# Adjust curriculum difficulty
python train_gimbal_curriculum.py curriculum_preset=aggressive

# Add learning rate scheduling
python train_gimbal_curriculum.py lr_schedule=exponential_decay

# Resume from checkpoint
python train_gimbal_curriculum.py checkpoint_dir=/path/to/snapshot_best_*.pt
```

### Output Location
`./exp_local/{YYYY.MM.DD}/{HHMMSS}_{overrides}/`

---

## 3. Codebase Structure

```
autolanding_jk/
├── train.py                          # Entry: 3D velocity control
├── train_gimbal.py                   # Entry: 5D action space (no curriculum)
├── train_gimbal_curriculum.py        # Entry: Curriculum learning (MAIN)
│
├── drqv2.py                          # DrQ-v2 agent for 3D actions
├── drqv2-w-new_nets.py               # DrQ-v2 with multiple architectures (5D)
├── dmc.py                            # Environment wrappers (observation pipeline)
├── replay_buffer.py                  # Episode-based buffer with n-step returns
├── utils.py                          # Utilities (seed, soft_update, schedule)
├── logger.py                         # CSV + TensorBoard logging
├── lr_scheduler.py                   # Learning rate scheduling classes
├── video.py                          # Video recording utilities
│
├── cfgs/                             # Hydra configuration system
│   ├── config.yaml                   # 3D baseline config
│   ├── config_gimbal.yaml            # 5D gimbal config
│   ├── config_gimbal_curriculum.yaml # Curriculum learning config (MAIN)
│   ├── task/                         # Task-specific configs
│   ├── curriculum_preset/            # Conservative/Balanced/Aggressive
│   ├── actor_critic_bundle/          # Architecture selection
│   ├── encoder_bundle/               # Encoder architecture
│   └── lr_schedule/                  # LR scheduling strategies
│
├── gym_pybullet_drones/
│   ├── envs/
│   │   ├── BaseAviary.py             # Physics engine base
│   │   └── single_agent_rl/
│   │       ├── LandingAviary.py      # Main: LandingAviary, LandingGimbalAviary,
│   │       │                         #       LandingGimbalCurriculumAviary (Lines 18-950)
│   │       └── odt_envs.py           # [OUTDATED - SKIP]
│   └── utils/
│       ├── utils.py                  # rgb2gray (Line 140), drone utilities
│       └── camera_visibility_checker.py
│
├── odt/                              # [OUTDATED - SKIP]
├── docker/                           # Docker setup
├── ACTOR_CRITIC_GUIDE.md             # Architecture documentation
└── LR_SCHEDULING_GUIDE.md            # LR schedule documentation
```

**Key File Paths** (for reference in future work):
- Agent (3D): `/home/user/autolanding_jk/drqv2.py:137-290`
- Agent (5D): `/home/user/autolanding_jk/drqv2-w-new_nets.py:787-1022`
- Curriculum Logic: `/home/user/autolanding_jk/train_gimbal_curriculum.py:182-555`
- Observation Wrappers: `/home/user/autolanding_jk/dmc.py:85-343`
- Environment Classes: `/home/user/autolanding_jk/gym_pybullet_drones/envs/single_agent_rl/LandingAviary.py:18-950`

---

## 4. Data Flow & Observation Pipeline

### Action Spaces

**3D (Simple Landing)**:
```
Action: [v_x, v_y, v_z] ∈ [-1, 1]³
Observation: images [3, 84, 84] + state [3, 7]
State per frame: [velocity (3D), quaternion (4D)]
```

**5D (Gimbal Control)**:
```
Action: [v_x, v_y, v_z, gimbal_pitch, gimbal_yaw] ∈ [-1, 1]⁵
Observation: images [3, 84, 84] + state [3, 11]
State per frame: [velocity (3D), quaternion (4D), gimbal_quaternion (4D)]

Gimbal Physical Ranges:
  - Pitch: ±47° (±5π/19 rad)
  - Yaw: ±90°
  - Roll: ±45° (currently fixed at 0)
```

### Observation Pipeline

```
┌─────────────────────────────────────────────────────────┐
│ 1. Environment (_computeObs in LandingAviary.py:392)   │
│    RGB 84×84 (4-channel RGBA) from PyBullet            │
│    self.rgb, _, _ = self._getDroneImages(0)            │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│ 2. Grayscale Conversion (utils.py:140-145)             │
│    rgb2gray(self.rgb) → [84, 84]                       │
│    Formula: 0.2989*R + 0.5870*G + 0.1140*B             │
│    [None,:] adds dimension → [1, 84, 84]               │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│ 3. Frame Stacking (dmc.py:85-162)                      │
│    FrameStackWrapper stacks num_frames=3:              │
│    observation = concat(frames) → [3, 84, 84]          │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│ 4. State Extraction (dmc.py:129-131, 141-143)          │
│    Per frame:                                           │
│    - velocity = env.vel[0,:] / SPEED_LIMIT → [3]       │
│    - attitude = env.quat[0,:] → [4]                    │
│    drone_state = concat((velocity, attitude)) → [7]    │
│    Stacked (3 frames): [3, 7] flattened → [21]         │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│ 5. Gimbal State Augmentation (dmc.py:165-205)          │
│    FrameStackWrapperWithGimbalState adds:              │
│    - gimbal_quat = env's gimbal quaternion → [4]       │
│    drone_state = concat((vel, quat, gimbal_quat)) → [11]│
│    Stacked (3 frames): [3, 11] flattened → [33]        │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│ 6. TimeStep Wrapper (dmc.py:215-261)                   │
│    ExtendedTimeStepWrapper creates:                    │
│    - observation: [3, 84, 84]                          │
│    - drone_state: [3, 7] or [3, 11]                   │
│    - action, reward, discount                          │
│    - landing_info: success/crash flag                  │
└─────────────────────────────────────────────────────────┘
```

**Critical Wrapper Functions**:
- `dmc.make()`: 3D action space → `FrameStackWrapper`
- `dmc.make_with_gimbal()`: 5D action space → `FrameStackWrapperWithGimbalState`

**State Normalization** (dmc.py:129, 141):
```python
velocity_normalized = env.vel[0,:] / env.SPEED_LIMIT  # → [0, 1] range
quaternion = env.quat[0,:]                             # → raw quaternion
```

---

## 5. Curriculum Learning System

**File**: `train_gimbal_curriculum.py`

### Stage Specification

**Dataclass** (LandingAviary.py:762-773):
```python
@dataclass
class CurriculumStageSpec:
    name: str                          # Stage identifier
    gimbal_enabled: bool               # Enable gimbal control
    lock_down: bool                    # Lock gimbal to downward position
    scale: Tuple[float, float, float]  # (pitch, roll, yaw) scaling [0, 1]
    include_viz_reward: bool           # Use visibility reward
    viz_weight: float                  # Alignment reward weight
    not_visible_penalty: float         # Penalty when target not in FOV
    smooth_penalty: float              # Gimbal jerk penalty λ*||Δangles||²
    yaw_only: bool = False             # Restrict to yaw-only control
    pitch_only: bool = False           # Restrict to pitch-only control
```

### Typical Curriculum Flow

```
Stage 0: S0_lock (Gimbal Disabled)
  ├─ gimbal_enabled: False, lock_down: True
  ├─ Reward: hover quality + landing success
  └─ Advance: 80+ episodes, 85% success rate
        ↓
Stage 1: S1_yaw_small (Yaw Only, Limited Range)
  ├─ yaw_only: True, scale: [0, 0, 0.15]
  ├─ not_visible_penalty: -0.010
  └─ Advance: 80+ episodes, 85% success rate
        ↓
Stage 2: S2_yaw_pitch_small (Both Axes, Moderate Range)
  ├─ scale: [0.25, 0, 0.35]
  ├─ include_viz_reward: True, viz_weight: 0.05
  └─ Advance: 80+ episodes, 85% success rate
        ↓
Stage 3: S3_mid (Expanded Ranges)
  ├─ scale: [0.65, 0, 0.65]
  ├─ viz_weight: 0.08, not_visible_penalty: -0.050
  └─ Advance: 80+ episodes, 85% success rate
        ↓
Stage 4: S4_full (Full Gimbal Control)
  ├─ scale: [1.0, 0, 1.0]
  ├─ viz_weight: 0.10, not_visible_penalty: -0.100
  └─ Final stage: train to convergence
```

### Curriculum Management Methods

**1. Episode Recording** (train_gimbal_curriculum.py:182-198):
```python
_record_episode_result(success: bool)
# Updates rolling window, tracks consecutive successful windows
```

**2. Advancement Check** (Lines 218-221):
```python
_should_advance_curriculum() -> bool
# Conditions: eligible_to_evaluate AND success_rate >= threshold AND consecutive_windows >= 4
```

**3. Stage Transition** (Lines 557-563):
```python
_advance_curriculum_stage()
# - Saves stage boundary checkpoint
# - Updates environment to next stage
# - Resets statistics (with optional bootstrap failures)
```

**4. Checkpoint Management** (Lines 456-505):
```python
_save_best_checkpoint_if_needed()
# Per-curriculum-stage top-K checkpoints:
# - Non-final stages: top 3 (SR >= 0.8)
# - Final stage: top 8 (SR >= 0.8)
# Metadata stored in best_checkpoints.json
```

### Configuration Parameters

**Curriculum Presets** (cfgs/curriculum_preset/):

| Preset | Window | Min Episodes | Success Rate | Max Stage |
|--------|--------|--------------|--------------|-----------|
| Conservative | 60 | 59 | 0.85 | 3 |
| Balanced | 100 | 80 | 0.85 | 4 |
| Aggressive | 40 | 39 | 0.82 | 3 |

**Key Config Fields**:
```yaml
curriculum:
  enable: true
  initial_stage: 0
  window: 100                       # Success rate sliding window size
  min_episodes: 80                  # Min episodes before checking advancement
  success_rate: 0.85                # Threshold for stage transition
  cooldown_episodes: 0              # Wait period after stage change
  require_consecutive_windows: 4    # Consecutive successful windows required
  bootstrap_failures: 0             # Pre-seed failures (for conservative approach)
```

---

## 6. Neural Network Architectures

### Basic Agent (drqv2.py)

**Encoder** (Lines 56-78):
```
Input: RGB images [B, 3, 84, 84] (values [0, 255])
  ↓
4× Conv2D layers (32 filters each) + ReLU
  ↓
Flatten → [B, 39200]
  ↓
Concat drone_state → [B, 39200 + state_dim]
  ↓
Linear(feature_dim=50) + LayerNorm → [B, 50]
```

**Actor** (Lines 81-103):
```
Trunk: Linear(feature_dim → hidden_dim) + LayerNorm
  ↓
Policy: Linear(hidden_dim → action_dim) → mean
  ↓
TruncatedNormal(mean, std=stddev_schedule)
  ↓
Tanh scaling to action bounds
```

**Critic (Twin Q-Networks)** (Lines 106-134):
```
Shared trunk: Linear(feature_dim + action_dim → hidden_dim) + LayerNorm
  ↓
Q1: Linear → scalar
Q2: Linear → scalar
  ↓
Target: min(Q1, Q2) for stability
```

### Advanced Architectures (drqv2-w-new_nets.py)

**Encoders**:
1. **EncoderBaseline** (Lines 52-90): Standard CNN + state concat
2. **EncoderTransformer** (Lines 216-303): Temporal state tokenization with multi-head attention
3. **EncoderViT** (Lines 304-330): Vision Transformer with image patches

**Actors** (5D action space support):
1. **ActorOriginal** (Lines 433-462): Standard MLP
2. **ActorAutoregressive** (Lines 503-621):
   - Sequential: Gimbal → Drone OR Drone → Gimbal
   - Conditioning: latter action depends on former
3. **ActorAttention** (Lines 624-717):
   - Multi-head attention fusion
   - Separate embeddings for drone vs gimbal actions

**Critics**:
1. **CriticOriginal** (Lines 464-498): Standard twin Q-networks
2. **CriticFactored** (Lines 719-785):
   - Separate Q-heads for drone and gimbal actions
   - Reduces interference between action subspaces

**Configuration Selection** (cfgs/actor_critic_bundle/):
```yaml
# Example: multihead_attention.yaml
actor_type: multihead_attention
actor_cfg:
  nhead: 4                  # Number of attention heads
  dropout: 0.1
critic_type: original
```

**Available Bundles**:
- `original.yaml`: Standard MLP actor/critic
- `autoregressive_gimbal_first.yaml`: Gimbal → Drone ordering
- `autoregressive_drone_first.yaml`: Drone → Gimbal ordering
- `multihead_attention.yaml`: Transformer-based actor
- `multihead_attention_factored.yaml`: Attention actor + factored critic

---

## 7. Checkpoint & Resume System

### Checkpoint Types

**1. Rolling Checkpoint** (train_gimbal_curriculum.py:382-389):
```
snapshot.pt
- Saved every episode end
- Contains: agent, timer, step counters, curriculum_state
- Lightweight, always overwritten
```

**2. Stage Boundary Checkpoint** (Lines 391-413):
```
snapshot_stage{idx}_{name}_{frame}_{tag}.pt
- Saved when advancing to new curriculum stage
- Full curriculum state serialized
- Used for stage-specific analysis
```

**3. Best Model Checkpoint** (Lines 456-505):
```
snapshot_best_sr{percent}_stage{idx}_{frame}.pt
- Saved when success_rate >= 0.8 at current stage
- Per-stage limits:
  * Non-final stages: keep top 3
  * Final stage: keep top 8
- Sorted by (success_rate DESC, frame DESC)
```

### Checkpoint Metadata

**best_checkpoints.json** (Lines 434-454):
```json
{
  "0": [
    {"sr": 0.95, "frame": 1000000, "path": "snapshot_best_sr95_stage0_1000000.pt"},
    {"sr": 0.88, "frame": 950000, "path": "snapshot_best_sr88_stage0_950000.pt"},
    {"sr": 0.82, "frame": 900000, "path": "snapshot_best_sr82_stage0_900000.pt"}
  ],
  "4": [
    {...}, {...}, ..., {...}  // Top 8 for final stage
  ]
}
```

### Auto-Resume Logic

**Priority Order** (train_gimbal_curriculum.py:508-555):
```
auto_resume_if_possible():
  1. Check snapshot.pt exists?
     → Load latest rolling checkpoint

  2. Check best_checkpoints.json exists?
     → Find newest checkpoint by frame across ALL stages
     → Load best model from any stage

  3. Check snapshot_stage*.pt files?
     → Find largest frame number across stage snapshots
     → Load stage boundary checkpoint

  4. Start fresh training if nothing found
```

**Manual Resume**:
```bash
python train_gimbal_curriculum.py checkpoint_dir=/path/to/snapshot_best_sr95_stage4_2000000.pt
```

### Checkpoint Update Logic (Recent Change)

**Git History Note** (commit 7669870):
- Changed from global top-8 to per-curriculum-stage top-8
- Final stage gets top-8, non-final stages get top-3
- Prevents early-stage checkpoints from dominating saved models

**Implementation** (Lines 456-505):
```python
# Determine top-K limit based on stage
is_final_stage = (self.current_stage == self.cfg.curriculum.max_stage)
top_k = 8 if is_final_stage else 3

# Load existing checkpoints for current stage
stage_key = str(self.current_stage)
stage_checkpoints = best_checkpoints.get(stage_key, [])

# Add new checkpoint if SR >= 0.8
if success_rate >= 0.80:
    stage_checkpoints.append({"sr": success_rate, "frame": frame, "path": path})
    stage_checkpoints.sort(key=lambda x: (-x["sr"], -x["frame"]))

    # Keep only top-K
    if len(stage_checkpoints) > top_k:
        to_remove = stage_checkpoints[top_k:]
        for ckpt in to_remove:
            os.remove(ckpt["path"])
        stage_checkpoints = stage_checkpoints[:top_k]
```

---

## 8. Extension Points: Adding New Features

### A. Add New Neural Network Architecture

**Steps**:
1. Define encoder/actor/critic in `drqv2-w-new_nets.py`
2. Create config file in `cfgs/actor_critic_bundle/my_architecture.yaml`:
   ```yaml
   actor_type: my_new_actor
   actor_cfg:
     custom_param: value
   critic_type: original  # or custom critic
   ```
3. Update agent instantiation in `train_gimbal_curriculum.py` if needed
4. Test with: `python train_gimbal_curriculum.py actor_critic_bundle=my_architecture`

**Example Reference**:
- Attention actor: `drqv2-w-new_nets.py:624-717`
- Config: `cfgs/actor_critic_bundle/multihead_attention.yaml`

### B. Add Learning Rate Scheduling

**Steps**:
1. Define scheduler class in `lr_scheduler.py`:
   ```python
   class MyScheduler:
       def __init__(self, optimizer, **kwargs):
           self.optimizer = optimizer

       def step(self, current_step):
           new_lr = self._compute_lr(current_step)
           for param_group in self.optimizer.param_groups:
               param_group['lr'] = new_lr
   ```
2. Create config in `cfgs/lr_schedule/my_schedule.yaml`:
   ```yaml
   type: my_scheduler
   kwargs:
     param1: value1
   ```
3. Reference in `train_gimbal_curriculum.py:155-170` for integration

**Example Reference**:
- Exponential decay: `lr_scheduler.py:26-40`
- Config: `cfgs/lr_schedule/exponential_decay.yaml`

### C. Create Evaluation Script

**Pattern** (based on `video.py`):
```python
# 1. Load checkpoint
agent = hydra.utils.instantiate(cfg.agent)
with open(checkpoint_path, 'rb') as f:
    payload = torch.load(f)
agent.load_state_dict(payload['agent'])

# 2. Create evaluation environment (no wrappers that add randomness)
env = dmc.make_with_gimbal(cfg.task_name, ...)

# 3. Run episodes
for episode in range(num_eval_episodes):
    time_step = env.reset()
    while not time_step.last():
        with torch.no_grad():
            action = agent.act(time_step.observation, time_step.drone_state,
                              step=999999, eval_mode=True)
        time_step = env.step(action)
    # Collect metrics
```

**Location Suggestion**: Create `eval.py` in root directory

### D. Modify Curriculum Stages

**File**: `gym_pybullet_drones/envs/single_agent_rl/LandingAviary.py:952-1020`

**Steps**:
1. Modify `_create_default_curriculum_specs()` method
2. Add/remove stages or adjust parameters:
   ```python
   CurriculumStageSpec(
       name="S5_extreme",
       gimbal_enabled=True,
       lock_down=False,
       scale=(1.0, 0.0, 1.0),
       include_viz_reward=True,
       viz_weight=0.15,
       not_visible_penalty=-0.200,
       smooth_penalty=0.01
   )
   ```
3. Update `max_stage` in curriculum config to match new stage count

### E. Add New Curriculum Preset

**Steps**:
1. Create `cfgs/curriculum_preset/my_preset.yaml`:
   ```yaml
   curriculum:
     enable: true
     initial_stage: 0
     window: 120
     min_episodes: 100
     success_rate: 0.90
     max_stage: 4
     cooldown_episodes: 10
     require_consecutive_windows: 5
     bootstrap_failures: 5
   ```
2. Use with: `python train_gimbal_curriculum.py curriculum_preset=my_preset`

---

## 9. Replay Buffer & Data Collection

**Architecture**:
```
ReplayBufferStorage (episode accumulation)
    ↓
Episode files: {timestamp}_{episode_id}_{length}.npz
    ↓
ReplayBuffer (sampling + n-step returns)
    ↓
PyTorch DataLoader (multi-worker parallel loading)
```

**Files**:
- Storage: `replay_buffer.py:32-73`
- Buffer: `replay_buffer.py:76-161`
- Factory: `replay_buffer.py:170-187`

**Episode Format** (.npz):
```
observation: [T, 3, 84, 84]      # Stacked grayscale images
action: [T, 3] or [T, 5]         # 3D or 5D actions
reward: [T, 1]                   # Scalar rewards
discount: [T, 1]                 # 0.0 if terminal, 1.0 otherwise
drone_state: [T, 7] or [T, 11]   # State observations
```

**N-Step Return Computation** (Lines 145-157):
```python
reward = 0
discount = 1
for i in range(nstep):
    step_reward = episode['reward'][idx + i]
    reward += discount * step_reward
    discount *= episode['discount'][idx + i] * cfg.discount
```

**Configuration**:
```yaml
replay_buffer_size: 400000       # Max transitions stored
nstep: 3                         # 3-step bootstrapping
batch_size: 512                  # Gimbal training (256 for simple)
replay_buffer_num_workers: 4     # Parallel episode loading
```

---

## 10. Known Complexities & Gotchas

### A. Outdated Code (Do Not Use)
- `odt/` directory: outdated experiments
- `gym_pybullet_drones/envs/single_agent_rl/odt_envs.py`: outdated environment

### B. Action Space Mismatch
**Issue**: Using wrong wrapper for agent type causes dimension errors.

**Solution**:
- 3D actions → `dmc.make()` + `drqv2.DrQV2Agent`
- 5D actions → `dmc.make_with_gimbal()` + `drqv2-w-new_nets.DrQV2Agent`

### C. Gimbal Initialization
**Issue**: `"gimbal_target is not set"` error.

**Cause**: Environment's `_getDroneImages()` called before `reset()`.

**Solution**: Always call `env.reset()` before stepping or rendering.

### D. Curriculum Not Advancing
**Common Causes**:
1. `success_rate` too high (try 0.80-0.85)
2. `window` too small (increase to 80-100)
3. `require_consecutive_windows` too high (reduce to 3-4)
4. `min_episodes` not reached yet

**Debug**: Check `train.csv` columns: `curriculum_stage`, `success_rate_window`

### E. Checkpoint Loading Failures
**Issue**: Resume fails or loads wrong stage.

**Solution**: Use `auto_resume_if_possible()` priority:
1. `snapshot.pt` (most recent)
2. `best_checkpoints.json` (best models)
3. `snapshot_stage*.pt` (stage boundaries)

**Manual override**: Specify `checkpoint_dir=path` explicitly.

### F. Per-Stage Checkpoint Logic
**Change (commit 7669870)**: Switched from global top-8 to per-stage top-K.

**Implications**:
- Old checkpoints may not load into new system
- Non-final stages: only 3 checkpoints kept
- Final stage: 8 checkpoints kept

**Best Practice**: When refactoring checkpoint logic, test both save and load paths.

### G. State Normalization Inconsistencies
**Issue**: Agent receives unnormalized gimbal angles.

**Check**: `dmc.py:129-143` for normalization:
```python
velocity = env.vel[0,:] / env.SPEED_LIMIT  # Normalized
quaternion = env.quat[0,:]                  # Raw (unit quaternion)
```

**Note**: Quaternions are unit vectors by definition, so no normalization needed.

### H. Image Resolution Assumptions
**Gotcha**: Code assumes 84×84 images throughout.

**Locations**:
- `BaseAviary.py:163`: `IMG_RES = np.array([84, 84])`
- `dmc.py:93`: `observation_spec` hardcodes `(num_frames, 84, 84)`

**If changing resolution**: Update all hardcoded references + encoder architecture.

### I. Soft Target Update Timing
**Detail**: Critic target updated with `tau=0.05` every `update_every_steps=2`.

**Location**: `drqv2.py:253-289`, `utils.py:39-43`

**Formula**: `θ_target = τ * θ + (1 - τ) * θ_target`

**Gotcha**: Too high `tau` → instability. Too low → slow learning.

---

## 11. Training Monitoring

### Output Structure
```
exp_local/{YYYY.MM.DD}/{HHMMSS}_{overrides}/
├── snapshot.pt                          # Rolling checkpoint
├── snapshot_best_sr{percent}_stage{idx}_{frame}.pt  # Best models
├── snapshot_stage{idx}_{name}_{frame}_{tag}.pt      # Stage transitions
├── best_checkpoints.json                # Checkpoint metadata
├── train.csv                            # Training metrics
├── eval.csv                             # Evaluation metrics
├── tb/                                  # TensorBoard logs
└── buffer/                              # Replay buffer episodes
```

### Key Metrics (train.csv, eval.csv)

| Metric | Description |
|--------|-------------|
| `fps` | Training frames per second |
| `episode_reward` | Total reward per episode |
| `episode_length` | Steps per episode |
| `success_rate_window` | Sliding window success rate (curriculum) |
| `curriculum_stage` | Current curriculum stage index |
| `critic_loss` | TD error loss for Q-networks |
| `actor_loss` | Policy gradient loss |
| `buffer_size` | Replay buffer transition count |

### TensorBoard
```bash
tensorboard --logdir exp_local/{date}/{time}/tb
```

---

## 12. Configuration System: Hydra

**Root Configs**:
- `cfgs/config.yaml`: 3D baseline
- `cfgs/config_gimbal.yaml`: 5D gimbal
- `cfgs/config_gimbal_curriculum.yaml`: Curriculum learning (MAIN)

**Composable Overrides**:
```yaml
defaults:
  - _self_
  - task@_global_: landing-aviary-v0-gimbal-curriculum
  - curriculum_preset: balanced
  - encoder_bundle@agent: original
  - actor_critic_bundle@agent: original
  - lr_schedule: constant
```

**Override Syntax**:
```bash
python train_gimbal_curriculum.py \
    key1=value1 \
    key2.nested=value2 \
    +new_key=value3        # Add new key
    ~existing_key          # Remove key
```

**Config Access in Code**:
```python
# In Workspace.__init__()
self.cfg = cfg  # DictConfig from Hydra
env = dmc.make_with_gimbal(self.cfg.task_name, ...)
agent = hydra.utils.instantiate(self.cfg.agent)  # Dynamic instantiation
```

---

## 13. Update Instructions (For Future Maintenance)

When modifying the codebase, update relevant sections:

### A. Adding New Architecture
- Update **Section 6** (Neural Network Architectures)
- Add new bundle to **Section 8.A** (Extension Points)
- Note file paths and line numbers

### B. Changing Curriculum Stages
- Update **Section 5** (Curriculum Learning System)
- Document new stage specifications
- Update **Section 10.D** if advancement logic changes

### C. Refactoring Checkpoint Logic
- Update **Section 7** (Checkpoint & Resume System)
- Document backward compatibility issues in **Section 10.F**
- Update auto-resume priority order if changed

### D. Modifying Data Flow
- Update **Section 4** (Data Flow & Observation Pipeline)
- Check for resolution changes in **Section 10.H**
- Document new wrapper classes

### E. Adding New Scripts
- Update **Section 3** (Codebase Structure)
- Add quick start command in **Section 2**
- Document in **Section 8** if it's an extension pattern

### F. Configuration Changes
- Update **Section 12** (Configuration System)
- Document new Hydra overrides
- Update **Section 2** (Quick Start) with examples

---

## 14. Quick Reference Tables

### Training Script Comparison

| Script | Action Space | Agent | Environment | Curriculum | Auto-Resume |
|--------|--------------|-------|-------------|------------|-------------|
| `train.py` | 3D | `drqv2.DrQV2Agent` | `LandingAviary` | No | No |
| `train_gimbal.py` | 5D | `drqv2-w-new_nets.DrQV2Agent` | `LandingGimbalAviary` | No | No |
| `train_gimbal_curriculum.py` | 5D | `drqv2-w-new_nets.DrQV2Agent` | `LandingGimbalCurriculumAviary` | Yes | Yes |

### Wrapper Selection

| Wrapper | Action Space | State Dim | Use Case |
|---------|--------------|-----------|----------|
| `FrameStackWrapper` | 3D | [3, 7] | Simple landing |
| `FrameStackWrapperWithGimbalState` | 5D | [3, 11] | Gimbal control |
| `FrameStackWrapperWithGimbalOracle` | 3D | [3, 7] | Oracle gimbal (legacy) |

### Encoder Options

| Type | File Location | Key Feature |
|------|---------------|-------------|
| `EncoderBaseline` | `drqv2.py:56-78` | Standard CNN |
| `EncoderTransformer` | `drqv2-w-new_nets.py:216-303` | Temporal attention |
| `EncoderViT` | `drqv2-w-new_nets.py:304-330` | Vision Transformer |

### Actor Options

| Type | File Location | Key Feature |
|------|---------------|-------------|
| `ActorOriginal` | `drqv2.py:81-103` | Standard MLP |
| `ActorAutoregressive` | `drqv2-w-new_nets.py:503-621` | Sequential action generation |
| `ActorAttention` | `drqv2-w-new_nets.py:624-717` | Multi-head attention |

### Critic Options

| Type | File Location | Key Feature |
|------|---------------|-------------|
| `CriticOriginal` | `drqv2.py:106-134` | Twin Q-networks |
| `CriticFactored` | `drqv2-w-new_nets.py:719-785` | Separate drone/gimbal Q-heads |

---

## 15. Common Development Tasks

### Task: Add Learning Rate Scheduling
1. Define scheduler in `lr_scheduler.py`
2. Create config in `cfgs/lr_schedule/`
3. Integrate in `train_gimbal_curriculum.py:155-170`
4. Test: `python train_gimbal_curriculum.py lr_schedule=my_schedule`

### Task: Create Evaluation Script
1. Load checkpoint with `torch.load()`
2. Create environment with `dmc.make_with_gimbal()`
3. Run episodes with `agent.act(..., eval_mode=True)`
4. Collect metrics: success rate, rewards, episode length
5. Reference: `video.py` for evaluation loop pattern

### Task: Modify Reward Function
1. Edit `LandingAviary.py` or `LandingGimbalAviary.py`
2. Locate `_computeReward()` method
3. For curriculum: modify `CurriculumStageSpec` parameters
4. Test with short training run to verify reward scaling

### Task: Add New Network Architecture
1. Define in `drqv2-w-new_nets.py` (encoder/actor/critic)
2. Create bundle config in `cfgs/actor_critic_bundle/`
3. Test: `python train_gimbal_curriculum.py actor_critic_bundle=my_arch`
4. Compare to baseline with tensorboard

### Task: Debug Curriculum Not Advancing
1. Check `train.csv`: `curriculum_stage`, `success_rate_window`
2. Reduce `success_rate` threshold in config (e.g., 0.80)
3. Verify `min_episodes` reached
4. Check `require_consecutive_windows` (reduce to 3)
5. Inspect `_should_advance_curriculum()` in `train_gimbal_curriculum.py:218-221`

### Task: Refactor Model Saving Logic
1. Locate `_save_best_checkpoint_if_needed()` in `train_gimbal_curriculum.py:456-505`
2. Modify top-K limits or metadata tracking
3. Test save and load paths with short run
4. Update `auto_resume_if_possible()` if resume logic changes
5. Document backward compatibility issues in this file (Section 10)

---

**End of Documentation**

*This guide is optimized for AI-assisted development. Prioritizes: data flow understanding, extension points, known gotchas, and token efficiency.*
