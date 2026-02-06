# Gimbal Control Modes Guide

This guide explains the different gimbal control modes available in the autonomous drone landing system and how to use them.

## Overview

The gimbal control system supports three different control modes, providing varying levels of realism for gimbal dynamics:

1. **Position Control (Default)** - Instant angle adjustment (backward compatible)
2. **Velocity Control** - Rate-limited gimbal movement
3. **Acceleration Control** - Inertial gimbal dynamics with velocity and acceleration limits

## Control Modes

### 1. Position Control (Default)

**Description:** The gimbal instantly adjusts to the commanded angle. This is the original behavior and maintains full backward compatibility with existing trained models.

**When to use:**
- When you need instant gimbal response
- For compatibility with existing trained models
- For simplified training experiments

**Action interpretation:**
- Action values directly specify the desired gimbal angle (normalized to [-1, 1])
- Gimbal immediately moves to the commanded position

**Configuration:**
```yaml
gimbal_control_mode: "position"  # This is the default, can be omitted
```

**Training command:**
```bash
python train_gimbal_curriculum.py curriculum_preset=balanced
```

---

### 2. Velocity Control

**Description:** The action specifies the desired gimbal angular velocity rather than position. The gimbal moves at the commanded velocity with configurable maximum velocity limits.

**When to use:**
- For more realistic gimbal dynamics
- When you want rate-limited gimbal movement
- For smoother gimbal trajectories

**Action interpretation:**
- Action values specify desired angular velocity (normalized to [-1, 1])
- Action range [-1, 1] maps to [-max_velocity, +max_velocity] in rad/s
- Gimbal position integrates the velocity over time

**Physical interpretation:**
- `gimbal_max_velocity`: Maximum angular velocity in rad/s
  - Typical value: 3.0 rad/s (~172°/s)
  - Range: 1.0 - 10.0 rad/s depending on desired responsiveness

**Configuration:**
```yaml
gimbal_control_mode: "velocity"
gimbal_max_velocity: 3.0  # rad/s
```

**Training command:**
```bash
python train_gimbal_curriculum.py \
    task@_global_=landing-aviary-v0-gimbal-curriculum-velocity \
    curriculum_preset=balanced
```

**Example scenario:**
If `gimbal_max_velocity = 3.0 rad/s` and action = `[..., 0.5, ...]` (for pitch):
- Desired velocity: 0.5 × 3.0 = 1.5 rad/s
- At 240 Hz physics: gimbal moves ~0.00625 rad per step
- Time to move ±47° (±0.82 rad): ~0.55 seconds

---

### 3. Acceleration Control

**Description:** The action specifies the desired gimbal angular acceleration. The gimbal has inertial dynamics with both velocity and acceleration limits, providing the most realistic gimbal behavior.

**When to use:**
- For highly realistic gimbal simulation
- When modeling actual gimbal motor dynamics
- For challenging RL training scenarios

**Action interpretation:**
- Action values specify desired angular acceleration (normalized to [-1, 1])
- Action range [-1, 1] maps to [-max_acceleration, +max_acceleration] in rad/s²
- Gimbal velocity integrates the acceleration
- Gimbal position integrates the velocity
- Both velocity and acceleration are clamped to their limits

**Physical interpretation:**
- `gimbal_max_velocity`: Maximum angular velocity in rad/s (same as velocity mode)
- `gimbal_max_acceleration`: Maximum angular acceleration in rad/s²
  - Typical value: 10.0 rad/s²
  - Range: 5.0 - 30.0 rad/s² depending on motor characteristics

**Configuration:**
```yaml
gimbal_control_mode: "acceleration"
gimbal_max_velocity: 3.0  # rad/s
gimbal_max_acceleration: 10.0  # rad/s²
```

**Training command:**
```bash
python train_gimbal_curriculum.py \
    task@_global_=landing-aviary-v0-gimbal-curriculum-acceleration \
    curriculum_preset=balanced
```

**Example scenario:**
If `gimbal_max_acceleration = 10.0 rad/s²` and constant action = `[..., 1.0, ...]`:
- Acceleration: 10.0 rad/s²
- Time to reach max velocity (3.0 rad/s): 0.3 seconds
- Time to move ±47° from rest: ~0.7 seconds (including acceleration and deceleration)

---

## Configuration Parameters

### Task Configuration

Add these parameters to your task YAML file (e.g., `cfgs/task/landing-aviary-v0-gimbal-curriculum.yaml`):

```yaml
# Gimbal control mode: "position", "velocity", or "acceleration"
gimbal_control_mode: "position"  # Default

# Maximum angular velocity (used in velocity and acceleration modes)
gimbal_max_velocity: 3.0  # rad/s

# Maximum angular acceleration (used only in acceleration mode)
gimbal_max_acceleration: 10.0  # rad/s²
```

### Environment Instantiation

When creating the environment programmatically:

```python
import gym
import gym_pybullet_drones

# Position mode (default)
env = gym.make('gimbal-curriculum-landing-aviary-v0')

# Velocity mode
env = gym.make('gimbal-curriculum-landing-aviary-v0',
               gimbal_control_mode='velocity',
               gimbal_max_velocity=3.0)

# Acceleration mode
env = gym.make('gimbal-curriculum-landing-aviary-v0',
               gimbal_control_mode='acceleration',
               gimbal_max_velocity=3.0,
               gimbal_max_acceleration=10.0)
```

---

## Comparison Table

| Feature | Position | Velocity | Acceleration |
|---------|----------|----------|--------------|
| **Realism** | Low | Medium | High |
| **Action meaning** | Angle | Velocity | Acceleration |
| **Response time** | Instant | ~0.5s for full range | ~0.7s for full range |
| **Dynamics complexity** | None | 1st-order | 2nd-order |
| **Training difficulty** | Easy | Medium | Hard |
| **Backward compatible** | Yes | No | No |
| **State tracking** | Target only | Position | Position + Velocity |

---

## Training Tips

### For Velocity Control:
1. **Adjust curriculum stages:** You may need to extend early curriculum stages as the agent needs to learn velocity control
2. **Exploration noise:** Consider slightly higher `stddev_schedule` initial values for better exploration
3. **Hyperparameters:** May need to tune learning rate and batch size

### For Acceleration Control:
1. **Extended training:** Expect 20-30% longer training time due to increased complexity
2. **Curriculum design:** Add intermediate stages with partial acceleration limits
3. **Reward shaping:** May benefit from additional rewards for smooth gimbal motion
4. **Initialization:** Consider warm-starting from a velocity-mode trained model

### General Recommendations:
- **Start simple:** Begin with position mode to verify your setup
- **Gradual transition:** Move to velocity mode, then acceleration mode
- **Monitor metrics:** Track gimbal tracking error and settling time
- **Tune parameters:** Adjust `gimbal_max_velocity` and `gimbal_max_acceleration` based on your scenario

---

## Implementation Details

### State Variables

The gimbal control system tracks:
- `gimbal_target`: Commanded target from agent action
- `gimbal_current_angles`: Actual current gimbal angles (normalized)
- `gimbal_current_velocity`: Current gimbal velocity (for acceleration mode)

### Update Frequency

Gimbal dynamics are updated at the physics simulation frequency:
- Default: 240 Hz (dt = 1/240 ≈ 0.00417 seconds)
- Configurable via `freq` parameter

### Angle Ranges

Gimbal physical limits (in radians):
- Pitch: ±5π/19 ≈ ±47°
- Roll: ±π/4 = ±45° (currently fixed at 0)
- Yaw: ±π/2 = ±90°

These limits are enforced regardless of control mode.

---

## Example Training Commands

### Position mode (default, backward compatible):
```bash
python train_gimbal_curriculum.py curriculum_preset=balanced
```

### Velocity mode with custom parameters:
```bash
python train_gimbal_curriculum.py \
    task@_global_=landing-aviary-v0-gimbal-curriculum-velocity \
    curriculum_preset=balanced \
    gimbal_max_velocity=4.0
```

### Acceleration mode with custom parameters:
```bash
python train_gimbal_curriculum.py \
    task@_global_=landing-aviary-v0-gimbal-curriculum-acceleration \
    curriculum_preset=balanced \
    gimbal_max_velocity=3.0 \
    gimbal_max_acceleration=15.0
```

### Overriding control mode on the fly:
```bash
# Use acceleration mode with the default task config
python train_gimbal_curriculum.py \
    curriculum_preset=balanced \
    gimbal_control_mode=acceleration \
    gimbal_max_velocity=3.0 \
    gimbal_max_acceleration=10.0
```

---

## Troubleshooting

### Issue: Training fails to converge with velocity/acceleration mode

**Solutions:**
1. Increase curriculum stage duration (more episodes per stage)
2. Reduce `gimbal_max_velocity` for smoother control
3. Add intermediate curriculum stages
4. Increase exploration noise duration

### Issue: Gimbal movement too slow

**Solutions:**
1. Increase `gimbal_max_velocity` (velocity and acceleration modes)
2. Increase `gimbal_max_acceleration` (acceleration mode only)
3. Verify your action values are reaching ±1 during training

### Issue: Gimbal oscillates or unstable

**Solutions:**
1. Decrease `gimbal_max_acceleration` (acceleration mode)
2. Add damping via reward shaping (penalize rapid changes)
3. Increase `smooth_penalty` in curriculum stages
4. Reduce learning rate

---

## Future Extensions

Potential enhancements to gimbal control:
1. **Friction/damping models:** Add velocity-dependent resistance
2. **Gear backlash:** Model mechanical slack in gimbal gears
3. **Motor saturation:** Non-linear torque curves
4. **Gimbal lock:** Singularity handling for extreme orientations
5. **External disturbances:** Wind gusts affecting gimbal stability

---

## References

- Main environment implementation: `gym_pybullet_drones/envs/single_agent_rl/LandingAviary.py`
- Control mode update logic: `LandingGimbalAviary._update_gimbal_dynamics()` (lines 687-757)
- Training script: `train_gimbal_curriculum.py`
- Task configurations: `cfgs/task/landing-aviary-v0-gimbal-curriculum-*.yaml`

For questions or issues, please refer to the main REPOSITORY_GUIDE.md or create an issue.
