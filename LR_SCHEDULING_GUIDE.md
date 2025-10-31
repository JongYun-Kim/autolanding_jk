# Learning Rate Scheduling Guide

This guide explains how to use learning rate scheduling in your RL training pipeline.

## Overview

Learning rate scheduling allows you to dynamically adjust learning rates during training for the encoder, actor, and critic networks independently. This can improve convergence, stability, and final performance.

## Quick Start

### Using Presets (Recommended)

The easiest way to use LR scheduling is through presets:

```bash
# Use gentle step decay (recommended for most cases)
python train_gimbal_curriculum.py lr_schedule=gentle_decay

# Use aggressive decay for faster training
python train_gimbal_curriculum.py lr_schedule=aggressive_decay

# Use curriculum-aligned scheduling
python train_gimbal_curriculum.py lr_schedule=stage_based

# Use no scheduling (default, backward compatible)
python train_gimbal_curriculum.py lr_schedule=constant
```

### Available Presets

| Preset | Type | Description | Best For |
|--------|------|-------------|----------|
| `constant` | None | No scheduling (default LR throughout) | Baseline, backward compatibility |
| `gentle_decay` | Step | Gradual 2-stage decay | Stable training, risk-averse |
| `aggressive_decay` | Step | Fast 4-stage decay | Quick convergence, exploration |
| `exponential_smooth` | Exponential | Continuous smooth decay | Sensitive tasks, avoiding instabilities |
| `stage_based` | Step | Aligned with curriculum stages | Curriculum learning scenarios |
| `differential` | Mixed | Different schedules per network | Advanced tuning, specific issues |

## Schedule Types

### 1. Step Decay

Learning rate changes to fixed values at specific step boundaries.

**Example:**
```yaml
lr_schedule:
  encoder:
    type: step_decay
    intervals:
      - {start: 0, end: 1000000, lr: 1.0e-4}      # 0-1M steps: LR = 1e-4
      - {start: 1000000, end: 3000000, lr: 5.0e-5}  # 1M-3M steps: LR = 5e-5
      - {start: 3000000, end: 6000000, lr: 1.0e-5}  # 3M-6M steps: LR = 1e-5
```

### 2. Exponential Decay

Learning rate decays exponentially: `lr = init_lr * (decay_rate ^ num_decays)`

**Example:**
```yaml
lr_schedule:
  encoder:
    type: exponential_decay
    intervals:
      - start: 0
        end: 1000000
        init_lr: 1.0e-4
        decay_rate: 0.999      # Multiply by 0.999 every decay_interval
        decay_interval: 1000    # Decay every 1000 steps
```

## Custom Configurations

### Per-Network Scheduling

Configure different schedules for encoder, actor, and critic:

```yaml
lr_schedule:
  # Encoder: Conservative (stable features)
  encoder:
    type: step_decay
    intervals:
      - {start: 0, end: 3000000, lr: 8.0e-5}

  # Actor: Moderate (exploration to exploitation)
  actor:
    type: exponential_decay
    intervals:
      - {start: 0, end: 2000000, init_lr: 1.2e-4, decay_rate: 0.999, decay_interval: 5000}

  # Critic: Aggressive (fast convergence)
  critic:
    type: step_decay
    intervals:
      - {start: 0, end: 1000000, lr: 1.5e-4}
      - {start: 1000000, end: 3000000, lr: 5.0e-5}
```

### Creating Custom Presets

1. Create a new file in `cfgs/lr_schedule/your_preset.yaml`:

```yaml
# @package _global_

lr_schedule:
  encoder:
    type: step_decay
    intervals:
      - {start: 0, end: 2000000, lr: 1.0e-4}
      - {start: 2000000, end: 6000000, lr: 5.0e-5}

  actor:
    type: step_decay
    intervals:
      - {start: 0, end: 2000000, lr: 1.0e-4}
      - {start: 2000000, end: 6000000, lr: 5.0e-5}

  critic:
    type: step_decay
    intervals:
      - {start: 0, end: 2000000, lr: 1.0e-4}
      - {start: 2000000, end: 6000000, lr: 5.0e-5}
```

2. Use it: `python train_gimbal_curriculum.py lr_schedule=your_preset`

### Override via Command Line

Override specific parameters:

```bash
# Override just the encoder schedule
python train_gimbal_curriculum.py \
  lr_schedule=gentle_decay \
  lr_schedule.encoder.intervals.0.lr=2.0e-4

# Change schedule type for actor
python train_gimbal_curriculum.py \
  lr_schedule=gentle_decay \
  lr_schedule.actor.type=exponential_decay
```

## Monitoring

Learning rates are automatically logged to TensorBoard under:
- `lr_encoder`
- `lr_actor`
- `lr_critic`

Monitor these curves to verify your schedule is working as expected.

## Recommendations

### For Standard Training (6M steps)

**Start with:** `gentle_decay`
- Proven safe and stable
- Good baseline for comparison
- 3-stage decay: 1e-4 → 5e-5 → 2.5e-5

### For Curriculum Learning

**Start with:** `stage_based`
- Aligned with curriculum progression
- Higher LR early, lower as difficulty increases
- Natural fit for staged learning

### For Quick Experiments

**Start with:** `aggressive_decay`
- Faster convergence
- More decay stages
- May sacrifice some final performance for speed

### For Sensitive/Unstable Training

**Start with:** `exponential_smooth`
- Gradual, continuous decay
- No sudden changes
- Better for avoiding training instabilities

### For Advanced Users

**Start with:** `differential`
- Different schedules per network
- Tune encoder, actor, critic independently
- Best when you know specific components need adjustment

## Backward Compatibility

**Default behavior:** If you don't specify `lr_schedule`, the system uses a constant learning rate (same as before this feature was added).

```bash
# These are equivalent (constant LR = 1e-4)
python train_gimbal_curriculum.py
python train_gimbal_curriculum.py lr_schedule=constant
```

## Troubleshooting

### Training becomes unstable

- Try `gentle_decay` or `exponential_smooth`
- Reduce initial learning rate: `lr=5.0e-5`
- Increase decay rate for exponential: `decay_rate=0.9998`

### Convergence too slow

- Try `aggressive_decay`
- Increase initial learning rate: `lr=2.0e-4`
- Delay first decay step

### Actor/Critic imbalance

- Use `differential` preset
- Adjust individual network schedules
- Monitor `lr_encoder`, `lr_actor`, `lr_critic` in TensorBoard

### Schedule not matching curriculum

- Use `stage_based` preset
- Manually align interval boundaries with your curriculum stages
- Check curriculum advancement with `curriculum/stage` metric

## Examples

### Example 1: Standard Training

```bash
python train_gimbal_curriculum.py \
  lr_schedule=gentle_decay \
  curriculum_preset=balanced \
  encoder_bundle@agent=enc_B
```

### Example 2: Fast Training

```bash
python train_gimbal_curriculum.py \
  lr_schedule=aggressive_decay \
  lr=2.0e-4 \
  num_train_frames=3000000
```

### Example 3: Custom Multi-Phase Training

```bash
python train_gimbal_curriculum.py \
  lr_schedule=differential \
  lr_schedule.encoder.intervals.0.lr=1.0e-4 \
  lr_schedule.actor.intervals.0.init_lr=1.5e-4 \
  lr_schedule.critic.intervals.0.lr=2.0e-4
```

### Example 4: Curriculum-Aligned

```bash
python train_gimbal_curriculum.py \
  lr_schedule=stage_based \
  curriculum_preset=conservative
```

## Technical Details

- Schedulers are implemented in `lr_scheduler.py`
- Integration in `drqv2-w-new_nets.py` (lines 509, 564-571, 666-670)
- LR updates happen every `update_every_steps` (default: 2)
- Schedulers use global step count for timing
- Intervals are validated to prevent overlaps

## Best Practices

1. **Start simple:** Use presets before creating custom configs
2. **Monitor closely:** Check TensorBoard LR curves match expectations
3. **Align with curriculum:** If using curriculum, consider `stage_based`
4. **Same schedule initially:** Use same schedule for all networks first
5. **Iterate gradually:** Make small adjustments, don't change everything at once
6. **Document experiments:** Keep track of which schedules work for your task
