# Learning Rate Scheduling Guide

This guide explains how to use and customize learning rate scheduling for encoder, actor, and critic networks.

## Quick Start

Use one of the available presets via command line:

```bash
# No scheduling (constant LR, default)
python train_gimbal_curriculum.py lr_schedule=constant

# Step decay
python train_gimbal_curriculum.py lr_schedule=step_decay

# Exponential decay
python train_gimbal_curriculum.py lr_schedule=exponential_decay

# Linear decay
python train_gimbal_curriculum.py lr_schedule=linear_decay

# Mixed types across networks
python train_gimbal_curriculum.py lr_schedule=mixed_schedule
```

The system is **backward compatible**: running without `lr_schedule` uses constant LR (same as before).

## Schedule Types

### 1. Step Decay

LR changes to fixed values at step boundaries.

**Equation:**
```
lr = lr_i    if start_i ≤ step < end_i
```

**Config:**
```yaml
lr_schedule:
  encoder:
    type: step_decay
    intervals:
      - {start: 0, end: 2000000, lr: 1.0e-4}
      - {start: 2000000, end: 4000000, lr: 5.0e-5}
```

### 2. Exponential Decay

LR decays exponentially at regular intervals.

**Equation:**
```
lr = init_lr × (decay_rate)^n
where n = (step - start) // decay_interval
```

**Config:**
```yaml
lr_schedule:
  encoder:
    type: exponential_decay
    intervals:
      - start: 0
        end: 3000000
        init_lr: 1.0e-4
        decay_rate: 0.9995       # Multiply by this every decay_interval
        decay_interval: 10000     # Decay every 10K steps
```

### 3. Linear Decay

LR interpolates linearly between init_lr and final_lr.

**Equation:**
```
lr = init_lr + (final_lr - init_lr) × progress
where progress = (step - start) / (end - start)
```

**Config:**
```yaml
lr_schedule:
  encoder:
    type: linear_decay
    intervals:
      - {start: 0, end: 3000000, init_lr: 1.0e-4, final_lr: 2.0e-5}
```

## Creating Custom Schedules

You can mix different schedule types for encoder, actor, and critic:

**Example: Custom mixed schedule**

Create `cfgs/lr_schedule/my_schedule.yaml`:

```yaml
# @package _global_

lr_schedule:
  # Encoder: Conservative step decay
  encoder:
    type: step_decay
    intervals:
      - {start: 0, end: 3000000, lr: 8.0e-5}
      - {start: 3000000, end: 10000000, lr: 4.0e-5}

  # Actor: Aggressive exponential decay
  actor:
    type: exponential_decay
    intervals:
      - start: 0
        end: 2000000
        init_lr: 1.5e-4
        decay_rate: 0.999
        decay_interval: 5000
      - start: 2000000
        end: 10000000
        init_lr: 7.0e-5
        decay_rate: 0.9995
        decay_interval: 10000

  # Critic: Smooth linear decay
  critic:
    type: linear_decay
    intervals:
      - {start: 0, end: 2500000, init_lr: 1.2e-4, final_lr: 3.0e-5}
      - {start: 2500000, end: 6000000, init_lr: 3.0e-5, final_lr: 1.0e-5}
```

Use it: `python train_gimbal_curriculum.py lr_schedule=my_schedule`

## Interval Rules

**Contiguous intervals (recommended):**
```yaml
intervals:
  - {start: 0, end: 1000000, lr: 1.0e-4}
  - {start: 1000000, end: 2000000, lr: 5.0e-5}  # end matches next start
```

**Gaps (allowed):** LR remains at previous value during gap.
```yaml
intervals:
  - {start: 0, end: 1000000, lr: 1.0e-4}
  - {start: 2000000, end: 3000000, lr: 5.0e-5}  # Gap: steps 1M-2M stay at 1e-4
```

**Overlaps (forbidden):** Training will fail with ValueError.
```yaml
intervals:
  - {start: 0, end: 2000000, lr: 1.0e-4}
  - {start: 1000000, end: 3000000, lr: 5.0e-5}  # ERROR: overlap at 1M-2M
```

## Monitoring

Learning rates are automatically logged to TensorBoard:
- `lr_encoder` - Encoder learning rate
- `lr_actor` - Actor learning rate
- `lr_critic` - Critic learning rate

Check these curves to verify your schedule is working as expected.

## Implementation Structure

**Overview:**
1. **Scheduler classes** (`lr_scheduler.py`): `StepDecayScheduler`, `ExponentialDecayScheduler`, `LinearDecayScheduler`, `ConstantLRScheduler`
2. **Agent integration** (`drqv2-w-new_nets.py`): Schedulers created in `__init__`, updated in `update()`
3. **Config management** (`cfgs/`): Hydra presets for easy composition

**Flow:**
1. Agent receives `lr_schedule` config (or `None` for constant LR)
2. `create_multi_optimizer_schedulers()` creates scheduler per optimizer
3. On each `agent.update(step)`:
   - Each scheduler calculates LR for current step
   - Updates optimizer's `param_groups[0]['lr']` if changed
   - Logs to TensorBoard if `use_tb=True`

**Key functions:**
- `create_lr_scheduler()`: Factory for single optimizer
- `create_multi_optimizer_schedulers()`: Factory for encoder/actor/critic
- `LRScheduler.step(global_step)`: Update LR based on current step
- `LRScheduler.get_lr(global_step)`: Calculate LR for given step

**Backward compatibility:**
- If `lr_schedule` is `None` or not provided, uses `ConstantLRScheduler`
- Returns constant LR equal to `lr` from config
- No behavior change from previous code

## Command Line Overrides

Override specific parameters:

```bash
# Change encoder's first interval LR
python train_gimbal_curriculum.py \
  lr_schedule=step_decay \
  lr_schedule.encoder.intervals.0.lr=2.0e-4

# Change actor's decay rate
python train_gimbal_curriculum.py \
  lr_schedule=exponential_decay \
  lr_schedule.actor.intervals.0.decay_rate=0.999
```
