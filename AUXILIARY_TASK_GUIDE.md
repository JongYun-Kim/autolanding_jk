# Auxiliary Gimbal Alignment Task Guide

## Overview

The auxiliary gimbal alignment task provides an additional supervised learning signal to help the agent learn gimbal control more effectively. By predicting the ideal gimbal orientation (pointing at the landing pad center) as an auxiliary task, the network receives more direct feedback beyond the sparse reward signal.

### Key Benefits
- **Stronger Learning Signal**: Direct supervision for gimbal control alongside RL rewards
- **Faster Convergence**: More explicit guidance on what the gimbal should be doing
- **Better Representations**: Encoder learns features useful for both value estimation and gimbal alignment

## How It Works

### High-Level Architecture

```
┌─────────────────────────────────────────────────────┐
│ Observation (images + drone state)                 │
└────────────────┬────────────────────────────────────┘
                 ↓
         ┌───────────────┐
         │    Encoder    │
         └───────┬───────┘
                 ↓
         [Representation h]
                 ↓
    ┌────────────┴────────────┐
    ↓                         ↓
┌─────────┐          ┌─────────────────┐
│ Actor & │          │ Auxiliary Head  │
│ Critic  │          │ (Gimbal Pred)   │
└─────────┘          └─────────────────┘
    ↓                         ↓
RL Loss              Auxiliary Loss (MSE)
 (Q-learning)        (vs. Oracle Gimbal)
    └─────────────┬─────────────┘
                  ↓
         Total Loss = RL Loss + λ * Aux Loss
```

### Training Process

1. **Oracle Computation**: Environment computes ideal gimbal angles pointing at landing pad
2. **Prediction**: Auxiliary head predicts gimbal angles from encoder features
3. **Auxiliary Loss**: MSE between predicted and oracle gimbal angles [pitch, yaw]
4. **Combined Training**: Total loss = `critic_loss + weight * auxiliary_loss`
5. **Shared Encoder**: Auxiliary gradients flow back through encoder, improving representations

### Oracle Gimbal

The oracle gimbal angles are computed geometrically by the environment:
- **Input**: Current drone position, gimbal camera offset, landing pad position
- **Output**: Ideal [pitch, yaw] angles (normalized to [-1, 1]) that point camera at pad center
- **Availability**: Computed every step during training (not available at inference)

## Configuration

### Default Configuration

The auxiliary task is **disabled by default** to maintain backward compatibility.

```yaml
# cfgs/config_gimbal_curriculum.yaml
agent:
  auxiliary_task:
    enable: false      # Disabled by default
    weight: 0.1        # Weight of auxiliary loss (λ)
    hidden_dim: 128    # MLP hidden dimension for prediction head
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable` | `false` | Enable/disable auxiliary task |
| `weight` | `0.1` | Relative weight of auxiliary loss (λ) |
| `hidden_dim` | `128` | Hidden dimension of auxiliary MLP head |

**Tuning `weight`:**
- **Lower (0.01-0.05)**: Subtle guidance, mostly RL-driven learning
- **Medium (0.1-0.2)**: Balanced contribution from both signals (recommended starting point)
- **Higher (0.5-1.0)**: Strong supervision, may bias towards imitation

## Usage

### Basic Usage

```bash
# Enable auxiliary task with default settings
python train_gimbal_curriculum.py agent.auxiliary_task.enable=true
```

### Custom Configuration

```bash
# Enable with custom weight
python train_gimbal_curriculum.py \
    agent.auxiliary_task.enable=true \
    agent.auxiliary_task.weight=0.2

# Full customization
python train_gimbal_curriculum.py \
    agent.auxiliary_task.enable=true \
    agent.auxiliary_task.weight=0.15 \
    agent.auxiliary_task.hidden_dim=256
```

### Combined with Other Settings

```bash
# Auxiliary task + specific architecture + curriculum preset
python train_gimbal_curriculum.py \
    agent.auxiliary_task.enable=true \
    agent.auxiliary_task.weight=0.1 \
    actor_critic_bundle=multihead_attention \
    curriculum_preset=aggressive
```

## Monitoring

### TensorBoard Metrics

When the auxiliary task is enabled, additional metrics are logged:

| Metric | Description |
|--------|-------------|
| `aux_loss` | MSE loss between predicted and oracle gimbal angles |
| `aux_gimbal_error` | Mean absolute error of gimbal predictions (in normalized space) |
| `total_loss` | Combined loss (critic + auxiliary) |
| `critic_loss` | Standard critic loss (for comparison) |

### Viewing Metrics

```bash
# Launch TensorBoard
tensorboard --logdir exp_local/{date}/{time}/tb

# Key metrics to monitor:
# - aux_loss: Should decrease over training
# - aux_gimbal_error: Should approach 0.0 (perfect alignment)
# - Compare aux_loss vs critic_loss to understand balance
```

### Interpreting Results

**Healthy Training:**
- `aux_loss` steadily decreases
- `aux_gimbal_error` < 0.1 after curriculum stage 2-3
- Both RL metrics (success rate) and auxiliary metrics improve

**Red Flags:**
- `aux_loss` plateaus at high values → Increase `weight` or check data pipeline
- `aux_gimbal_error` remains > 0.3 after many episodes → Network capacity issue
- Success rate degrades with auxiliary task → Reduce `weight`

## Best Practices

### When to Use

✅ **Recommended for:**
- Initial training of gimbal control policies
- Curriculum learning (especially early stages)
- Complex gimbal control tasks
- When sample efficiency is critical

❌ **May not help for:**
- Fine-tuning well-trained policies
- Tasks where gimbal reward is already very dense
- When you want pure RL without imitation bias

### Curriculum Integration

The auxiliary task works well across curriculum stages:

```yaml
# Stage 0 (lock): Auxiliary task has no effect (gimbal locked)
# Stage 1-2 (partial): Strong auxiliary signal helps bootstrap gimbal learning
# Stage 3-4 (full): Auxiliary task refines alignment while RL optimizes landing
```

**Recommendation**: Enable from the beginning and keep consistent across stages.

### Hyperparameter Tuning

Start with defaults, then adjust `weight` based on metrics:

1. **Baseline**: Run with `weight=0.1`
2. **Monitor**: Check `aux_loss` and success rate after 100k-200k steps
3. **Adjust**:
   - If `aux_loss` dominates but success rate is low → Reduce weight to 0.05
   - If gimbal errors persist → Increase weight to 0.2
   - If both metrics improve → Keep current setting

## Example Training Session

```bash
# 1. Start training with auxiliary task enabled
python train_gimbal_curriculum.py \
    curriculum_preset=balanced \
    actor_critic_bundle=original \
    agent.auxiliary_task.enable=true \
    seed=42

# 2. Monitor training
tensorboard --logdir exp_local/

# 3. Expected metrics after Stage 2:
#    - aux_gimbal_error: ~0.05-0.1
#    - success_rate: ~0.85+
#    - aux_loss: ~0.01-0.02
```

## Backward Compatibility

The auxiliary task is **fully backward compatible**:

- ✅ Disabled by default (`enable: false`)
- ✅ Old checkpoints load without issues
- ✅ Existing training scripts work unchanged
- ✅ No performance impact when disabled

To verify backward compatibility:
```bash
# Standard training (auxiliary task disabled)
python train_gimbal_curriculum.py

# Should produce identical results to previous version
```

## Technical Notes

### Oracle Gimbal Computation

- Computed using geometric projection in environment
- Accounts for drone position, orientation, camera offset
- Clipped to gimbal physical limits (±47° pitch, ±90° yaw)
- Available in `info['oracle_gimbal']` during training only

### Data Pipeline

Oracle gimbal data flows through:
1. **Environment**: Computed in `LandingGimbalAviary.step()`
2. **Wrappers**: Extracted in `ExtendedTimeStepWrapper`
3. **Replay Buffer**: Stored in episode files (`*.npz`)
4. **Sampling**: Returned with standard batch data
5. **Agent**: Used in `update_critic()` for auxiliary loss

### Network Architecture

```python
AuxiliaryGimbalHead(
    Linear(repr_dim → hidden_dim),
    LayerNorm,
    Tanh,
    Linear(hidden_dim → 2)  # Output: [pitch, yaw]
)
```

- Shared encoder gradients improve representations
- Lightweight head (128-256 hidden units recommended)
- Optimized together with encoder during critic updates

## FAQ

**Q: Does this make the agent just imitate oracle gimbal?**
A: No. The auxiliary loss is only one component (typically 10% of total loss). The RL objective still dominates and optimizes for landing success.

**Q: Why not use oracle gimbal as the action directly?**
A: The auxiliary task provides *learning guidance*, not direct control. The agent still learns to balance gimbal control with drone control for successful landing.

**Q: Can I enable this mid-training?**
A: Yes, but it's recommended to start from the beginning for best results. The encoder learns better representations when trained with the auxiliary task from scratch.

**Q: What if I don't see improvement?**
A: Try adjusting the `weight` parameter. Too high may bias the policy, too low may have minimal effect. Monitor both `aux_loss` and `success_rate` to find the sweet spot.

**Q: Does this work with all encoder/actor types?**
A: Yes. The auxiliary head sits on top of the encoder representation, so it's compatible with all architecture bundles (original, attention, transformer, etc.).

---

For implementation details, see:
- `drqv2-w-new_nets.py:45-71` - Auxiliary head architecture
- `drqv2-w-new_nets.py:984-1027` - Loss computation
- `LandingAviary.py:522-638` - Oracle gimbal computation
