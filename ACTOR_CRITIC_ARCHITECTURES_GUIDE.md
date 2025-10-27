# Actor/Critic Architectures - Usage Guide

## Overview

This project now supports multiple actor and critic architectures for ablation studies and performance comparison. The architectures can be easily switched via Hydra configuration files.

## Available Architectures

### Actor Types

#### 1. **Original** (`original`)
- **Description**: Baseline joint action actor
- **Action Generation**: All 5 actions `[vx, vy, vz, gimbal_pitch, gimbal_yaw]` generated simultaneously
- **Pros**: Simple, well-tested baseline
- **Cons**: No explicit coupling between drone and gimbal actions
- **Use Case**: Baseline for comparisons

#### 2. **Autoregressive Gimbal-First** (`autoregressive_gimbal_first`)
- **Description**: Point-then-move strategy
- **Action Generation**:
  1. First samples gimbal action `[pitch, yaw]`
  2. Then samples drone action `[vx, vy, vz]` conditioned on gimbal
- **Pros**: Natural for camera-centric tasks, gimbal stabilizes first
- **Cons**: Sequential sampling (slightly slower)
- **Use Case**: Tasks where camera pointing is critical

#### 3. **Autoregressive Drone-First** (`autoregressive_drone_first`)
- **Description**: Move-then-point strategy
- **Action Generation**:
  1. First samples drone action `[vx, vy, vz]`
  2. Then samples gimbal action `[pitch, yaw]` conditioned on drone
- **Pros**: Good for aggressive flight, gimbal compensates
- **Cons**: Camera may lag behind optimal pointing
- **Use Case**: Navigation-heavy tasks

#### 4. **Multi-Head Attention** (`multihead_attention`)
- **Description**: Parallel but contextually-aware actions (RECOMMENDED)
- **Action Generation**:
  - Separate query embeddings for drone and gimbal
  - Cross-attention with observation
  - Self-attention between drone and gimbal for coordination
  - Parallel action generation
- **Pros**:
  - Most flexible, can learn any coordination strategy
  - Parallel generation (fast)
  - Bidirectional information flow
- **Cons**: More parameters, may need more data
- **Use Case**: Best default choice for new experiments
- **Configuration**:
  ```yaml
  actor_cfg:
    nhead: 4  # Number of attention heads (default: 4)
  ```

### Critic Types

#### 1. **Original** (`original`)
- **Description**: Baseline joint critic
- **Q-Value**: `Q(s, a_full)` - evaluates full joint action
- **Use Case**: Works with all actor types

#### 2. **Factored** (`factored`)
- **Description**: Factored Q-value for autoregressive actors
- **Q-Value**: `Q(s, a) = Q_drone(s, a_drone) + Q_joint(s, a_full)`
- **Pros**: Separates drone base value from gimbal coordination value
- **Use Case**: Recommended for autoregressive actors, also works with others

## Configuration Files

### Using Pre-defined Bundles

Configuration bundles are in `cfgs/actor_critic_bundle/`:

1. **`original.yaml`** - Baseline
   ```yaml
   actor_type: original
   actor_cfg: null
   critic_type: original
   ```

2. **`autoregressive_gimbal_first.yaml`**
   ```yaml
   actor_type: autoregressive_gimbal_first
   actor_cfg: null
   critic_type: factored
   ```

3. **`autoregressive_drone_first.yaml`**
   ```yaml
   actor_type: autoregressive_drone_first
   actor_cfg: null
   critic_type: factored
   ```

4. **`multihead_attention.yaml`** (Recommended)
   ```yaml
   actor_type: multihead_attention
   actor_cfg:
     nhead: 4
   critic_type: original
   ```

5. **`multihead_attention_factored.yaml`**
   ```yaml
   actor_type: multihead_attention
   actor_cfg:
     nhead: 4
   critic_type: factored
   ```

### Selecting an Architecture

In `cfgs/config_gimbal_curriculum.yaml`:

```yaml
defaults:
  - ...
  - encoder_bundle@agent: enc_B
  - actor_critic_bundle@agent: multihead_attention  # Change this line
  - ...
```

Or via command line:

```bash
python train_gimbal_curriculum.py actor_critic_bundle@agent=multihead_attention
```

## Usage Examples

### 1. Train with Original Architecture (Baseline)

```bash
python train_gimbal_curriculum.py \
  actor_critic_bundle@agent=original
```

### 2. Train with Multi-Head Attention Actor (Recommended)

```bash
python train_gimbal_curriculum.py \
  actor_critic_bundle@agent=multihead_attention
```

### 3. Train with Autoregressive Gimbal-First

```bash
python train_gimbal_curriculum.py \
  actor_critic_bundle@agent=autoregressive_gimbal_first
```

### 4. Combine with Different Encoders

```bash
# Multi-head attention actor + Encoder B
python train_gimbal_curriculum.py \
  encoder_bundle@agent=enc_B \
  actor_critic_bundle@agent=multihead_attention

# Autoregressive actor + Encoder A
python train_gimbal_curriculum.py \
  encoder_bundle@agent=enc_A \
  actor_critic_bundle@agent=autoregressive_gimbal_first
```

### 5. Custom Actor Configuration

Create a new config file `cfgs/actor_critic_bundle/my_custom.yaml`:

```yaml
actor_type: multihead_attention
actor_cfg:
  nhead: 8  # More attention heads
critic_type: factored
```

Then use it:

```bash
python train_gimbal_curriculum.py \
  actor_critic_bundle@agent=my_custom
```

### 6. Ablation Study

Run multiple experiments with different architectures:

```bash
# Baseline
python train_gimbal_curriculum.py \
  actor_critic_bundle@agent=original \
  experiment=ablation_original

# Autoregressive gimbal-first
python train_gimbal_curriculum.py \
  actor_critic_bundle@agent=autoregressive_gimbal_first \
  experiment=ablation_autoregressive_gf

# Autoregressive drone-first
python train_gimbal_curriculum.py \
  actor_critic_bundle@agent=autoregressive_drone_first \
  experiment=ablation_autoregressive_df

# Multi-head attention
python train_gimbal_curriculum.py \
  actor_critic_bundle@agent=multihead_attention \
  experiment=ablation_multihead

# Multi-head attention with factored critic
python train_gimbal_curriculum.py \
  actor_critic_bundle@agent=multihead_attention_factored \
  experiment=ablation_multihead_factored
```

## Architecture Comparison

| Architecture | Coordination | Speed | Flexibility | Complexity |
|-------------|-------------|-------|------------|------------|
| Original | None | Fast | Low | Low |
| Autoregressive (G→D) | Sequential | Medium | Medium | Medium |
| Autoregressive (D→G) | Sequential | Medium | Medium | Medium |
| Multi-Head Attention | Parallel | Fast | High | High |

## Recommendations

### For New Experiments
**Start with**: `multihead_attention`
- Most flexible
- Can learn optimal coordination strategy
- Good performance potential

### For Camera-Critical Tasks
**Use**: `autoregressive_gimbal_first`
- Prioritizes camera pointing
- Good for visual tracking

### For Fast Baseline
**Use**: `original`
- Simple and fast
- Good for debugging

### For Ablation Studies
**Test all four** actor types with both critic types (8 combinations total):
1. `original` + `original`
2. `original` + `factored`
3. `autoregressive_gimbal_first` + `original`
4. `autoregressive_gimbal_first` + `factored`
5. `autoregressive_drone_first` + `original`
6. `autoregressive_drone_first` + `factored`
7. `multihead_attention` + `original`
8. `multihead_attention` + `factored`

## Implementation Details

### Actor Forward Pass

All actors implement the same interface:

```python
def forward(self, obs, std):
    """
    Args:
        obs: [B, repr_dim] - encoded observation
        std: scalar or tensor - action noise std
    Returns:
        dist: TruncatedNormal distribution over [B, 5] joint actions
    """
```

### Critic Forward Pass

All critics implement the same interface:

```python
def forward(self, obs, action):
    """
    Args:
        obs: [B, repr_dim] - encoded observation
        action: [B, 5] - full joint action [drone(3), gimbal(2)]
    Returns:
        Q1, Q2: [B, 1] - twin Q-values
    """
```

### Compatibility

- All actors work with all critics
- All encoder types work with all actor/critic combinations
- Checkpoints are compatible as long as architecture doesn't change

## Monitoring Training

The agent type is logged during initialization:

```
[DrQV2Agent] Creating actor with type: multihead_attention
[DrQV2Agent] Creating critic with type: factored
```

Check logs to confirm correct architecture is loaded.

## Troubleshooting

### Issue: "Unknown actor_type: X"
**Solution**: Check spelling in config file. Valid values:
- `original`
- `autoregressive_gimbal_first`
- `autoregressive_drone_first`
- `multihead_attention`

### Issue: "Unknown critic_type: X"
**Solution**: Check spelling in config file. Valid values:
- `original`
- `factored`

### Issue: Training is slower with autoregressive actors
**Expected**: Autoregressive actors have sequential sampling (gimbal→drone or drone→gimbal), which is slightly slower than joint sampling. Consider using `multihead_attention` for faster parallel action generation.

### Issue: Multi-head attention not converging
**Solutions**:
1. Reduce number of attention heads: `actor_cfg.nhead: 2`
2. Try factored critic: `critic_type: factored`
3. Adjust learning rate or increase batch size

## Next Steps

1. **Baseline**: Train with `original` architecture
2. **Best Performance**: Train with `multihead_attention`
3. **Ablation**: Compare all architectures
4. **Analysis**: Use Tensorboard to compare learning curves
5. **Fine-tuning**: Adjust `actor_cfg.nhead` for multi-head attention

## References

- Implementation: `drqv2-w-new_nets.py`
- Analysis: `ANALYSIS_ACTOR_CRITIC_AND_FIXES.md`
- Main README: `README.md`
