# Actor/Critic Architecture Guide

## Available Architectures

### Actors

1. **`Actor`** (Original)
   - Joint action generation for all 5 actions simultaneously

2. **`AutoregressiveActorGimbalFirst`**
   - Autoregressive: Gimbal → Drone
   - Gimbal action sampled first, then drone action conditioned on gimbal

3. **`AutoregressiveActorDroneFirst`**
   - Autoregressive: Drone → Gimbal
   - Drone action sampled first, then gimbal action conditioned on drone

4. **`MultiHeadAttentionActor`**
   - Parallel action generation with cross-attention and self-attention
   - Learnable query embeddings for drone and gimbal coordination

### Critics

1. **`Critic`** (Original)
   - Joint Q-value estimation

2. **`FactoredCritic`**
   - Factored Q-value: Q(s,a) = Q_drone(s,a_drone) + Q_joint(s,a_full)

## Implementation Location

**File**: `drqv2-w-new_nets.py`

- `Actor` class (original actor)
- `Critic` class (original critic)
- `AutoregressiveActorGimbalFirst` class
- `AutoregressiveActorDroneFirst` class
- `MultiHeadAttentionActor` class
- `FactoredCritic` class
- `DrQV2Agent` class (selection logic in `__init__` method)

## Configuration Location

**Main config**: `cfgs/config_gimbal_curriculum.yaml`

**Actor/Critic bundles**: `cfgs/actor_critic_bundle/`
- `original.yaml`
- `autoregressive_gimbal_first.yaml`
- `autoregressive_drone_first.yaml`
- `multihead_attention.yaml`
- `multihead_attention_factored.yaml`

## Configuration Options

### Actor Types

In `actor_critic_bundle/*.yaml` or `DrQV2Agent.__init__()`:

**`actor_type`** (string):
- `'original'` - Baseline joint action actor
- `'autoregressive_gimbal_first'` - Gimbal → Drone autoregressive
- `'autoregressive_drone_first'` - Drone → Gimbal autoregressive
- `'multihead_attention'` - Multi-head attention actor

**`actor_cfg`** (dict):
- For `MultiHeadAttentionActor`:
  - `nhead` (int): Number of attention heads (default: 4)
- For other actors: `null` (no additional config)

### Critic Types

In `actor_critic_bundle/*.yaml` or `DrQV2Agent.__init__()`:

**`critic_type`** (string):
- `'original'` - Baseline joint critic
- `'factored'` - Factored Q-value critic

No additional configuration options for critics.

## Usage

Command line:
```bash
python train_gimbal_curriculum.py actor_critic_bundle@agent=multihead_attention
```

Or modify `cfgs/config_gimbal_curriculum.yaml`:
```yaml
defaults:
  - actor_critic_bundle@agent: multihead_attention
```
