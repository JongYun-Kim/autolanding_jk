# Analysis: High Success Rates in Early Curriculum Stages

## Executive Summary

This document analyzes the phenomenon where early curriculum stages can produce models with unexpectedly high success rates (sometimes reaching 100%), even though curriculum advancement is triggered at a lower threshold (e.g., 85% in the `balanced.yaml` configuration).

## Background

The training script uses curriculum learning with 5 stages (S0 to S4):
- **Stage 0 (S0_lock)**: Drone control only, gimbal disabled, locked position
- **Stage 1 (S1_yaw_small)**: Small yaw range (15% scale)
- **Stage 2 (S2_yaw_pitch_small)**: Small yaw+pitch range (25%/35% scale)
- **Stage 3 (S3_mid)**: Medium range (60% scale)
- **Stage 4 (S4_full)**: Full range (100% scale) - Final deployment target

The curriculum advances to the next stage when the success rate reaches 85% over a rolling window of 100 episodes (with at least 80 episodes completed), maintained across 4 consecutive evaluation windows.

## Why Early Stages Achieve Unexpectedly High Success Rates

### 1. **Task Simplicity and Reduced Complexity**

Early curriculum stages are intentionally designed to be simpler:

- **Stage 0**: No gimbal control required, locked initial position with zero spatial variation (`scale: [0.0, 0.0, 0.0]`). The agent only needs to learn basic drone landing dynamics.
- **Stage 1**: Only yaw control with limited range (15%). Single degree of freedom for gimbal control.

The reduced task complexity means:
- Smaller effective state space to explore
- Simpler optimal policies
- Faster convergence to near-optimal behavior
- Less potential for failure modes

### 2. **Restricted Action Space**

Early stages have significantly constrained action spaces:

- Stage 0: 4D action space (drone control only, no gimbal)
- Stage 1: 5D action space (drone + yaw only) with limited range
- Stage 4: 6D action space (drone + pitch + yaw) with full range

**Impact**: Smaller action spaces are exponentially easier for RL agents to optimize. With fewer actions to choose from and smaller ranges, the agent can:
- More quickly discover successful action sequences
- Experience less exploration variance
- Achieve more consistent performance

### 3. **Curriculum Advancement Lag and Continued Training**

The curriculum uses multiple stability mechanisms that introduce advancement lag:

```yaml
min_episodes: 80           # Minimum episodes before checking
require_consecutive_windows: 4  # Must pass 4 consecutive checks
cooldown_episodes: (if configured)  # Cooldown period after stage change
```

**Consequence**: After the agent first reaches 85% success rate, it continues training for:
- At least 3 more evaluation windows (300+ episodes) to satisfy consecutive requirements
- Additional episodes during cooldown periods
- Episodes accumulated during the stage transition process

During this extended training period, the agent can:
- Continue optimizing and exceed the 85% threshold significantly
- Reach performance plateaus at 90-100% success rates
- Overfit to the simplified stage conditions

### 4. **Simplified Reward Structure**

Early stages have progressively simpler reward structures:

| Stage | Viz Reward | Viz Weight | Not Visible Penalty | Smooth Penalty |
|-------|-----------|------------|---------------------|----------------|
| S0    | No        | 0.0        | 0.0                 | 0.0            |
| S1    | No        | 0.0        | -0.010              | 0.0001         |
| S2    | Yes       | 0.05       | -0.015              | 0.0003         |
| S4    | Yes       | 0.10       | -0.025              | 0.0010         |

**Analysis**:
- Stage 0 has no visibility or smoothness penalties - only basic landing reward
- Stage 1 has minimal penalties compared to later stages
- Fewer penalty terms mean fewer ways to fail
- Simpler reward landscapes are easier to optimize

### 5. **Overfitting to Stage-Specific Conditions**

Early stage success rates may not translate to later stages because:

- **Position initialization**: Stage 0 uses locked positions; learned policies may not generalize to varied initial positions
- **Sensor constraints**: Without needing to manage gimbal effectively, the agent might learn vision-based policies that fail when gimbal control becomes critical
- **Reduced penalties**: Agents trained without smoothness penalties may learn jerky control policies that work in early stages but fail in later stages with higher penalty weights

**Example**: An agent achieving 100% in Stage 1 (yaw-only) might fail dramatically in Stage 2 when pitch control is introduced, as its policy never learned to handle the additional degree of freedom.

### 6. **Statistical Factors**

Several statistical factors can contribute to high success rates:

- **Lucky evaluation windows**: With 100-episode windows, random variation can produce 95-100% success rates even if the true policy performance is 90%
- **Episode correlation**: If the replay buffer or training dynamics introduce temporal correlation, consecutive episodes might have similar outcomes
- **Convergence to local optima**: Simple stages may have obvious local optima that are easy to find and achieve near-perfect performance

### 7. **Diminishing Returns in Simple Environments**

The learning curve in simple environments often shows:
- Rapid initial improvement (0% → 85%): 10,000-50,000 steps
- Slow refinement (85% → 95%): 50,000-150,000 steps
- Asymptotic convergence (95% → 100%): 150,000+ steps

Because the agent continues training well beyond the 85% threshold due to advancement lag, it has time to reach the asymptotic region where 95-100% performance is achievable.

## Implications for Model Selection

### Previous Global Top-8 Approach

**Problem**: With global ranking, early-stage models with 95-100% success rates would occupy 6-7 of the 8 saved slots, leaving only 1-2 slots for final-stage models.

**Why this is suboptimal**:
- Early-stage models have limited deployment value (trained on simplified tasks)
- Final-stage models are more valuable (trained on full-complexity task)
- High early-stage SR doesn't indicate better model quality for deployment

### New Per-Stage Approach

**Solution**: Save top 3 models per non-final stage, top 8 for final stage

**Benefits**:
- Preserves some high-performing early-stage models (useful for analysis/debugging)
- Ensures at least 8 final-stage models are saved (most valuable for deployment)
- Prevents early-stage models from crowding out late-stage models
- Maintains historical performance record across curriculum progression

## Recommendations for Future Work

While the current modification addresses the model saving issue, future improvements could consider:

1. **Adaptive stage advancement thresholds**: Increase the success rate threshold for earlier stages (e.g., 90% for S0-S1, 85% for S2-S4)

2. **Stage-specific advancement criteria**: Use different `require_consecutive_windows` values per stage (e.g., 2 windows for early stages, 4 for later stages)

3. **Performance monitoring**: Track and log whether high early-stage success rates correlate with better final-stage performance

4. **Curriculum debugging mode**: Add option to cap training time per stage to prevent excessive overfitting

5. **Transfer learning analysis**: Investigate whether models with 100% success in stage N perform better or worse in stage N+1 compared to models with 85-90% success

## Conclusion

High success rates in early curriculum stages are expected and result from:
- Intentionally simplified task complexity
- Reduced action/state spaces
- Continued training during curriculum advancement lag
- Simpler reward structures with fewer penalties
- Natural convergence behavior in simple environments

The per-stage model saving approach appropriately handles this phenomenon by:
- Limiting early-stage model storage (top 3) while preserving some for analysis
- Prioritizing final-stage model storage (top 8) for deployment value
- Preventing early-stage performance from masking final-stage model quality

This approach ensures that the most valuable models (final-stage) are adequately preserved without completely discarding early-stage high-performers that may still be useful for curriculum analysis and debugging.
