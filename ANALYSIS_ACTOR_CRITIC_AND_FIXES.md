# Analysis: Actor/Critic Architecture & Division by Zero Fix

## Task 2: Actor/Critic Architecture Analysis

### Current Architecture

**Location**: `drqv2-w-new_nets.py:432-491`

#### Actor (lines 432-458)
```python
class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.Tanh()
        )
        self.policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_shape[0])  # Output: 5D joint action
        )

    def forward(self, obs, std):
        h = self.trunk(obs)
        mu = self.policy(h)  # [B, 5] - joint action mean
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std
        dist = utils.TruncatedNormal(mu, std)
        return dist
```

**Issues**:
1. **Joint Action Output**: Produces all 5 actions `[vx, vy, vz, gimbal_pitch, gimbal_yaw]` simultaneously
2. **No Coupling**: Drone actions and gimbal actions are independent in the network
3. **Suboptimal for Sequential Tasks**: Doesn't capture the natural dependency where:
   - Gimbal should point at target based on drone's intended trajectory
   - OR drone should move knowing where gimbal is pointing

#### Critic (lines 461-491)
```python
class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.Tanh()
        )
        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),  # feature + 5 actions
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        self.Q2 = ...  # Same as Q1
```

**Issues**:
1. **Joint Q-value**: Evaluates the full joint action `[drone, gimbal]` together
2. **Cannot Exploit Structure**: Doesn't leverage the fact that drone and gimbal have different roles

---

## Proposed Improvements

### Option 1: Autoregressive Actor (Gimbal → Drone)

**Motivation**: "Point-then-move" strategy
- First decide where to point the camera
- Then decide drone movement given camera orientation

**Architecture**:
```python
class AutoregressiveActorGimbalFirst(nn.Module):
    def __init__(self, repr_dim, feature_dim, hidden_dim):
        super().__init__()
        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.Tanh()
        )

        # Gimbal policy (unconditional)
        self.gimbal_policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2)  # [pitch, yaw]
        )

        # Drone policy (conditioned on gimbal)
        self.drone_policy = nn.Sequential(
            nn.Linear(feature_dim + 2, hidden_dim),  # +2 for gimbal actions
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3)  # [vx, vy, vz]
        )

    def forward(self, obs, std):
        h = self.trunk(obs)  # [B, feature_dim]

        # Step 1: Sample gimbal action
        gimbal_mu = self.gimbal_policy(h)  # [B, 2]
        gimbal_mu = torch.tanh(gimbal_mu)
        gimbal_std = torch.ones_like(gimbal_mu) * std
        gimbal_dist = utils.TruncatedNormal(gimbal_mu, gimbal_std)
        gimbal_action = gimbal_dist.sample()  # [B, 2]

        # Step 2: Sample drone action conditioned on gimbal
        h_drone = torch.cat([h, gimbal_action], dim=-1)  # [B, feature_dim + 2]
        drone_mu = self.drone_policy(h_drone)  # [B, 3]
        drone_mu = torch.tanh(drone_mu)
        drone_std = torch.ones_like(drone_mu) * std
        drone_dist = utils.TruncatedNormal(drone_mu, drone_std)

        # Return joint distribution (for compatibility)
        action = torch.cat([drone_action, gimbal_action], dim=-1)  # [B, 5]
        return action, gimbal_dist, drone_dist
```

**Pros**:
- Natural for "stabilize camera first" approach
- Camera can focus on target regardless of drone motion
- Drone can then adjust trajectory knowing camera state

**Cons**:
- May be slower to converge (sequential sampling)
- Gimbal policy gets no direct feedback from drone actions

---

### Option 2: Autoregressive Actor (Drone → Gimbal)

**Motivation**: "Move-then-point" strategy
- First decide where the drone should go
- Then adjust gimbal to keep target in view

**Architecture**:
```python
class AutoregressiveActorDroneFirst(nn.Module):
    def __init__(self, repr_dim, feature_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.Tanh()
        )

        # Drone policy (unconditional)
        self.drone_policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3)  # [vx, vy, vz]
        )

        # Gimbal policy (conditioned on drone)
        self.gimbal_policy = nn.Sequential(
            nn.Linear(feature_dim + 3, hidden_dim),  # +3 for drone actions
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2)  # [pitch, yaw]
        )

    def forward(self, obs, std):
        h = self.trunk(obs)

        # Step 1: Sample drone action
        drone_mu = self.drone_policy(h)
        drone_mu = torch.tanh(drone_mu)
        drone_std = torch.ones_like(drone_mu) * std
        drone_dist = utils.TruncatedNormal(drone_mu, drone_std)
        drone_action = drone_dist.sample()

        # Step 2: Sample gimbal action conditioned on drone
        h_gimbal = torch.cat([h, drone_action], dim=-1)
        gimbal_mu = self.gimbal_policy(h_gimbal)
        gimbal_mu = torch.tanh(gimbal_mu)
        gimbal_std = torch.ones_like(gimbal_mu) * std
        gimbal_dist = utils.TruncatedNormal(gimbal_mu, gimbal_std)

        action = torch.cat([drone_action, gimbal_action], dim=-1)
        return action, drone_dist, gimbal_dist
```

**Pros**:
- Natural for "navigate first" approach
- Gimbal compensates for drone movements
- Good for aggressive flight maneuvers

**Cons**:
- Camera may lag behind optimal pointing
- Drone doesn't know where camera will point

---

### Option 3: Multi-head Attention Actor

**Motivation**: Parallel but contextually-aware actions
- Both drone and gimbal policies see each other's intent
- Cross-attention allows coordination without sequential dependency

**Architecture**:
```python
class MultiHeadAttentionActor(nn.Module):
    def __init__(self, repr_dim, feature_dim, hidden_dim, nhead=4):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.Tanh()
        )

        # Create drone and gimbal query embeddings
        self.drone_query = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.gimbal_query = nn.Parameter(torch.randn(1, 1, feature_dim))

        # Cross-attention between drone/gimbal intents
        self.cross_attn = nn.MultiheadAttention(
            feature_dim, nhead, batch_first=True
        )

        # Separate policy heads
        self.drone_policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3)
        )

        self.gimbal_policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, obs, std):
        B = obs.shape[0]
        h = self.trunk(obs)  # [B, feature_dim]

        # Expand queries for batch
        drone_q = self.drone_query.expand(B, -1, -1)   # [B, 1, feature_dim]
        gimbal_q = self.gimbal_query.expand(B, -1, -1) # [B, 1, feature_dim]

        # Stack queries
        queries = torch.cat([drone_q, gimbal_q], dim=1)  # [B, 2, feature_dim]

        # Use observation as key/value
        h_kv = h.unsqueeze(1)  # [B, 1, feature_dim]

        # Cross-attention
        attn_out, _ = self.cross_attn(
            queries, h_kv, h_kv
        )  # [B, 2, feature_dim]

        # Extract drone and gimbal features
        h_drone = attn_out[:, 0, :]   # [B, feature_dim]
        h_gimbal = attn_out[:, 1, :]  # [B, feature_dim]

        # Self-attention between drone and gimbal
        queries_self = torch.stack([h_drone, h_gimbal], dim=1)  # [B, 2, feature_dim]
        h_coordinated, _ = self.cross_attn(
            queries_self, queries_self, queries_self
        )  # [B, 2, feature_dim]

        h_drone_final = h_coordinated[:, 0, :]
        h_gimbal_final = h_coordinated[:, 1, :]

        # Generate actions
        drone_mu = torch.tanh(self.drone_policy(h_drone_final))
        gimbal_mu = torch.tanh(self.gimbal_policy(h_gimbal_final))

        # Create distributions
        drone_std = torch.ones_like(drone_mu) * std
        gimbal_std = torch.ones_like(gimbal_mu) * std

        drone_dist = utils.TruncatedNormal(drone_mu, drone_std)
        gimbal_dist = utils.TruncatedNormal(gimbal_mu, gimbal_std)

        action = torch.cat([drone_mu, gimbal_mu], dim=-1)
        return action, drone_dist, gimbal_dist
```

**Pros**:
- Parallel action generation (faster sampling)
- Bidirectional information flow
- Most flexible for learning coordination
- Can learn both "point-then-move" and "move-then-point"

**Cons**:
- More complex architecture
- More parameters to train
- May require more data to converge

---

## Recommended Approach

**Recommendation: Option 3 (Multi-head Attention Actor)**

**Reasoning**:
1. **Flexibility**: Can learn optimal coordination strategy from data
2. **Efficiency**: Parallel action generation (no sequential sampling overhead)
3. **Expressiveness**: Cross-attention naturally models dependencies
4. **Curriculum-friendly**: Can adapt strategy as gimbal control complexity increases
5. **Modern**: Aligns with current deep RL best practices (e.g., Decision Transformer)

**Implementation Priority**:
1. Start with Option 3 (best performance potential)
2. Fall back to Option 1 if training instability occurs
3. Option 2 is least recommended (gimbal should typically lead, not follow)

---

## Critic Modifications

For autoregressive actors, consider **factored critic**:

```python
class FactoredCritic(nn.Module):
    """
    Q(s, a_drone, a_gimbal) = Q_drone(s, a_drone) + Q_gimbal(s, a_drone, a_gimbal)
    """
    def __init__(self, repr_dim, feature_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.Tanh()
        )

        # Q for drone action
        self.Q_drone = nn.Sequential(
            nn.Linear(feature_dim + 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

        # Q for joint action (additive correction)
        self.Q_joint = nn.Sequential(
            nn.Linear(feature_dim + 5, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        h = self.trunk(obs)
        drone_action = action[:, :3]

        q_drone = self.Q_drone(torch.cat([h, drone_action], dim=-1))
        q_joint = self.Q_joint(torch.cat([h, action], dim=-1))

        return q_drone + q_joint
```

This allows the critic to separately evaluate:
- Base value of drone actions
- Additional value from gimbal coordination

---

## Task 3: Division by Zero Issues in LandingAviary.py

### Problem Location

**File**: `gym_pybullet_drones/envs/single_agent_rl/LandingAviary.py`

#### Issue 1: `_rad_to_norm` method (line 395)

```python
def _rad_to_norm(self, angles_rad):
    """angles_rad shape (3,): [pitch(rad), roll(rad), yaw(rad)] -> [-1,1]^3"""
    lo = self.gimbal_angle_ranges[:, 0]
    hi = self.gimbal_angle_ranges[:, 1]
    return 2.0 * (np.asarray(angles_rad) - lo) / (hi - lo) - 1.0  # DIVISION BY ZERO!
```

**When does this occur?**
- When curriculum stage has `scale = (0.0, 0.0, 0.0)` or any axis with scale=0
- In `_apply_stage` (line 861-869), if `scale[i] = 0.0`:
  ```python
  new_lo = -scale * base_half  # = 0.0
  new_hi = +scale * base_half  # = 0.0
  # => hi - lo = 0.0 => DIVISION BY ZERO
  ```

**Where is this called?**
1. `compute_oracle_gimbal()` line 585
2. Any reward computation using normalized angles

#### Issue 2: `_norm_to_rad` method (line 401)

```python
def _norm_to_rad(self, angles_norm):
    """[-1,1]^3 -> radians (3,) in order [pitch, roll, yaw]"""
    lo = self.gimbal_angle_ranges[:, 0]
    hi = self.gimbal_angle_ranges[:, 1]
    return lo + 0.5 * (np.asarray(angles_norm) + 1.0) * (hi - lo)
```

**This doesn't cause division by zero**, but when `hi = lo = 0`:
- Output is always `0.0` regardless of input
- This is actually correct behavior for locked axes

### Root Cause Analysis

The issue occurs in **curriculum Stage 0** (and any stage with disabled axes):
```python
CurriculumStageSpec(
    name="S0_lock",
    gimbal_enabled=False,
    lock_down=True,
    scale=(0.0, 0.0, 0.0),  # ALL AXES HAVE ZERO RANGE
    ...
)
```

When `scale[i] = 0.0`, the range becomes `[0, 0]`, causing:
- `_rad_to_norm`: Division by zero
- Oracle gimbal computation attempts to normalize angles into a zero-width range

### Proposed Fix

**Option A: Guard in normalization functions (Safest)**

```python
def _rad_to_norm(self, angles_rad):
    """angles_rad shape (3,): [pitch(rad), roll(rad), yaw(rad)] -> [-1,1]^3"""
    lo = self.gimbal_angle_ranges[:, 0]
    hi = self.gimbal_angle_ranges[:, 1]

    # Guard against division by zero
    range_width = hi - lo
    safe_range = np.where(
        np.abs(range_width) < 1e-9,
        np.ones_like(range_width),  # Use 1.0 to avoid div/0
        range_width
    )

    normalized = 2.0 * (np.asarray(angles_rad) - lo) / safe_range - 1.0

    # For zero-width ranges, always return 0.0 (centered)
    normalized = np.where(
        np.abs(range_width) < 1e-9,
        np.zeros_like(normalized),
        normalized
    )

    return normalized
```

**Option B: Prevent zero-width ranges in `_apply_stage`**

```python
def _apply_stage(self, spec: CurriculumStageSpec):
    """Stage application with minimum range enforcement"""
    base = self._base_gimbal_angle_ranges
    base_half = np.minimum(np.abs(base[:, 0]), np.abs(base[:, 1]))
    scale = np.array(spec.scale, dtype=np.float32)

    # Ensure minimum range to avoid singularity
    MIN_RANGE = 1e-6  # Small but non-zero
    scale = np.maximum(scale, MIN_RANGE)

    new_lo = -scale * base_half
    new_hi = +scale * base_half
    self.gimbal_angle_ranges = np.stack([new_lo, new_hi], axis=1).astype(np.float32)

    if spec.lock_down or not spec.gimbal_enabled:
        self.gimbal_target = self.initial_gimbal_target.copy()
    self._prev_gimbal_target = self.gimbal_target.copy()
```

**Option C: Skip normalization when axis is locked (Most Principled)**

```python
def _rad_to_norm(self, angles_rad):
    """angles_rad shape (3,): [pitch(rad), roll(rad), yaw(rad)] -> [-1,1]^3"""
    lo = self.gimbal_angle_ranges[:, 0]
    hi = self.gimbal_angle_ranges[:, 1]
    range_width = hi - lo

    # Check which axes are locked (zero range)
    locked_axes = np.abs(range_width) < 1e-9

    # For unlocked axes, normalize normally
    result = np.zeros_like(angles_rad, dtype=np.float32)
    result[~locked_axes] = (
        2.0 * (np.asarray(angles_rad)[~locked_axes] - lo[~locked_axes])
        / range_width[~locked_axes] - 1.0
    )
    # For locked axes, return 0.0 (centered, though value doesn't matter)
    result[locked_axes] = 0.0

    return result
```

### Recommended Fix: **Option C**

**Reasoning**:
1. **Most principled**: Explicitly handles locked vs. unlocked axes
2. **Preserves semantics**: Locked axes return meaningful default (0.0)
3. **No magic numbers**: No arbitrary MIN_RANGE or epsilon thresholds
4. **Clear intent**: Code documents that zero-range axes are intentionally locked
5. **Robust**: Works for any combination of locked/unlocked axes

### Additional Safeguards

Also add assertion in `compute_oracle_gimbal`:

```python
def compute_oracle_gimbal(self):
    """..."""
    # ... existing code ...

    # Before normalization, clip to ACTUAL ranges (not zero-width)
    lo = self.gimbal_angle_ranges[:, 0]
    hi = self.gimbal_angle_ranges[:, 1]

    # Only clip axes with non-zero range
    range_width = hi - lo
    unlocked = np.abs(range_width) >= 1e-9

    angles_rad_clipped = np.array(angles_rad, copy=True)
    angles_rad_clipped[unlocked] = np.clip(
        angles_rad[unlocked],
        lo[unlocked],
        hi[unlocked]
    )

    # For locked axes, force to center (0.0)
    angles_rad_clipped[~unlocked] = 0.0

    angles_norm = self._rad_to_norm(angles_rad_clipped)

    # ...
```

---

## Summary

### Task 2: Actor/Critic Improvements
- **Current**: Joint action actor/critic, no coupling between drone and gimbal
- **Best Option**: Multi-head attention actor (Option 3)
  - Parallel action generation
  - Bidirectional coordination via cross-attention
  - Most flexible and powerful
- **Alternative**: Autoregressive gimbal-first (Option 1) if stability issues

### Task 3: Division by Zero Fix
- **Problem**: Curriculum stages with zero-range axes cause division by zero in `_rad_to_norm`
- **Location**: `LandingAviary.py:395-399`
- **Root Cause**: Stage 0 uses `scale=(0.0, 0.0, 0.0)` → `hi - lo = 0`
- **Best Fix**: Option C - explicitly handle locked axes in normalization
- **Additional**: Add safeguards in oracle gimbal computation

Both issues are **critical** and should be addressed before training with curriculum learning.
