# Mavic 3 Drone Model Support

This document describes the DJI Mavic 3 drone model added to the gym-pybullet-drones simulation alongside the existing CF2X (Crazyflie 2.0 X-config).

## Parameter Derivations

### Physical Dimensions

| Parameter | Value | Source |
|-----------|-------|--------|
| Mass | 0.895 kg | DJI Mavic 3 spec sheet |
| Motor rectangle | 347.5 x 283 mm | Mavic 3 dimensions |
| Arm length | 0.225 m | Half-diagonal: sqrt(0.34752 + 0.2832)/2 |
| Prop diameter | 238.8 mm | Mavic 3 propeller spec |
| Prop radius | 0.1194 m | Diameter / 2 |

### Thrust and Torque Coefficients

| Parameter | Value | Derivation |
|-----------|-------|------------|
| kf | 2.25e-7 | CF2X kf (3.16e-10) scaled by (prop_diam ratio)^4 = (238.8/46.3)^4 = 711x |
| km | 5.65e-9 | Same km/kf ratio as CF2X: 7.94e-12/3.16e-10 = 0.0251; 0.0251 * 2.25e-7 = 5.65e-9 |
| thrust2weight | 2.0 | Specified for comparable flight envelope to CF2X (2.25) |
| max_speed_kmh | 30 | Matched to CF2X so SPEED_LIMIT is identical |

### Inertia (Box Approximation)

Using a uniform box approximation at 50% of full-box inertia (accounting for mass distribution):

| Axis | Formula | Value |
|------|---------|-------|
| Ixx | 0.5 * (1/12) * M * (w^2 + h^2) = 0.5 * (1/12) * 0.895 * (0.283^2 + 0.1077^2) | 0.0034 kg*m^2 |
| Iyy | 0.5 * (1/12) * M * (l^2 + h^2) = 0.5 * (1/12) * 0.895 * (0.3475^2 + 0.1077^2) | 0.0049 kg*m^2 |
| Izz | 0.5 * (1/12) * M * (l^2 + w^2) = 0.5 * (1/12) * 0.895 * (0.3475^2 + 0.283^2) | 0.0075 kg*m^2 |

(Dimensions: 347.5 x 283 x 107.7 mm)

### Drag Coefficients

Matched normalized drag deceleration to CF2X (drag per unit mass should produce similar velocity decay):

| Parameter | CF2X | Mavic3 | Rationale |
|-----------|------|--------|-----------|
| drag_coeff_xy | 9.1785e-7 | 1.41e-4 | Scaled by mass ratio and prop-speed ratio to match deceleration |
| drag_coeff_z | 10.311e-7 | 1.58e-4 | Same approach for vertical axis |

### Other Coefficients

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| gnd_eff_coeff | 11.36859 | Same as CF2X; model parameterized by kf and prop_radius |
| dw_coeff_1/2/3 | 2267.18/0.16/-0.11 | Same as CF2X; single-agent scenarios, no downwash effect |

### Collision Geometry

| Parameter | Value | Derivation |
|-----------|-------|------------|
| Cylinder radius | 0.25 m | Slightly larger than arm length for safety margin |
| Cylinder length | 0.11 m | Approximate body height |
| Z offset | -0.04 m | Center collision volume on body |

### Visual Representation

The visual mesh uses a scaled cf2.dae (scale 5.67x) as a placeholder. The arm ratio is 0.225/0.0397 = 5.67. Physics correctness depends on the collision cylinder, not the visual mesh.

## Controllers

Two PID controllers are available for Mavic3, selectable via the `controller_type` parameter:

| Controller | Class | Pipeline | Selection |
|------------|-------|----------|-----------|
| SimplePID (default) | `Mavic3PIDControl` | Force-based PID + nnlsRPM mixing | `controller_type=None` |
| DSL | `Mavic3DSLPIDControl` | PWM-based thrust + mixer matrix | `controller_type="dsl"` |

### Controller Selection

```python
LandingAviary(drone_model=DroneModel.MAVIC3, controller_type=None)   # SimplePID (default)
LandingAviary(drone_model=DroneModel.MAVIC3, controller_type="dsl")  # DSL-style
```

For CF2X/CF2P/HB, the `controller_type` parameter is ignored.

---

## Mavic3PIDControl (SimplePID-based, tuned)

Based on SimplePIDControl with gains tuned for Mavic3 responsiveness and ground effect compensation. Uses force-based PID with nnlsRPM motor mixing and X-config allocation matrix.

### Position Gains (5x HB baseline)

| Gain | Original (1.79x HB) | Tuned (5x HB) |
|------|---------------------|----------------|
| P_pos | [.18, .18, .36] | [.90, .90, 1.80] |
| I_pos | [.0002, .0002, .0002] | [.001, .001, .01] |
| D_pos | [.9, .9, 1.43] | [4.5, 4.5, 7.15] |

The Z integrator gain (I_pos[2]) is 10x higher than XY to compensate for ground effect. Z integrator clamp widened from [-0.15, 0.15] to [-0.5, 0.5].

### Attitude Gains (2x HB baseline)

| Gain | Original | Tuned |
|------|----------|-------|
| P_att | [.52, .60, .009] | [1.04, 1.20, .018] |
| D_att | [.045, .05, 0] | [.090, .10, 0] |

### Torque Clip

Per-axis clipping at 95% of physical maximum:
- XY: `0.95 * MAX_XY_TORQUE` (~1.33 Nm)
- Z: `0.95 * MAX_Z_TORQUE` (~0.21 Nm)

### X-Config Specifics

Unlike SimplePIDControl (plus-config for HB), Mavic3PIDControl uses:
- X-config allocation matrix A with 1/sqrt(2) terms
- X-config MAX_XY_TORQUE formula: (2*L*KF*MAX_RPM^2)/sqrt(2)

---

## Mavic3DSLPIDControl (DSL-style)

PWM-based thrust pipeline adapted from DSLPIDControl (CF2X). Uses direct PWM-to-RPM conversion and mixer matrix instead of nnlsRPM.

### PWM2RPM Constants

Derived from Mavic3 RPM range mapped to [20000, 65535] PWM range:

| Constant | Value | Derivation |
|----------|-------|------------|
| PWM2RPM_SCALE | 0.05242 | (MAX_RPM - MIN_RPM) / (MAX_PWM - MIN_PWM) |
| PWM2RPM_CONST | 982.6 | MIN_RPM - SCALE * MIN_PWM |
| MIN_PWM | 20000 | Standard DSL minimum |
| MAX_PWM | 65535 | Standard DSL maximum |

Verification: hover PWM = (3125 - 982.6) / 0.05242 = 40,871 (in valid range).

### Position Gains

Scaled from CF2X by mass ratio (33.15x) then reduced for stability:
- XY: /6.6 from mass-ratio scaling (effective 5x)
- Z: /1.5 from mass-ratio scaling (effective 22x, stronger for ground effect)

| Gain | CF2X | Mavic3 DSL |
|------|------|------------|
| P_pos | [.4, .4, 1.25] | [2.0, 2.0, 27.63] |
| I_pos | [.05, .05, .05] | [.25, .25, 1.10] |
| D_pos | [.2, .2, .5] | [1.0, 1.0, 11.05] |

### Attitude Gains

Scaled by inertia_ratio / authority_ratio, then damped 30% for stability:

| Gain | CF2X | Mavic3 DSL |
|------|------|------------|
| P_att | [70000, 70000, 60000] | [70000, 100000, 480000] |
| I_att | [0, 0, 500] | [0, 0, 4000] |
| D_att | [20000, 20000, 12000] | [20000, 29000, 96000] |

Torque clip: 37000 (11.5x CF2X yaw-authority ratio).

### Mixer Matrix

X-config (same as CF2X):
```
[[.5, -.5, -1], [.5, .5, 1], [-.5, .5, -1], [-.5, -.5, 1]]
```

---

## Expected Values

From the URDF parameters:
- HOVER_RPM = sqrt(M*g / (4*kf)) = sqrt(0.895*9.8 / (4*2.25e-7)) = sqrt(9765556) ~ 3125
- MAX_RPM = sqrt(thrust2weight * M*g / (4*kf)) = sqrt(2.0 * 0.895*9.8 / (4*2.25e-7)) ~ 4418

## Controller Comparison Results

Test: 3 controllers x 8 maneuvers, 50 steps each, ActionType.VEL.

### Responsiveness (peak speed ratio vs CF2X-DSL)

| Controller | Mean Ratio | Status |
|------------|-----------|--------|
| Mavic3-SimplePID | 0.94 | PASS (>= 0.80) |
| Mavic3-DSL | 2.01 | PASS (>= 0.80) |

### Hover Drift

| Controller | Drift (m) | Status |
|------------|-----------|--------|
| CF2X-DSL | 0.0001 | Baseline |
| Mavic3-SimplePID | 0.011 | Improved from ~0.05-0.10 |
| Mavic3-DSL | 0.005 | PASS (< 0.01) |

### T80 Response Time (steps to 80% peak speed)

| Controller | Mean T80 |
|------------|----------|
| CF2X-DSL | 5.3 |
| Mavic3-SimplePID | 10.0 |
| Mavic3-DSL | 17.6 |

### Key Findings

- **Mavic3-DSL** has the best hover stability (0.005m drift) and highest responsiveness (2.0x CF2X)
- **Mavic3-SimplePID** has balanced performance with 0.94x CF2X responsiveness
- Ground effect compensation improved dramatically with higher Z integrator gains
- DSL XY is over-responsive due to lower aerodynamic drag at Mavic3 scale; gains intentionally reduced to prevent instability
- No torque-out-of-range warnings with per-axis clipping

## Training Pipeline Integration

Drone model and controller type are configurable via Hydra config. Both `dmc.make()` (non-gimbal) and `dmc.make_with_gimbal()` (gimbal) pass environment kwargs built from the config.

### Usage

```bash
# Default (CF2X, SimplePID)
python train.py

# Mavic3 with SimplePID (default controller)
python train.py drone_model=mavic3

# Mavic3 with DSL controller
python train.py drone_model=mavic3 controller_type=dsl

# Mavic3 in gimbal mode with curriculum
python train.py --config-name=config_gimbal_curriculum drone_model=mavic3

# Mavic3 in gimbal mode without curriculum
python train.py --config-name=config_gimbal drone_model=mavic3 controller_type=dsl
```

### Config Fields

All three base configs (`config.yaml`, `config_gimbal.yaml`, `config_gimbal_curriculum.yaml`) include:

```yaml
drone_model: null      # "cf2x", "mavic3", "hb", "cf2p" (null = constructor default)
controller_type: null  # null (default) or "dsl" (Mavic3 only)
```

### Flow

1. `train.py:Workspace.setup()` calls `_build_env_kwargs_from_cfg()` to extract `drone_model` and `controller_type` from config
2. `drone_model` string is converted to `DroneModel` enum (e.g. `DroneModel("mavic3")`)
3. `env_kwargs` dict is passed to `dmc.make()` or `dmc.make_with_gimbal()`
4. `dmc.make()` passes kwargs to `gym.make(name, **env_config)`
5. Aviary constructor receives `drone_model` and `controller_type`, dispatches to the appropriate PID controller

---

## Tuning Guidance

If the Mavic3 behaves incorrectly during training:

1. **Drone oscillates or is unstable**: Reduce P_att gains (P_COEFF_TOR). Start by reducing all by 20%.
2. **Drone is sluggish to respond**: Increase P_pos gains (P_COEFF_FOR) by 10-20%.
3. **Drone overshoots target positions**: Increase D_pos gains (D_COEFF_FOR) or reduce P_pos.
4. **Drone drifts upward (ground effect)**: Increase I_pos[2] and widen Z integrator clamp.
5. **Yaw instability**: Adjust P_att[2] and D_att[2] (currently minimal as yaw is locked to 0).
6. **Torque saturation (SimplePID)**: Per-axis clip at 95% MAX_XY/Z_TORQUE prevents nnlsRPM torque warnings.
7. **Switch controller**: Use `controller_type="dsl"` for better hover stability, `None` for simpler/predictable behavior.

## Files Modified/Created

### Created
- `gym_pybullet_drones/assets/mavic3.urdf` - Physical parameters and geometry
- `gym_pybullet_drones/control/Mavic3PIDControl.py` - Tuned SimplePID controller (5x gains, ground effect compensation)
- `gym_pybullet_drones/control/Mavic3DSLPIDControl.py` - DSL-style PWM controller (mass-ratio scaled)
- `test_mavic3.py` - 3-controller comparison test script
- `MAVIC3_SUPPORT.md` - This documentation

### Modified
- `gym_pybullet_drones/envs/BaseAviary.py` - Added MAVIC3 to DroneModel enum and X-config branches
- `gym_pybullet_drones/envs/single_agent_rl/BaseSingleAgentAviary.py` - Added `controller_type` param, DSL controller dispatch
- `gym_pybullet_drones/envs/single_agent_rl/LandingAviary.py` - Added `controller_type` pass-through in all three aviary classes
- `dmc.py` - Added `env_config` parameter to `make()` for passing kwargs to `gym.make()`
- `train.py` - Build env_kwargs for both gimbal and non-gimbal paths; extract `drone_model`/`controller_type` from config
- `cfgs/config.yaml` - Added `drone_model`, `controller_type` fields
- `cfgs/config_gimbal.yaml` - Added `drone_model`, `controller_type` fields
- `cfgs/config_gimbal_curriculum.yaml` - Added `drone_model`, `controller_type` fields
