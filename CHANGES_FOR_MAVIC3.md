# MAVIC3 Drone Model Support

This document describes the changes made to add DJI Mavic 3 support to the gym-pybullet-drones simulation framework.

## Overview

A new drone model option `DroneModel.MAVIC3` has been added, allowing the environment to be instantiated with a proof-of-concept Mavic 3 drone. The implementation preserves backward compatibility with existing drone models (CF2X, CF2P, HB) and the existing physics/control pipeline.

## How to Select DroneModel.MAVIC3

```python
from gym_pybullet_drones.envs.BaseAviary import DroneModel
from gym_pybullet_drones.envs.single_agent_rl.LandingAviary import LandingAviary

env = LandingAviary(
    drone_model=DroneModel.MAVIC3,
    # ... other parameters
)
```

## Files Changed/Added

### 1. `gym_pybullet_drones/envs/BaseAviary.py`

**Changes:**
- Added `MAVIC3 = "mavic3"` to the `DroneModel` enum
- Updated `MAX_XY_TORQUE` calculation to include MAVIC3 in the X-configuration branch (same formula as CF2X)
- Updated `DYNAMICS_ATTR` mixer matrix `A` to include MAVIC3 in the X-configuration branch
- Updated `_dynamics()` method torque calculations to include MAVIC3 in the X-configuration branch

**Rationale:** MAVIC3 uses an X-configuration rotor layout, similar to CF2X, so it shares the same torque/dynamics equations.

### 2. `gym_pybullet_drones/assets/mavic3.urdf` (NEW)

**Created:** New URDF file for the Mavic 3 drone model.

**Key parameters:**
| Parameter | Value | Notes |
|-----------|-------|-------|
| Mass | 0.895 kg | Mavic 3 unfolded mass |
| Arm length (L) | 0.19005 m | Half of motor-to-motor diagonal (0.3801 m) |
| Propeller radius | 0.1194 m | Mavic 3 propeller radius |
| max_speed_kmh | 30 | **Identical to CF2X** (as required) |
| thrust2weight | 2.25 | Same as CF2X |
| Ixx | 6.84e-3 kg·m² | Box approximation |
| Iyy | 9.87e-3 kg·m² | Box approximation |
| Izz | 1.497e-2 kg·m² | Box approximation |
| kf | 1.048e-8 | Scaled from CF2X by mass ratio |
| km | 2.633e-10 | Scaled from CF2X by mass ratio |

### 3. `gym_pybullet_drones/control/DSLPIDControl.py`

**Changes:**
- Updated model validation to accept `DroneModel.MAVIC3`
- Added MAVIC3 to X-configuration mixer matrix branch (same as CF2X)

**Rationale:** MAVIC3 uses the same X-configuration mixer matrix as CF2X and shares the same PID gains as a starting point.

### 4. `gym_pybullet_drones/envs/single_agent_rl/BaseSingleAgentAviary.py`

**Changes:**
- Added controller selection for `DroneModel.MAVIC3` using `DSLPIDControl`
- Added TUN (tuning) action type support with same initial gains as CF2X

### 5. `test_mavic3.py` (NEW)

**Created:** Smoke test script to validate MAVIC3 support.

**Tests:**
1. DroneModel.MAVIC3 enum exists
2. URDF loads and parses correctly
3. DSLPIDControl works with MAVIC3
4. Environment creation succeeds
5. MAVIC3 and CF2X have identical max_speed_kmh

## Approximations Made

### Visual Mesh
- **Approach:** Reuses `cf2.dae` with a scale factor of 4.787 (arm length ratio: 0.19005/0.0397)
- **Limitation:** Visual appearance resembles a scaled Crazyflie, not an actual Mavic 3
- **Recommendation:** For realistic visuals, create a custom Mavic 3 mesh

### Collision Geometry
- **Shape:** Cylinder (same type as CF2X to maintain parser compatibility)
- **Dimensions:** radius=0.20m, length=0.08m (approximates Mavic 3 body envelope)

### Inertia Approximation
- **Method:** Box approximation using unfolded dimensions
- **Dimensions used:**
  - Length: 0.3475 m
  - Width: 0.2830 m
  - Height: 0.1077 m
- **Formulas:**
  - Ixx = (1/12) × m × (width² + height²)
  - Iyy = (1/12) × m × (length² + height²)
  - Izz = (1/12) × m × (length² + width²)

### kf/km Strategy
- **Approach:** Scaled CF2X values by mass ratio (0.895/0.027 ≈ 33.15)
- **Result:** Similar hover RPM characteristics (~14,465 RPM)
- **Rationale:** Maintains controller stability with existing PID gains

### Other Coefficients
- Ground effect, drag, and downwash coefficients are kept identical to CF2X
- These may need tuning for realistic Mavic 3 behavior

## Known Limitations

1. **Visual Fidelity:** The visual mesh is a scaled Crazyflie, not a proper Mavic 3 model
2. **Aerodynamic Coefficients:** Ground effect, drag, and downwash coefficients are CF2X values
3. **PID Tuning:** Uses CF2X PID gains as a starting point; may need tuning for optimal MAVIC3 performance
4. **Rotor Dynamics:** No Mavic-specific rotor response modeling

## Suggested Tuning Knobs

For improved MAVIC3 behavior, consider tuning:

1. **PID Gains** (in `DSLPIDControl`):
   - `P_COEFF_FOR`, `I_COEFF_FOR`, `D_COEFF_FOR` (position control)
   - `P_COEFF_TOR`, `I_COEFF_TOR`, `D_COEFF_TOR` (attitude control)

2. **Thrust/Torque Coefficients** (in `mavic3.urdf`):
   - `kf`: Thrust coefficient
   - `km`: Torque coefficient

3. **Aerodynamic Effects** (in `mavic3.urdf`):
   - `gnd_eff_coeff`: Ground effect coefficient
   - `drag_coeff_xy`, `drag_coeff_z`: Drag coefficients
   - `dw_coeff_1`, `dw_coeff_2`, `dw_coeff_3`: Downwash coefficients

## Running the Smoke Test

```bash
python test_mavic3.py
```

Expected output: All 5 tests should pass.

## Backward Compatibility

- All existing drone models (CF2X, CF2P, HB) continue to work unchanged
- No changes to existing URDF files
- No changes to existing controller behavior for other models
