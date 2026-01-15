# Mavic 3 Drone Model Support

This document describes the addition of DJI Mavic 3 drone support to the gym-pybullet-drones simulation environment.

## Overview

The Mavic 3 is a significantly larger and heavier drone compared to the default CF2X (Crazyflie 2.0). This implementation allows training RL agents with the Mavic 3 model while maintaining compatibility with existing CF2X-based workflows.

## Physical Specifications

### DJI Mavic 3 (Real-world)
| Parameter | Value |
|-----------|-------|
| Mass | 0.895 kg |
| Max Takeoff Weight | 1.050 kg |
| Unfolded Dimensions (L×W×H) | 347.5 × 283 × 107.7 mm |
| Folded Dimensions (L×W×H) | 221 × 96.3 × 90.3 mm |
| Propeller Diameter | 238.8 mm (9.4 inches) |
| Propeller Pitch | 134.6 mm (5.3 inches) |

### Comparison: CF2X vs MAVIC3

| Parameter | CF2X | MAVIC3 | Scale Factor |
|-----------|------|--------|--------------|
| Mass | 0.027 kg | 0.895 kg | 33.1× |
| Arm Length | 0.0397 m | 0.112 m | 2.8× |
| Ixx | 1.4e-5 kg·m² | 6.84e-3 kg·m² | 489× |
| Iyy | 1.4e-5 kg·m² | 9.88e-3 kg·m² | 706× |
| Izz | 2.17e-5 kg·m² | 1.50e-2 kg·m² | 691× |
| KF (thrust coeff) | 3.16e-10 | 7.5e-8 | 237× |
| KM (moment coeff) | 7.94e-12 | 1.88e-9 | 237× |
| Thrust-to-Weight | 2.25 | 2.0 | 0.89× |
| Max Speed | 30 km/h | 30 km/h | 1.0× |
| Prop Radius | 0.0231 m | 0.12 m | 5.2× |
| Collision Radius | 0.06 m | 0.24 m | 4.0× |
| Collision Height | 0.025 m | 0.10 m | 4.0× |

## Parameter Derivations

### Inertia (Box Approximation)
Using unfolded body dimensions with uniform mass distribution:
```
Ixx = (1/12) × m × (W² + H²) = (1/12) × 0.895 × (0.283² + 0.1077²) = 6.84e-3 kg·m²
Iyy = (1/12) × m × (L² + H²) = (1/12) × 0.895 × (0.3475² + 0.1077²) = 9.88e-3 kg·m²
Izz = (1/12) × m × (L² + W²) = (1/12) × 0.895 × (0.3475² + 0.283²) = 1.50e-2 kg·m²
```

### Arm Length
Estimated from unfolded diagonal dimensions for X-configuration:
```
arm = sqrt((L/2)² + (W/2)²) / 2 ≈ 0.112 m
```

### Thrust Coefficient (KF)
Back-calculated from thrust-to-weight ratio assumption:
```
Max thrust = T/W × mass × g = 2.0 × 0.895 × 9.81 = 17.56 N (total)
Per motor = 4.39 N

Assuming max RPM ≈ 7500:
KF = thrust_per_motor / RPM² = 4.39 / 7500² ≈ 7.5e-8
```

### Moment Coefficient (KM)
Using CF2X ratio (KM/KF ≈ 1/40):
```
KM = 7.5e-8 / 40 = 1.88e-9
```

### Drag Coefficients
Scaled by frontal area ratio (~10×):
```
drag_coeff_xy = 9.18e-7 × 10 = 9.18e-6
drag_coeff_z = 10.31e-7 × 10 = 1.03e-5
```

### Ground Effect and Downwash
Kept identical to CF2X due to lack of empirical data:
- `gnd_eff_coeff`: 11.36859
- `dw_coeff_1`: 2267.18
- `dw_coeff_2`: 0.16
- `dw_coeff_3`: -0.11

## Controller Design (Mavic3PIDControl)

### PID Gain Scaling Rationale

The attitude PID gains were scaled from CF2X by a factor of ~0.75, based on:

1. **Inertia increase**: ~500× (requires more torque for same angular acceleration)
2. **Torque capability increase**: ~670× (from KF × arm length scaling)
3. **Net effect**: Torque capability > inertia increase, so gains are slightly LOWER

```python
# CF2X attitude gains
P_COEFF_TOR_CF2X = [70000, 70000, 60000]

# MAVIC3 attitude gains (×0.75)
P_COEFF_TOR_MAVIC3 = [52500, 52500, 45000]
```

### PWM to RPM Conversion
Estimated for Mavic 3 motors with max RPM ~7500:
```python
PWM2RPM_SCALE = 0.1646
PWM2RPM_CONST = -3292.0
```

## Files Changed

### New Files
| File | Description |
|------|-------------|
| `gym_pybullet_drones/assets/mavic3.urdf` | Mavic 3 drone model definition |
| `gym_pybullet_drones/control/Mavic3PIDControl.py` | PID controller for Mavic 3 |
| `test_mavic3_comparison.py` | Test script for comparing drones |
| `MAVIC3_SUPPORT.md` | This documentation |

### Modified Files
| File | Changes |
|------|---------|
| `gym_pybullet_drones/envs/BaseAviary.py` | Added `DroneModel.MAVIC3` enum, X-config mixer matrix |
| `gym_pybullet_drones/envs/single_agent_rl/BaseSingleAgentAviary.py` | Added Mavic3PIDControl import and selection |

## Usage

### Basic Usage
```python
from gym_pybullet_drones.envs.BaseAviary import DroneModel
from gym_pybullet_drones.envs.single_agent_rl.LandingAviary import LandingAviary

# Create environment with Mavic 3
env = LandingAviary(
    drone_model=DroneModel.MAVIC3,
    # ... other parameters
)
```

### Running Comparison Test
```bash
# Basic test (URDF parsing and controller)
python test_mavic3_comparison.py --test-only

# Full comparison with video recording
python test_mavic3_comparison.py

# Custom velocity action
python test_mavic3_comparison.py --action 0.5 0.0 0.0

# With GUI (if available)
python test_mavic3_comparison.py --gui
```

## Tuning Guide

### Thrust-to-Weight Ratio

The default value is **2.0**, which is a conservative estimate. Effects of different values:

| T/W Ratio | Behavior |
|-----------|----------|
| < 1.5 | Cannot hover stably, sluggish response, may crash |
| 1.5-2.0 | Conservative flight, slower acceleration, good for landing tasks |
| 2.0-2.5 | Balanced performance, responsive but stable |
| > 2.5 | Aggressive response, may overshoot, potential instability |

**To change**: Edit `thrust2weight` in `gym_pybullet_drones/assets/mavic3.urdf`

### PID Gains

If drone behavior is unsatisfactory:

| Symptom | Solution |
|---------|----------|
| Too sluggish | Increase `P_COEFF_TOR` in `Mavic3PIDControl.py` |
| Oscillating/aggressive | Decrease `P_COEFF_TOR` or increase `D_COEFF_TOR` |
| Drifting | Increase `I_COEFF_FOR` |
| Overshooting position | Increase `D_COEFF_FOR` |

### KF (Thrust Coefficient)

If hover is unstable or requires unreasonable RPM:
- **Too low KF**: Drone needs very high RPM to hover
- **Too high KF**: Drone hovers at very low RPM, may be unstable

Current value: `7.5e-8` (back-calculated from T/W=2.0 and estimated max RPM of 7500)

## Known Limitations

1. **No real Mavic 3 mesh**: Uses scaled CF2X mesh for visualization
2. **Estimated motor parameters**: KF, KM, and max RPM are estimates without real data
3. **Simplified inertia**: Box approximation may not match actual mass distribution
4. **Ground effect/downwash**: Uses CF2X coefficients (no Mavic 3-specific data)

## Future Improvements

- [ ] Obtain real Mavic 3 motor thrust curves for accurate KF
- [ ] Create accurate 3D mesh for visualization
- [ ] Validate against real Mavic 3 flight data
- [ ] Add wind disturbance modeling specific to larger airframe
- [ ] Tune PID gains through systematic optimization

## References

- DJI Mavic 3 Specifications: https://www.dji.com/mavic-3/specs
- Original gym-pybullet-drones: https://github.com/utiasDSL/gym-pybullet-drones
- Crazyflie 2.0 modeling: http://mikehamer.info/assets/papers/Crazyflie%20Modelling.pdf
