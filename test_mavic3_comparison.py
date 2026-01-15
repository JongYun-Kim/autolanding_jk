#!/usr/bin/env python3
"""
Test script to compare CF2X and MAVIC3 drone behavior in LandingAviary.

This script creates two separate environments (one for each drone model),
applies the same constant velocity action to both, and records videos
for visual comparison.

Usage:
    python test_mavic3_comparison.py

    # With custom action (normalized velocity command)
    python test_mavic3_comparison.py --action 0.5 0.0 0.0

    # With GUI (if available)
    python test_mavic3_comparison.py --gui

Output:
    Videos saved to files/videos/ directory for each drone model.
"""

import os
import sys
import argparse
import numpy as np
import time
from datetime import datetime

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.envs.single_agent_rl.LandingAviary import LandingAviary


def run_comparison(action: np.ndarray,
                   duration_sec: float = 10.0,
                   gui: bool = False,
                   record: bool = True):
    """
    Run a comparison between CF2X and MAVIC3 drones.

    Parameters
    ----------
    action : np.ndarray
        Normalized velocity action [vx, vy, vz] in range [-1, 1]
    duration_sec : float
        Duration of the test in seconds
    gui : bool
        Whether to show GUI (only works for one env at a time)
    record : bool
        Whether to record videos
    """

    drone_models = [DroneModel.CF2X, DroneModel.MAVIC3]
    results = {}

    for drone_model in drone_models:
        print(f"\n{'='*60}")
        print(f"Testing {drone_model.name}")
        print(f"{'='*60}")

        # Create environment
        env = LandingAviary(
            drone_model=drone_model,
            initial_xyzs=np.array([[0, 0, 1.5]]),  # Start at 1.5m height
            initial_rpys=np.array([[0, 0, 0]]),    # Level orientation
            physics=Physics.PYB_GND_DRAG_DW,
            freq=240,
            aggregate_phy_steps=10,
            gui=gui and (drone_model == drone_models[0]),  # GUI only for first
            record=record,
            obs=ObservationType.KIN,  # Use kinematic obs for faster testing
            act=ActionType.VEL,
            episode_len_sec=int(duration_sec) + 2
        )

        # Print drone parameters
        print(f"\nDrone Parameters:")
        print(f"  Mass: {env.M:.4f} kg")
        print(f"  Arm length: {env.L:.4f} m")
        print(f"  KF: {env.KF:.2e}")
        print(f"  KM: {env.KM:.2e}")
        print(f"  Max RPM: {env.MAX_RPM:.1f}")
        print(f"  Hover RPM: {env.HOVER_RPM:.1f}")
        print(f"  Max speed (km/h): {env.MAX_SPEED_KMH}")
        print(f"  Speed limit (m/s): {env.SPEED_LIMIT}")
        print(f"  Inertia Ixx: {env.J[0,0]:.2e}")
        print(f"  Inertia Iyy: {env.J[1,1]:.2e}")
        print(f"  Inertia Izz: {env.J[2,2]:.2e}")

        # Reset environment
        obs = env.reset()

        # Run simulation
        print(f"\nRunning simulation with action: {action}")

        positions = []
        velocities = []
        times = []

        steps_per_sec = env.SIM_FREQ / env.AGGR_PHY_STEPS
        total_steps = int(duration_sec * steps_per_sec)

        start_time = time.time()

        for step in range(total_steps):
            # Apply constant velocity action
            obs, reward, done, info = env.step(action)

            # Record state
            if hasattr(env, 'pos') and hasattr(env, 'vel'):
                positions.append(env.pos[0].copy())
                velocities.append(env.vel[0].copy())
                times.append(step / steps_per_sec)

            if done:
                print(f"  Episode ended at step {step}")
                break

        elapsed_time = time.time() - start_time

        # Calculate statistics
        positions = np.array(positions)
        velocities = np.array(velocities)

        print(f"\nResults for {drone_model.name}:")
        print(f"  Simulation time: {elapsed_time:.2f}s (real time)")
        print(f"  Steps completed: {len(positions)}")

        if len(positions) > 0:
            print(f"  Final position: [{positions[-1,0]:.3f}, {positions[-1,1]:.3f}, {positions[-1,2]:.3f}] m")
            print(f"  Final velocity: [{velocities[-1,0]:.3f}, {velocities[-1,1]:.3f}, {velocities[-1,2]:.3f}] m/s")
            print(f"  Max velocity reached: [{np.max(np.abs(velocities[:,0])):.3f}, {np.max(np.abs(velocities[:,1])):.3f}, {np.max(np.abs(velocities[:,2])):.3f}] m/s")
            print(f"  Position range X: [{np.min(positions[:,0]):.3f}, {np.max(positions[:,0]):.3f}] m")
            print(f"  Position range Y: [{np.min(positions[:,1]):.3f}, {np.max(positions[:,1]):.3f}] m")
            print(f"  Position range Z: [{np.min(positions[:,2]):.3f}, {np.max(positions[:,2]):.3f}] m")

        results[drone_model.name] = {
            'positions': positions,
            'velocities': velocities,
            'times': times,
            'mass': env.M,
            'max_rpm': env.MAX_RPM,
            'hover_rpm': env.HOVER_RPM,
            'speed_limit': env.SPEED_LIMIT.copy(),
        }

        env.close()

    # Print comparison summary
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"Action applied: {action}")
    print()

    for name, data in results.items():
        print(f"{name}:")
        print(f"  Mass: {data['mass']:.4f} kg")
        print(f"  Speed limit: {data['speed_limit']}")
        if len(data['positions']) > 0:
            final_pos = data['positions'][-1]
            final_vel = data['velocities'][-1]
            print(f"  Final position: [{final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}] m")
            print(f"  Final velocity: [{final_vel[0]:.3f}, {final_vel[1]:.3f}, {final_vel[2]:.3f}] m/s")
        print()

    return results


def test_urdf_parsing():
    """Test that MAVIC3 URDF parses correctly."""
    print("\n" + "="*60)
    print("Testing URDF Parsing")
    print("="*60)

    try:
        env = LandingAviary(
            drone_model=DroneModel.MAVIC3,
            initial_xyzs=np.array([[0, 0, 1]]),
            gui=False,
            record=False,
            obs=ObservationType.KIN,
            act=ActionType.VEL,
        )

        print(f"\nMAVIC3 URDF parsed successfully!")
        print(f"  Mass: {env.M} kg (expected: 0.895)")
        print(f"  Arm: {env.L} m (expected: 0.112)")
        print(f"  KF: {env.KF} (expected: 7.5e-8)")
        print(f"  KM: {env.KM} (expected: 1.88e-9)")
        print(f"  Inertia Ixx: {env.J[0,0]} (expected: 6.84e-3)")
        print(f"  Inertia Iyy: {env.J[1,1]} (expected: 9.88e-3)")
        print(f"  Inertia Izz: {env.J[2,2]} (expected: 1.50e-2)")
        print(f"  Collision radius: {env.COLLISION_R} m (expected: 0.24)")
        print(f"  Collision height: {env.COLLISION_H} m (expected: 0.10)")
        print(f"  Prop radius: {env.PROP_RADIUS} m (expected: 0.12)")

        # Verify values are in expected ranges
        assert 0.8 < env.M < 1.0, f"Mass {env.M} outside expected range"
        assert 0.1 < env.L < 0.15, f"Arm {env.L} outside expected range"
        assert 1e-9 < env.KF < 1e-6, f"KF {env.KF} outside expected range"
        assert 1e-10 < env.KM < 1e-7, f"KM {env.KM} outside expected range"

        env.close()
        print("\nAll URDF parsing tests passed!")
        return True

    except Exception as e:
        print(f"\nURDF parsing failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_controller():
    """Test that Mavic3PIDControl works correctly."""
    print("\n" + "="*60)
    print("Testing Mavic3PIDControl")
    print("="*60)

    try:
        from gym_pybullet_drones.control.Mavic3PIDControl import Mavic3PIDControl

        ctrl = Mavic3PIDControl(drone_model=DroneModel.MAVIC3)
        print(f"\nMavic3PIDControl created successfully!")
        print(f"  GRAVITY: {ctrl.GRAVITY}")
        print(f"  KF: {ctrl.KF}")
        print(f"  KM: {ctrl.KM}")
        print(f"  P_COEFF_FOR: {ctrl.P_COEFF_FOR}")
        print(f"  P_COEFF_TOR: {ctrl.P_COEFF_TOR}")
        print(f"  PWM2RPM_SCALE: {ctrl.PWM2RPM_SCALE}")
        print(f"  PWM2RPM_CONST: {ctrl.PWM2RPM_CONST}")

        # Test control computation
        rpm, pos_e, yaw_e = ctrl.computeControl(
            control_timestep=1/240,
            cur_pos=np.array([0, 0, 1]),
            cur_quat=np.array([0, 0, 0, 1]),
            cur_vel=np.array([0, 0, 0]),
            cur_ang_vel=np.array([0, 0, 0]),
            target_pos=np.array([0, 0, 1.1]),
        )

        print(f"\nControl output for hovering test:")
        print(f"  RPM: {rpm}")
        print(f"  Position error: {pos_e}")
        print(f"  Yaw error: {yaw_e}")

        assert np.all(rpm > 0), "RPM should be positive"
        assert np.all(rpm < 10000), "RPM should be reasonable for Mavic3"

        print("\nAll controller tests passed!")
        return True

    except Exception as e:
        print(f"\nController test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare CF2X and MAVIC3 drone behavior')
    parser.add_argument('--action', type=float, nargs=3, default=[0.5, 0.0, 0.0],
                        help='Normalized velocity action [vx, vy, vz] (default: [0.5, 0.0, 0.0])')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='Test duration in seconds (default: 10.0)')
    parser.add_argument('--gui', action='store_true',
                        help='Show GUI (only for first drone)')
    parser.add_argument('--no-record', action='store_true',
                        help='Disable video recording')
    parser.add_argument('--test-only', action='store_true',
                        help='Only run URDF and controller tests, skip comparison')

    args = parser.parse_args()

    print("="*60)
    print("MAVIC3 Drone Model Comparison Test")
    print("="*60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Run URDF parsing test
    urdf_ok = test_urdf_parsing()

    # Run controller test
    ctrl_ok = test_controller()

    if not args.test_only:
        if urdf_ok and ctrl_ok:
            # Run comparison
            action = np.array(args.action)
            results = run_comparison(
                action=action,
                duration_sec=args.duration,
                gui=args.gui,
                record=not args.no_record
            )

            print("\n" + "="*60)
            print("Test completed!")
            if not args.no_record:
                print("Check files/videos/ directory for recorded videos.")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("Skipping comparison due to failed tests.")
            print("="*60)
            sys.exit(1)
    else:
        print("\n" + "="*60)
        if urdf_ok and ctrl_ok:
            print("All tests passed!")
        else:
            print("Some tests failed!")
            sys.exit(1)
        print("="*60)
