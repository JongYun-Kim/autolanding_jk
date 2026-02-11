"""Controller comparison test: CF2X-DSL vs Mavic3-SimplePID (tuned) vs Mavic3-DSL.

Runs 3 controllers x 8 maneuvers with fixed seed for reproducibility.
Measures peak velocity, T80 (response time), and hover drift.

Usage:
    python test_mavic3.py
    python test_mavic3.py --steps 80
    python test_mavic3.py --gui
"""

import argparse
import numpy as np
import sys

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.envs.single_agent_rl.LandingAviary import LandingAviary


# Controller configurations: (label, drone_model, controller_type)
CONTROLLERS = [
    ("CF2X-DSL",        DroneModel.CF2X,   None),
    ("Mavic3-SimplePID", DroneModel.MAVIC3, None),
    ("Mavic3-DSL",      DroneModel.MAVIC3, "dsl"),
]

# Maneuver definitions: (name, velocity_action [vx, vy, vz])
MANEUVERS = [
    ("hover",    [0.0,  0.0,  0.0]),
    ("+X",       [1.0,  0.0,  0.0]),
    ("-X",       [-1.0, 0.0,  0.0]),
    ("+Y",       [0.0,  1.0,  0.0]),
    ("-Y",       [0.0, -1.0,  0.0]),
    ("+Z",       [0.0,  0.0,  1.0]),
    ("-Z",       [0.0,  0.0, -1.0]),
    ("diagonal", [0.577, 0.577, -0.577]),
]


def make_env(drone_model, controller_type, gui=False):
    """Create a LandingAviary environment."""
    return LandingAviary(
        drone_model=drone_model,
        physics=Physics.PYB_GND_DRAG_DW,
        obs=ObservationType.RGB,
        act=ActionType.VEL,
        gui=gui,
        record=False,
        controller_type=controller_type,
    )


def run_maneuver(env, action, num_steps):
    """Run a single maneuver, return per-step velocities and positions."""
    env.reset()
    # Reset controller state (integral errors) for clean maneuver comparison
    if hasattr(env, 'ctrl'):
        env.ctrl.reset()
    positions = []
    velocities = []

    action = np.array(action, dtype=np.float32)

    for step in range(num_steps):
        obs, reward, done, info = env.step(action)
        state = env._getDroneStateVector(0)
        positions.append(state[0:3].copy())
        velocities.append(state[10:13].copy())
        if done:
            break

    return np.array(positions), np.array(velocities)


def compute_metrics(positions, velocities, maneuver_name):
    """Compute peak velocity, T80, and hover drift from trajectory data."""
    n = len(velocities)

    # Peak velocity per axis
    peak_vel = np.max(np.abs(velocities), axis=0)
    peak_speed = np.max(np.linalg.norm(velocities, axis=1))

    # T80: steps to reach 80% of peak speed
    threshold = 0.8 * peak_speed
    t80 = n  # default: never reached
    if peak_speed > 1e-4:
        speeds = np.linalg.norm(velocities, axis=1)
        for i, s in enumerate(speeds):
            if s >= threshold:
                t80 = i + 1
                break

    # Hover drift: RMS position change over last 50% of trajectory
    half = max(1, n // 2)
    pos_last_half = positions[half:]
    if len(pos_last_half) > 1:
        hover_drift = np.sqrt(np.mean(np.sum((pos_last_half - pos_last_half[0])**2, axis=1)))
    else:
        hover_drift = 0.0

    return {
        "peak_vel": peak_vel,
        "peak_speed": peak_speed,
        "t80": t80,
        "hover_drift": hover_drift,
    }


def print_table_header():
    """Print the comparison table header."""
    print(f"\n{'Maneuver':<10} {'Controller':<18} {'Peak Vx':>8} {'Peak Vy':>8} {'Peak Vz':>8} "
          f"{'PeakSpd':>8} {'T80':>5} {'HvrDrift':>9}")
    print("-" * 95)


def print_metrics_row(maneuver, controller, metrics):
    """Print one row of the comparison table."""
    pv = metrics["peak_vel"]
    print(f"{maneuver:<10} {controller:<18} {pv[0]:8.4f} {pv[1]:8.4f} {pv[2]:8.4f} "
          f"{metrics['peak_speed']:8.4f} {metrics['t80']:5d} {metrics['hover_drift']:9.5f}")


def main():
    parser = argparse.ArgumentParser(description="Controller comparison: CF2X-DSL vs Mavic3-SimplePID vs Mavic3-DSL")
    parser.add_argument("--gui", action="store_true", help="Enable PyBullet GUI")
    parser.add_argument("--steps", type=int, default=50, help="Steps per maneuver (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Collect all results: results[controller_label][maneuver_name] = metrics
    all_results = {}

    for ctrl_label, drone_model, ctrl_type in CONTROLLERS:
        print(f"\n{'='*60}")
        print(f"  Controller: {ctrl_label}")
        print(f"{'='*60}")

        env = make_env(drone_model, ctrl_type, gui=args.gui)
        print(f"  M={env.M:.4f}kg  HOVER_RPM={env.HOVER_RPM:.1f}  "
              f"KF={env.KF:.2e}  KM={env.KM:.2e}  L={env.L:.4f}m")

        all_results[ctrl_label] = {}

        for man_name, man_vel in MANEUVERS:
            positions, velocities = run_maneuver(env, man_vel, args.steps)
            metrics = compute_metrics(positions, velocities, man_name)
            all_results[ctrl_label][man_name] = metrics

        env.close()

    # Print comparison table
    print_table_header()
    for man_name, _ in MANEUVERS:
        for ctrl_label, _, _ in CONTROLLERS:
            print_metrics_row(man_name, ctrl_label, all_results[ctrl_label][man_name])
        print()

    # Summary: responsiveness score (directional maneuvers only)
    directional = ["+X", "-X", "+Y", "-Y", "+Z", "-Z", "diagonal"]
    cf2x_label = CONTROLLERS[0][0]

    print(f"\n{'='*60}")
    print(f"  Responsiveness Summary (peak_speed ratio vs {cf2x_label})")
    print(f"{'='*60}")

    for ctrl_label, _, _ in CONTROLLERS[1:]:  # skip CF2X itself
        ratios = []
        for man_name in directional:
            cf2x_speed = all_results[cf2x_label][man_name]["peak_speed"]
            other_speed = all_results[ctrl_label][man_name]["peak_speed"]
            if cf2x_speed > 1e-6:
                ratios.append(other_speed / cf2x_speed)
        mean_ratio = np.mean(ratios) if ratios else 0.0
        print(f"  {ctrl_label:<18} mean ratio = {mean_ratio:.3f}  "
              f"({'PASS' if mean_ratio >= 0.8 else 'FAIL'}: target >= 0.80)")
        for i, man_name in enumerate(directional):
            print(f"    {man_name:<10} {ratios[i]:.3f}")

    # Hover drift summary
    print(f"\n{'='*60}")
    print(f"  Hover Drift Summary")
    print(f"{'='*60}")
    for ctrl_label, _, _ in CONTROLLERS:
        drift = all_results[ctrl_label]["hover"]["hover_drift"]
        target = 0.01 if "Mavic3-DSL" in ctrl_label else None
        status = ""
        if target is not None:
            status = f"  ({'PASS' if drift < target else 'FAIL'}: target < {target})"
        print(f"  {ctrl_label:<18} hover_drift = {drift:.5f} m{status}")

    # T80 summary
    print(f"\n{'='*60}")
    print(f"  T80 Response Time Summary (steps to 80% peak speed)")
    print(f"{'='*60}")
    for ctrl_label, _, _ in CONTROLLERS:
        t80_vals = [all_results[ctrl_label][m]["t80"] for m in directional]
        print(f"  {ctrl_label:<18} mean T80 = {np.mean(t80_vals):.1f}  "
              f"per-maneuver: {', '.join(f'{m}={t}' for m, t in zip(directional, t80_vals))}")


if __name__ == "__main__":
    main()
