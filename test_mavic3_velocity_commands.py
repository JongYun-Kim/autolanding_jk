#!/usr/bin/env python3
"""Test script for MAVIC3 velocity command responses.

This script tests how the MAVIC3 drone reacts to velocity commands (RL actions).
It allows interactive selection of:
- Direction: x, y, z, -x, -y, -z
- Magnitude: 0.0, 0.1, 0.5
- Repeat: whether to repeat the action until end of episode

Outputs two videos:
- World view (external camera showing drone motion)
- Observation view (what the agent sees - grayscale camera)

Usage:
    python test_mavic3_velocity_commands.py [--drone MAVIC3|CF2X] [--duration 10]

    # Non-interactive mode with preset action
    python test_mavic3_velocity_commands.py --direction z --magnitude 0.1 --repeat
"""

import os
import sys
import argparse
import numpy as np
import pybullet as p
import pybullet_data

try:
    import cv2
except ImportError:
    print("[ERROR] OpenCV required. Install with: pip install opencv-python")
    sys.exit(1)

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, ImageType
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import (
    BaseSingleAgentAviary, ActionType, ObservationType
)
from gym_pybullet_drones.utils.utils import rgb2gray


class VelocityTestAviary(BaseSingleAgentAviary):
    """Test environment for velocity commands without external resource dependencies.

    This is a simplified version that doesn't require landing pad textures.
    It provides RGB observations and accepts velocity commands.
    """

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.MAVIC3,
        initial_xyzs=None,
        initial_rpys=None,
        physics: Physics = Physics.PYB,
        freq: int = 240,
        aggregate_phy_steps: int = 10,
        gui: bool = False,
    ):
        if initial_xyzs is None:
            initial_xyzs = np.array([[0.0, 0.0, 3.0]])
        if initial_rpys is None:
            initial_rpys = np.array([[0, 0, 0]])

        super().__init__(
            drone_model=drone_model,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            freq=freq,
            aggregate_phy_steps=aggregate_phy_steps,
            gui=gui,
            record=False,
            obs=ObservationType.RGB,
            act=ActionType.VEL,
        )

        # Ensure required attributes
        if self.IMG_RES is None:
            self.IMG_RES = np.array([84, 84])
        if not hasattr(self, 'IMG_CAPTURE_FREQ') or self.IMG_CAPTURE_FREQ is None:
            self.IMG_CAPTURE_FREQ = 1

        self._init_xyzs = initial_xyzs.copy()
        self._init_rpys = initial_rpys.copy()

        # Disable ground vehicle logic
        self.v_des = 0
        self.steering = 0
        self.gv_velocity = [0, 0, 0, 0]
        self.tr_shape = 0
        self.num_step_repeats = 1

    def step(self, action):
        """Simplified step bypassing ground vehicle logic."""
        self._saveLastAction(action)
        clipped_action = np.reshape(self._preprocessAction(action), (self.NUM_DRONES, 4))

        for _ in range(self.AGGR_PHY_STEPS):
            if self.AGGR_PHY_STEPS > 1 and self.PHYSICS in [Physics.DYN, Physics.PYB_GND, Physics.PYB_DRAG, Physics.PYB_DW, Physics.PYB_GND_DRAG_DW]:
                self._updateAndStoreKinematicInformation()

            for i in range(self.NUM_DRONES):
                if self.PHYSICS == Physics.PYB:
                    self._physics(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.DYN:
                    self._dynamics(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_GND:
                    self._physics(clipped_action[i, :], i)
                    self._groundEffect(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_DRAG:
                    self._physics(clipped_action[i, :], i)
                    self._drag(self.last_clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_DW:
                    self._physics(clipped_action[i, :], i)
                    self._downwash(i)
                elif self.PHYSICS == Physics.PYB_GND_DRAG_DW:
                    self._physics(clipped_action[i, :], i)
                    self._groundEffect(clipped_action[i, :], i)
                    self._drag(self.last_clipped_action[i, :], i)
                    self._downwash(i)

            if self.PHYSICS != Physics.DYN:
                p.stepSimulation(physicsClientId=self.CLIENT)

        self._updateAndStoreKinematicInformation()
        self.step_counter += 1

        obs = self._computeObs()
        reward = self._computeReward()
        done = self._computeDone()
        info = self._computeInfo()

        return obs, reward, done, info

    def reset(self):
        """Reset without external resource dependencies."""
        p.resetSimulation(physicsClientId=self.CLIENT)
        p.setGravity(0, 0, -self.G, physicsClientId=self.CLIENT)

        # Load ground plane
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.PLANE_ID = p.loadURDF("plane.urdf", physicsClientId=self.CLIENT)

        # Add ground markers
        self._add_ground_markers()

        # Load drone
        self.DRONE_IDS = np.array([
            p.loadURDF(
                os.path.dirname(os.path.abspath(__file__))
                + f"/gym_pybullet_drones/assets/{self.URDF}",
                self._init_xyzs[0],
                p.getQuaternionFromEuler(self._init_rpys[0]),
                flags=p.URDF_USE_INERTIA_FROM_FILE,
                physicsClientId=self.CLIENT,
            )
        ])

        # Reset state
        self.step_counter = 0
        self.pos = np.zeros((self.NUM_DRONES, 3))
        self.quat = np.zeros((self.NUM_DRONES, 4))
        self.rpy = np.zeros((self.NUM_DRONES, 3))
        self.vel = np.zeros((self.NUM_DRONES, 3))
        self.ang_v = np.zeros((self.NUM_DRONES, 3))
        self.last_clipped_action = np.zeros((self.NUM_DRONES, 4))

        self._updateAndStoreKinematicInformation()

        if hasattr(self, 'ctrl'):
            self.ctrl.reset()

        # Initialize rgb buffer - matches LandingAviary format: (1, H, W) grayscale
        self.rgb = np.zeros((1, self.IMG_RES[1], self.IMG_RES[0]), dtype=np.uint8)
        self.dep = [np.zeros((self.IMG_RES[1], self.IMG_RES[0]))]
        self.seg = [np.zeros((self.IMG_RES[1], self.IMG_RES[0]))]

        return self._computeObs()

    def _add_ground_markers(self):
        """Add visual markers for reference."""
        # Concentric circles
        for i, (radius, color) in enumerate(zip(
            [0.5, 1.0, 1.5, 2.0],
            [[1, 0, 0, 1], [1, 1, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
        )):
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=0.01,
                                      rgbaColor=color, physicsClientId=self.CLIENT)
            p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                            basePosition=[0, 0, 0.005 + i * 0.001],
                            physicsClientId=self.CLIENT)

        # Corner markers
        for pos in [[2, 2, 0.02], [-2, 2, 0.02], [-2, -2, 0.02], [2, -2, 0.02]]:
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.02],
                                      rgbaColor=[1, 1, 1, 1], physicsClientId=self.CLIENT)
            p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                            basePosition=pos, physicsClientId=self.CLIENT)

        # Direction arrows on ground
        arrow_length = 1.5
        arrow_colors = {
            '+X': [1, 0, 0, 1],  # Red
            '+Y': [0, 1, 0, 1],  # Green
        }
        # X arrow
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[arrow_length/2, 0.05, 0.01],
                                  rgbaColor=arrow_colors['+X'], physicsClientId=self.CLIENT)
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                        basePosition=[arrow_length/2, 0, 0.02], physicsClientId=self.CLIENT)
        # Y arrow
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, arrow_length/2, 0.01],
                                  rgbaColor=arrow_colors['+Y'], physicsClientId=self.CLIENT)
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                        basePosition=[0, arrow_length/2, 0.02], physicsClientId=self.CLIENT)

    def _computeObs(self, done=False):
        """Compute observation - grayscale image matching LandingAviary format."""
        if self.step_counter % self.IMG_CAPTURE_FREQ == 0 or done:
            rgb_raw, _, _ = self._getDroneImages(0, segmentation=False)
            # Store raw RGB for video capture
            self.rgb_raw = rgb_raw.copy()
            # Convert to grayscale and reshape to (1, H, W) to match LandingAviary
            self.rgb = rgb2gray(rgb_raw)[None, :]
        return self.rgb

    def _computeReward(self):
        return 0.0

    def _computeDone(self):
        pos = self.pos[0]
        if pos[2] < 0.1 or pos[2] > 15.0:
            return True
        return False

    def _computeInfo(self):
        return {
            "pos": self.pos[0].copy(),
            "vel": self.vel[0].copy(),
            "rpy": self.rpy[0].copy(),
        }

    def _clipAndNormalizeState(self, state):
        return state

    def get_external_camera_image(self, width=640, height=480):
        """Get external view of drone."""
        drone_pos = self.pos[0]
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=drone_pos,
            distance=5.0,
            yaw=45,
            pitch=-30,
            roll=0,
            upAxisIndex=2,
            physicsClientId=self.CLIENT,
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=width/height, nearVal=0.1, farVal=100.0
        )
        _, _, rgb, _, _ = p.getCameraImage(
            width=width, height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.CLIENT,
        )
        return np.array(rgb, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]


def create_video_writer(filename, fps, width, height):
    """Create OpenCV video writer."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(filename, fourcc, fps, (width, height))


def get_action_from_direction(direction: str, magnitude: float) -> np.ndarray:
    """Convert direction string and magnitude to action array.

    Returns 3D velocity command [vx, vy, vz].
    """
    direction_map = {
        'x': np.array([1.0, 0.0, 0.0]),
        '-x': np.array([-1.0, 0.0, 0.0]),
        'y': np.array([0.0, 1.0, 0.0]),
        '-y': np.array([0.0, -1.0, 0.0]),
        'z': np.array([0.0, 0.0, 1.0]),
        '-z': np.array([0.0, 0.0, -1.0]),
        'hover': np.array([0.0, 0.0, 0.0]),
    }
    return direction_map.get(direction, np.zeros(3)) * magnitude


def interactive_action_selection():
    """Interactive menu for action selection."""
    print("\n" + "=" * 50)
    print("Action Selection Menu")
    print("=" * 50)

    # Direction selection
    print("\nSelect direction:")
    print("  1. +X (forward)")
    print("  2. -X (backward)")
    print("  3. +Y (left)")
    print("  4. -Y (right)")
    print("  5. +Z (up)")
    print("  6. -Z (down)")
    print("  7. Hover (no movement)")

    dir_map = {'1': 'x', '2': '-x', '3': 'y', '4': '-y', '5': 'z', '6': '-z', '7': 'hover'}

    while True:
        choice = input("\nEnter choice (1-7): ").strip()
        if choice in dir_map:
            direction = dir_map[choice]
            break
        print("Invalid choice. Please enter 1-7.")

    # Magnitude selection
    print("\nSelect magnitude:")
    print("  1. 0.0 (no velocity)")
    print("  2. 0.1 (slow)")
    print("  3. 0.5 (medium)")

    mag_map = {'1': 0.0, '2': 0.1, '3': 0.5}

    while True:
        choice = input("\nEnter choice (1-3): ").strip()
        if choice in mag_map:
            magnitude = mag_map[choice]
            break
        print("Invalid choice. Please enter 1-3.")

    # Repeat selection
    print("\nAction mode:")
    print("  1. Repeat - repeat same action until end of episode")
    print("  2. Step-by-step - select new action at each step (interactive)")

    while True:
        choice = input("\nEnter choice (1-2): ").strip()
        if choice == '1':
            repeat = 'repeat'
            break
        elif choice == '2':
            repeat = 'step'
            break
        print("Invalid choice. Please enter 1 or 2.")

    return direction, magnitude, repeat


def quick_action_selection():
    """Full action selection menu for step-by-step mode.

    Returns (direction, magnitude, continue) tuple.
    - direction: 'x', '-x', 'y', '-y', 'z', '-z', 'hover'
    - magnitude: 0.0, 0.1, 0.5
    - continue: True to continue, False to exit episode
    """
    print("\n" + "-" * 40)
    print("Step Action Selection")
    print("-" * 40)

    # Direction selection
    print("\nSelect direction:")
    print("  1. +X (forward)    2. -X (backward)")
    print("  3. +Y (left)       4. -Y (right)")
    print("  5. +Z (up)         6. -Z (down)")
    print("  7. Hover           x. Exit episode")

    dir_map = {'1': 'x', '2': '-x', '3': 'y', '4': '-y', '5': 'z', '6': '-z', '7': 'hover'}

    while True:
        choice = input("Direction (1-7, x=exit): ").strip().lower()
        if choice == 'x':
            return None, None, False  # Exit episode
        if choice in dir_map:
            direction = dir_map[choice]
            break
        print("Invalid. Enter 1-7 or x.")

    # Magnitude selection
    print("\nSelect magnitude:")
    print("  1. 0.0 (no velocity)")
    print("  2. 0.1 (slow)")
    print("  3. 0.5 (medium)")

    mag_map = {'1': 0.0, '2': 0.1, '3': 0.5}

    while True:
        choice = input("Magnitude (1-3): ").strip()
        if choice in mag_map:
            magnitude = mag_map[choice]
            break
        print("Invalid. Enter 1-3.")

    return direction, magnitude, True  # Continue episode


def run_velocity_test(
    drone_model: DroneModel = DroneModel.MAVIC3,
    direction: str = 'z',
    magnitude: float = 0.1,
    mode: str = 'repeat',  # 'repeat' or 'step'
    duration_sec: float = 10.0,
    gui: bool = False,
    output_dir: str = ".",
):
    """Run velocity command test and record videos.

    Parameters
    ----------
    mode : str
        'repeat' - repeat same action until end
        'step' - interactive action selection each step
    """

    print(f"\n{'='*60}")
    print(f"Velocity Command Test - {drone_model.name}")
    print(f"{'='*60}")
    print(f"Initial Direction: {direction}")
    print(f"Initial Magnitude: {magnitude}")
    print(f"Mode: {mode}")
    print(f"Duration: {duration_sec}s")

    # Create environment
    env = VelocityTestAviary(
        drone_model=drone_model,
        initial_xyzs=np.array([[0.0, 0.0, 3.0]]),
        physics=Physics.PYB,
        gui=gui,
    )

    print(f"\nDrone: {drone_model.name}")
    print(f"  Mass: {env.M:.4f} kg")
    print(f"  Arm: {env.L:.4f} m")

    # Video settings
    world_fps = 30
    obs_fps = 30
    world_size = (640, 480)
    obs_size = (256, 256)  # Upscaled for visibility

    model_name = drone_model.name.lower()
    mode_suffix = "interactive" if mode == 'step' else f"{direction}_{magnitude}"
    world_path = os.path.join(output_dir, f"velocity_test_world_{model_name}_{mode_suffix}.mp4")
    obs_path = os.path.join(output_dir, f"velocity_test_obs_{model_name}_{mode_suffix}.mp4")

    world_writer = create_video_writer(world_path, world_fps, world_size[0], world_size[1])
    obs_writer = create_video_writer(obs_path, obs_fps, obs_size[0], obs_size[1])

    print(f"\nRecording to:")
    print(f"  World: {world_path}")
    print(f"  Obs: {obs_path}")

    # Calculate steps
    env_step_time = env.AGGR_PHY_STEPS / env.SIM_FREQ
    total_steps = int(duration_sec / env_step_time)
    frame_interval = max(1, int(1.0 / (world_fps * env_step_time)))

    print(f"\nSimulation: {total_steps} steps")

    if mode == 'step':
        print("\n[INTERACTIVE MODE] You will select action at each step.")
        print("Press Enter after each key. Press 'x' to exit early.\n")

    obs = env.reset()

    # Current action state
    current_direction = direction
    current_magnitude = magnitude
    current_action = get_action_from_direction(current_direction, current_magnitude)

    frame_count = 0
    for step in range(total_steps):
        # In step mode, prompt for new action at each step
        if mode == 'step':
            new_dir, new_mag, should_continue = quick_action_selection()

            if not should_continue:
                print("Exiting episode early...")
                break

            current_direction = new_dir
            current_magnitude = new_mag
            current_action = get_action_from_direction(current_direction, current_magnitude)
            print(f"  -> Action: {current_direction} @ {current_magnitude} = {current_action}")

        obs, reward, done, info = env.step(current_action)

        # Record frames
        if step % frame_interval == 0:
            # World view
            world_img = env.get_external_camera_image(*world_size)
            world_bgr = cv2.cvtColor(world_img, cv2.COLOR_RGB2BGR)

            pos = info["pos"]
            vel = info["vel"]
            cv2.putText(world_bgr, f"{drone_model.name} - Dir:{direction} Mag:{magnitude}",
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(world_bgr, f"Pos: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})",
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(world_bgr, f"Vel: ({vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f})",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(world_bgr, f"Action: [{current_action[0]:.1f}, {current_action[1]:.1f}, {current_action[2]:.1f}]",
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(world_bgr, f"Step: {step}/{total_steps}",
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            world_writer.write(world_bgr)

            # Observation view - self.rgb is (1, H, W) grayscale
            # Convert grayscale (1, H, W) to displayable format
            obs_gray = env.rgb[0]  # Shape: (H, W)
            obs_gray_uint8 = obs_gray.astype(np.uint8)

            # Upscale for visibility
            obs_upscaled = cv2.resize(obs_gray_uint8, obs_size, interpolation=cv2.INTER_NEAREST)

            # Convert grayscale to BGR for video
            obs_bgr = cv2.cvtColor(obs_upscaled, cv2.COLOR_GRAY2BGR)

            # Add overlay
            cv2.putText(obs_bgr, f"Drone Obs - {drone_model.name}", (5, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(obs_bgr, f"Alt: {pos[2]:.2f}m", (5, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            obs_writer.write(obs_bgr)
            frame_count += 1

        if mode == 'repeat' and step % (total_steps // 10) == 0:
            print(f"  Progress: {100*step//total_steps}%")

        if done:
            print(f"  Episode ended at step {step}")
            break

    world_writer.release()
    obs_writer.release()
    env.close()

    print(f"\nCompleted! Recorded {frame_count} frames")
    print(f"Output: {world_path}")
    print(f"        {obs_path}")

    return world_path, obs_path


def main():
    parser = argparse.ArgumentParser(description="Test MAVIC3 velocity command responses")
    parser.add_argument("--drone", type=str, default="MAVIC3", choices=["MAVIC3", "CF2X"])
    parser.add_argument("--direction", type=str, default=None,
                       choices=['x', '-x', 'y', '-y', 'z', '-z', 'hover'])
    parser.add_argument("--magnitude", type=float, default=None, choices=[0.0, 0.1, 0.5])
    parser.add_argument("--mode", type=str, default=None, choices=['repeat', 'step'],
                       help="'repeat': same action all episode, 'step': select action each step")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Use interactive menu for initial action selection")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model_map = {"MAVIC3": DroneModel.MAVIC3, "CF2X": DroneModel.CF2X}
    drone_model = model_map[args.drone]

    # Determine action parameters
    if args.interactive or (args.direction is None and args.magnitude is None and args.mode is None):
        direction, magnitude, mode = interactive_action_selection()
    else:
        direction = args.direction or 'z'
        magnitude = args.magnitude if args.magnitude is not None else 0.1
        mode = args.mode or 'repeat'

    run_velocity_test(
        drone_model=drone_model,
        direction=direction,
        magnitude=magnitude,
        mode=mode,
        duration_sec=args.duration,
        gui=args.gui,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
