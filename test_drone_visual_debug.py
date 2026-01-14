#!/usr/bin/env python3
"""Visual diagnostic script for drone model verification.

This script creates two videos to verify drone behavior:
1. External world view - shows the drone moving in 3D space (third-person)
2. Drone observation view - shows what the onboard camera sees (first-person)

This helps diagnose issues where the drone body or propellers might be
blocking the camera observation due to size, camera placement, or physics.

Usage:
    python test_drone_visual_debug.py [--drone MAVIC3|CF2X] [--duration 10]

Requirements:
    pip install opencv-python numpy pybullet gym scipy

Output:
    - drone_world_view_<model>.mp4   (external simulation view)
    - drone_obs_view_<model>.mp4     (drone camera observations)
"""

import os
import sys
import argparse
import numpy as np
import pybullet as p
import pybullet_data
from datetime import datetime

try:
    import cv2
except ImportError:
    print("[ERROR] OpenCV required. Install with: pip install opencv-python")
    sys.exit(1)

from gym import spaces
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, ImageType
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import (
    BaseSingleAgentAviary, ActionType, ObservationType
)
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl


class VisualDebugAviary(BaseSingleAgentAviary):
    """Minimal environment for visual debugging without external resources.

    This environment:
    - Spawns a single drone at a fixed starting position
    - Provides RGB camera observations from the drone's POV
    - Does NOT require external landing pad textures
    - Adds a simple ground plane and optional visual markers
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
        record: bool = False,
    ):
        """Initialize the visual debug environment.

        Parameters
        ----------
        drone_model : DroneModel
            The drone model to use (MAVIC3, CF2X, etc.)
        initial_xyzs : ndarray, optional
            Initial XYZ position. Defaults to [0, 0, 2.0].
        initial_rpys : ndarray, optional
            Initial roll, pitch, yaw. Defaults to [0, 0, 0].
        physics : Physics
            Physics simulation mode.
        freq : int
            Simulation frequency in Hz.
        aggregate_phy_steps : int
            Number of physics steps per env step.
        gui : bool
            Whether to show PyBullet GUI.
        record : bool
            Whether to record onboard images.
        """
        if initial_xyzs is None:
            initial_xyzs = np.array([[0.0, 0.0, 2.0]])
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
            record=record,
            obs=ObservationType.RGB,
            act=ActionType.VEL,
        )

        # Ensure vision attributes are set (may be set by parent, but ensure non-None)
        if self.IMG_RES is None:
            self.IMG_RES = np.array([84, 84])
        if not hasattr(self, 'IMG_CAPTURE_FREQ') or self.IMG_CAPTURE_FREQ is None:
            self.IMG_CAPTURE_FREQ = 1

        # Store initial position for reset
        self._init_xyzs = initial_xyzs.copy()
        self._init_rpys = initial_rpys.copy()

        # Disable ground vehicle (landing pad) logic - set required attributes
        self.v_des = 0
        self.steering = 0
        self.gv_velocity = [0, 0, 0, 0]
        self.tr_shape = 0
        self.num_step_repeats = 1

    def step(self, action):
        """Simplified step that bypasses ground vehicle logic."""
        # Save, preprocess, and clip the action
        self._saveLastAction(action)
        clipped_action = np.reshape(self._preprocessAction(action), (self.NUM_DRONES, 4))

        # Repeat for aggregate physics steps
        for _ in range(self.AGGR_PHY_STEPS):
            # Update kinematic info if needed
            if self.AGGR_PHY_STEPS > 1 and self.PHYSICS in [Physics.DYN, Physics.PYB_GND, Physics.PYB_DRAG, Physics.PYB_DW, Physics.PYB_GND_DRAG_DW]:
                self._updateAndStoreKinematicInformation()

            # Apply physics
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

            # Step simulation
            if self.PHYSICS != Physics.DYN:
                p.stepSimulation(physicsClientId=self.CLIENT)

        # Update and store kinematic information
        self._updateAndStoreKinematicInformation()

        # Increment step counter
        self.step_counter += 1

        # Compute observation, reward, done, info
        obs = self._computeObs()
        reward = self._computeReward()
        done = self._computeDone()
        info = self._computeInfo()

        return obs, reward, done, info

    def reset(self):
        """Reset the environment without external resource dependencies."""
        # Reset simulation
        p.resetSimulation(physicsClientId=self.CLIENT)
        p.setGravity(0, 0, -self.G, physicsClientId=self.CLIENT)

        # Reload ground plane from pybullet_data
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.PLANE_ID = p.loadURDF(
            "plane.urdf", physicsClientId=self.CLIENT
        )

        # Add visual reference markers on the ground
        self._add_ground_markers()

        # Reload drone
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

        # Reset state variables
        self.step_counter = 0
        self.pos = np.zeros((self.NUM_DRONES, 3))
        self.quat = np.zeros((self.NUM_DRONES, 4))
        self.rpy = np.zeros((self.NUM_DRONES, 3))
        self.vel = np.zeros((self.NUM_DRONES, 3))
        self.ang_v = np.zeros((self.NUM_DRONES, 3))
        self.last_clipped_action = np.zeros((self.NUM_DRONES, 4))

        # Initialize kinematic info
        self._updateAndStoreKinematicInformation()

        # Reset controller
        if hasattr(self, 'ctrl'):
            self.ctrl.reset()

        # Initialize RGB buffer
        self.rgb = np.zeros((1, self.IMG_RES[1], self.IMG_RES[0]))
        self.dep = [np.zeros((self.IMG_RES[1], self.IMG_RES[0]))]
        self.seg = [np.zeros((self.IMG_RES[1], self.IMG_RES[0]))]

        return self._computeObs()

    def _add_ground_markers(self):
        """Add visual reference markers on the ground plane."""
        # Create a target circle pattern using visual shapes
        marker_colors = [
            [1, 0, 0, 1],    # Red
            [1, 1, 0, 1],    # Yellow
            [0, 1, 0, 1],    # Green
            [0, 0, 1, 1],    # Blue
        ]

        # Add concentric circles as markers
        for i, (radius, color) in enumerate(zip([0.5, 1.0, 1.5, 2.0], marker_colors)):
            visual_id = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=radius,
                length=0.01,
                rgbaColor=color,
                physicsClientId=self.CLIENT,
            )
            p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=visual_id,
                basePosition=[0, 0, 0.005 + i * 0.001],
                physicsClientId=self.CLIENT,
            )

        # Add corner markers to show scale
        corner_positions = [
            [2, 2, 0.02],
            [-2, 2, 0.02],
            [-2, -2, 0.02],
            [2, -2, 0.02],
        ]
        for pos in corner_positions:
            visual_id = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[0.1, 0.1, 0.02],
                rgbaColor=[1, 1, 1, 1],
                physicsClientId=self.CLIENT,
            )
            p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=visual_id,
                basePosition=pos,
                physicsClientId=self.CLIENT,
            )

    def _computeObs(self, done=False):
        """Compute and return RGB observation from drone camera."""
        if self.step_counter % self.IMG_CAPTURE_FREQ == 0 or done:
            self.rgb, self.dep[0], self.seg[0] = self._getDroneImages(
                0, segmentation=False
            )
            # Convert to grayscale for standard observation format
            if len(self.rgb.shape) == 3:
                # Convert RGBA to grayscale
                gray = np.mean(self.rgb[:, :, :3], axis=2).astype(np.uint8)
                self.rgb_raw = self.rgb.copy()  # Keep raw RGB for video
                self.rgb = gray[None, :]
        return self.rgb

    def _computeReward(self):
        """Minimal reward function - just return 0."""
        return 0.0

    def _computeDone(self):
        """Check if episode is done (e.g., drone crashed)."""
        pos = self.pos[0]
        # Done if drone is too low or too high
        if pos[2] < 0.1 or pos[2] > 10.0:
            return True
        return False

    def _computeInfo(self):
        """Return additional info."""
        return {
            "pos": self.pos[0].copy(),
            "rpy": self.rpy[0].copy(),
            "vel": self.vel[0].copy(),
        }

    def _clipAndNormalizeState(self, state):
        """Normalize state for observation space."""
        # Simple normalization
        return state

    def get_external_camera_image(self, width=640, height=480):
        """Capture an external view of the drone in the world.

        Returns
        -------
        ndarray
            RGB image of shape (height, width, 3)
        """
        drone_pos = self.pos[0]

        # Camera positioned to see the drone from above and behind
        camera_distance = 4.0
        camera_yaw = 45  # degrees
        camera_pitch = -30  # degrees (negative = looking down)

        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=drone_pos,
            distance=camera_distance,
            yaw=camera_yaw,
            pitch=camera_pitch,
            roll=0,
            upAxisIndex=2,
            physicsClientId=self.CLIENT,
        )

        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=width / height,
            nearVal=0.1,
            farVal=100.0,
        )

        _, _, rgb, _, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.CLIENT,
        )

        # Convert to numpy RGB (remove alpha channel)
        rgb_array = np.array(rgb, dtype=np.uint8).reshape(height, width, 4)
        return rgb_array[:, :, :3]

    def get_drone_camera_image(self, width=84, height=84):
        """Capture the drone's onboard camera view.

        Returns
        -------
        ndarray
            RGB image of shape (height, width, 3)
        """
        # Use the stored raw RGB if available
        if hasattr(self, 'rgb_raw') and self.rgb_raw is not None:
            img = self.rgb_raw[:, :, :3]
            if img.shape[0] != height or img.shape[1] != width:
                img = cv2.resize(img, (width, height))
            return img

        # Otherwise capture fresh
        rgb, _, _ = self._getDroneImages(0, segmentation=False)
        return rgb[:, :, :3]


def create_video_writer(filename, fps, width, height):
    """Create an OpenCV video writer.

    Parameters
    ----------
    filename : str
        Output video filename.
    fps : int
        Frames per second.
    width : int
        Video width in pixels.
    height : int
        Video height in pixels.

    Returns
    -------
    cv2.VideoWriter
        OpenCV video writer object.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(filename, fourcc, fps, (width, height))


def run_visual_debug(
    drone_model: DroneModel = DroneModel.MAVIC3,
    duration_sec: float = 10.0,
    gui: bool = False,
    output_dir: str = ".",
):
    """Run the visual debug test and save videos.

    Parameters
    ----------
    drone_model : DroneModel
        The drone model to test.
    duration_sec : float
        Duration of the test in seconds.
    gui : bool
        Whether to show PyBullet GUI.
    output_dir : str
        Directory to save output videos.
    """
    print(f"\n{'='*60}")
    print(f"Visual Debug Test for {drone_model.name}")
    print(f"{'='*60}")

    # Create environment
    print(f"\nCreating environment with {drone_model.name}...")
    env = VisualDebugAviary(
        drone_model=drone_model,
        initial_xyzs=np.array([[0.0, 0.0, 3.0]]),  # Start 3m high
        initial_rpys=np.array([[0, 0, 0]]),
        physics=Physics.PYB,
        freq=240,
        aggregate_phy_steps=10,
        gui=gui,
        record=False,
    )

    # Print drone parameters
    print(f"\nDrone Parameters:")
    print(f"  Mass: {env.M:.4f} kg")
    print(f"  Arm length: {env.L:.5f} m")
    print(f"  Prop radius: {env.PROP_RADIUS:.4f} m")
    print(f"  Hover RPM: {env.HOVER_RPM:.1f}")
    print(f"  Max speed: {env.MAX_SPEED_KMH:.1f} km/h")

    # Video settings
    world_fps = 30
    obs_fps = 30
    world_size = (640, 480)
    obs_size = (256, 256)  # Upscaled from 84x84 for visibility

    # Create video writers
    model_name = drone_model.name.lower()
    world_video_path = os.path.join(output_dir, f"drone_world_view_{model_name}.mp4")
    obs_video_path = os.path.join(output_dir, f"drone_obs_view_{model_name}.mp4")

    world_writer = create_video_writer(
        world_video_path, world_fps, world_size[0], world_size[1]
    )
    obs_writer = create_video_writer(
        obs_video_path, obs_fps, obs_size[0], obs_size[1]
    )

    print(f"\nRecording videos:")
    print(f"  World view: {world_video_path}")
    print(f"  Drone observation: {obs_video_path}")

    # Calculate number of steps
    env_step_time = env.AGGR_PHY_STEPS / env.SIM_FREQ
    total_steps = int(duration_sec / env_step_time)
    frame_interval = max(1, int(1.0 / (world_fps * env_step_time)))

    print(f"\nSimulation settings:")
    print(f"  Duration: {duration_sec} seconds")
    print(f"  Total steps: {total_steps}")
    print(f"  Step time: {env_step_time*1000:.2f} ms")
    print(f"  Frame interval: every {frame_interval} steps")

    # Reset environment
    print(f"\nRunning simulation...")
    obs = env.reset()

    # Define a simple motion pattern: hover with small oscillations
    motion_phase = 0.0
    motion_speed = 0.5  # rad/sec

    frame_count = 0
    for step in range(total_steps):
        # Create oscillating velocity commands
        motion_phase += motion_speed * env_step_time

        # Circular horizontal motion + vertical bobbing
        vx = 0.3 * np.sin(motion_phase)
        vy = 0.3 * np.cos(motion_phase)
        vz = 0.1 * np.sin(2 * motion_phase)  # Gentle vertical oscillation

        action = np.array([vx, vy, vz])

        # Step environment
        obs, reward, done, info = env.step(action)

        # Record frames at specified interval
        if step % frame_interval == 0:
            # Capture external world view
            world_img = env.get_external_camera_image(
                width=world_size[0], height=world_size[1]
            )
            # Convert RGB to BGR for OpenCV
            world_img_bgr = cv2.cvtColor(world_img, cv2.COLOR_RGB2BGR)

            # Add overlay text
            pos = info["pos"]
            overlay_text = f"Pos: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
            cv2.putText(
                world_img_bgr, overlay_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            cv2.putText(
                world_img_bgr, f"Model: {drone_model.name}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            cv2.putText(
                world_img_bgr, f"Step: {step}/{total_steps}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )

            world_writer.write(world_img_bgr)

            # Capture drone observation view
            obs_img = env.get_drone_camera_image(width=84, height=84)
            # Upscale for visibility
            obs_img_upscaled = cv2.resize(
                obs_img, obs_size, interpolation=cv2.INTER_NEAREST
            )
            obs_img_bgr = cv2.cvtColor(obs_img_upscaled, cv2.COLOR_RGB2BGR)

            # Add overlay text
            cv2.putText(
                obs_img_bgr, f"Drone Camera - {drone_model.name}", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
            cv2.putText(
                obs_img_bgr, f"Alt: {pos[2]:.2f}m", (5, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

            obs_writer.write(obs_img_bgr)
            frame_count += 1

        # Progress indicator
        if step % (total_steps // 10) == 0:
            pct = int(100 * step / total_steps)
            print(f"  Progress: {pct}% (step {step}/{total_steps})")

        if done:
            print(f"  Episode ended early at step {step}")
            break

    # Cleanup
    world_writer.release()
    obs_writer.release()
    env.close()

    print(f"\nCompleted!")
    print(f"  Recorded {frame_count} frames")
    print(f"\nOutput files:")
    print(f"  {world_video_path}")
    print(f"  {obs_video_path}")

    return world_video_path, obs_video_path


def compare_drone_models(duration_sec: float = 8.0, output_dir: str = "."):
    """Run visual debug for both CF2X and MAVIC3 for comparison.

    Parameters
    ----------
    duration_sec : float
        Duration for each test.
    output_dir : str
        Output directory.
    """
    print("\n" + "=" * 60)
    print("Comparing CF2X and MAVIC3 drone models")
    print("=" * 60)

    results = {}

    for model in [DroneModel.CF2X, DroneModel.MAVIC3]:
        try:
            world_path, obs_path = run_visual_debug(
                drone_model=model,
                duration_sec=duration_sec,
                gui=False,
                output_dir=output_dir,
            )
            results[model.name] = {
                "world_view": world_path,
                "obs_view": obs_path,
                "status": "success",
            }
        except Exception as e:
            print(f"[ERROR] Failed for {model.name}: {e}")
            results[model.name] = {"status": "failed", "error": str(e)}

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for model_name, result in results.items():
        if result["status"] == "success":
            print(f"\n{model_name}:")
            print(f"  World view: {result['world_view']}")
            print(f"  Obs view:   {result['obs_view']}")
        else:
            print(f"\n{model_name}: FAILED - {result.get('error', 'Unknown error')}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visual diagnostic script for drone model verification"
    )
    parser.add_argument(
        "--drone",
        type=str,
        default="MAVIC3",
        choices=["MAVIC3", "CF2X", "CF2P", "HB", "compare"],
        help="Drone model to test, or 'compare' to test CF2X and MAVIC3",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Duration of test in seconds",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Show PyBullet GUI window",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save output videos",
    )

    args = parser.parse_args()

    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)

    if args.drone == "compare":
        compare_drone_models(
            duration_sec=args.duration,
            output_dir=args.output_dir,
        )
    else:
        # Map string to DroneModel enum
        model_map = {
            "MAVIC3": DroneModel.MAVIC3,
            "CF2X": DroneModel.CF2X,
            "CF2P": DroneModel.CF2P,
            "HB": DroneModel.HB,
        }
        drone_model = model_map[args.drone]

        run_visual_debug(
            drone_model=drone_model,
            duration_sec=args.duration,
            gui=args.gui,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
