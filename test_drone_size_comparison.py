#!/usr/bin/env python3
"""Test script for comparing CF2X and MAVIC3 drone sizes side by side.

This script spawns both CF2X and MAVIC3 drones in the same scene to
visually compare their sizes and behavior.

Outputs two videos:
- World view (external camera showing both drones)
- Observation view (side-by-side grayscale camera views from both drones)

Usage:
    python test_drone_size_comparison.py [--duration 10]
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

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import (
    BaseSingleAgentAviary, ActionType, ObservationType
)
from gym_pybullet_drones.utils.utils import rgb2gray


class DualDroneAviary(BaseSingleAgentAviary):
    """Environment with two drones (CF2X and MAVIC3) for size comparison.

    This creates a single environment with two drones side by side.
    """

    def __init__(
        self,
        physics: Physics = Physics.PYB,
        freq: int = 240,
        aggregate_phy_steps: int = 10,
        gui: bool = False,
        separation: float = 2.0,  # Horizontal separation between drones
    ):
        # Store models for both drones
        self.drone_models = [DroneModel.CF2X, DroneModel.MAVIC3]
        self.separation = separation

        # Initialize with CF2X as the "main" drone (for parent class)
        # We'll manually add MAVIC3
        initial_xyzs = np.array([[0.0, -separation/2, 3.0]])
        initial_rpys = np.array([[0, 0, 0]])

        super().__init__(
            drone_model=DroneModel.CF2X,
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

        if self.IMG_RES is None:
            self.IMG_RES = np.array([84, 84])
        if not hasattr(self, 'IMG_CAPTURE_FREQ') or self.IMG_CAPTURE_FREQ is None:
            self.IMG_CAPTURE_FREQ = 1

        self._init_xyzs = initial_xyzs.copy()
        self._init_rpys = initial_rpys.copy()

        # Disable ground vehicle
        self.v_des = 0
        self.steering = 0
        self.gv_velocity = [0, 0, 0, 0]
        self.tr_shape = 0
        self.num_step_repeats = 1

        # Store second drone info
        self.mavic3_id = None
        self.mavic3_pos = np.zeros(3)
        self.mavic3_quat = np.zeros(4)
        self.mavic3_rpy = np.zeros(3)
        self.mavic3_vel = np.zeros(3)

        # Load MAVIC3 URDF parameters for reference
        self._load_mavic3_params()

    def _load_mavic3_params(self):
        """Load MAVIC3 parameters from URDF."""
        import xml.etree.ElementTree as etxml
        urdf_path = os.path.dirname(os.path.abspath(__file__)) + "/gym_pybullet_drones/assets/mavic3.urdf"
        tree = etxml.parse(urdf_path)
        root = tree.getroot()
        props = root[0]
        self.mavic3_L = float(props.attrib['arm'])
        self.mavic3_M = float(root[1][0][1].attrib['value'])
        self.mavic3_KF = float(props.attrib['kf'])

    def step(self, action):
        """Step both drones with the same action."""
        self._saveLastAction(action)
        clipped_action = np.reshape(self._preprocessAction(action), (self.NUM_DRONES, 4))

        for _ in range(self.AGGR_PHY_STEPS):
            if self.AGGR_PHY_STEPS > 1 and self.PHYSICS in [Physics.DYN, Physics.PYB_GND, Physics.PYB_DRAG, Physics.PYB_DW, Physics.PYB_GND_DRAG_DW]:
                self._updateAndStoreKinematicInformation()

            # Apply physics to CF2X
            for i in range(self.NUM_DRONES):
                if self.PHYSICS == Physics.PYB:
                    self._physics(clipped_action[i, :], i)

            # Apply same thrust pattern to MAVIC3 (scaled)
            if self.mavic3_id is not None:
                self._apply_mavic3_physics(clipped_action[0, :])

            if self.PHYSICS != Physics.DYN:
                p.stepSimulation(physicsClientId=self.CLIENT)

        self._updateAndStoreKinematicInformation()
        self._update_mavic3_state()
        self.step_counter += 1

        obs = self._computeObs()
        reward = self._computeReward()
        done = self._computeDone()
        info = self._computeInfo()

        return obs, reward, done, info

    def _apply_mavic3_physics(self, rpm):
        """Apply physics to MAVIC3 drone with scaled forces."""
        if self.mavic3_id is None:
            return

        # Scale RPM to account for different kf
        # MAVIC3 has larger kf, so same RPM produces more thrust
        # But the drone is also heavier
        pos, quat = p.getBasePositionAndOrientation(self.mavic3_id, physicsClientId=self.CLIENT)
        rot = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)

        # Use hover RPM ratio to scale
        mavic3_hover_rpm = np.sqrt((self.mavic3_M * self.G) / (4 * self.mavic3_KF))
        cf2x_hover_rpm = self.HOVER_RPM
        rpm_scale = mavic3_hover_rpm / cf2x_hover_rpm

        scaled_rpm = rpm * rpm_scale

        # Apply forces at each prop position
        forces = np.array(scaled_rpm**2) * self.mavic3_KF
        thrust = forces.sum()

        # Apply thrust along drone's z-axis
        thrust_world = np.dot(rot, np.array([0, 0, thrust]))
        p.applyExternalForce(
            self.mavic3_id, -1, thrust_world, pos,
            p.WORLD_FRAME, physicsClientId=self.CLIENT
        )

    def _update_mavic3_state(self):
        """Update MAVIC3 state from PyBullet."""
        if self.mavic3_id is None:
            return
        pos, quat = p.getBasePositionAndOrientation(self.mavic3_id, physicsClientId=self.CLIENT)
        vel, ang_v = p.getBaseVelocity(self.mavic3_id, physicsClientId=self.CLIENT)
        self.mavic3_pos = np.array(pos)
        self.mavic3_quat = np.array(quat)
        self.mavic3_rpy = np.array(p.getEulerFromQuaternion(quat))
        self.mavic3_vel = np.array(vel)

    def reset(self):
        """Reset environment with both drones."""
        p.resetSimulation(physicsClientId=self.CLIENT)
        p.setGravity(0, 0, -self.G, physicsClientId=self.CLIENT)

        # Load ground
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.PLANE_ID = p.loadURDF("plane.urdf", physicsClientId=self.CLIENT)

        # Add markers
        self._add_ground_markers()

        # Load CF2X (left side)
        cf2x_pos = [0.0, -self.separation/2, 3.0]
        self.DRONE_IDS = np.array([
            p.loadURDF(
                os.path.dirname(os.path.abspath(__file__))
                + "/gym_pybullet_drones/assets/cf2x.urdf",
                cf2x_pos,
                p.getQuaternionFromEuler([0, 0, 0]),
                flags=p.URDF_USE_INERTIA_FROM_FILE,
                physicsClientId=self.CLIENT,
            )
        ])

        # Load MAVIC3 (right side)
        mavic3_pos = [0.0, self.separation/2, 3.0]
        self.mavic3_id = p.loadURDF(
            os.path.dirname(os.path.abspath(__file__))
            + "/gym_pybullet_drones/assets/mavic3.urdf",
            mavic3_pos,
            p.getQuaternionFromEuler([0, 0, 0]),
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=self.CLIENT,
        )

        # Add labels above drones
        self._add_drone_labels()

        # Reset states
        self.step_counter = 0
        self.pos = np.zeros((self.NUM_DRONES, 3))
        self.quat = np.zeros((self.NUM_DRONES, 4))
        self.rpy = np.zeros((self.NUM_DRONES, 3))
        self.vel = np.zeros((self.NUM_DRONES, 3))
        self.ang_v = np.zeros((self.NUM_DRONES, 3))
        self.last_clipped_action = np.zeros((self.NUM_DRONES, 4))

        self._updateAndStoreKinematicInformation()
        self._update_mavic3_state()

        if hasattr(self, 'ctrl'):
            self.ctrl.reset()

        # Initialize RGB buffers
        self.rgb = np.zeros((1, self.IMG_RES[1], self.IMG_RES[0]), dtype=np.uint8)
        self.rgb_cf2x = np.zeros((1, self.IMG_RES[1], self.IMG_RES[0]), dtype=np.uint8)
        self.rgb_mavic3 = np.zeros((1, self.IMG_RES[1], self.IMG_RES[0]), dtype=np.uint8)
        self.dep = [np.zeros((self.IMG_RES[1], self.IMG_RES[0]))]
        self.seg = [np.zeros((self.IMG_RES[1], self.IMG_RES[0]))]

        return self._computeObs()

    def _add_ground_markers(self):
        """Add ground reference markers."""
        # Scale reference (1 meter grid)
        for i in range(-3, 4):
            for j in range(-3, 4):
                if (i + j) % 2 == 0:
                    vis = p.createVisualShape(
                        p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.005],
                        rgbaColor=[0.8, 0.8, 0.8, 1], physicsClientId=self.CLIENT
                    )
                else:
                    vis = p.createVisualShape(
                        p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.005],
                        rgbaColor=[0.3, 0.3, 0.3, 1], physicsClientId=self.CLIENT
                    )
                p.createMultiBody(
                    baseMass=0, baseVisualShapeIndex=vis,
                    basePosition=[i, j, 0.005], physicsClientId=self.CLIENT
                )

        # Add scale markers
        for meter in [1, 2]:
            vis = p.createVisualShape(
                p.GEOM_BOX, halfExtents=[0.02, 0.02, meter/2],
                rgbaColor=[1, 0, 0, 1], physicsClientId=self.CLIENT
            )
            p.createMultiBody(
                baseMass=0, baseVisualShapeIndex=vis,
                basePosition=[-2, 0, meter/2], physicsClientId=self.CLIENT
            )

    def _add_drone_labels(self):
        """Add text labels near drones (using visual shapes as placeholders)."""
        # CF2X marker (left)
        vis = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.1,
            rgbaColor=[0, 0, 1, 1], physicsClientId=self.CLIENT
        )
        p.createMultiBody(
            baseMass=0, baseVisualShapeIndex=vis,
            basePosition=[0, -self.separation/2, 4.5], physicsClientId=self.CLIENT
        )

        # MAVIC3 marker (right)
        vis = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.1,
            rgbaColor=[1, 0, 0, 1], physicsClientId=self.CLIENT
        )
        p.createMultiBody(
            baseMass=0, baseVisualShapeIndex=vis,
            basePosition=[0, self.separation/2, 4.5], physicsClientId=self.CLIENT
        )

    def _computeObs(self, done=False):
        """Compute observations from both drones."""
        if self.step_counter % self.IMG_CAPTURE_FREQ == 0 or done:
            # CF2X observation
            rgb_cf2x, _, _ = self._getDroneImages(0, segmentation=False)
            self.rgb_cf2x_raw = rgb_cf2x.copy()
            self.rgb_cf2x = rgb2gray(rgb_cf2x)[None, :]

            # MAVIC3 observation
            rgb_mavic3 = self._get_mavic3_camera_image()
            self.rgb_mavic3_raw = rgb_mavic3.copy()
            self.rgb_mavic3 = rgb2gray(rgb_mavic3)[None, :]

            # Combined for standard output
            self.rgb = self.rgb_cf2x

        return self.rgb

    def _get_mavic3_camera_image(self):
        """Get camera image from MAVIC3's perspective."""
        if self.mavic3_id is None:
            return np.zeros((self.IMG_RES[1], self.IMG_RES[0], 4), dtype=np.uint8)

        rot_mat = np.array(p.getMatrixFromQuaternion(self.mavic3_quat)).reshape(3, 3)
        target = np.dot(rot_mat, np.array([0, 0, -1000])) + self.mavic3_pos

        cam_view = p.computeViewMatrix(
            cameraEyePosition=self.mavic3_pos - np.array([0, 0, 0.15]) + np.array([0, 0, self.mavic3_L]),
            cameraTargetPosition=target,
            cameraUpVector=[1, 0, 0],
            physicsClientId=self.CLIENT
        )
        cam_pro = p.computeProjectionMatrixFOV(
            fov=85.7, aspect=1.0, nearVal=self.mavic3_L, farVal=1000.0
        )

        _, _, rgb, _, _ = p.getCameraImage(
            width=self.IMG_RES[0], height=self.IMG_RES[1],
            viewMatrix=cam_view, projectionMatrix=cam_pro,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.CLIENT
        )

        return np.array(rgb, dtype=np.uint8).reshape(self.IMG_RES[1], self.IMG_RES[0], 4)

    def _computeReward(self):
        return 0.0

    def _computeDone(self):
        pos = self.pos[0]
        if pos[2] < 0.1 or pos[2] > 15.0:
            return True
        if self.mavic3_pos[2] < 0.1 or self.mavic3_pos[2] > 15.0:
            return True
        return False

    def _computeInfo(self):
        return {
            "cf2x_pos": self.pos[0].copy(),
            "cf2x_vel": self.vel[0].copy(),
            "mavic3_pos": self.mavic3_pos.copy(),
            "mavic3_vel": self.mavic3_vel.copy(),
        }

    def _clipAndNormalizeState(self, state):
        return state

    def get_comparison_camera_image(self, width=800, height=600):
        """Get camera view showing both drones for size comparison."""
        # Camera positioned to see both drones
        center = np.array([0, 0, 3.0])
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=center,
            distance=8.0,
            yaw=90,  # Looking from +X direction
            pitch=-15,
            roll=0,
            upAxisIndex=2,
            physicsClientId=self.CLIENT,
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=50, aspect=width/height, nearVal=0.1, farVal=100.0
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
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(filename, fourcc, fps, (width, height))


def run_size_comparison(
    duration_sec: float = 10.0,
    gui: bool = False,
    output_dir: str = ".",
):
    """Run size comparison test and record videos."""

    print(f"\n{'='*60}")
    print("Drone Size Comparison: CF2X vs MAVIC3")
    print(f"{'='*60}")

    env = DualDroneAviary(
        physics=Physics.PYB,
        gui=gui,
        separation=3.0,  # 3 meter separation
    )

    print(f"\nCF2X Parameters:")
    print(f"  Mass: {env.M:.4f} kg")
    print(f"  Arm: {env.L:.5f} m")
    print(f"  Prop radius: {env.PROP_RADIUS:.5f} m")

    print(f"\nMAVIC3 Parameters:")
    print(f"  Mass: {env.mavic3_M:.4f} kg")
    print(f"  Arm: {env.mavic3_L:.5f} m")

    print(f"\nSize ratio (MAVIC3/CF2X):")
    print(f"  Mass: {env.mavic3_M/env.M:.2f}x")
    print(f"  Arm length: {env.mavic3_L/env.L:.2f}x")

    # Video settings
    fps = 30
    world_size = (800, 600)
    obs_size = (512, 256)  # Side by side observations

    world_path = os.path.join(output_dir, "size_comparison_world.mp4")
    obs_path = os.path.join(output_dir, "size_comparison_obs.mp4")

    world_writer = create_video_writer(world_path, fps, world_size[0], world_size[1])
    obs_writer = create_video_writer(obs_path, fps, obs_size[0], obs_size[1])

    print(f"\nRecording to:")
    print(f"  World: {world_path}")
    print(f"  Obs: {obs_path}")

    env_step_time = env.AGGR_PHY_STEPS / env.SIM_FREQ
    total_steps = int(duration_sec / env_step_time)
    frame_interval = max(1, int(1.0 / (fps * env_step_time)))

    print(f"\nSimulation: {total_steps} steps")

    obs = env.reset()

    frame_count = 0
    motion_phase = 0.0

    for step in range(total_steps):
        # Gentle hover with small oscillation
        motion_phase += 0.5 * env_step_time
        vz = 0.05 * np.sin(motion_phase)  # Very gentle vertical motion
        action = np.array([0.0, 0.0, vz])

        obs, reward, done, info = env.step(action)

        if step % frame_interval == 0:
            # World view
            world_img = env.get_comparison_camera_image(*world_size)
            world_bgr = cv2.cvtColor(world_img, cv2.COLOR_RGB2BGR)

            # Add labels
            cv2.putText(world_bgr, "CF2X vs MAVIC3 Size Comparison", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cf2x_pos = info["cf2x_pos"]
            mavic3_pos = info["mavic3_pos"]

            cv2.putText(world_bgr, f"CF2X (blue marker): z={cf2x_pos[2]:.2f}m", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
            cv2.putText(world_bgr, f"MAVIC3 (red marker): z={mavic3_pos[2]:.2f}m", (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)

            cv2.putText(world_bgr, f"Grid: 1m squares | Red poles: 1m and 2m height", (10, world_size[1]-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.putText(world_bgr, f"Step: {step}/{total_steps}", (world_size[0]-150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            world_writer.write(world_bgr)

            # Observation view - side by side
            # CF2X obs: (1, H, W) -> (H, W)
            cf2x_gray = env.rgb_cf2x[0].astype(np.uint8)
            cf2x_upscaled = cv2.resize(cf2x_gray, (obs_size[0]//2, obs_size[1]),
                                       interpolation=cv2.INTER_NEAREST)
            cf2x_bgr = cv2.cvtColor(cf2x_upscaled, cv2.COLOR_GRAY2BGR)

            # MAVIC3 obs: (1, H, W) -> (H, W)
            mavic3_gray = env.rgb_mavic3[0].astype(np.uint8)
            mavic3_upscaled = cv2.resize(mavic3_gray, (obs_size[0]//2, obs_size[1]),
                                         interpolation=cv2.INTER_NEAREST)
            mavic3_bgr = cv2.cvtColor(mavic3_upscaled, cv2.COLOR_GRAY2BGR)

            # Combine side by side
            obs_combined = np.hstack([cf2x_bgr, mavic3_bgr])

            # Add labels
            cv2.putText(obs_combined, "CF2X Camera", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
            cv2.putText(obs_combined, "MAVIC3 Camera", (obs_size[0]//2 + 10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)

            # Add dividing line
            cv2.line(obs_combined, (obs_size[0]//2, 0), (obs_size[0]//2, obs_size[1]),
                    (255, 255, 255), 2)

            obs_writer.write(obs_combined)
            frame_count += 1

        if step % (total_steps // 10) == 0:
            print(f"  Progress: {100*step//total_steps}%")

        if done:
            print(f"  Episode ended at step {step}")
            break

    world_writer.release()
    obs_writer.release()
    env.close()

    print(f"\nCompleted! Recorded {frame_count} frames")
    print(f"\nOutput files:")
    print(f"  {world_path}")
    print(f"  {obs_path}")

    return world_path, obs_path


def main():
    parser = argparse.ArgumentParser(description="Compare CF2X and MAVIC3 drone sizes")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--output-dir", type=str, default=".")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    run_size_comparison(
        duration_sec=args.duration,
        gui=args.gui,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
