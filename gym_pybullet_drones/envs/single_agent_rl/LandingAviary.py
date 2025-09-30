import numpy as np
from gym import spaces
import pybullet as p

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, ImageType
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType, BaseSingleAgentAviary
from gym_pybullet_drones.utils.utils import rgb2gray
from gym_pybullet_drones.utils.camera_visibility_checker import CameraVisibilityChecker

import inspect
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


class LandingAviary(BaseSingleAgentAviary):
    """Single agent RL problem: take-off."""
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB_GND_DRAG_DW,
                 freq: int= 240,
                 aggregate_phy_steps: int=10,
                 gui=False,
                 record=False, 
                 obs: ObservationType=ObservationType.RGB,
                 act: ActionType=ActionType.VEL,
                 episode_len_sec: int=18,
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        freq : int, optional
            The frequency (Hz) at which the physics engine steps.
        aggregate_phy_steps : int, optional
            The number of physics steps within one call to `BaseAviary.step()`.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        print("In [gym_pybullet_drones] LandingAviary inits..$")
        super().__init__(drone_model=drone_model,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act,
                         episode_len_sec=episode_len_sec,
                         )

        self.num_step_repeats = 4
        self.IMG_CAPTURE_FREQ = self.AGGR_PHY_STEPS * self.num_step_repeats
        self.IMG_FRAME_PER_SEC = self.SIM_FREQ / self.IMG_CAPTURE_FREQ

        self.desired_landing_altitude = 0.2745
    
    def video_camera(self):
        nth_drone = 0
        gv_pos = np.array(self._get_vehicle_position()[0])
        #### Set target point, camera view and projection matrices #
        target = gv_pos#np.dot(rot_mat, np.array([0, 0, -1000])) + np.array(self.pos[nth_drone, :])

        DRONE_CAM_VIEW = p.computeViewMatrix(cameraEyePosition=self.pos[nth_drone, :] + np.array([0, 0, 0.5]) +np.array([0, 0, self.L]),
                                             cameraTargetPosition=target,
                                             cameraUpVector=[1, 0, 0],
                                             physicsClientId=self.CLIENT
                                             )
        DRONE_CAM_PRO =  p.computeProjectionMatrixFOV(fov=60.0,
                                                      aspect=1.0,
                                                      nearVal=self.L,
                                                      farVal=1000.0,
                                                      physicsClientId=self.CLIENT
                                                      )
        SEG_FLAG = True
        [w, h, rgb, dep, seg] = p.getCameraImage(width=128,
                                                 height=128,
                                                 shadow=0,
                                                 viewMatrix=DRONE_CAM_VIEW,
                                                 projectionMatrix=DRONE_CAM_PRO,
                                                 flags=SEG_FLAG,
                                                 physicsClientId=self.CLIENT
                                                 )

        return rgb
    
    def _computeReward_evil(self):
        """Computes the current reward value.
        Returns
        -------
        float
            The reward.
        """
        vz_max = -0.5
        gv_pos = np.array(self._get_vehicle_position()[0])
        drone_state = self._getDroneStateVector(0)
        drone_pos = drone_state[0:3]
        drone_v = drone_state[10:13]
        error_xy = np.linalg.norm(drone_pos[0:2]-gv_pos[0:2])
        error_z = np.linalg.norm(drone_pos[2]-gv_pos[2])

        flag_land = drone_pos[2] >= 0.05 and p.getContactPoints(bodyA=1, physicsClientId=self.CLIENT) != ()
        flag_crash = drone_pos[2] < 0.05 and p.getContactPoints(bodyA=1, physicsClientId=self.CLIENT) != ()

        error_z = error_z + 4

        theta_l = 20
        theta = np.rad2deg(np.arctan2(abs(error_xy), error_z))
        r = (error_xy ** 2 + error_z ** 2) ** 0.5

        reward_r = 2 - r / 5
        reward_theta = 4 / (1 + np.e ** ((theta - theta_l) / 3)) + 2 / (1 + np.e**((theta-6)/2)) - 2

        if drone_v[2]>0:
            reward_vz = -1
        elif drone_v[2]>-0.3:
            reward_vz = -0.3
        elif drone_v[2]>-0.6:
            reward_vz = 1
        elif drone_v[2]>-0.8:
            reward_vz = -0.8
        elif np.linalg.norm(drone_v)/self.SPEED_LIMIT[2] > 1.2:
            reward_vz = -0.8
        else:
            reward_vz = -1.5


        reward = reward_r + reward_theta + reward_vz


        if flag_crash:
            print('crashed!')
            reward = -300

        if flag_land:
            print('landed!')
            reward = 5000 * np.e ** (-abs(drone_v[2] + 0.5))

        return reward

    def _computeReward(self):
        reward = self._computeReward_good()
        return reward

    def _computeReward_good(self):
        """Computes the current reward value.
        Returns
        -------
        float
            The reward.
        """
        lambda_error = 1/3
        desired_z_velocity = -0.5
        #eventually it will become speed of ground vehicle
        desired_xy_velocity = 0.0
        alpha = 30
        UGV_pos = np.array(self._get_vehicle_position()[0])
        UGV_vel = self._get_vehicle_velocity()
        drone_state = self._getDroneStateVector(0)
        drone_position = drone_state[0:3]
        drone_velocity = drone_state[10:13]
        velocity_error = np.linalg.norm(drone_velocity)
        #velocity_reward = velocity_error
        #distance_xy = np.linalg.norm(drone_position[0:2]-UGV_pos[0:2])
        #distance_z = np.linalg.norm(drone_position[2]-UGV_pos[2])
        #distance_reward = (alpha*distance_xy+beta*distance_z)/10
        #combined_reward = -(gamma*distance_reward**2+zeta*velocity_error**2)
        position_errors = np.abs(drone_position - UGV_pos)
        distance_xy = np.linalg.norm(drone_position[0:2]-UGV_pos[0:2])
        distance_z = np.linalg.norm(drone_position[2:3]-UGV_pos[2:3])
        velocity_z_flag = (0 > drone_velocity[2]) * (drone_velocity[2] > desired_z_velocity)
        reward_z_velocity = (alpha**(drone_velocity[2]/desired_z_velocity) -1)/(alpha -1)
        angle = np.rad2deg(np.arctan2(distance_xy,distance_z))
        #punishment for excessive z velocity
        if velocity_z_flag == False:
            if drone_velocity[2] < desired_z_velocity:
                reward_z_velocity = -0.01#-abs(drone_velocity[2]/self.SPEED_LIMIT[2])**2
            else:
                reward_z_velocity = -0.1#- 10*drone_velocity[2]
            if abs(drone_velocity[2])/self.SPEED_LIMIT[2] > 1.1:
                reward_z_velocity = 0#reward_z_velocity -5
        #reward_xy_velocity = np.sum(-np.abs(drone_velocity[0:2]- desired_xy_velocity))
        if distance_xy < 10:
            normalized_distance_xy = 0.1*(10 - distance_xy)
            reward_xy = (30**normalized_distance_xy -1)/(30 -1)
        else:
            reward_xy = 0 #-distance_xy
        if distance_z < 10:
            normalized_distance_z = 0.1*(10-distance_z)
            reward_z = (30**normalized_distance_z -1)/(30 -1)
        else:
            reward_z = 0
        combined_reward = 0.6*reward_xy + 1.0*reward_z_velocity#+ 0.2*reward_z + reward_z_velocity #np.tanh(reward_z_velocity) #+ reward_xy_velocity
        #print(distance_xy)
        #combined_reward = np.sum(combined_reward)
        #if combined_reward < 0:
        #    print(drone_velocity)
        #    exit()
        if drone_position[2] >= self.desired_landing_altitude and p.getContactPoints(bodyA=1, physicsClientId=self.CLIENT) != ():
            print('landed!')
            combined_reward =  140 + combined_reward
        elif drone_position[2]  < self.desired_landing_altitude and p.getContactPoints(bodyA=1, physicsClientId=self.CLIENT) != ():
            print('crashed!')
            combined_reward = -1 #normalized_distance_xy * 10 #0#5*distance_xy + combined_reward
        else:
            combined_reward =  combined_reward
        distance_x = np.abs(drone_position[0]-UGV_pos[0])
        distance_y = np.abs(drone_position[1]-UGV_pos[1])
        if np.abs(angle) > 30 and (distance_y > 0.8 and distance_x > 0.8):
            combined_reward = -0.01
        #print(combined_reward)
        #print('z velocity reward')
        #print(reward_z_velocity)
        #print('z distance reward')
        #print(reward_z)
        #print('xy error reward')
        #print(reward_xy)
        #print('combined reward')
        #print(combined_reward)
        return combined_reward

    def _computeDone(self):
        if p.getContactPoints(bodyA=1, physicsClientId=self.CLIENT) != ():
            drone_altitude = self._getDroneStateVector(0)[2]
            if drone_altitude >= 0.2745:
                print('_computeDone: Landed!')
            else:
                print('_computeDone: Crashed!')
            return True
        if self.step_counter >= self.EPISODE_LEN_SEC * self.SIM_FREQ:
            print("_computeDone: TimeOut!")
            return True
        else:
            return False

    
    def _computeInfo(self):
        """Computes the current info dict(s).
        """
        drone_state = self._getDroneStateVector(0)
        drone_position = drone_state[0:3]
        UGV_pos = np.array(self._get_vehicle_position()[0])
        x_pos_error = np.linalg.norm(drone_position[0]-UGV_pos[0])
        y_pos_error = np.linalg.norm(drone_position[1]-UGV_pos[1])
        if drone_position[2] >= self.desired_landing_altitude and p.getContactPoints(bodyA=1, physicsClientId=self.CLIENT) != ():
            landing_flag = True
        else:
            landing_flag = False
        #episode end returns true when landing ends up on the ground i.e. the episode should truely finish
        if p.getContactPoints(bodyA=1, physicsClientId=self.CLIENT) != ():
            episode_end = True
        else:
            episode_end = False
        return {"landing": landing_flag,
                "episode end flag": episode_end,
                "x error": x_pos_error,
                "y error": y_pos_error,
                # "drone_state": [self.pos[0], self.rpy[0], self.quat[0]],
                # "GV_state": UGV_pos
                }

    def _resetPosition(self):
        PYB_CLIENT = self.getPyBulletClient()
        DRONE_IDS = self.getDroneIds()
        p.resetBasePositionAndOrientation(DRONE_IDS[0],
                                  [0, 0, 1],
                                  p.getQuaternionFromEuler([0, 0, 0]),
                                  physicsClientId= PYB_CLIENT
                                  )
        p.resetBaseVelocity(DRONE_IDS[0],
                    [0, 0, 0],
                    physicsClientId= PYB_CLIENT
                    )

    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.

        """
        MAX_LIN_VEL_XY = 3 
        MAX_LIN_VEL_Z = 1

        MAX_XY = MAX_LIN_VEL_XY*self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z*self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi # Full range

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        if self.GUI:
            self._clipAndNormalizeStateWarning(state,
                                               clipped_pos_xy,
                                               clipped_pos_z,
                                               clipped_rp,
                                               clipped_vel_xy,
                                               clipped_vel_z
                                               )

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = state[13:16]/np.linalg.norm(state[13:16]) if np.linalg.norm(state[13:16]) != 0 else state[13:16]

        norm_and_clipped = np.hstack([normalized_pos_xy,
                                      normalized_pos_z,
                                      state[3:7],
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      state[16:20]
                                      ]).reshape(20,)
        return norm_and_clipped
    
    def _clipAndNormalizeStateWarning(self,
                                      state,
                                      clipped_pos_xy,
                                      clipped_pos_z,
                                      clipped_rp,
                                      clipped_vel_xy,
                                      clipped_vel_z,
                                      ):
        """Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if values in a state vector is out of the clipping range.
        
        """
        if not(clipped_pos_xy == np.array(state[0:2])).all():
            print("[WARNING] it", self.step_counter, "in TakeoffAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0], state[1]))
        if not(clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter, "in TakeoffAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not(clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter, "in TakeoffAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7], state[8]))
        if not(clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter, "in TakeoffAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10], state[11]))
        if not(clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter, "in TakeoffAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))
    # for cross compatibility with dm gym


class LandingGimbalAviary(LandingAviary):
    """Single agent RL: LandingAviary with gimbal control."""
    """
    Todos
    - [o] Expand the action space to include gimbal control
    - [o] Expand the observation space to include gimbal angles
    - [o] Implement the gimbal control logic in the step or get_obs 
    - [o] Design a reward function to keep the gimbal pointing the ground vehicle
    - [o] Add a visibility checker to see if the ground vehicle is in the camera's FOV (margin included)
    - [o] Check if IMG_CAPTURE_FREQ is properly set !!!!!!! Zebra
    """
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB_GND_DRAG_DW,
                 freq: int=240,
                 aggregate_phy_steps: int=10,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.RGB,
                 act: ActionType=ActionType.VEL,
                 episode_len_sec: int=18,
                 ):
        """Initialization of a single agent RL environment with gimbal control."""
        print("In [gym_pybullet_drones] LandingGimbalAviary inits..$")

        self._cam_dir_local = np.array([0.0, 0.0, -1.0], dtype=np.float64)  # camera forward (-z)
        self._cam_up_local  = np.array([1.0, 0.0, 0.0], dtype=np.float64)   # camera up (+x)
        self._cam_offset_local = np.array([0.0, 0.0, -0.0034], dtype=np.float64)  # 아래로 약간 오프셋

        self._cam_fwd_world = None
        self._cam_up_world = None
        self._cam_pos_world = None

        # FOV/프로젝션 (오라클과 동일하게)
        self._fov_deg = 85.7
        self._near = 0.03
        self._far = 100.0
        self._aspect = 1.0

        # Angles: [pitch,  roll,  yaw]
        # gimbal_target: target gimbal angles in normalized [-1, 1] range
        self.gimbal_target = None  # action[3:]
        self.initial_gimbal_target = np.array([0.0, 0, 0.0])  # camera pointing down, no roll, no yaw

        # Gimbal specs
        eps = 1e-6
        self.gimbal_angle_ranges = np.array([       # (3,2)
            [-(5*np.pi/19)+0.004  +eps, (5*np.pi/19)-0.004 -eps],  # Pitch : +-50-ish degrees (no horizon at hovering)
            [-np.pi/4.                , np.pi/4                ],  # Roll  :  -45   to  45     degrees
            [-np.pi/2             +eps, np.pi/2            -eps],  # Yaw   :  -90   to  90     degrees
        ], dtype=np.float32)  # Ranges for gimbal angles in radians

        # gimbal_state_quat: current gimbal angles in quaternion format
        self.gimbal_state_quat = None

        # Visibility checker
        margin_deg_in_checker = 2.0  # margin to avoid checking at the very edge of FOV
        margin_deg_in_checker = np.clip(margin_deg_in_checker, 0.0, min(self._fov_deg / 2, self._fov_deg / 2 * self._aspect) - 1e-2)
        fov_deg_in_checker = self._fov_deg - (2 * margin_deg_in_checker)
        self.min_viz_ratio_in_checker = 0.0014  # minimum visible ratio to consider the target visible 3*3 in 80*80
        self.min_viz_ratio_in_checker = np.clip(self.min_viz_ratio_in_checker, 0.0, 1.0)
        self.visibility_checker = CameraVisibilityChecker(fov_deg=fov_deg_in_checker, aspect=self._aspect)

        super().__init__(drone_model=drone_model,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act,
                         episode_len_sec=episode_len_sec)

    def _observationSpace(self):
        if self.OBS_TYPE == ObservationType.RGB:
            return spaces.Box(low=0, high=255, shape=(self.IMG_RES[1], self.IMG_RES[0], 4), dtype=np.uint8)
        else:
            print(f"[ERROR] in {self.__class__.__name__}.{inspect.currentframe().f_code.co_name}()")

    def _actionSpace(self):
        if self.ACT_TYPE == ActionType.VEL:
            size = 5
            return spaces.Box(low=-1*np.ones(size), high=np.ones(size), dtype=np.float32)
        else:
            print(f"[ERROR] in {self.__class__.__name__}.{inspect.currentframe().f_code.co_name}()")
            exit()

    def _computeObs(self, done = False):
        if self.OBS_TYPE == ObservationType.RGB:
            if self.step_counter%self.IMG_CAPTURE_FREQ == 0 or done == True:
                self.rgb, _, _ = self._getDroneImages(0, segmentation=False)
                if self.RECORD:
                    self._exportImage(img_type=ImageType.RGB,
                                      img_input=self.rgb,
                                      path=self.ONBOARD_IMG_PATH,
                                      frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ)
                                      )
                self.rgb = rgb2gray(self.rgb)[None,:]
            return self.rgb
        else:
            print(f"[ERROR] in {self.__class__.__name__}.{inspect.currentframe().f_code.co_name}()")

    def _rad_to_norm(self, angles_rad):
        """angles_rad shape (3,): [pitch(rad), roll(rad), yaw(rad)] -> [-1,1]^3"""
        lo = self.gimbal_angle_ranges[:, 0]
        hi = self.gimbal_angle_ranges[:, 1]
        return 2.0 * (np.asarray(angles_rad) - lo) / (hi - lo) - 1.0

    def _norm_to_rad(self, angles_norm):
        """[-1,1]^3 -> radians (3,) in order [pitch, roll, yaw]"""
        lo = self.gimbal_angle_ranges[:, 0]
        hi = self.gimbal_angle_ranges[:, 1]
        return lo + 0.5 * (np.asarray(angles_norm) + 1.0) * (hi - lo)

    def _gimbal_rot_from_angles(self, pitch_rad, roll_rad, yaw_rad):
        """R_local = Rz(yaw) @ Ry(pitch); roll은 현재 0으로 고정해 사용"""
        yaw_rot = np.array([
            [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
            [np.sin(yaw_rad), np.cos(yaw_rad), 0],
            [0, 0, 1]
        ], dtype=np.float32)
        pitch_rot = np.array([
            [np.cos(pitch_rad), 0, -np.sin(pitch_rad)],
            [0, 1, 0],
            [np.sin(pitch_rad), 0, np.cos(pitch_rad)]
        ], dtype=np.float32)
        # roll은 0으로 두므로 별도 회전 불가피시 np.eye 사용
        return yaw_rot @ pitch_rot

    def _getDroneImages(self, nth_drone, segmentation: bool = True):
        """
        - self.gimbal_target([-1,1]^3) -> (pitch,roll,yaw)[rad] -> R_local = Rz@Ry
        - world forward/up를 만든 후 PyBullet view/projection으로 렌더
        - self.gimbal_state_quat은 실제 사용한 R_local로부터 생성(정합 보장)
        """
        if self.gimbal_target is None:
            raise ValueError("gimbal_target is not set. Call reset() or step() first.")
        if self.IMG_RES is None:
            print(f"[ERROR] in {self.__class__.__name__}.{inspect.currentframe().f_code.co_name}(), "
                  f"remember to set self.IMG_RES to np.array([width, height])")
            exit()

        # 1) 정규화 -> 라디안
        pitch_rad, roll_rad, yaw_rad = self._norm_to_rad(self.gimbal_target)

        # 2) 드론 자세
        drone_pos = np.array(p.getBasePositionAndOrientation(self.DRONE_IDS[0], physicsClientId=self.CLIENT)[0])
        drone_quat = np.array(p.getBasePositionAndOrientation(self.DRONE_IDS[0], physicsClientId=self.CLIENT)[1])
        rot_drone = np.array(p.getMatrixFromQuaternion(drone_quat)).reshape(3, 3)

        # 3) 짐벌 로컬 회전(Rz @ Ry)
        gimbal_rot_local = self._gimbal_rot_from_angles(pitch_rad, roll_rad, yaw_rad)
        self.gimbal_state_quat = R.from_matrix(gimbal_rot_local).as_quat()

        # 4) 카메라 월드 기준 방향/업벡터
        cam_pos_world = np.array(drone_pos) + rot_drone @ self._cam_offset_local
        cam_dir_world = rot_drone @ (gimbal_rot_local @ self._cam_dir_local)
        cam_up_world = rot_drone @ (gimbal_rot_local @ self._cam_up_local)
        self._cam_fwd_world = cam_dir_world
        self._cam_up_world = cam_up_world
        self._cam_pos_world = cam_pos_world

        target_world = cam_pos_world + cam_dir_world * 1000.0

        DRONE_CAM_VIEW = p.computeViewMatrix(
            cameraEyePosition=cam_pos_world,
            cameraTargetPosition=target_world,
            cameraUpVector=cam_up_world,
            physicsClientId=self.CLIENT
        )
        DRONE_CAM_PRO = p.computeProjectionMatrixFOV(
            fov=self._fov_deg,
            aspect=self._aspect,
            nearVal=self._near,
            farVal=self._far
        )

        SEG_FLAG = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX if segmentation else p.ER_NO_SEGMENTATION_MASK

        [_, _, rgb, _, _] = p.getCameraImage(
            width=self.IMG_RES[0],
            height=self.IMG_RES[1],
            shadow=1,
            viewMatrix=DRONE_CAM_VIEW,
            projectionMatrix=DRONE_CAM_PRO,
            flags=SEG_FLAG,
            physicsClientId=self.CLIENT
        )

        if self.distortion:
            raise NotImplementedError("Distortion is not implemented for LandingGimbalAviary.")

        rgb = np.moveaxis(rgb, -1, 0)  # (C,H,W)
        return rgb, None, None

    def compute_oracle_gimbal(self):
        """
        Compute the ideal (oracle) gimbal orientation that points the camera optical axis at the helipad target.

        Frames & axes:
          - World: PyBullet Z-up, right-handed.
          - Drone/body: rotation to world is rot_drone; to convert world→drone, use rot_drone.T.
          - Camera local (pre-gimbal, fixed): forward = -Z, up = +X, right = +Y.
          - Gimbal rotation: R_gimbal = Rz(yaw) @ Ry(pitch), applied to camera-local axes to yield camera in drone frame.
            Roll is unused (kept 0).

        Output:
          - 'quat': camera-local→drone rotation as quaternion [x,y,z,w].
          - 'angles_rad': np.array([pitch, roll(=0), yaw]) in radians, clipped to self.gimbal_angle_ranges.
          - 'angles_norm': normalized angles in [-1,1]^3 via _rad_to_norm.
          - 'oracle_forward_local': desired camera forward in drone frame (unit), i.e., where (-Z_cam) should point.

        Guarantees:
          - Uses atan2-based extraction; numerically stable.
          - Handles near-singular cases (target nearly collinear with up_ref).
          - Quaternion and vectors are normalized.
        """
        # 1) Positions/orientation
        UGV_pos = np.array(self._get_vehicle_position()[0], dtype=np.float64)  # helipad top-center
        drone_pos, drone_quat = p.getBasePositionAndOrientation(self.DRONE_IDS[0], physicsClientId=self.CLIENT)
        drone_pos, drone_quat = np.array(drone_pos, dtype=np.float64), np.array(drone_quat, dtype=np.float64)
        rot_drone = np.array(p.getMatrixFromQuaternion(drone_quat)).reshape(3, 3).astype(np.float64)

        cam_pos_world = drone_pos + rot_drone @ self._cam_offset_local

        # 2) Desired forward in world (from camera to target)
        to_target_world = UGV_pos - cam_pos_world
        dist = float(np.linalg.norm(to_target_world))
        if dist < 1e-9:
            # Degenerate: keep current viewing direction
            f_world = rot_drone @ self._cam_dir_local
        else:
            f_world = to_target_world / dist

        # 3) Choose a robust world up reference and build orthonormal frame about f_world
        up_ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(np.dot(f_world, up_ref)) > 0.95:
            up_ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        right_world = np.cross(up_ref, f_world)
        n = np.linalg.norm(right_world)
        right_world = right_world / (n + 1e-12)

        up_world = np.cross(f_world, right_world)
        up_world = up_world / (np.linalg.norm(up_world) + 1e-12)

        # 4) Bring basis into drone frame
        f_local = rot_drone.T @ f_world
        f_local = f_local / (np.linalg.norm(f_local) + 1e-12)

        u_local = rot_drone.T @ up_world
        u_local = u_local / (np.linalg.norm(u_local) + 1e-12)

        r_local = np.cross(u_local, f_local)
        r_local = r_local / (np.linalg.norm(r_local) + 1e-12)

        # Re-orthogonalize u_local to avoid drift
        u_local = np.cross(f_local, r_local)
        u_local = u_local / (np.linalg.norm(u_local) + 1e-12)

        # 5) Rotation camera-local→drone: columns are images of cam axes (X=up, Y=right, Z=back)
        #    cam_x(+X up) -> u_local, cam_y(+Y right) -> r_local, cam_z(+Z back) -> -f_local (since forward is -Z)
        R_local = np.column_stack([u_local, r_local, -f_local])
        # Numerical cleanup
        U, _, Vt = np.linalg.svd(R_local)
        R_local = U @ Vt  # nearest proper rotation
        quat = R.from_matrix(R_local).as_quat()  # [x,y,z,w]

        # 6) Extract yaw/pitch for Rz(yaw) @ Ry(pitch) such that R_gimbal @ (-Z) = f_local
        fx, fy, fz = f_local.astype(np.float64)
        # Clamp for safety
        fx = float(np.clip(fx, -1.0, 1.0))
        fy = float(np.clip(fy, -1.0, 1.0))
        fz = float(np.clip(fz, -1.0, 1.0))

        yaw = float(np.arctan2(fy, fx))  # [-pi, pi]
        s = float(np.sqrt(max(fx * fx + fy * fy, 0.0)))
        pitch = float(np.arctan2(s, -fz))  # in [0, pi]

        # 7) Roll fixed to 0; clip to gimbal limits then normalize
        roll = 0.0
        angles_rad = np.array([pitch, roll, yaw], dtype=np.float32)

        # Clip to configured ranges to keep oracle within actuator limits
        lo = self.gimbal_angle_ranges[:, 0].astype(np.float64)
        hi = self.gimbal_angle_ranges[:, 1].astype(np.float64)
        # Wrap yaw to (-pi, pi]
        yaw_wrapped = (angles_rad[2] + np.pi) % (2 * np.pi) - np.pi
        angles_rad = np.array([angles_rad[0], angles_rad[1], yaw_wrapped], dtype=np.float64)
        # Clip
        angles_rad = np.minimum(np.maximum(angles_rad, lo), hi).astype(np.float32)

        angles_norm = self._rad_to_norm(angles_rad)

        return {
            'quat': quat,
            'angles_rad': angles_rad,
            'angles_norm': angles_norm,
            'oracle_forward_local': f_local
        }

    def is_target_visible(self, drone_position, drone_quaternion, pad_position):

        # Pad geometry
        # pad_pos = np.array(self._get_vehicle_position()[0])
        # pad_hf = 0.2875 #2875gives0.1margin  # Helipad - visual shape: (0.675, 0.675, 0), collision shape: (0.5, 0.5, 0)
        pad_hf = 0.3375
        pad_corners = np.array([
            [ pad_hf,  pad_hf, 0.0],
            [ pad_hf, -pad_hf, 0.0],
            [-pad_hf, -pad_hf, 0.0],
            [-pad_hf,  pad_hf, 0.0],
        ]) + pad_position

        # Drone rotation
        rot_drone = np.array(p.getMatrixFromQuaternion(drone_quaternion)).reshape(3, 3).astype(np.float64)

        # Camera
        pitch_rad, _, yaw_rad = self._norm_to_rad(self.gimbal_target)
        R_local = self._gimbal_rot_from_angles(pitch_rad, 0.0, yaw_rad)
        cam_pos_world = drone_position + rot_drone @ self._cam_offset_local
        cam_fwd_world = rot_drone @ (R_local @ self._cam_dir_local)
        cam_up_world = rot_drone @ (R_local @ self._cam_up_local)

        # Visibility
        visibility = self.visibility_checker.is_visible(
            rect_xyz=pad_corners,
            cam_pos=cam_pos_world,
            cam_forward=cam_fwd_world,
            cam_up=cam_up_world,
            min_fraction=self.min_viz_ratio_in_checker,
        )
        return visibility

    def reset(self):
        self.gimbal_target = self.initial_gimbal_target
        return super().reset()

    def step(self, action):
        # ★ gimbal action 반영: pitch=action[3], yaw=action[4], roll은 0 고정
        self.gimbal_target = np.array([action[3], 0.0, action[4]], dtype=np.float32)
        if not np.all(np.abs(self.gimbal_target) <= 1.0 + 1e-9):
            raise ValueError(f"Gimbal target angles must be in [-1, 1], got {self.gimbal_target}")
        vel_cmd = action[0:3]
        return super().step(vel_cmd)

    def _compute_hv_rewards(self, drone_position, drone_velocity, pad_position):
        desired_z_velocity = -0.5
        alpha = 30
        distance_xy = np.linalg.norm(drone_position[0:2] - pad_position[0:2])
        velocity_z_flag = (0 > drone_velocity[2]) * (drone_velocity[2] > desired_z_velocity)
        reward_z_velocity = (alpha ** (drone_velocity[2] / desired_z_velocity) - 1) / (alpha - 1)
        # punishment for excessive z velocity
        if velocity_z_flag == False:
            if drone_velocity[2] < desired_z_velocity:
                reward_z_velocity = -0.01  # -abs(drone_velocity[2]/self.SPEED_LIMIT[2])**2
            else:
                reward_z_velocity = -0.1  # - 10*drone_velocity[2]
            if abs(drone_velocity[2]) / self.SPEED_LIMIT[2] > 1.1:
                reward_z_velocity = 0  # reward_z_velocity -5
        if distance_xy < 8:
            normalized_distance_xy = 0.125 * (8 - distance_xy)
            reward_xy = (30 ** normalized_distance_xy - 1) / (30 - 1)
        else:
            reward_xy = 0  # -distance_xy
        combined_reward = 0.56 * reward_xy + 1.0 * reward_z_velocity

        return combined_reward

    def _forward_from_norm_angles(self, angles_norm: np.ndarray) -> np.ndarray:
        """현재 규약(Rz(yaw) @ Ry(pitch))에서 카메라 로컬 forward(+x)가
           드론 로컬 좌표에서 어디를 가리키는지(단위벡터) 반환."""
        pitch_rad, _, yaw_rad = self._norm_to_rad(angles_norm)
        Rg = self._gimbal_rot_from_angles(pitch_rad, 0.0, yaw_rad)  # (3,3) in camera-local basis
        f_local = Rg @ self._cam_dir_local  # camera-local forward mapped in camera local (여기선 같지만 명시)
        # 위 f_local은 '카메라 로컬 기준' 벡터. 보상에서 오라클과 같은 기준이면 충분합니다.
        # 굳이 드론 로컬로 옮기고 싶다면: f_drone = f_local (카메라축 자체가 드론 로컬 축에 정의됨)
        return f_local / (np.linalg.norm(f_local) + 1e-12)

    def _compute_viz_reward(self, is_visible):
        if is_visible:
            # Get oracle
            oracle = self.compute_oracle_gimbal()
            R_oracle = R.from_quat(oracle['quat']).as_matrix()  # camera-local rotation in drone frame
            f_oracle = R_oracle @ self._cam_dir_local
            f_oracle /= (np.linalg.norm(f_oracle) + 1e-12)

            # Get gimbal forward (현재 액션/타겟에서 직접 계산)
            f_curr = self._forward_from_norm_angles(self.gimbal_target)

            # Compute cosine alignment
            cos_align = float(np.clip(np.dot(f_curr, f_oracle), -1.0, 1.0))
            cos_align = cos_align**5  # let's make it sharper with cheaper computation
            # -1(정반대), 0(직교), 1(정렬)
            # print(f"cos_align: {cos_align:.4f}")
            # print(f"    f_curr:   {f_curr}")
            # print(f"    f_oracle: {f_oracle}")
            # print(f"    ")
            return 0.1 * cos_align  # 스케일은 필요에 따라 튜닝
        else:
            return -0.025

    def _computeReward(self):
        # 0. 드론 상태
        drone_pos, drone_quat = p.getBasePositionAndOrientation(self.DRONE_IDS[0], physicsClientId=self.CLIENT)
        drone_pos, drone_quat = np.array(drone_pos, dtype=np.float64), np.array(drone_quat, dtype=np.float64)
        drone_vel = np.array(p.getBaseVelocity(self.DRONE_IDS[0], physicsClientId=self.CLIENT)[0], dtype=np.float64)
        pad_pos = np.array(self._get_vehicle_position()[0])
        drone_has_touched = p.getContactPoints(bodyA=1, physicsClientId=self.CLIENT) != ()

        # 1. 착륙, 충돌의 경우 큰 보상/페널티
        if drone_has_touched:
            landed, crashed = (True, False) if drone_pos[2] >= self.desired_landing_altitude else (False, True)
            if crashed:
                return -1.0
            elif landed:
                # ONLY FOR DEBUGGING REMOVE the below LATER !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                if not self.is_target_visible(drone_pos, drone_quat, pad_pos):
                    print("Check self.rgb, drone pos, pad pos etc !!")
                    print("Drone has landed but target not visible!!!")
                # ONLY FOR DEBUGGING REMOVE the above LATER !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # No viz reward unfortunately due to the bug in BaseAviary making the drone go below the pad
                return 140.0 + self._compute_hv_rewards(drone_pos, drone_vel, pad_pos)  # + self._compute_viz_reward(is_visible=True)
            else:
                raise ValueError("Logic error in landed/crashed check.")

        # 2. 아무 접촉이 없이 진행  중인 경우
        if self.is_target_visible(drone_pos, drone_quat, pad_pos):
            return self._compute_hv_rewards(drone_pos, drone_vel, pad_pos) + self._compute_viz_reward(is_visible=True)
        else:
            return self._compute_viz_reward(is_visible=False)


def show_rgb(rgb):
    # rgb shape: (C,H,W) with RGBA; matplotlib은 (H,W,3) RGB를 선호
    if rgb.shape[0] == 4:
        img = np.moveaxis(rgb[:3, :, :], 0, -1)  # drop alpha
    elif rgb.shape[0] == 3:
        img = np.moveaxis(rgb, 0, -1)
    else:
        # grayscale (1,H,W) 인 경우
        img = np.moveaxis(np.repeat(rgb, 3, axis=0), 0, -1)
    plt.imshow(img.astype(np.uint8))
    plt.title("env.rgb")
    plt.axis('off')
    plt.show()


def main():
    env = LandingGimbalAviary(episode_len_sec=6, )

    obs = env.reset()
    print("reset gimbal_state_quat:", getattr(env, "gimbal_state_quat", None))
    print(f"  Target Visibility: {env.is_target_visible()}")
    show_rgb(env.rgb)

    # 몇 스텝 전진/무작위 속도 명령으로 진행해 보며 계속 패드를 가운데에 두는지 확인
    for t in range(128):
        # 예: 전진살짝 + 고도유지
        action = np.array([0.0, 0.0, 0.01, 0, (-1+(t/18))], dtype=np.float32)
        obs, rew, done, info = env.step(action)

        # 쿼터니안 → 오일러 (yaw, pitch, roll)
        quat = env.gimbal_state_quat
        euler = R.from_quat(quat).as_euler('ZYX', degrees=True)
        yaw, pitch, roll = euler  # 'zyx' 이므로 [yaw, pitch, roll] 순서

        print(f"step {t+1} gimbal_euler [deg]: yaw={yaw:.2f}, pitch={pitch:.2f}, roll={roll:.2f}; quat: {quat}")
        print(f"  Target Visibility: {env.is_target_visible()}; Reward: {rew:.3f}")

        # print(f"step {t+1}: step counter={env.step_counter}")
        # show_rgb(env.rgb)

        if t % 6 == 0:
            show_rgb(env.rgb)
            # print(env.is_target_visible())
        if done:
            break

    env.close()


if __name__ == "__main__":
    main()
