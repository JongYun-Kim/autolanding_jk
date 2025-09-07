import numpy as np
from gym import spaces

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary, ImageType
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType, BaseSingleAgentAviary
import pybullet as p
from gym_pybullet_drones.utils.specs import BoundedArray

from gym_pybullet_drones.utils.utils import rgb2gray
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
        if drone_position[2] >= 0.275 and p.getContactPoints(bodyA=1, physicsClientId=self.CLIENT) != ():
            print('landed!')
            combined_reward =  140 + combined_reward
        elif drone_position[2]  < 0.275 and p.getContactPoints(bodyA=1, physicsClientId=self.CLIENT) != ():
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
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        if p.getContactPoints(bodyA=1, physicsClientId=self.CLIENT) != ():
            drone_altitude = self._getDroneStateVector(0)[2]
            if drone_altitude >= 0.275:
                print('_computeDone: Landed!')
            else:
                print('_computeDone: Crashed!')
            return True
        if self.step_counter/self.SIM_FREQ > self.EPISODE_LEN_SEC:
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
        if drone_position[2] >= 0.275 and p.getContactPoints(bodyA=1, physicsClientId=self.CLIENT) != ():
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


class ToddlerLandingAviary(LandingAviary):
    """
    1. New reward functions (function components)
    2. Curriculum(s)
    3. Curriculum handler
    4. Difficulty setter

    Difficulty
    - ddd
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
                 difficulty: int=1,
                 ):
        self.difficulty = difficulty  # Use self.set_difficulty to set difficulty in episode handlers

        super().__init__(
            drone_model=drone_model,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            freq=freq,
            aggregate_phy_steps=aggregate_phy_steps,
            gui=gui,
            record=record,
            obs=obs,
            act=act
        )

    def set_difficulty(self, new_diff: int):
        """This setter is provided to change the difficulty during the training."""
        self.difficulty = new_diff

    def _computeReward(self):
        if self.difficulty == 1:
            return self._compute_reward_vis_only()
        elif self.difficulty == 2:
            return self._compute_reward_vis_n_lc()
        elif self.difficulty == 3:
            return self._compute_reward_full()
        else:
            raise ValueError(f"Difficulty {self.difficulty} not supported.")

    def _compute_reward_vis_only(self):
        # Visibility only
        UGV_pos = np.array(self._get_vehicle_position()[0])
        drone_state = self._getDroneStateVector(0)
        drone_position = drone_state[0:3]
        distance_xy = np.linalg.norm(drone_position[0:2]-UGV_pos[0:2])
        distance_z = np.linalg.norm(drone_position[2:3]-UGV_pos[2:3])
        angle = np.rad2deg(np.arctan2(distance_xy,distance_z))

        combined_reward = 0
        distance_x = np.abs(drone_position[0]-UGV_pos[0])
        distance_y = np.abs(drone_position[1]-UGV_pos[1])
        if np.abs(angle) > 30 and (distance_y > 0.8 and distance_x > 0.8):
            combined_reward = -0.01
        return combined_reward

    def _compute_reward_vis_n_lc(self):
        # Visibility + landing/crash(only-if visible)
        UGV_pos = np.array(self._get_vehicle_position()[0])
        drone_state = self._getDroneStateVector(0)
        drone_position = drone_state[0:3]
        distance_xy = np.linalg.norm(drone_position[0:2]-UGV_pos[0:2])
        distance_z = np.linalg.norm(drone_position[2:3]-UGV_pos[2:3])
        angle = np.rad2deg(np.arctan2(distance_xy,distance_z))

        combined_reward = 0
        if drone_position[2] >= 0.275 and p.getContactPoints(bodyA=1, physicsClientId=self.CLIENT) != ():
            # print('landed!')
            combined_reward =  140 + combined_reward
        elif drone_position[2]  < 0.275 and p.getContactPoints(bodyA=1, physicsClientId=self.CLIENT) != ():
            # print('crashed!')
            combined_reward = -1
        else:
            combined_reward =  combined_reward
        distance_x = np.abs(drone_position[0]-UGV_pos[0])
        distance_y = np.abs(drone_position[1]-UGV_pos[1])
        if np.abs(angle) > 30 and (distance_y > 0.8 and distance_x > 0.8):
            combined_reward = -0.01
        return combined_reward

    def _compute_reward_full(self):
        lambda_error = 1/3
        desired_z_velocity = -0.5
        desired_xy_velocity = 0.0
        alpha = 30
        UGV_pos = np.array(self._get_vehicle_position()[0])
        UGV_vel = self._get_vehicle_velocity()
        drone_state = self._getDroneStateVector(0)
        drone_position = drone_state[0:3]
        drone_velocity = drone_state[10:13]
        velocity_error = np.linalg.norm(drone_velocity)
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
        if drone_position[2] >= 0.275 and p.getContactPoints(bodyA=1, physicsClientId=self.CLIENT) != ():
            # print('landed!')
            combined_reward =  140 + combined_reward
        elif drone_position[2]  < 0.275 and p.getContactPoints(bodyA=1, physicsClientId=self.CLIENT) != ():
            # print('crashed!')
            combined_reward = -1 #normalized_distance_xy * 10 #0#5*distance_xy + combined_reward
        else:
            combined_reward =  combined_reward
        distance_x = np.abs(drone_position[0]-UGV_pos[0])
        distance_y = np.abs(drone_position[1]-UGV_pos[1])
        if np.abs(angle) > 30 and (distance_y > 0.8 and distance_x > 0.8):
            combined_reward = -0.01
        return combined_reward


class ReflectiveToddlerLandingAviary(ToddlerLandingAviary):
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
                 difficulty: int=1,
                 ):
        self.r_list = None

        super().__init__(
            drone_model=drone_model,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            freq=freq,
            aggregate_phy_steps=aggregate_phy_steps,
            gui=gui,
            record=record,
            obs=obs,
            act=act,
            difficulty=difficulty
        )

    def _computeReward(self):
        r_vis_only = self._compute_reward_vis_only()
        r_vis_n_lc = self._compute_reward_vis_n_lc()
        r_full = self._compute_reward_full()

        r_list = [r_vis_only, r_vis_n_lc, r_full]  # Retrieve the rewards for reflective learning in replay buffer
        self.r_list = np.array(r_list)

        # TODO: Check if the replay buffer takes which reward during training: scalar (r_xx) or array (r_list)
        if self.difficulty == 1:
            return r_vis_only
        elif self.difficulty == 2:
            return r_vis_n_lc
        elif self.difficulty == 3:
            return r_full
        else:
            raise ValueError(f"Difficulty {self.difficulty} not supported.")


class ToddlerLandingAviaryS1(ToddlerLandingAviary):

    def __init__(self, drone_model: DroneModel=DroneModel.CF2X, initial_xyzs=None, initial_rpys=None, physics: Physics=Physics.PYB_GND_DRAG_DW, freq: int=240, aggregate_phy_steps: int=10, gui=False, record=False, obs: ObservationType=ObservationType.RGB, act: ActionType=ActionType.VEL,
                 difficulty: int=1,
                 ):
        super().__init__(
            drone_model=drone_model,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            freq=freq,
            aggregate_phy_steps=aggregate_phy_steps,
            gui=gui,
            record=record,
            obs=obs,
            act=act,
            difficulty=difficulty
        )

    def _computeDone(self):
        if p.getContactPoints(bodyA=1, physicsClientId=self.CLIENT) != ():
            drone_altitude = self._getDroneStateVector(0)[2]
            if drone_altitude >= 0.275:
                print('_computeDone: Landed!')
            else:
                print('_computeDone: Crashed!')

            if self.difficulty == 1:
                pass
                # return False  # This MUST be COMMENTED OUT. OTHERWISE, the episode will NEVER END; please allow PASS
            elif self.difficulty == 2:
                pass
                # return False  # This MUST be COMMENTED OUT. OTHERWISE, the episode will NEVER END; please allow PASS
            elif self.difficulty == 3:
                return True
            elif self.difficulty == 4:
                return True
            else:
                raise ValueError(f"Difficulty {self.difficulty} not supported.")

        if self.step_counter/self.SIM_FREQ > self.EPISODE_LEN_SEC:
            print("_computeDone: TimeOut!")
            return True
        else:
            return False

    def _compute_reward_vis_n_lc_no_penalty(self):
        # Visibility + landing/crash(only-if visible)
        UGV_pos = np.array(self._get_vehicle_position()[0])
        drone_state = self._getDroneStateVector(0)
        drone_position = drone_state[0:3]
        distance_xy = np.linalg.norm(drone_position[0:2]-UGV_pos[0:2])
        distance_z = np.linalg.norm(drone_position[2:3]-UGV_pos[2:3])
        angle = np.rad2deg(np.arctan2(distance_xy,distance_z))

        combined_reward = 0
        if drone_position[2] >= 0.275 and p.getContactPoints(bodyA=1, physicsClientId=self.CLIENT) != ():
            # print('landed!')
            combined_reward =  140 + combined_reward
        # elif drone_position[2] < 0.275 and p.getContactPoints(bodyA=1, physicsClientId=self.CLIENT) != ():
        #     print('crashed!')
        #     combined_reward = combined_reward  ## This is the difference!!
        else:
            combined_reward =  combined_reward
        distance_x = np.abs(drone_position[0]-UGV_pos[0])
        distance_y = np.abs(drone_position[1]-UGV_pos[1])
        if np.abs(angle) > 30 and (distance_y > 0.8 and distance_x > 0.8):
            combined_reward = -0.01
        return combined_reward

    def _computeReward(self):
        if self.difficulty == 1:
            return self._compute_reward_vis_only()
        elif self.difficulty == 2:
            return self._compute_reward_vis_n_lc_no_penalty()
        elif self.difficulty == 3:
            return self._compute_reward_vis_n_lc()
        elif self.difficulty == 4:
            return self._compute_reward_full()
        else:
            raise ValueError(f"Difficulty {self.difficulty} not supported.")


class LandingGimbalAviary(LandingAviary):
    """Single agent RL: LandingAviary with gimbal control."""
    """
    Todos
    - [o] Expand the action space to include gimbal control
    - [o] Expand the observation space to include gimbal angles
    - [o] Implement the gimbal control logic in the step or get_obs 
    - [o] Design a reward function to keep the gimbal pointing the ground vehicle
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

        self._cam_dir_local = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # camera forward (+x)
        self._cam_up_local = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # camera up (+z)
        self._cam_offset_local = np.array([0.0, 0.0, -0.05], dtype=np.float32)  # 아래로 5cm

        # FOV/프로젝션 (오라클과 동일하게)
        self._fov_deg = 85.7
        self._near = 0.03
        self._far = 200.0
        self._aspect = 1.0

        # Angles: [pitch,  roll,  yaw]
        # gimbal_target: target gimbal angles in normalized [-1, 1] range
        self.gimbal_target = None  # action[3:]
        self.initial_gimbal_target = np.array([1.0, 0.0, 0.0])  # camera pointing down, no roll, no yaw

        # Gimbal specs
        self.gimbal_angle_ranges = np.array([  # (3,2)
            [0       , np.pi/2],  # Pitch: 0    to 90  degrees
            [-np.pi/4, np.pi/4],  # Roll : -45  to 45  degrees
            [-np.pi  , np.pi  ]   # Yaw  : -180 to 180 degrees
        ], dtype=np.float32)  # Ranges for gimbal angles in radians

        # gimbal_state_quat: current gimbal angles in quaternion format
        self.gimbal_state_quat = None

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
            [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
            [0, 1, 0],
            [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
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
        drone_pos = self.pos[nth_drone, :]
        drone_quat = self.quat[nth_drone, :]
        rot_drone = np.array(p.getMatrixFromQuaternion(drone_quat)).reshape(3, 3)

        # 3) 짐벌 로컬 회전(Rz @ Ry)
        gimbal_rot_local = self._gimbal_rot_from_angles(pitch_rad, roll_rad, yaw_rad)
        self.gimbal_state_quat = R.from_matrix(gimbal_rot_local).as_quat()

        # 4) 카메라 월드 기준 방향/업벡터
        cam_pos_world = np.array(drone_pos) + rot_drone @ self._cam_offset_local
        cam_dir_world = rot_drone @ (gimbal_rot_local @ self._cam_dir_local)
        cam_up_world = rot_drone @ (gimbal_rot_local @ self._cam_up_local)

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
        현재 드론 상태에서 'UGV를 정중앙'에 두도록 하는 오라클 짐벌을 계산만 합니다.
        - env 상태는 바꾸지 않음 (self.gimbal_target 변경 없음)
        - 반환:
          dict {
            'quat': np.ndarray(4,)  # [x,y,z,w] 로컬(드론기준) 카메라 회전
            'angles_rad': np.ndarray(3,)  # [pitch, roll(=0), yaw], radians
            'angles_norm': np.ndarray(3,) # 위 angles_rad 를 [-1,1]^3 범위로 정규화
          }
        """
        # 1) 드론/UGV/카메라 위치/자세
        UGV_pos = np.array(self._get_vehicle_position()[0], dtype=np.float64)
        drone_state = self._getDroneStateVector(0)
        drone_pos = drone_state[0:3].astype(np.float64)
        drone_quat = self.quat[0, :]
        rot_drone = np.array(p.getMatrixFromQuaternion(drone_quat)).reshape(3, 3).astype(np.float64)

        cam_pos_world = drone_pos + rot_drone @ self._cam_offset_local

        # 2) 목표 방향(월드)
        to_target_world = UGV_pos - cam_pos_world
        dist = np.linalg.norm(to_target_world)
        if dist < 1e-9:
            # 거의 같은 위치면 이전 값에 의존해야 하지만, 여기서는 정면으로 가정
            f_world = rot_drone @ self._cam_dir_local
        else:
            f_world = to_target_world / dist

        # 3) 안정적인 월드 업 기준
        up_ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(np.dot(f_world, up_ref)) > 0.95:
            up_ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        # 4) 월드 기준 right/up
        right_world = np.cross(up_ref, f_world)
        right_world /= (np.linalg.norm(right_world) + 1e-12)
        up_world = np.cross(f_world, right_world)
        up_world /= (np.linalg.norm(up_world) + 1e-12)

        # 5) 드론 로컬로 가져와 카메라 로컬 회전행렬(R_local) 구성
        f_local = rot_drone.T @ f_world
        u_local = rot_drone.T @ up_world
        f_local /= (np.linalg.norm(f_local) + 1e-12)
        r_local = np.cross(u_local, f_local)
        r_local /= (np.linalg.norm(r_local) + 1e-12)
        u_local = np.cross(f_local, r_local)
        u_local /= (np.linalg.norm(u_local) + 1e-12)

        R_local = np.column_stack([f_local, r_local, u_local])  # e_x->f, e_y->r, e_z->u
        quat = R.from_matrix(R_local).as_quat()

        # 6) 오라클 규약에서의 (yaw,pitch) 추출, roll=0
        #    forward=+x, right=+y, up=+z 에서:
        #    yaw   = atan2(f_local_y, f_local_x)
        #    pitch = atan2(-f_local_z, sqrt(f_local_x^2 + f_local_y^2))
        fx, fy, fz = f_local
        yaw = float(np.arctan2(fy, fx))
        pitch = float(np.arctan2(-fz, np.sqrt(fx * fx + fy * fy)))
        roll = 0.0

        angles_rad = np.array([pitch, roll, yaw], dtype=np.float32)
        angles_norm = self._rad_to_norm(angles_rad)

        return {'quat': quat, 'angles_rad': angles_rad, 'angles_norm': angles_norm, 'oracle_forward_local': f_local}

    def is_target_visible(self, margin_deg=0.0, return_details=False):
        """
        현재 짐벌 상태/카메라 파라미터에서 UGV가 프레임 내에 있는지 판정.
        - margin_deg: 여유각(양수면 더 엄격한 프레임 내 판정)
        - return_details=True 이면 보조정보(픽셀좌표 등) dict도 반환
        반환:
          - return_details=False: bool
          - return_details=True : (bool, dict)
        """
        # 1) 기하 확보
        UGV_pos = np.array(self._get_vehicle_position()[0], dtype=np.float64)
        drone_pos = self.pos[0, :].astype(np.float64)
        drone_quat = self.quat[0, :]
        rot_drone = np.array(p.getMatrixFromQuaternion(drone_quat)).reshape(3, 3).astype(np.float64)

        # 현재 짐벌 회전
        pitch_rad, roll_rad, yaw_rad = self._norm_to_rad(self.gimbal_target)
        R_local = self._gimbal_rot_from_angles(pitch_rad, roll_rad, yaw_rad)

        cam_pos_world = drone_pos + rot_drone @ self._cam_offset_local
        f_world = rot_drone @ (R_local @ self._cam_dir_local)
        u_world = rot_drone @ (R_local @ self._cam_up_local)
        r_world = np.cross(u_world, f_world)

        to_target = UGV_pos - cam_pos_world
        if np.linalg.norm(to_target) < 1e-9:
            visible = True
            details = {'pixel': (self.IMG_RES[0] // 2, self.IMG_RES[1] // 2)}
            return (visible, details) if return_details else visible

        # 2) 카메라 좌표계(+x fwd, +y right, +z up) 성분
        x_cam = np.dot(to_target, f_world)
        y_cam = np.dot(to_target, r_world)
        z_cam = np.dot(to_target, u_world)

        # 3) FOV 판정
        vfov_half = np.deg2rad(self._fov_deg * 0.5)
        hfov_half = np.arctan(np.tan(vfov_half) * self._aspect)

        # 여유각 적용
        vfov_half_eff = vfov_half - np.deg2rad(margin_deg)
        hfov_half_eff = hfov_half - np.deg2rad(margin_deg)
        vfov_half_eff = max(vfov_half_eff, 1e-6)
        hfov_half_eff = max(hfov_half_eff, 1e-6)

        if x_cam <= 0.0:
            visible = False
            det = {'reason': 'behind camera', 'pixel': None}
            return (visible, det) if return_details else visible

        yaw_offset = np.arctan2(y_cam, x_cam)  # 좌우 각
        pitch_offset = np.arctan2(z_cam, x_cam)  # 상하 각

        visible = (abs(yaw_offset) <= hfov_half_eff) and (abs(pitch_offset) <= vfov_half_eff)

        details = None
        if return_details:
            # 4) 픽셀 좌표 추정 (정확한 pinhole 투영 기반)
            #    u_ndc = (y/x)/tan(hfov_half), v_ndc = (z/x)/tan(vfov_half)  in [-1,1]
            u_ndc = (y_cam / x_cam) / np.tan(hfov_half)
            v_ndc = (z_cam / x_cam) / np.tan(vfov_half)
            # NDC(-1..1) -> pixel(0..W/H); v는 위가 작아야 하므로 반전
            W, H = int(self.IMG_RES[0]), int(self.IMG_RES[1])
            u_px = (u_ndc * 0.5 + 0.5) * W
            v_px = (1.0 - (v_ndc * 0.5 + 0.5)) * H
            details = {
                'pixel': (float(u_px), float(v_px)),
                'yaw_offset_deg': np.rad2deg(yaw_offset),
                'pitch_offset_deg': np.rad2deg(pitch_offset),
                'x_cam_y_z': (float(x_cam), float(y_cam), float(z_cam)),
                'ndc': (float(u_ndc), float(v_ndc))
            }

        return (visible, details) if return_details else visible

    def reset(self):
        self.gimbal_target = self.initial_gimbal_target
        # _ = super().reset()
        # # ★ 초기 gimbal로 한 번 캡처해서 env.rgb를 일치시킴
        # rgb, _, _ = self._getDroneImages(0, segmentation=False)
        # self.rgb = rgb
        # return self.rgb
        return super().reset()

    def step(self, action):
        # ★ gimbal action 반영: pitch=action[3], yaw=action[4], roll은 0 고정
        self.gimbal_target = np.array([action[3], 0.0, action[4]], dtype=np.float32)
        if not np.all(np.abs(self.gimbal_target) <= 1.0 + 1e-9):
            raise ValueError(f"Gimbal target angles must be in [-1, 1], got {self.gimbal_target}")

        vel_cmd = action[0:3]
        return super().step(vel_cmd)

    def _computeReward_good(self):
        desired_z_velocity = -0.5
        alpha = 30
        UGV_pos = np.array(self._get_vehicle_position()[0])
        drone_state = self._getDroneStateVector(0)
        drone_position = drone_state[0:3]
        drone_velocity = drone_state[10:13]
        distance_xy = np.linalg.norm(drone_position[0:2] - UGV_pos[0:2])
        distance_z = np.linalg.norm(drone_position[2:3] - UGV_pos[2:3])
        velocity_z_flag = (0 > drone_velocity[2]) * (drone_velocity[2] > desired_z_velocity)
        reward_z_velocity = (alpha ** (drone_velocity[2] / desired_z_velocity) - 1) / (alpha - 1)
        angle = np.rad2deg(np.arctan2(distance_xy, distance_z))
        # punishment for excessive z velocity
        if velocity_z_flag == False:
            if drone_velocity[2] < desired_z_velocity:
                reward_z_velocity = -0.01  # -abs(drone_velocity[2]/self.SPEED_LIMIT[2])**2
            else:
                reward_z_velocity = -0.1  # - 10*drone_velocity[2]
            if abs(drone_velocity[2]) / self.SPEED_LIMIT[2] > 1.1:
                reward_z_velocity = 0  # reward_z_velocity -5
        # reward_xy_velocity = np.sum(-np.abs(drone_velocity[0:2]- desired_xy_velocity))
        if distance_xy < 8:
            normalized_distance_xy = 0.125 * (8 - distance_xy)
            reward_xy = (30 ** normalized_distance_xy - 1) / (30 - 1)
        else:
            reward_xy = 0  # -distance_xy
        combined_reward = 0.56 * reward_xy + 1.0 * reward_z_velocity
        if drone_position[2] >= 0.275 and p.getContactPoints(bodyA=1, physicsClientId=self.CLIENT) != ():
            print('landed!')
            combined_reward = 140 + combined_reward
        elif drone_position[2] < 0.275 and p.getContactPoints(bodyA=1, physicsClientId=self.CLIENT) != ():
            print('crashed!')
            combined_reward = -1  # normalized_distance_xy * 10 #0#5*distance_xy + combined_reward
        else:
            combined_reward = combined_reward
        # distance_x = np.abs(drone_position[0] - UGV_pos[0])
        # distance_y = np.abs(drone_position[1] - UGV_pos[1])
        # if np.abs(angle) > 30 and (distance_y > 0.8 and distance_x > 0.8):
        #     combined_reward = -0.01

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

    def _computeReward(self):
        # 1) 기본 보상
        if self.is_target_visible():
            reward = self._computeReward_good()

            # 2) 오라클 forward
            oracle = self.compute_oracle_gimbal()
            R_oracle = R.from_quat(oracle['quat']).as_matrix()  # camera-local rotation in drone frame
            f_oracle = R_oracle @ self._cam_dir_local
            f_oracle /= (np.linalg.norm(f_oracle) + 1e-12)

            # 3) 현재 짐벌 forward (현재 액션/타겟에서 직접 계산)
            f_curr = self._forward_from_norm_angles(self.gimbal_target)

            # 4) 의도한 '코사인 정렬' 보상
            cos_align = float(np.clip(np.dot(f_curr, f_oracle), -1.0, 1.0))
            cos_align = cos_align**5  # 보상 변화가 더 뚜렸하게 함.
            # -1(정반대), 0(직교), 1(정렬)
            reward += 0.1 * cos_align  # 스케일은 필요에 따라 튜닝
        else:
            reward = -0.025  # 시야 밖 페널티

        return reward


class LandingGimbalOracleAviary(LandingGimbalAviary):
    """
    Oracle gimbal controller version:
    - RL은 드론의 velocity 명령만 학습 (action[0:3])
    - Gimbal yaw/pitch는 매 step마다 오라클이 착륙패드(UGV) 중심을 바라보도록 자동 설정
    - Roll은 0으로 고정
    - 짐벌 각도 범위(self.gimbal_angle_ranges)를 그대로 사용하고, 내부적으로 [-1,1] 정규화 사용
    todos:
    - [o] Implement the gimbal oracle logic in the step (or get_obs)
    - [o] Implement the gimbal oracle logic in the reset
    - [ ] Implement reward function that properly reflects LOS rewarding
    """

    def __init__(self, is_noisy_gimbal=False, margin_size=None, *args, **kwargs):
        print("In [gym_pybullet_drones] LandingGimbalOracleAviary inits..$")
        super().__init__(*args, **kwargs)

        self.contain_in_frame = is_noisy_gimbal  # 기본: 기존과 동일하게 항상 중심 조준
        if self.contain_in_frame:
            if margin_size is None:
                self.contain_margin = 0.15  # 기본값
            else:
                self.contain_margin = margin_size  # 프레임 경계 여유(0~<1), contain 모드에서만 사용
            assert 0.0 <= self.contain_margin < 1.0, "margin_size must be in [0,1)"
        else:
            if margin_size is not None:
                for _ in range(64):
                    print("[Warning] margin_size is ignored when contain_in_frame is False.")
        for _ in range(16):
            print(f"noisy gimbal mode is {'ON' if is_noisy_gimbal else 'OFF'} !!!")
            print(f"  margin size is {self.contain_margin} !!!")

        # 오라클에서 사용할 카메라 로컬 기준 벡터(전방/업방향)
        # 전방을 +x로 두고, 업을 +z로 두면 yaw(z)->pitch(y)로 자연스러운 pan-tilt가 가능
        self._cam_dir_local = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # forward
        self._cam_up_local  = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # up

        # 드론 본체 기준의 카메라 위치 오프셋 (아래로 5cm)
        self._cam_offset_local = np.array([0.0, 0.0, -0.05], dtype=np.float32)

        # FOV/프로젝션 (원본과 동일/유사)
        self._fov_deg = 85.7
        self._near    = 0.03
        self._far     = 200.0
        self._aspect  = 1.0

    # --- 스페이스 정의: 기존과 호환 유지 (gimbal 차원은 무시됨) ---
    def _actionSpace(self):
        if self.ACT_TYPE == ActionType.VEL:
            size = 3  # 기존과 동일하게 유지 (정책/리플레이 호환)
            return spaces.Box(low=-1*np.ones(size), high=np.ones(size), dtype=np.float32)
        else:
            print(f"[ERROR] in {self.__class__.__name__}.{inspect.currentframe().f_code.co_name}()")
            exit()

    # --- 공용 유틸: 라디안→정규화 / 정규화→라디안 ---
    def _rad_to_norm(self, angles_rad):
        """angles_rad shape (3,): [pitch(rad), roll(rad), yaw(rad)] -> [-1,1]^3"""
        lo = self.gimbal_angle_ranges[:, 0]
        hi = self.gimbal_angle_ranges[:, 1]
        return 2.0 * (np.asarray(angles_rad) - lo) / (hi - lo) - 1.0

    def _norm_to_rad(self, angles_norm):
        """[-1,1]^3 -> radians (3,)"""
        lo = self.gimbal_angle_ranges[:, 0]
        hi = self.gimbal_angle_ranges[:, 1]
        return lo + 0.5 * (np.asarray(angles_norm) + 1.0) * (hi - lo)

    def _update_gimbal_oracle(self):
        """
        contain_in_frame=False(기본): 기존대로 항상 중심 조준(oracle-center)
        contain_in_frame=True       : 타겟이 프레임 안에만 있도록 유지(경계 밖일 때만 따라감)
        - 회전 순서: yaw(z) -> pitch(y) (roll=0 고정)
        - 카메라 로컬 축: +x forward, +z up, +y right
        - gimbal_state_quat: 드론 로컬 좌표계에서의 카메라 회전
        """
        # 안전: 옵션 기본값 보장(기존 코드와 호환)
        if not hasattr(self, "contain_in_frame"):
            self.contain_in_frame = False
            print("[Warning] contain_in_frame not set, default to False (center-oracle mode).")
        if not hasattr(self, "contain_margin"):
            self.contain_margin = 0.10
            print("[Warning] contain_margin not set, default to 0.10.")

        # --- 1) 기하 준비 ---
        UGV_pos = np.array(self._get_vehicle_position()[0], dtype=np.float64)  # (3,)
        drone_state = self._getDroneStateVector(0)
        drone_pos = drone_state[0:3].astype(np.float64)
        drone_quat = self.quat[0, :]  # [x,y,z,w]
        rot_drone = np.array(p.getMatrixFromQuaternion(drone_quat)).reshape(3, 3).astype(np.float64)

        # 카메라 월드 위치
        cam_pos_world = drone_pos + rot_drone @ self._cam_offset_local

        # 타겟 방향(월드)
        to_target_world = UGV_pos - cam_pos_world
        dist = np.linalg.norm(to_target_world)
        if dist < 1e-9:
            return
        f_world_target = to_target_world / dist

        # 드론 로컬로 변환한 타겟 방향
        f_local_target = rot_drone.T @ f_world_target
        f_local_target /= (np.linalg.norm(f_local_target) + 1e-12)

        # --- 2) 목표 yaw*, pitch* (로컬) ---
        vx, vy, vz = float(f_local_target[0]), float(f_local_target[1]), float(f_local_target[2])
        yaw_star = np.arctan2(vy, vx)  # z축 기준 회전
        horiz = np.sqrt(max(vx * vx + vy * vy, 1e-18))
        pitch_star = np.arctan2(-vz, horiz)  # y축 기준 회전(머리 끄덕임; 아래 보면 양수)

        # --- 3) contain 모드가 아니면: 항상 중심 조준 ---
        if not self.contain_in_frame:
            yaw_cmd = yaw_star
            pitch_cmd = pitch_star
        else:
            # --- 4) contain(=noisy) 모드: 프레임 내부 임의 위치로 '의도적으로' 배치 ---
            # 4-1) FOV/마진 → 허용 NDC 범위 설정
            fov_v = np.deg2rad(self._fov_deg)
            fov_h = 2.0 * np.arctan(np.tan(fov_v * 0.5) * self._aspect)
            # contain_margin=0.06이면, NDC 안쪽 94% 영역만 사용
            u_bound = 1.0 - float(np.clip(self.contain_margin, 0.0, 0.99))

            # 4-2) 프레임 내부의 목표 NDC 좌표를 무작위 샘플
            # Uniform 또는 약간 중앙을 피하고 싶으면 작은 inner gap 둬도 됨
            u_ndc = np.random.uniform(-u_bound, u_bound)  # 좌우 [-1,1]
            v_ndc = np.random.uniform(-u_bound, u_bound)  # 상하 [-1,1]

            # 4-3) 원하는 NDC에 해당하는 '각 오프셋' 계산
            # u_ndc = (y/x)/tan(hfov)  → yaw_offset = atan(u_ndc * tan(hfov))
            # v_ndc = (z/x)/tan(vfov)  → pitch_offset = atan(v_ndc * tan(vfov))
            yaw_offset = np.arctan(u_ndc * np.tan(fov_h * 0.5))
            pitch_offset = np.arctan(v_ndc * np.tan(fov_v * 0.5))

            # 4-4) 중앙 조준 각(yaw*, pitch*)에서 '반대 부호'로 보정해서 원하는 픽셀로 보냄
            yaw_cmd = yaw_star - yaw_offset
            pitch_cmd = pitch_star - pitch_offset

            # # 프레임 안정성을 위해 작은 가우시안 지터
            # yaw_cmd   += np.random.normal(0.0, np.deg2rad(0.2))
            # pitch_cmd += np.random.normal(0.0, np.deg2rad(0.2))

        # --- 5) gimbal 각 범위 클램프 (roll=0, pitch/yaw만) ---
        # self.gimbal_angle_ranges: [ [pitch_lo, pitch_hi], [roll_lo, roll_hi], [yaw_lo, yaw_hi] ]
        p_lo, p_hi = float(self.gimbal_angle_ranges[0, 0]), float(self.gimbal_angle_ranges[0, 1])
        y_lo, y_hi = float(self.gimbal_angle_ranges[2, 0]), float(self.gimbal_angle_ranges[2, 1])

        # wrap 후 클램프
        def wrap_pi(a):
            return (a + np.pi) % (2 * np.pi) - np.pi

        pitch_cmd = np.clip(pitch_cmd, p_lo, p_hi)
        yaw_cmd = np.clip(wrap_pi(yaw_cmd), y_lo, y_hi)

        # --- 6) 최종 로컬 회전행렬 (R = Rz(yaw) @ Ry(pitch), roll=0) ---
        cz, sz = np.cos(yaw_cmd), np.sin(yaw_cmd)
        cy, sy = np.cos(pitch_cmd), np.sin(pitch_cmd)
        Rz = np.array([[cz, -sz, 0.0],
                       [sz, cz, 0.0],
                       [0.0, 0.0, 1.0]], dtype=np.float64)
        Ry = np.array([[cy, 0.0, sy],
                       [0.0, 1.0, 0.0],
                       [-sy, 0.0, cy]], dtype=np.float64)
        R_local = Rz @ Ry

        # --- 7) 상태/렌더 값 갱신 ---
        # 드론 로컬 기준 forward/up
        f_local = R_local @ np.array([1.0, 0.0, 0.0], dtype=np.float64)  # +x
        u_local = R_local @ np.array([0.0, 0.0, 1.0], dtype=np.float64)  # +z

        # 월드 forward/up (렌더용)
        self._oracle_forward_world = (rot_drone @ f_local).astype(np.float64)
        self._oracle_up_world = (rot_drone @ u_local).astype(np.float64)

        # 쿼터니언(드론 로컬에서의 카메라 회전)
        self.gimbal_state_quat = R.from_matrix(R_local).as_quat()

        # 보상/가시성에서 쓰는 gimbal_target(정규화 각)을 동기화
        angles_rad = np.array([pitch_cmd, 0.0, yaw_cmd], dtype=np.float32)
        self.gimbal_target = self._rad_to_norm(angles_rad)

        # 내부 각 상태 저장 (yaw, pitch)  — roll은 항상 0
        self._gimbal_yaw_pitch = np.array([yaw_cmd, pitch_cmd], dtype=np.float64)

    def _getDroneImages(self, nth_drone, segmentation: bool = True):
        """
        - 오라클이 만든 월드 f/up 벡터를 그대로 사용해 패드를 바라보는 뷰를 만든다.
        - self.gimbal_state_quat 은 이미 _update_gimbal_oracle 에서 일치하게 갱신됨.
        """
        if self.IMG_RES is None:
            print(f"[ERROR] in {self.__class__.__name__}.{inspect.currentframe().f_code.co_name}(), "
                  f"remember to set self.IMG_RES to np.array([width, height])")
            exit()

        # 안전장치: 오라클이 아직 안 돌면 한 번 갱신
        if not hasattr(self, "_oracle_forward_world"):
            self._update_gimbal_oracle()

        drone_pos = self.pos[nth_drone, :]
        drone_quat = self.quat[nth_drone, :]
        rot_drone = np.array(p.getMatrixFromQuaternion(drone_quat)).reshape(3, 3)

        # 카메라 위치(월드)
        cam_pos_world = np.array(drone_pos) + rot_drone @ self._cam_offset_local

        # 오라클이 만든 카메라 forward/up (월드)
        f_world = self._oracle_forward_world
        u_world = self._oracle_up_world

        target_world = cam_pos_world + f_world * 1000.0

        DRONE_CAM_VIEW = p.computeViewMatrix(
            cameraEyePosition=cam_pos_world,
            cameraTargetPosition=target_world,
            cameraUpVector=u_world,
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
            raise NotImplementedError("Distortion is not implemented for LandingGimbalOracleAviary.")

        rgb = np.moveaxis(rgb, -1, 0)
        return rgb, None, None

    # --- Gym API ---
    def reset(self):
        obs = super().reset()
        # 초기에는 오라클 한 번 갱신해서 gimbal_target을 유효하게 만들어둠
        self._update_gimbal_oracle()
        return obs

    def step(self, action):
        """
        - gimbal 관련 action 차원은 무시됨
        - 오라클로 gimbal_target 업데이트 후, 부모 클래스의 step 호출
        """
        # 1) 오라클로 짐벌 갱신
        self._update_gimbal_oracle()

        # 2) 드론 속도 명령만 전달
        vel_cmd = action[0:3]
        return super(LandingGimbalAviary, self).step(vel_cmd)  # LandingGimbalAviary의 부모(LandingAviary)의 step 호출

    def _computeReward(self):
       # 1) 기본 보상
       if self.is_target_visible():
           reward = self._computeReward_good()
       else:
           reward = -0.025  # 시야 밖 페널티

       return reward


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
    env = LandingGimbalOracleAviary(episode_len_sec=20, is_noisy_gimbal=True)
    # env = LandingGimbalAviary(episode_len_sec=10, )

    obs = env.reset()
    print("reset gimbal_state_quat:", getattr(env, "gimbal_state_quat", None))
    show_rgb(env.rgb)

    # 몇 스텝 전진/무작위 속도 명령으로 진행해 보며 계속 패드를 가운데에 두는지 확인
    for t in range(128):
        # 예: 전진살짝 + 고도유지
        action = np.array([0.0, 0.0, 0.01], dtype=np.float32)
        # action = np.array([0.0, 0.0, 0.01, 1, 0], dtype=np.float32)
        obs, rew, done, info = env.step(action)

        # 쿼터니안 → 오일러 (yaw, pitch, roll)
        quat = env.gimbal_state_quat
        euler = R.from_quat(quat).as_euler('ZYX', degrees=True)
        yaw, pitch, roll = euler  # 'zyx' 이므로 [yaw, pitch, roll] 순서

        print(f"step {t+1} gimbal_euler [deg]: yaw={yaw:.2f}, pitch={pitch:.2f}, roll={roll:.2f}; quat: {quat}")

        if t % 5 == 0:
            show_rgb(env.rgb)
            print(env.is_target_visible())
        if done:
            break

    env.close()


def oracle_test():
    env = LandingGimbalAviary(episode_len_sec=10, )

    obs = env.reset()
    quat = env.gimbal_state_quat
    euler = R.from_quat(quat).as_euler('zyx', degrees=True)
    yaw, pitch, roll = euler  # 'zyx' 이므로 [yaw, pitch, roll] 순서
    print(f"reset gimbal_euler [deg]: yaw={yaw:.2f}, pitch={pitch:.2f}, roll={roll:.2f}; quat: {quat}")
    show_rgb(env.rgb)

    for t in range(200):
        oracle_dict = env.compute_oracle_gimbal()
        vel_cmd = np.array([0.0, 0.0, 0.01], dtype=np.float32)
        action = np.concatenate([vel_cmd, oracle_dict['angles_norm']])
        obs, rew, done, info = env.step(action)

        quat = env.gimbal_state_quat
        euler = R.from_quat(quat).as_euler('zyx', degrees=True)
        yaw, pitch, roll = euler  # 'zyx' 이므로 [yaw, pitch, roll] 순서
        print(f"step {t+1} gimbal_euler [deg]: yaw={yaw:.2f}, pitch={pitch:.2f}, roll={roll:.2f}; quat: {quat}")

        if t % 5 == 0:
            show_rgb(env.rgb)
        if done:
            break

    env.close()


def gimbal_env_test():
    def _unit(v):
        v = np.asarray(v, dtype=float)
        n = np.linalg.norm(v)
        return v if n == 0 else v / n

    def _plot_vectors_3d(f_curr, f_oracle, rel_pos_world_norm=None, rgb=None, visibility=None):
        """3D quiver(왼쪽) + 옵션: RGB 이미지(오른쪽) 나란히 표시.
           rgb는 (1,H,W) 권장(필수는 아님). visibility가 주어지면 이미지 제목에 표시."""
        f_curr = np.asarray(f_curr, dtype=float)
        f_oracle = np.asarray(f_oracle, dtype=float)

        # Determine axis limits from both vectors
        vmax = float(np.max(np.abs(np.concatenate([f_curr, f_oracle]))))
        vmax = 1.0 if vmax < 1e-9 else min(max(1.0, 1.1 * vmax), 1.5)

        if rgb is None:
            # 기존과 동일한 단일 3D 플롯
            fig = plt.figure(figsize=(16, 14))
            ax = fig.add_subplot(111, projection='3d')
            ax_img = None
        else:
            # 3D(왼쪽) + 이미지(오른쪽) 나란히
            fig = plt.figure(figsize=(20, 10))
            gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0], wspace=0.15)
            ax = fig.add_subplot(gs[0, 0], projection='3d')
            ax_img = fig.add_subplot(gs[0, 1])

            # --- RGB 준비: (1|3|4, H, W) -> (H, W, 3) ---
            rgb_arr = np.asarray(rgb)
            if rgb_arr.ndim != 3:
                raise ValueError(f"`rgb`는 (C,H,W) 여야 합니다. 받은 shape={rgb_arr.shape}")

            C = rgb_arr.shape[0]
            if C == 1:
                rgb_arr = np.repeat(rgb_arr, 3, axis=0)
            elif C == 4:
                rgb_arr = rgb_arr[:3, :, :]  # alpha drop
            elif C != 3:
                raise ValueError(f"`rgb`의 채널 수는 1/3/4만 지원합니다. 받은 C={C}")

            img = np.moveaxis(rgb_arr, 0, -1)

            # dtype 정리: 0~1 float이면 0~255로 스케일 후 uint8, 그 외엔 클리핑
            if img.dtype != np.uint8:
                vmax_img = float(np.nanmax(img))
                if vmax_img <= 1.0:
                    img_disp = np.clip(img, 0.0, 1.0)
                    img_disp = (img_disp * 255.0).astype(np.uint8)
                else:
                    img_disp = np.clip(img, 0.0, 255.0).astype(np.uint8)
            else:
                img_disp = img

            ax_img.imshow(img_disp)
            ax_img.axis('off')
            title_vis = f"env.rgb — visibility: {visibility:.3f}" if visibility is not None else "env.rgb"
            ax_img.set_title(title_vis)

        # --- 3D quiver ---
        ax.quiver(0, 0, 0, f_curr[0], f_curr[1], f_curr[2], length=1.0, normalize=True, color='orange', label='curr')
        ax.quiver(0, 0, 0, f_oracle[0], f_oracle[1], f_oracle[2], length=1.0, normalize=True, color='purple',
                  label='oracle')

        if rel_pos_world_norm is not None:
            rel_pos_world_norm = np.asarray(rel_pos_world_norm, dtype=float)
            ax.quiver(0, 0, 0, rel_pos_world_norm[0], rel_pos_world_norm[1], rel_pos_world_norm[2],
                      length=1.0, normalize=True, color='cyan', label='rel_pos')

        ax.set_xlim([-vmax, vmax])
        ax.set_ylim([-vmax, vmax])
        ax.set_zlim([-vmax, vmax])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Gimbal Forward (curr) vs Oracle Forward')

        # Reference axes
        ax.quiver(0, 0, 0, 1, 0, 0, length=1.0, normalize=True, color='r')  # +X
        ax.quiver(0, 0, 0, 0, 1, 0, length=1.0, normalize=True, color='g')  # +Y
        ax.quiver(0, 0, 0, 0, 0, 1, length=1.0, normalize=True, color='b')  # +Z

        # 범례(필요시)
        try:
            ax.legend(loc='upper left')
        except Exception:
            pass

        plt.tight_layout()
        plt.show()

    def compare_gimbal_to_oracle(env, show_plot=True):
        """Numerically & visually compare oracle vs current gimbal forward vectors.

        Args:
            env: LandingGimbalAviary-like instance (already reset/stepped to a consistent state).
            show_plot: whether to render a 3D plot.
        Returns:
            dict with 'cosine', 'angle_deg', 'visible', 'pixel', 'f_curr', 'f_oracle'.
        """
        # Set the current gimbal target consistent with your step convention (no env.step here)
        gimbal_target = env.gimbal_target
        assert gimbal_target.shape == (3,)
        assert gimbal_target[1] < 1e-6, "Roll should be zero in this convention."

        # Current forward (from your helper, same convention Rz(yaw) @ Ry(pitch))
        f_curr = env._forward_from_norm_angles(gimbal_target)

        # Oracle forward
        oracle = env.compute_oracle_gimbal()
        f_oracle = oracle['oracle_forward_local']
        roll_f_oracle = oracle['angles_rad'][1]
        assert abs(roll_f_oracle) < 1e-6, "Oracle roll should be zero in this convention."

        # Metrics
        cosine = float(np.clip(np.dot(_unit(f_curr), _unit(f_oracle)), -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cosine)))

        # Visibility (optional details)
        vis, det = env.is_target_visible(return_details=True)
        pixel = det.get('pixel', None) if isinstance(det, dict) else None

        # Relative position vector between drone (not cam) and target (in world frame)
        drone_position = np.asarray(env._getDroneStateVector(0)[0:3], dtype=float)
        target_position = np.asarray(env._get_vehicle_position()[0], dtype=float)
        rel_pos_world = target_position - drone_position
        rel_pos_world_normalized = _unit(rel_pos_world)

        print("\n[Gimbal vs Oracle Alignment]")
        print(f"cosine alignment : {cosine: .6f}")
        print(f"angle difference : {angle_deg: .3f} deg")
        print(f"visible          : {vis}")
        print(f"pixel estimate   : {pixel}")
        print(f"f_curr           : {f_curr}")
        print(f"f_oracle         : {f_oracle}")
        print(f"rel_pos_world      : {rel_pos_world}")
        print(f"rel_pos_world_norm    : {rel_pos_world_normalized}")

        out = {
            'cosine': cosine,
            'angle_deg': angle_deg,
            'visible': bool(vis),
            'pixel': pixel,
            'f_curr': np.asarray(f_curr, dtype=float),
            'f_oracle': np.asarray(f_oracle, dtype=float),
            'rel_pos_world_normalized': rel_pos_world_normalized,
        }

        if show_plot:
            _plot_vectors_3d(out['f_curr'], out['f_oracle'], rel_pos_world_normalized, env.rgb, vis)

        return out

    env = LandingGimbalAviary(episode_len_sec=20, )
    obs = env.reset()

    for t in range(128):
        action = np.array([0.06, 0.0, 0.01, 0.82, 0.0])  # Example action with pitch_norm = -1.0, yaw_norm = 0.0
        obs, reward, done, _ = env.step(action)
        if t % 8 == 0 or done:
            result = compare_gimbal_to_oracle(env, show_plot=True)
            print("\nStep", t)
            print(reward, result['angle_deg'], result['visible'], result['pixel'])

    env.close()


if __name__ == "__main__":
    main()
    # oracle_test()
    # gimbal_env_test()
