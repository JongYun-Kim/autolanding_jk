import numpy as np
from gym import spaces

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary, ImageType
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType, BaseSingleAgentAviary
import pybullet as p
from gym_pybullet_drones.utils.specs import BoundedArray

from ...utils.utils import rgb2gray
import inspect
from scipy.spatial.transform import Rotation as R

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
                 episode_len_sec: int=30,
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
    # for cross compatbility with dm gym


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
                 episode_len_sec: int=30,
                 ):
        """Initialization of a single agent RL environment with gimbal control."""
        print("In [gym_pybullet_drones] LandingGimbalAviary inits..$")

        # Angles: [pitch,  roll,  yaw]
        # gimbal_target: target gimbal angles in normalized [-1, 1] range
        self.gimbal_target = None  # action[3:]
        self.initial_gimbal_target = np.array([-1.0, 0.0, 0.0])  # camera pointing down, no roll, no yaw

        # Gimbal specs
        self.gimbal_angle_ranges = np.array([  # (3,2)
            [-np.pi/2, 0      ],  # Pitch: -90  to 0   degrees
            [-np.pi/4, np.pi/4],  # Roll : -45  to 45  degrees
            [-np.pi,   np.pi  ]   # Yaw  : -180 to 180 degrees
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
            size = 6
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

    def _getDroneImages(self, nth_drone, segmentation: bool=True):
        # Default gimbal target if not defined
        if self.gimbal_target is not None:
            gimbal_target = self.gimbal_target  # (3,) in [-1, 1]
        else:
            raise ValueError(f"gimbal_target is not set. Please set it before calling {self.__class__.__name__}.{inspect.currentframe().f_code.co_name}()")

        # Convert normalized gimbal_target (-1 to 1) to actual target angle ranges
        # Pitch: self.gimbal_angle_ranges[0,0] to self.gimbal_angle_ranges[0,1] in radians
        # Roll : self.gimbal_angle_ranges[1,0] to self.gimbal_angle_ranges[1,1] in radians
        # Yaw  : self.gimbal_angle_ranges[2,0] to self.gimbal_angle_ranges[2,1] in radians
        normalized_angles = (gimbal_target + 1) / 2.0  # (3,) in [0, 1]
        angle_ranges = self.gimbal_angle_ranges[:, 1] - self.gimbal_angle_ranges[:, 0]  # (3,) in radians
        gimbal_rad = self.gimbal_angle_ranges[:, 0] + normalized_angles * angle_ranges
        pitch_rad, roll_rad, yaw_rad = gimbal_rad

        # Update gimbal state quaternion based on target angles
        # yaw→pitch→roll = intrinsic ZYX Euler:
        self.gimbal_state_quat = R.from_euler('ZYX',[yaw_rad, pitch_rad, roll_rad]).as_quat()   # [x,y,z,w]; nparray (4,)

        # Get drone orientation
        drone_pos = self.pos[nth_drone, :]
        drone_orientation = self.quat[nth_drone, :]
        rot_drone = np.array(p.getMatrixFromQuaternion(drone_orientation)).reshape(3, 3)

        # Apply camera offset in drone local frame (below drone)
        local_offset = np.array([0.0, 0.0, -0.05])
        world_offset = rot_drone @ local_offset
        camera_pos = np.array(drone_pos) + world_offset

        # Gimbal rotation matrices
        # Pitch (around x-axis)
        pitch_rot = np.array([
            [1, 0, 0],
            [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
            [0, np.sin(pitch_rad), np.cos(pitch_rad)]
        ], dtype=np.float32)
        # Roll (around y-axis)
        roll_rot = np.array([
            [np.cos(roll_rad), 0, np.sin(roll_rad)],
            [0, 1, 0],
            [-np.sin(roll_rad), 0, np.cos(roll_rad)]
        ], dtype=np.float32)
        # Yaw (around z-axis)
        yaw_rot = np.array([
            [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
            [np.sin(yaw_rad), np.cos(yaw_rad), 0],
            [0, 0, 1]
        ], dtype=np.float32)

        # Apply rotation in Yaw → Pitch → Roll order (항공 시스템 convention)
        gimbal_rot = roll_rot @ pitch_rot @ yaw_rot

        # Local camera direction and up vectors
        camera_dir_local = np.array([0.0, 0.0, -1.0])  #np.array([0, 0, -1])
        camera_up_local = np.array([0.0, 1.0, 0.0])  #np.array([1, 0, 0])

        # Rotate by gimbal
        camera_dir_rot = gimbal_rot @ camera_dir_local
        camera_up_rot = gimbal_rot @ camera_up_local

        # Then rotate by drone orientation
        camera_dir = rot_drone @ camera_dir_rot
        camera_up = rot_drone @ camera_up_rot

        # Camera target (far ahead in direction)
        target = camera_pos + camera_dir * 1000.0

        if self.IMG_RES is None:
            print(f"[ERROR] in {self.__class__.__name__}.{inspect.currentframe().f_code.co_name}(), "
                  f"remember to set self.IMG_RES to np.array([width, height])")
            exit()

        DRONE_CAM_VIEW = p.computeViewMatrix(
            cameraEyePosition=camera_pos,
            cameraTargetPosition=target,
            cameraUpVector=camera_up,
            physicsClientId=self.CLIENT
        )

        DRONE_CAM_PRO = p.computeProjectionMatrixFOV(
            fov=85.7,
            aspect=1.0,
            nearVal=0.03,
            farVal=200.0
        )

        # Segmentation flag
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

        rgb = np.moveaxis(rgb, -1, 0)

        return rgb, None, None

    def _getDroneImages_myog(self, nth_drone, segmentation: bool=True):
        # target angles: # 0: pitch, 1: roll, 2: yaw
        initial_gimbal_target = np.zeros(3)
        gimbal_target = self.gimbal_target if self.gimbal_target is not None else initial_gimbal_target

        if self.IMG_RES is None:
            print(f"[ERROR] in {self.__class__.__name__}.{inspect.currentframe().f_code.co_name}(), "
                  f"remember to set self.IMG_RES to np.array([width, height])")
            exit()

        rot_mat = np.array(p.getMatrixFromQuaternion(self.quat[nth_drone, :])).reshape(3, 3)

        # Set target point, camera view and projection matrices #
        target = np.dot(rot_mat, np.array([0, 0, -1000])) + np.array(self.pos[nth_drone, :])

        # Image fov and res
        fov = 85.7
        img_res = self.IMG_RES

        DRONE_CAM_VIEW = p.computeViewMatrix(cameraEyePosition=self.pos[nth_drone, :] - np.array([0, 0, 0.15]) +np.array([0, 0, self.L]),
                                             cameraTargetPosition=target,
                                             cameraUpVector=[1, 0, 0],
                                             physicsClientId=self.CLIENT
                                             )
        DRONE_CAM_PRO =  p.computeProjectionMatrixFOV(fov=fov,
                                                      aspect=1.0,
                                                      nearVal=self.L,
                                                      farVal=1000.0
                                                      )
        SEG_FLAG = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX if segmentation else p.ER_NO_SEGMENTATION_MASK

        [_, _, rgb, _, _] = p.getCameraImage(width=img_res[0],
                                                 height=img_res[1],
                                                 shadow=1,
                                                 viewMatrix=DRONE_CAM_VIEW,
                                                 projectionMatrix=DRONE_CAM_PRO,
                                                 flags=SEG_FLAG,
                                                 physicsClientId=self.CLIENT
                                                 )
        if self.distortion:
            raise NotImplementedError("Distortion is not implemented for LandingGimbalAviary.")

        rgb = np.moveaxis(rgb, -1, 0)

        return rgb, None, None

    def reset(self):
        self.gimbal_target = self.initial_gimbal_target
        return super().reset()

    def step(self, action):
        self.gimbal_target = np.array([action[3], 0.0, action[4]])  # force roll to 0.0; smaller action space!
        if not np.all(np.abs(self.gimbal_target) <= 1):
            raise ValueError(f"Gimbal target angles must be in [-1, 1], got {self.gimbal_target}")

        vel_cmd = action[0:3]
        return super().step(vel_cmd)

