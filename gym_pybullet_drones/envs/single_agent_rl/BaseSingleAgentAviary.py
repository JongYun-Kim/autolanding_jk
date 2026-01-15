import os
from enum import Enum
from gym import spaces
import pybullet as p
import numpy as np
from gym_pybullet_drones.utils.utils import rgb2gray
import random
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, ImageType, BaseAviary
from gym_pybullet_drones.utils.utils import nnlsRPM
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from gym_pybullet_drones.control.Mavic3PIDControl import Mavic3PIDControl
import cv2


class ActionType(Enum):
    """Action type enumeration class."""
    RPM = "rpm"                 # RPMS
    DYN = "dyn"                 # Desired thrust and torques
    PID = "pid"                 # PID control
    VEL = "vel"                 # Velocity input (using PID control)
    TUN = "tun"                 # Tune the coefficients of a PID controller
    ONE_D_RPM = "one_d_rpm"     # 1D (identical input to all motors) with RPMs
    ONE_D_DYN = "one_d_dyn"     # 1D (identical input to all motors) with desired thrust and torques
    ONE_D_PID = "one_d_pid"     # 1D (identical input to all motors) with PID control


class ObservationType(Enum):
    """Observation type enumeration class."""
    KIN = "kin"     # Kinematic information (pose, linear and angular velocities)
    RGB = "rgb"     # RGB camera capture in each drone's POV


class BaseSingleAgentAviary(BaseAviary):
    """Base single drone environment class for reinforcement learning."""

    def __init__(self,
                 drone_model: DroneModel=DroneModel.HB,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 freq: int=240,
                 aggregate_phy_steps: int=10,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM,
                 episode_len_sec: int=30,
                 gv_path_type: str="straight",
                 gv_sinusoidal_amplitude: float=2.0,
                 gv_sinusoidal_frequency: float=0.5
                 ):
        """Initialization of a generic single agent RL environment.

        Attribute `num_drones` is automatically set to 1; `vision_attributes`
        and `dynamics_attributes` are selected based on the choice of `obs`
        and `act`; `obstacles` is set to True and overridden with landmarks for
        vision applications; `user_debug_gui` is set to False for performance.

        Parameters
        --
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3) array; the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3) array; the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        freq : int, optional
            The frequency (Hz) at which the physics engine steps.
        aggregate_phy_steps : int, optional
            The number of physics steps within one call to `BaseAviary.step()`.
        gui : bool, optional
        record : bool, optional
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, waypoint or velocity with PID control; etc.)
        gv_path_type : str, optional
            Type of path for the ground vehicle/landing pad. Options: "straight" (default), "sinusoidal".
        gv_sinusoidal_amplitude : float, optional
            Amplitude of sinusoidal oscillation in meters (only used if gv_path_type="sinusoidal"). Default: 2.0.
        gv_sinusoidal_frequency : float, optional
            Frequency of sinusoidal oscillation in Hz (only used if gv_path_type="sinusoidal"). Default: 0.5.
        """
        vision_attributes = True if obs == ObservationType.RGB else False
        dynamics_attributes = True if act in [ActionType.DYN, ActionType.ONE_D_DYN] else False
        self.OBS_TYPE = obs
        self.ACT_TYPE = act
        self.EPISODE_LEN_SEC = episode_len_sec
        # Create integrated controllers
        if act in [ActionType.PID, ActionType.VEL, ActionType.TUN, ActionType.ONE_D_PID]:
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
            if drone_model in [DroneModel.CF2X, DroneModel.CF2P]:
                self.ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
                if act == ActionType.TUN:
                    self.TUNED_P_POS = np.array([.4, .4, 1.25])
                    self.TUNED_I_POS = np.array([.05, .05, .05])
                    self.TUNED_D_POS = np.array([.2, .2, .5])
                    self.TUNED_P_ATT = np.array([70000., 70000., 60000.])
                    self.TUNED_I_ATT = np.array([.0, .0, 500.])
                    self.TUNED_D_ATT = np.array([20000., 20000., 12000.])
            elif drone_model == DroneModel.MAVIC3:
                self.ctrl = Mavic3PIDControl(drone_model=DroneModel.MAVIC3)
                if act == ActionType.TUN:
                    # Mavic3 tuning defaults (scaled from CF2X)
                    self.TUNED_P_POS = np.array([.4, .4, 1.25])
                    self.TUNED_I_POS = np.array([.05, .05, .05])
                    self.TUNED_D_POS = np.array([.3, .3, .6])
                    self.TUNED_P_ATT = np.array([52500., 52500., 45000.])
                    self.TUNED_I_ATT = np.array([.0, .0, 375.])
                    self.TUNED_D_ATT = np.array([15000., 15000., 9000.])
            elif drone_model == DroneModel.HB:
                self.ctrl = SimplePIDControl(drone_model=DroneModel.HB)
                if act == ActionType.TUN:
                    self.TUNED_P_POS = np.array([.1, .1, .2])
                    self.TUNED_I_POS = np.array([.0001, .0001, .0001])
                    self.TUNED_D_POS = np.array([.3, .3, .4])
                    self.TUNED_P_ATT = np.array([.3, .3, .05])
                    self.TUNED_I_ATT = np.array([.0001, .0001, .0001])
                    self.TUNED_D_ATT = np.array([.3, .3, .5])
            else:
                print("[ERROR] in BaseSingleAgentAviary.__init()__, no controller is available for the specified drone_model")
        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record,
                         obstacles=False, # Add obstacles for RGB observations and/or FlyThruGate
                         user_debug_gui=False, # Remove of RPM sliders from all single agent learning aviaries
                         vision_attributes=vision_attributes,
                         dynamics_attributes=dynamics_attributes,
                         gv_path_type=gv_path_type,
                         gv_sinusoidal_amplitude=gv_sinusoidal_amplitude,
                         gv_sinusoidal_frequency=gv_sinusoidal_frequency
                         )
        # Set a limit on the maximum target speed
        #Z needs its own speed limit
        if act == ActionType.VEL:
            self.SPEED_LIMIT = np.array([0.17 * self.MAX_SPEED_KMH * (1000/3600), 0.17 * self.MAX_SPEED_KMH * (1000/3600), 0.06 * self.MAX_SPEED_KMH * (1000/3600)])
        # Try _trajectoryTrackingRPMs exists IFF ActionType.TUN
        if act == ActionType.TUN and not (hasattr(self.__class__, '_trajectoryTrackingRPMs') and callable(getattr(self.__class__, '_trajectoryTrackingRPMs'))):
                print("[ERROR] in BaseSingleAgentAviary.__init__(), ActionType.TUN requires an implementation of _trajectoryTrackingRPMs in the instantiated subclass")
                exit()
        self.max_episode_steps = int(self.EPISODE_LEN_SEC*self.SIM_FREQ/self.AGGR_PHY_STEPS)

    def _addObstacles(self):
        if self.OBS_TYPE == ObservationType.RGB:
            p.loadURDF("block.urdf",
                       [1, 0, .1],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
            p.loadURDF("cube_small.urdf",
                       [0, 1, .1],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
            p.loadURDF("duck_vhacd.urdf",
                       [-1, 0, .1],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
            p.loadURDF("teddy_vhacd.urdf",
                       [0, -1, .1],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
        else:
            pass

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        --
        ndarray
            A Box() of size 1, 3, 4, or 6 depending on the action type.
        """
        if self.ACT_TYPE == ActionType.TUN:
            size = 6
        elif self.ACT_TYPE in [ActionType.RPM, ActionType.DYN, ActionType.VEL]:
            size = 3
        elif self.ACT_TYPE == ActionType.PID:
            size = 3
        elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_DYN, ActionType.ONE_D_PID]:
            size = 1
        else:
            print("[ERROR] in BaseSingleAgentAviary._actionSpace()")
            exit()
        return spaces.Box(low=-1*np.ones(size),
                          high=np.ones(size),
                          dtype=np.float32
                          )

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Parameter `action` is processed differenly for each of the different
        action types: `action` can be of length 1, 3, 4, or 6 and represent 
        RPMs, desired thrust and torques, the next target position to reach 
        using PID control, a desired velocity vector, new PID coefficients, etc.

        Parameters
        --
        action : ndarray
            The input action for each drone, to be translated into RPMs.

        Returns
        --
        ndarray
            (4,)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        if self.ACT_TYPE == ActionType.TUN:
            self.ctrl.setPIDCoefficients(p_coeff_pos=(action[0]+1)*self.TUNED_P_POS,
                                         i_coeff_pos=(action[1]+1)*self.TUNED_I_POS,
                                         d_coeff_pos=(action[2]+1)*self.TUNED_D_POS,
                                         p_coeff_att=(action[3]+1)*self.TUNED_P_ATT,
                                         i_coeff_att=(action[4]+1)*self.TUNED_I_ATT,
                                         d_coeff_att=(action[5]+1)*self.TUNED_D_ATT
                                         )
            return self._trajectoryTrackingRPMs() 
        elif self.ACT_TYPE == ActionType.RPM:
            return np.array(self.HOVER_RPM * (1+0.05*action))
        elif self.ACT_TYPE == ActionType.DYN:
            return nnlsRPM(thrust=(self.GRAVITY*(action[0]+1)),
                           x_torque=(0.05*self.MAX_XY_TORQUE*action[1]),
                           y_torque=(0.05*self.MAX_XY_TORQUE*action[2]),
                           z_torque=(0.05*self.MAX_Z_TORQUE*action[3]),
                           counter=self.step_counter,
                           max_thrust=self.MAX_THRUST,
                           max_xy_torque=self.MAX_XY_TORQUE,
                           max_z_torque=self.MAX_Z_TORQUE,
                           a=self.A,
                           inv_a=self.INV_A,
                           b_coeff=self.B_COEFF,
                           gui=self.GUI
                           )
        elif self.ACT_TYPE == ActionType.PID: 
            state = self._getDroneStateVector(0)
            rpm, _, _ = self.ctrl.computeControl(control_timestep=self.AGGR_PHY_STEPS*self.TIMESTEP, 
                                                 cur_pos=state[0:3],
                                                 cur_quat=state[3:7],
                                                 cur_vel=state[10:13],
                                                 cur_ang_vel=state[13:16],
                                                 target_pos=state[0:3]+0.1*action
                                                 )
            return rpm
        elif self.ACT_TYPE == ActionType.VEL:
            #implemement deadzones
            deadzone_limit = 0.005
            action[np.abs(action)<deadzone_limit] = 0 
            state = self._getDroneStateVector(0)
            if np.linalg.norm(action[0:3]) != 0:
                v_unit_vector = action[0:3] / np.linalg.norm(action[0:3])
            else:
                v_unit_vector = np.zeros(3)
            # It is necessary to limit velocity changes in smooth manner, otherwise controller goes crazy
            target_vel = 0.15 * (self.SPEED_LIMIT * action[0:3]) + 0.85 * state[10:13]
            rpm, _, _ = self.ctrl.computeControl(control_timestep=self.AGGR_PHY_STEPS*self.TIMESTEP,
                                                 cur_pos=state[0:3],
                                                 cur_quat=state[3:7],
                                                 cur_vel=state[10:13],
                                                 cur_ang_vel=state[13:16],
                                                 target_pos=state[0:3], # same as the current position
                                                 target_rpy=np.array([0,0,0]),#,state[9]]), # keep current yaw
                                                 target_vel= target_vel#self.SPEED_LIMIT * action[0:3]#np.abs(action[3]) * v_unit_vector # target the desired velocity vector
                                                 )
            return rpm
        elif self.ACT_TYPE == ActionType.ONE_D_RPM:
            return np.repeat(self.HOVER_RPM * (1+0.05*action), 4)
        elif self.ACT_TYPE == ActionType.ONE_D_DYN:
            return nnlsRPM(thrust=(self.GRAVITY*(1+0.05*action[0])),
                           x_torque=0,
                           y_torque=0,
                           z_torque=0,
                           counter=self.step_counter,
                           max_thrust=self.MAX_THRUST,
                           max_xy_torque=self.MAX_XY_TORQUE,
                           max_z_torque=self.MAX_Z_TORQUE,
                           a=self.A,
                           inv_a=self.INV_A,
                           b_coeff=self.B_COEFF,
                           gui=self.GUI
                           )
        elif self.ACT_TYPE == ActionType.ONE_D_PID:
            state = self._getDroneStateVector(0)
            rpm, _, _ = self.ctrl.computeControl(control_timestep=self.AGGR_PHY_STEPS*self.TIMESTEP, 
                                                 cur_pos=state[0:3],
                                                 cur_quat=state[3:7],
                                                 cur_vel=state[10:13],
                                                 cur_ang_vel=state[13:16],
                                                 target_pos=state[0:3]+0.1*np.array([0,0,action[0]])
                                                 )
            return rpm
        else:
            print("[ERROR] in BaseSingleAgentAviary._preprocessAction()")

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        --
        ndarray
            A Box() of shape (H,W,4) or (12,) depending on the observation type.
        """
        if self.OBS_TYPE == ObservationType.RGB:
            return spaces.Box(low=0,
                              high=255,
                              shape=(self.IMG_RES[1], self.IMG_RES[0], 4),
                              dtype=np.uint8
                              )
        elif self.OBS_TYPE == ObservationType.KIN:
            return spaces.Box(low=np.array([-1,-1,0, -1,-1,-1, -1,-1,-1, -1,-1,-1]),
                              high=np.array([1,1,1, 1,1,1, 1,1,1, 1,1,1]),
                              dtype=np.float32
                              )
        else:
            print("[ERROR] in BaseSingleAgentAviary._observationSpace()")
    
    def change_mean_std(self, image, new_mean=125, new_std=30):
        img_float = image.astype(np.float32)
        old_mean = np.mean(img_float)
        old_std = np.std(img_float)
        img_normalized = (img_float - old_mean) / old_std
        img_scaled = (img_normalized * new_std) + new_mean
        img_result = np.clip(img_scaled, 0, 255).astype(image.dtype)

        return img_result

    def find_spikes(self, img, spike_tr = 50):
        spikes = []
        hist, bins = np.histogram(img.ravel(), bins=256, range=[0, 256])

        for i in range(len(hist)):
            if i == 0 and hist[0] > hist[1] + spike_tr:
                spikes.append(0)
            
            elif i == len(hist) and hist[-2] + spike_tr < hist[-1]:
                spikes.append(255)
            
            elif hist[i-1] + spike_tr < hist[i] and hist[i] > hist[i+1] + spike_tr:
                spikes.append(i)
        return spikes

    def distribute_spikes(self, img, spike_tr=50, blur=15):
        spikes = self.find_spikes(img, spike_tr=spike_tr)
        for y in range(len(img[0])):
            for x in range(len(img[1])):
                color = img[y][x]
                if color in spikes:
                    if color > blur:
                        img[y][x] += random.randint(-blur,blur)
                    else:
                        img[y][x] += random.randint(-color,blur)
        return img

    def apply_clahe(self, image, clip_limit=2.0, tile_grid_size=(8, 8)):

        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image.copy()
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        equalized_image = clahe.apply(gray_image)
        return equalized_image

    def _computeObs(self, done = False):
        """Returns the current observation of the environment.

        Returns
        --
        ndarray
            A Box() of shape (H,W,4) or (12,) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            if self.step_counter%self.IMG_CAPTURE_FREQ == 0 or done == True: 
                
                self.rgb, self.dep[0], self.seg[0] = self._getDroneImages(0,
                                                                             segmentation=False
                                                                             )
                if self.RECORD:
                    self._exportImage(img_type=ImageType.RGB,
                                      img_input=self.rgb,
                                      path=self.ONBOARD_IMG_PATH,
                                      frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ)
                                      )
                self.rgb = rgb2gray(self.rgb)[None,:]
            return self.rgb
        elif self.OBS_TYPE == ObservationType.KIN: 
            obs = self._clipAndNormalizeState(self._getDroneStateVector(0))
            return np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16]]).reshape(12,)
        else:
            print("[ERROR] in BaseSingleAgentAviary._computeObs()")

    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a drone's state to the [-1,1] range.

        Must be implemented in a subclass.

        Parameters
        --
        state : ndarray
            Array containing the non-normalized state of a single drone.
        """
        raise NotImplementedError
