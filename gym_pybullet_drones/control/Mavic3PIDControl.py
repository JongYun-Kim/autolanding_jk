import math
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.envs.BaseAviary import DroneModel, BaseAviary


class Mavic3PIDControl(BaseControl):
    """PID control class for DJI Mavic 3.

    Adapted from DSLPIDControl (UTIAS' DSL) with parameters scaled for the
    heavier Mavic 3 drone (0.895 kg vs CF2X's 0.027 kg).

    Parameter Scaling Notes:
    ------------------------
    - PWM2RPM constants: Estimated for Mavic 3 motors with max RPM ~7500
    - Position PID (outer loop): Similar to CF2X with slightly increased damping
    - Attitude PID (inner loop): Scaled by ~0.75 factor accounting for:
        * Higher inertia (~500x): requires more torque for same angular accel
        * Higher KF (~237x): each RPM produces more thrust
        * Longer arm (~2.8x): more torque per thrust differential
        * Net effect: torque capability increase (670x) > inertia increase (500x)
          so gains are slightly LOWER than CF2X

    If the drone response is:
    - Too sluggish: Increase P_COEFF_TOR values
    - Too aggressive/oscillating: Decrease P_COEFF_TOR or increase D_COEFF_TOR
    - Drifting: Increase I_COEFF_FOR values
    """

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel,
                 g: float = 9.8
                 ):
        """Mavic3PIDControl initialization.

        Parameters
        ----------
        drone_model : DroneModel
            The type of drone to control (must be DroneModel.MAVIC3).
        g : float, optional
            The gravitational acceleration in m/s^2.

        """
        super().__init__(drone_model=drone_model, g=g)
        if self.DRONE_MODEL != DroneModel.MAVIC3:
            print("[ERROR] in Mavic3PIDControl.__init__(), Mavic3PIDControl requires DroneModel.MAVIC3")
            exit()

        # Position PID coefficients (outer loop)
        # Similar to CF2X - these produce desired angles from position error
        self.P_COEFF_FOR = np.array([.4, .4, 1.25])
        self.I_COEFF_FOR = np.array([.05, .05, .05])
        self.D_COEFF_FOR = np.array([.3, .3, .6])  # Slightly higher damping for larger inertia

        # Attitude PID coefficients (inner loop)
        # Scaled by ~0.75 from CF2X (70000 -> 52500, etc.)
        # See class docstring for scaling rationale
        self.P_COEFF_TOR = np.array([52500., 52500., 45000.])
        self.I_COEFF_TOR = np.array([.0, .0, 375.])
        self.D_COEFF_TOR = np.array([15000., 15000., 9000.])

        # PWM to RPM conversion for Mavic 3 motors
        # Estimated for max RPM ~7500 over PWM range [20000, 65535]
        # RPM = PWM2RPM_SCALE * PWM + PWM2RPM_CONST
        # At PWM=20000: RPM ~0, At PWM=65535: RPM ~7500
        self.PWM2RPM_SCALE = 0.1646
        self.PWM2RPM_CONST = -3292.0
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535

        # Motor mixer matrix for X-configuration (same geometry as CF2X)
        # Prop layout:  0(+,+)  1(-,+)
        #               3(+,-)  2(-,-)
        self.MIXER_MATRIX = np.array([
            [.5, -.5, -1],   # Motor 0: front-right
            [.5, .5, 1],     # Motor 1: front-left
            [-.5, .5, -1],   # Motor 2: rear-left
            [-.5, -.5, 1]    # Motor 3: rear-right
        ])

        self.reset()

    ################################################################################

    def reset(self):
        """Resets the control class.

        The previous step's and integral errors for both position and attitude are set to zero.

        """
        super().reset()
        # Store the last roll, pitch, and yaw
        self.last_rpy = np.zeros(3)
        # Initialized PID control variables
        self.last_pos_e = np.zeros(3)
        self.integral_pos_e = np.zeros(3)
        self.last_rpy_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)

    ################################################################################

    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3)
                       ):
        """Computes the PID control action (as RPMs) for a single drone.

        This method sequentially calls `_dslPIDPositionControl()` and `_dslPIDAttitudeControl()`.
        Parameter `cur_ang_vel` is unused.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        cur_ang_vel : ndarray
            (3,1)-shaped array of floats containing the current angular velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray, optional
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray, optional
            (3,1)-shaped array of floats containing the desired velocity.
        target_rpy_rates : ndarray, optional
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.
        ndarray
            (3,1)-shaped array of floats containing the current XYZ position error.
        float
            The current yaw error.

        """
        self.control_counter += 1
        thrust, computed_target_rpy, pos_e = self._dslPIDPositionControl(control_timestep,
                                                                         cur_pos,
                                                                         cur_quat,
                                                                         cur_vel,
                                                                         target_pos,
                                                                         target_rpy,
                                                                         target_vel
                                                                         )
        rpm = self._dslPIDAttitudeControl(control_timestep,
                                          thrust,
                                          cur_quat,
                                          computed_target_rpy,
                                          target_rpy_rates
                                          )
        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        return rpm, pos_e, computed_target_rpy[2] - cur_rpy[2]

    ################################################################################

    def _dslPIDPositionControl(self,
                               control_timestep,
                               cur_pos,
                               cur_quat,
                               cur_vel,
                               target_pos,
                               target_rpy,
                               target_vel
                               ):
        """Mavic 3 PID position control (adapted from DSL's CF2.x).

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray
            (3,1)-shaped array of floats containing the desired velocity.

        Returns
        -------
        float
            The target thrust along the drone z-axis.
        ndarray
            (3,1)-shaped array of floats containing the target roll, pitch, and yaw.
        float
            The current position error.

        """
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        pos_e = target_pos - cur_pos
        vel_e = target_vel - cur_vel
        self.integral_pos_e = self.integral_pos_e + pos_e * control_timestep
        self.integral_pos_e = np.clip(self.integral_pos_e, -2., 2.)
        self.integral_pos_e[2] = np.clip(self.integral_pos_e[2], -0.15, .15)

        # PID target thrust
        target_thrust = np.multiply(self.P_COEFF_FOR, pos_e) \
                        + np.multiply(self.I_COEFF_FOR, self.integral_pos_e) \
                        + np.multiply(self.D_COEFF_FOR, vel_e) + np.array([0, 0, self.GRAVITY])
        scalar_thrust = max(0., np.dot(target_thrust, cur_rotation[:, 2]))
        thrust = (math.sqrt(scalar_thrust / (4 * self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        target_z_ax = target_thrust / np.linalg.norm(target_thrust)
        target_x_c = np.array([math.cos(target_rpy[2]), math.sin(target_rpy[2]), 0])
        target_y_ax = np.cross(target_z_ax, target_x_c) / np.linalg.norm(np.cross(target_z_ax, target_x_c))
        target_x_ax = np.cross(target_y_ax, target_z_ax)
        target_rotation = (np.vstack([target_x_ax, target_y_ax, target_z_ax])).transpose()

        # Target rotation
        target_euler = (Rotation.from_matrix(target_rotation)).as_euler('XYZ', degrees=False)
        if np.any(np.abs(target_euler) > math.pi):
            print("\n[ERROR] ctrl it", self.control_counter,
                  "in Mavic3PIDControl._dslPIDPositionControl(), values outside range [-pi,pi]")
        return thrust, target_euler, pos_e

    ################################################################################

    def _dslPIDAttitudeControl(self,
                               control_timestep,
                               thrust,
                               cur_quat,
                               target_euler,
                               target_rpy_rates
                               ):
        """Mavic 3 PID attitude control (adapted from DSL's CF2.x).

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        thrust : float
            The target thrust along the drone z-axis.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        target_euler : ndarray
            (3,1)-shaped array of floats containing the computed target Euler angles.
        target_rpy_rates : ndarray
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.

        """
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        target_quat = (Rotation.from_euler('XYZ', target_euler, degrees=False)).as_quat()
        w, x, y, z = target_quat
        target_rotation = (Rotation.from_quat([w, x, y, z])).as_matrix()
        rot_matrix_e = np.dot((target_rotation.transpose()), cur_rotation) - np.dot(cur_rotation.transpose(),
                                                                                     target_rotation)
        rot_e = np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]])
        rpy_rates_e = target_rpy_rates - (cur_rpy - self.last_rpy) / control_timestep
        self.last_rpy = cur_rpy
        self.integral_rpy_e = self.integral_rpy_e - rot_e * control_timestep
        self.integral_rpy_e = np.clip(self.integral_rpy_e, -1500., 1500.)
        self.integral_rpy_e[0:2] = np.clip(self.integral_rpy_e[0:2], -1., 1.)

        # PID target torques
        target_torques = - np.multiply(self.P_COEFF_TOR, rot_e) \
                         + np.multiply(self.D_COEFF_TOR, rpy_rates_e) \
                         + np.multiply(self.I_COEFF_TOR, self.integral_rpy_e)
        target_torques = np.clip(target_torques, -3200, 3200)
        pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
        return self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST

    ################################################################################

    def _one23DInterface(self,
                         thrust
                         ):
        """Utility function interfacing 1, 2, or 3D thrust input use cases.

        Parameters
        ----------
        thrust : ndarray
            Array of floats of length 1, 2, or 4 containing a desired thrust input.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the PWM (not RPMs) to apply to each of the 4 motors.

        """
        DIM = len(np.array(thrust))
        pwm = np.clip((np.sqrt(np.array(thrust) / (self.KF * (4 / DIM))) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE,
                      self.MIN_PWM, self.MAX_PWM)
        if DIM in [1, 4]:
            return np.repeat(pwm, 4 / DIM)
        elif DIM == 2:
            return np.hstack([pwm, np.flip(pwm)])
        else:
            print("[ERROR] in Mavic3PIDControl._one23DInterface()")
            exit()
