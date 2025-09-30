import numpy as np
from gym import spaces
import pybullet as p

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.envs.single_agent_rl.LandingAviary import LandingAviary, LandingGimbalAviary

import inspect
from scipy.spatial.transform import Rotation as R


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

        # Note: Check if the replay buffer takes which reward during training: scalar (r_xx) or array (r_list)
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


if __name__ == "__main__":
    print("This is an outdated script. Get yours elsewhere.")
