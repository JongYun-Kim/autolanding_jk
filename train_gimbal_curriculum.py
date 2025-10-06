import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path
import hydra
import torch
import dmc
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import VideoRecorder
from collections import deque
from gym_pybullet_drones.envs.single_agent_rl.LandingAviary import CurriculumStageSpec

torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                self.cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

        # Curriculum learning state
        self.curr_enabled = getattr(self.cfg, "curriculum", None) is not None and self.cfg.curriculum.enable
        self.curr_stage = (self.cfg.curriculum.initial_stage if self.curr_enabled else None)
        self.success_hist = deque(maxlen=(self.cfg.curriculum.window if self.curr_enabled else 1))

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        # create envs
        # self.train_env = dmc.make_with_gimbal(self.cfg.task_name, self.cfg.frame_stack,
        #                           self.cfg.action_repeat, self.cfg.seed)
        # self.eval_env = dmc.make_with_gimbal(self.cfg.task_name, self.cfg.frame_stack,
        #                          self.cfg.action_repeat, self.cfg.seed)
        env_kwargs = self._build_env_kwargs_from_cfg(self.cfg.curriculum_preset)
        self.train_env = dmc.make_with_gimbal(self.cfg.task_name, self.cfg.frame_stack, self.cfg.action_repeat,
                                              self.cfg.seed, env_kwargs)
        self.eval_env = dmc.make_with_gimbal(self.cfg.task_name, self.cfg.frame_stack, self.cfg.action_repeat,
                                             self.cfg.seed, env_kwargs)

        # 커리큘럼 초기 스테이지를 양쪽 env에 적용
        if getattr(self, "curr_enabled", False):
            self.train_env.set_curriculum_stage(self.cfg.curriculum.initial_stage)
            self.eval_env.set_curriculum_stage(self.cfg.curriculum.initial_stage)

        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      self.train_env.reward_spec(),
                      self.train_env.discount_spec(),
                      self.train_env.drone_state_spec())

        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')

        self.replay_loader = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount)
        self._replay_iter = None
        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)
        self.train_video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_train_video else None)

    def _build_env_kwargs_from_cfg(self, cfg):
        # 커리큘럼 비사용: 빈 kwargs
        if not (hasattr(cfg, "curriculum") and cfg.curriculum.enable):
            return {}
        # Hydra YAML -> dataclass 리스트
        stages = []
        for s in cfg.curriculum.stages:
            stages.append(CurriculumStageSpec(
                name = s.name,
                gimbal_enabled = bool(s.gimbal_enabled),
                lock_down = bool(s.lock_down),
                scale = tuple(float(x) for x in s.scale),  # [pitch, roll, yaw] in [0,1]
                include_viz_reward = bool(s.include_viz_reward),
                viz_weight = float(s.viz_weight),
                not_visible_penalty = float(s.not_visible_penalty),
                smooth_penalty = float(getattr(s, "smooth_penalty", 0.0)),
                yaw_only = bool(getattr(s, "yaw_only", False)),
                pitch_only = bool(getattr(s, "pitch_only", False)),
            ))
        return {
            "use_curriculum": True,
            "curriculum_cfg": stages,
        }

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            time_step.drone_state,
                                            eval_mode=True)
                #  action = np.append(action, -0.45).astype(np.float32)    # meraj
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')
        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)
            if getattr(self, "curr_enabled", False):
                log('curriculum_stage', self.curr_stage)

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step)
        self.train_video_recorder.init(self.train_env)
        metrics = None
        action = None

        while train_until_step(self.global_step):

            if time_step.last():
                if self._global_episode == 5:
                    self.train_video_recorder.save(f'{self.global_frame}.mp4')
                self._global_episode += 1

                # 커리큘럼: 에피소드 성공/실패 집계 및 스테이지 전환
                if getattr(self, "curr_enabled", False):
                    ep_success = bool(time_step.landing_info)  # 마지막 타임스텝에 설정됨
                    self.success_hist.append(1 if ep_success else 0)
                    # seed 단계 지난 이후에만 승급 평가
                    if (not seed_until_step(self.global_step)) and self._should_advance_curriculum():
                        self._advance_curriculum_stage()

                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)
                        if getattr(self, "curr_enabled", False):
                            log('curriculum_stage', self.curr_stage)
                            if len(self.success_hist) > 0:
                                log('success_rate_window', sum(self.success_hist) / len(self.success_hist))

                # Reset env
                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)
                self.train_video_recorder.init(self.train_env)
                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # Eval
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(), self.global_frame)
                self.eval()

            # Sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        self.global_step,
                                        time_step.drone_state,
                                        eval_mode=False)

            # Update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step)
            if self._global_episode == 5:
                self.train_video_recorder.record(self.train_env)
            episode_step += 1
            self._global_step += 1

        # Finalize: eval and save snapshot
        self.logger.log('eval_total_time', self.timer.total_time(), self.global_frame)
        self.eval()
        # 스냅샷 저장 (커리큘럼 상태 포함)
        if getattr(self, "curr_enabled", False):
            self.save_stage_checkpoint(tag="final")

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        # 커리큘럼 상태 포함
        if getattr(self, "curr_enabled", False):
            self.curriculum_state = {
                "enabled": True,
                "stage": self.curr_stage,
                "success_hist": list(self.success_hist),
            }
            keys_to_save.append("curriculum_state")

        payload = {k: self.__dict__[k] for k in keys_to_save}

        # 커리큘럼 상태 포함
        if getattr(self, "curr_enabled", False):
            payload["curriculum_state"] = {
                "enabled": True,
                "stage": self.curr_stage,
                "success_hist": list(self.success_hist),
            }

        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def save_stage_checkpoint(self, tag: str = ""):
        """
        커리큘럼 스테이지 종료 시 저장하는 전용 체크포인트.
        파일명 예: snapshot_stage2_S3_mid_120000.pt
        """

        if not getattr(self, "curr_enabled", False):
            return
        # 스테이지 이름 얻기 (eval_env/train_env 어느 쪽이든 동일 스테이지를 유지)
        try:
            info = self.train_env.get_curriculum_info()
            stage_idx = info["stage_idx"]
            stage_name = info["stage_name"]
        except Exception:
            stage_idx = int(self.curr_stage) if self.curr_stage is not None else -1
            stage_name = f"S{stage_idx}"
        safe_name = str(stage_name).replace("/", "_").replace(" ", "")
        fname = f"snapshot_stage{stage_idx}_{safe_name}_{self.global_frame}"
        if tag:
            fname += f"_{tag}"
        fname += ".pt"
        path = self.work_dir / fname
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        payload["curriculum_state"] = {
            "enabled": True,
            "stage": self.curr_stage,
            "success_hist": list(self.success_hist),
            "stage_name": stage_name,
        }
        with path.open('wb') as f:
            torch.save(payload, f)
        print(f"[Curriculum] Saved stage checkpoint: {path}")

    def load_snapshot(self, checkpoint_dir=None):
        if checkpoint_dir is None:
            snapshot = self.work_dir / 'snapshot.pt'
        else:
            snapshot = Path(checkpoint_dir)
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v

        # 로드 후 스테이지 재적용
        if getattr(self, "curriculum_state", None):
            self.curr_enabled = True
            self.curr_stage = self.curriculum_state.get("stage", 0)
            # env가 아직 없을 수 있으니 setup 이후에도 한 번 더 호출됨에 유의
            try:
                self.train_env.set_curriculum_stage(self.curr_stage)
                self.eval_env.set_curriculum_stage(self.curr_stage)
            except Exception:
                pass

    # 커리큘럼 판단/승급 로직
    def _should_advance_curriculum(self):
        cfgc = self.cfg.curriculum
        if self.curr_stage >= cfgc.max_stage:
            return False
        if len(self.success_hist) < cfgc.min_episodes:
            return False
        sr = sum(self.success_hist) / len(self.success_hist)
        return sr >= cfgc.success_rate

    def _advance_curriculum_stage(self):
        # 1) 현재 스테이지를 "마감"하며 스냅샷 저장
        self.save_stage_checkpoint(tag="end_of_stage")
        # 2) 다음 스테이지로 승급
        self.curr_stage += 1
        self.train_env.set_curriculum_stage(self.curr_stage)
        self.eval_env.set_curriculum_stage(self.curr_stage)
        print(f"[Curriculum] advanced to stage {self.curr_stage}")


@hydra.main(config_path='cfgs', config_name='config_gimbal_curriculum')
def main(cfg):
    root_dir = Path.cwd()
    workspace = Workspace(cfg)
    if getattr(cfg, "checkpoint_dir", None):
        workspace.load_snapshot(cfg.checkpoint_dir)
    else:
        snapshot = root_dir / 'snapshot.pt'
        if snapshot.exists():
            print(f'resuming: {snapshot}')
            workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()
