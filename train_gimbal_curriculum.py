import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path
import json
import glob
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

        # === Curriculum config/state ===
        self.curr_enabled = getattr(self.cfg, "curriculum_preset", None) is not None and self.cfg.curriculum_preset.curriculum.enable
        self.curr_stage = (self.cfg.curriculum_preset.curriculum.initial_stage if self.curr_enabled else None)

        if self.curr_enabled:
            cfgc = self.cfg.curriculum_preset.curriculum
            # base knobs
            self._c_window = int(cfgc.window)
            self._c_min_episodes = int(cfgc.min_episodes)
            self._c_success_rate = float(cfgc.success_rate)
            self._c_max_stage = int(cfgc.max_stage)
            # stability knobs (new, with back-compat defaults)
            self._c_cooldown_episodes = int(getattr(cfgc, "cooldown_episodes", 0))
            self._c_min_stage_episodes = int(getattr(cfgc, "min_stage_episodes", self._c_min_episodes))
            self._c_bootstrap_failures = int(getattr(cfgc, "bootstrap_failures", 0))
            self._c_require_consec = int(getattr(cfgc, "require_consecutive_windows", 4))
            assert self._c_min_episodes <= self._c_window, "curriculum.min_episodes must not be greater than curriculum.window"
            assert self._c_require_consec >= 1

        # runtime stats
        self.success_hist = deque(maxlen=(self._c_window if self.curr_enabled else 1))
        self.stage_episode_count = 0
        self.cooldown_left = 0
        self.consecutive_passes = 0

        if self.curr_enabled:
            print(f"[Curriculum] Enabled. Starting at stage {self.curr_stage}.")
            self._reset_stage_statistics(new_stage=self.curr_stage, bootstrap=(self._c_bootstrap_failures > 0))

        # Best checkpoints metadata file (per-stage structure)
        self.best_meta_path = self.work_dir / "best_checkpoints.json"
        if not self.best_meta_path.exists():
            self._write_best_meta({})

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        # create envs
        env_kwargs = self._build_env_kwargs_from_cfg(self.cfg.curriculum_preset)
        self.train_env = dmc.make_with_gimbal(self.cfg.task_name, self.cfg.frame_stack, self.cfg.action_repeat,
                                              self.cfg.seed, env_kwargs)
        self.eval_env = dmc.make_with_gimbal(self.cfg.task_name, self.cfg.frame_stack, self.cfg.action_repeat,
                                             self.cfg.seed, env_kwargs)

        # 커리큘럼 초기 스테이지를 양쪽 env에 적용
        if getattr(self, "curr_enabled", False):
            init_stage = self.cfg.curriculum_preset.curriculum.initial_stage
            self.train_env.set_curriculum_stage(init_stage)
            self.eval_env.set_curriculum_stage(init_stage)

        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      self.train_env.reward_spec(),
                      self.train_env.discount_spec(),
                      self.train_env.drone_state_spec())

        self.replay_storage = ReplayBufferStorage(data_specs, self.work_dir / 'buffer')
        self.replay_loader = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount)
        self._replay_iter = None
        self.video_recorder = VideoRecorder(self.work_dir if self.cfg.save_video else None)
        self.train_video_recorder = VideoRecorder(self.work_dir if self.cfg.save_train_video else None)

    def _build_env_kwargs_from_cfg(self, cfg):
        # 커리큘럼 비사용: 빈 kwargs
        preset = cfg
        if not (hasattr(preset, "curriculum") and preset.curriculum.enable):
            return {}
        # Hydra YAML -> dataclass 리스트
        stages = []
        for s in preset.curriculum.stages:
            stages.append(CurriculumStageSpec(
                name=s.name,
                gimbal_enabled=bool(s.gimbal_enabled),
                lock_down=bool(s.lock_down),
                scale=tuple(float(x) for x in s.scale),
                include_viz_reward=bool(s.include_viz_reward),
                viz_weight=float(s.viz_weight),
                not_visible_penalty=float(s.not_visible_penalty),
                smooth_penalty=float(getattr(s, "smooth_penalty", 0.0)),
                yaw_only=bool(getattr(s, "yaw_only", False)),
                pitch_only=bool(getattr(s, "pitch_only", False)),
            ))
        return {"use_curriculum": True, "curriculum_cfg": stages}

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

    # Curriculum helpers
    def _record_episode_result(self, time_step):
        if not getattr(self, "curr_enabled", False):
            return
        # 성공/실패 기록
        ep_success = bool(getattr(time_step, "landing_info", False))
        self.success_hist.append(1 if ep_success else 0)
        self.stage_episode_count += 1
        # Cooldown 감소
        if self.cooldown_left > 0:
            self.cooldown_left -= 1
        # consecutive_passes 업데이트
        if len(self.success_hist) >= self._c_min_episodes:
            sr = self._window_success_rate()
            if sr >= self._c_success_rate:
                self.consecutive_passes += 1
            else:
                self.consecutive_passes = 0

    def _window_success_rate(self):
        return (sum(self.success_hist) / len(self.success_hist)) if len(self.success_hist) > 0 else 0.0

    def _eligible_to_evaluate(self, seed_until_step):
        if not getattr(self, "curr_enabled", False):
            return False
        if seed_until_step(self.global_step):
            return False
        if self.cooldown_left > 0:
            return False
        if self.stage_episode_count < self._c_min_stage_episodes:
            return False
        if len(self.success_hist) < self._c_min_episodes:
            return False
        if self.curr_stage >= self._c_max_stage:
            return False
        return True

    def _should_advance_curriculum(self, seed_until_step):
        if not self._eligible_to_evaluate(seed_until_step):
            return False
        return self.consecutive_passes >= self._c_require_consec

    def _reset_stage_statistics(self, new_stage, bootstrap=False):
        self.curr_stage = int(new_stage)
        self.success_hist = deque(maxlen=self._c_window)
        self.stage_episode_count = 0
        self.consecutive_passes = 0
        self.cooldown_left = int(self._c_cooldown_episodes)
        if bootstrap:
            for _ in range(self._c_bootstrap_failures):
                self.success_hist.append(0)

    # ---- state (de)serialization helpers ----
    def _build_curriculum_state(self):
        if not getattr(self, "curr_enabled", False):
            return None
        return {
            "enabled": True,
            "stage": int(self.curr_stage),
            "success_hist": list(self.success_hist),
            "stage_episode_count": int(self.stage_episode_count),
            "cooldown_left": int(self.cooldown_left),
            "consecutive_passes": int(self.consecutive_passes),
            "window": int(self._c_window),
            "min_episodes": int(self._c_min_episodes),
            "success_rate": float(self._c_success_rate),
            "max_stage": int(self._c_max_stage),
            "cooldown_episodes": int(self._c_cooldown_episodes),
            "min_stage_episodes": int(self._c_min_stage_episodes),
            "bootstrap_failures": int(self._c_bootstrap_failures),
            "require_consecutive_windows": int(self._c_require_consec),
        }

    def _apply_curriculum_state(self, st):
        # knobs
        self._c_window = int(st.get("window", self._c_window))
        self._c_min_episodes = int(st.get("min_episodes", self._c_min_episodes))
        self._c_success_rate = float(st.get("success_rate", self._c_success_rate))
        self._c_max_stage = int(st.get("max_stage", self._c_max_stage))
        self._c_cooldown_episodes = int(st.get("cooldown_episodes", self._c_cooldown_episodes))
        self._c_min_stage_episodes = int(st.get("min_stage_episodes", self._c_min_stage_episodes))
        self._c_bootstrap_failures = int(st.get("bootstrap_failures", self._c_bootstrap_failures))
        self._c_require_consec = int(st.get("require_consecutive_windows", self._c_require_consec))
        # runtime
        self.curr_stage = int(st.get("stage", self.curr_stage or 0))
        hist_list = st.get("success_hist", [])
        self.success_hist = deque(hist_list, maxlen=self._c_window)
        self.stage_episode_count = int(st.get("stage_episode_count", 0))
        self.cooldown_left = int(st.get("cooldown_left", 0))
        self.consecutive_passes = int(st.get("consecutive_passes", 0))
        # reapply to envs
        try:
            self.train_env.set_curriculum_stage(self.curr_stage)
            self.eval_env.set_curriculum_stage(self.curr_stage)
        except Exception:
            pass

    def train(self):
        train_until_step = utils.Until(self.cfg.num_train_frames, self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames, self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames, self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step)
        self.train_video_recorder.init(self.train_env)
        metrics = None
        action = None

        # training loop
        while train_until_step(self.global_step):

            # (-1) at the end of episode:
            if time_step.last():
                if self._global_episode == 5:
                    self.train_video_recorder.save(f'{self.global_frame}.mp4')
                self._global_episode += 1

                # curriculum bookkeeping
                if getattr(self, "curr_enabled", False):
                    self._record_episode_result(time_step)
                    if self._should_advance_curriculum(seed_until_step):
                        self._advance_curriculum_stage()

                # logging
                if metrics is not None:
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame, ty='train') as log:
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
                                sr = self._window_success_rate()
                                log('success_rate_window', sr)
                                log('stage_episode_count', self.stage_episode_count)
                                log('cooldown_left', self.cooldown_left)
                                log('consecutive_passes', self.consecutive_passes)

                # snapshotting
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                    if getattr(self, "curr_enabled", False):
                        # Top-8 best checkpoints when SR >= 0.8
                        sr = self._window_success_rate()
                        self._save_best_checkpoint_if_needed(sr_threshold=0.8, sr=sr)

                # reset env/episode
                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)
                self.train_video_recorder.init(self.train_env)
                episode_step = 0
                episode_reward = 0

            # (0) periodic eval
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(), self.global_frame)
                self.eval()

            # (1) act
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        self.global_step,
                                        time_step.drone_state,
                                        eval_mode=False)

            # (2) learn
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # (3) env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step)
            if self._global_episode == 5:
                self.train_video_recorder.record(self.train_env)
            episode_step += 1
            self._global_step += 1

        # finalize training
        self.logger.log('eval_total_time', self.timer.total_time(), self.global_frame)
        self.eval()
        if getattr(self, "curr_enabled", False):
            self.save_stage_checkpoint(tag="final")

    # Snapshot I/O
    def _base_payload(self):
        return {
            'agent': self.agent,
            'timer': self.timer,
            '_global_step': self._global_step,
            '_global_episode': self._global_episode,
        }

    def save_snapshot(self):
        """Lightweight rolling checkpoint."""
        snapshot = self.work_dir / 'snapshot.pt'
        payload = self._base_payload()
        if getattr(self, "curr_enabled", False):
            payload['curriculum_state'] = self._build_curriculum_state()
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def save_stage_checkpoint(self, tag: str = ""):
        """Checkpoint at stage boundaries."""
        if not getattr(self, "curr_enabled", False):
            return
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

        payload = self._base_payload()
        payload["curriculum_state"] = self._build_curriculum_state()
        with path.open('wb') as f:
            torch.save(payload, f)
        print(f"[Curriculum] Saved stage checkpoint: {path}")

    def load_snapshot(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = self.work_dir / 'snapshot.pt'
        else:
            checkpoint_path = Path(checkpoint_path)
        with checkpoint_path.open('rb') as f:
            payload = torch.load(f)
        # restore base
        self.agent = payload['agent']
        self.timer = payload['timer']
        self._global_step = payload.get('_global_step', 0)
        self._global_episode = payload.get('_global_episode', 0)
        # restore curriculum
        st = payload.get("curriculum_state", None)
        if st:
            self.curr_enabled = True
            self._apply_curriculum_state(st)

    # Per-stage best checkpoints logic
    def _read_best_meta(self):
        """Read per-stage best checkpoints metadata.
        Returns dict: {stage_idx: [{"sr": ..., "frame": ..., "path": ...}, ...]}
        """
        try:
            with self.best_meta_path.open('r') as f:
                data = json.load(f)
                # Handle legacy format (list) by converting to new format
                if isinstance(data, list):
                    print("[Checkpoint] Converting legacy best_checkpoints.json format to per-stage format")
                    return {}
                return data
        except Exception:
            return {}

    def _write_best_meta(self, items):
        """Write per-stage best checkpoints metadata.
        items: dict {stage_idx: [{"sr": ..., "frame": ..., "path": ...}, ...]}
        """
        with self.best_meta_path.open('w') as f:
            json.dump(items, f, indent=2)

    def _save_best_checkpoint_if_needed(self, sr_threshold, sr):
        """Save best checkpoints per curriculum stage.
        - For non-final stages: keep top 3 models
        - For final stage: keep top 8 models
        Models do not compete across stages.
        """
        if sr < sr_threshold:
            return

        # Determine how many models to keep for this stage
        is_final_stage = (self.curr_stage >= self._c_max_stage)
        max_keep = 8 if is_final_stage else 3

        # create checkpoint path
        sr_pct = int(round(sr * 100))
        fname = f"snapshot_best_sr{sr_pct}_stage{self.curr_stage}_{self.global_frame}.pt"
        path = self.work_dir / fname

        payload = self._base_payload()
        if getattr(self, "curr_enabled", False):
            payload["curriculum_state"] = self._build_curriculum_state()
        with path.open('wb') as f:
            torch.save(payload, f)

        # Update per-stage leaderboard
        all_stages = self._read_best_meta()
        stage_key = str(self.curr_stage)

        # Get or initialize list for this stage
        stage_items = all_stages.get(stage_key, [])
        stage_items.append({"sr": sr, "frame": int(self.global_frame), "path": str(path)})

        # Sort: sr desc, frame desc
        stage_items.sort(key=lambda x: (x["sr"], x["frame"]), reverse=True)

        # Keep only top N for this stage; delete extras on disk
        extras = stage_items[max_keep:]
        stage_items = stage_items[:max_keep]
        for ex in extras:
            try:
                Path(ex["path"]).unlink(missing_ok=True)
            except Exception:
                pass

        # Update the stage entry
        all_stages[stage_key] = stage_items
        self._write_best_meta(all_stages)

        stage_type = "FINAL" if is_final_stage else f"stage {self.curr_stage}"
        print(f"[Checkpoint] Saved BEST (sr={sr:.3f}) for {stage_type} -> {path.name}. Kept top-{max_keep} for this stage.")

    # Auto resume: does not always resume the best checkpoint
    def auto_resume_if_possible(self):
        """
        Try checkpoints in order:
        1) snapshot.pt
        2) best_checkpoints.json -> pick the newest (by frame) across all stages
        3) latest snapshot_stage*.pt by frame number
        """
        snap = self.work_dir / 'snapshot.pt'
        if snap.exists():
            print(f"[Resume] Found snapshot.pt -> {snap}")
            self.load_snapshot(snap)
            return True

        # try best checkpoints from all stages
        all_stages = self._read_best_meta()
        if all_stages:
            # Collect all checkpoints from all stages
            all_checkpoints = []
            for stage_key, stage_items in all_stages.items():
                all_checkpoints.extend(stage_items)

            if all_checkpoints:
                # pick newest by frame
                all_checkpoints.sort(key=lambda x: x["frame"], reverse=True)
                for cand in all_checkpoints:
                    p = Path(cand["path"])
                    if p.exists():
                        print(f"[Resume] Found BEST checkpoint -> {p}")
                        self.load_snapshot(p)
                        return True

        # try stage snapshots (pick largest frame in filename)
        candidates = list(self.work_dir.glob("snapshot_stage*_*_*.pt"))
        def _frame_from_name(pth: Path):
            try:
                # pattern ..._<frame>.pt (frame is last token before .pt)
                return int(pth.stem.split("_")[-1])
            except Exception:
                return -1
        candidates.sort(key=_frame_from_name, reverse=True)
        for p in candidates:
            if p.exists():
                print(f"[Resume] Found stage checkpoint -> {p}")
                self.load_snapshot(p)
                return True

        print("[Resume] No checkpoint found. Starting fresh.")
        return False

    def _advance_curriculum_stage(self):
        self.save_stage_checkpoint(tag="end_of_stage")
        next_stage = self.curr_stage + 1
        self.train_env.set_curriculum_stage(next_stage)
        self.eval_env.set_curriculum_stage(next_stage)
        print(f"[Curriculum] advanced to stage {next_stage}")
        self._reset_stage_statistics(new_stage=next_stage, bootstrap=(self._c_bootstrap_failures > 0))


@hydra.main(config_path='cfgs', config_name='config_gimbal_curriculum')
def main(cfg):
    workspace = Workspace(cfg)

    # Explicit user-provided checkpoint takes precedence
    if getattr(cfg, "checkpoint_dir", None):
        workspace.load_snapshot(cfg.checkpoint_dir)
    else:
        # try snapshot.pt, best, or stage snapshots
        workspace.auto_resume_if_possible()

    workspace.train()


if __name__ == '__main__':
    main()