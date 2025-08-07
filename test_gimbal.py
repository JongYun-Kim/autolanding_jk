import os
from pathlib import Path
import hydra
from omegaconf import OmegaConf
import torch
from train import Workspace
from video import VideoRecorder  # NEW: for recording evaluation videos
import csv


def evaluate_workspace(workspace, num_episodes: int = 128, video_every: int = 16):
    """Run `num_episodes` episodes in evaluation mode.

    A video of the environment is recorded every `video_every`‑th episode and
    saved into ``<workspace>/eval_videos``.
    """
    # ------------------------------------------------------------------
    # Set up a dedicated folder for evaluation videos
    video_root = workspace.work_dir / "eval_videos"
    video_root.mkdir(parents=True, exist_ok=True)

    total_reward = 0.0
    landing_count = 0
    crash_count = 0
    timeout_count = 0
    episodes = []

    for ep in range(1, num_episodes + 1):
        time_step = workspace.eval_env.reset()
        episode_reward = 0.0
        step_count = 0

        # --------------------------------------------------------------
        # Decide whether to capture this run
        record_video = (ep % video_every == 0)
        if record_video:
            video_recorder = VideoRecorder(video_root)
            video_recorder.init(workspace.eval_env, enabled=True)

        # Env Loop
        while not time_step.last():
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                action = workspace.agent.act(
                    time_step.observation,
                    workspace.global_step,
                    time_step.drone_state,
                    eval_mode=True,
                )
            time_step = workspace.eval_env.step(action)
            episode_reward += float(time_step.reward)
            step_count += 1

            if record_video:
                video_recorder.record(workspace.eval_env)

        if record_video:
            video_recorder.save(f"eval_ep{ep:04d}.mp4")

        # --------------------------------------------------------------
        # Determine outcome flags
        landing_flag = bool(getattr(time_step, "landing_info", False))
        if landing_flag:
            crash_flag = False
            timeout_flag = False
        else:
            info = workspace.eval_env.env.env.env._computeInfo()
            crash_flag = bool(info.get("episode end flag", False))
            timeout_flag = not crash_flag

        # --------------------------------------------------------------
        # Accumulate statistics
        total_reward += episode_reward
        if landing_flag:
            landing_count += 1
        elif crash_flag:
            crash_count += 1
        else:
            timeout_count += 1

        # Record per‑episode metrics
        episodes.append(
            {
                "episode": ep,
                "reward": episode_reward,
                "landing": int(landing_flag),
                "crash": int(crash_flag),
                "timeout": int(timeout_flag),
                "length": step_count,
            }
        )

        print(
            f"Episode {ep:3d} Reward: {episode_reward:.2f}, Length: {step_count:3d}")
        print(
            f"Episode {ep:3d} Flags: landed={landing_flag}, crashed={crash_flag}, timeout={timeout_flag}\n"
        )

    # ------------------------------------------------------------------
    # Aggregate metrics across all episodes
    avg_reward = total_reward / num_episodes
    landing_rate = landing_count / num_episodes
    crash_rate = crash_count / num_episodes
    timeout_rate = timeout_count / num_episodes

    return episodes, {
        "average_reward": avg_reward,
        "landing_rate": landing_rate,
        "crash_rate": crash_rate,
        "timeout_rate": timeout_rate,
    }


@hydra.main(config_path="cfgs", config_name="test_config")
# ----------------------------------------------------------------------
# No change to the Hydra wrapper, but we now call `evaluate_workspace` with
# the new `video_every` parameter (default 16)
# ----------------------------------------------------------------------
def my_tests(cfg):
    exps = setup_exp()

    for exp in exps:
        print(f"\n\n======== Running Experiment: {exp['model_type']} ========")

        model_type = exp["model_type"]
        target_dir = exp["target_dir"]
        num_episodes = exp["num_episodes"]

        snapshot_name = "snapshot.pt"
        snapshot_path = os.path.join(target_dir, snapshot_name)

        if not os.path.exists(target_dir):
            print(f"Directory {target_dir} does not exist. Skipping...")
            continue

        workspace = Workspace(cfg)
        workspace.load_snapshot(checkpoint_dir=snapshot_path)
        print(f"Loaded snapshot from {snapshot_path}")
        workspace.agent.eval()

        # --------------------------------------------------------------
        # NEW: pass `video_every=16` to enable periodic recording
        episodes, summary = evaluate_workspace(
            workspace, num_episodes=num_episodes, video_every=16
        )

        # --------------------------------------------------------------
        # Save per‑episode results
        results_file = Path(f"test_results_{model_type}.csv")
        with results_file.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "episode",
                    "reward",
                    "landing",
                    "crash",
                    "timeout",
                    "length",
                ],
            )
            writer.writeheader()
            for ep_data in episodes:
                writer.writerow(ep_data)

        # --------------------------------------------------------------
        # Save summary statistics
        summary["model_dir"] = snapshot_path
        summary["model_type"] = model_type
        summary["num_train_frames"] = workspace.cfg.num_train_frames
        summary["curriculum"] = (
            workspace.cfg.curriculum_config if "toddler" in model_type else "None"
        )
        summary["num_episodes"] = num_episodes

        summary_file = Path(f"summary_results_{model_type}.txt")
        with summary_file.open("w") as f:
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")

        print(f"Saved results for {model_type}.")

        # Print summary to console
        print("\n========= Summary of the Test =========")
        print(f"Model Dir: {summary['model_dir']}")
        print(f"Model Type: {summary['model_type']}")
        print(f"Curriculum: {summary['curriculum']}")
        print(f"Num Train Steps: {summary['num_train_frames']}")
        print(f"Average Reward: {summary['average_reward']:.2f}")
        print(f"Landing Rate : {summary['landing_rate']:.2%}")
        print(f"Crashed Rate : {summary['crash_rate']:.2%}")
        print(f"Timeout Rate : {summary['timeout_rate']:.2%}")


def setup_exp():
    num_episodes = 1024

    experiment_settings = [
        # {
        #     "model_type": "drqv2-mv-toddler-base", #160-400-3200",
        #     "target_dir": "/home/user/landing/exp_local/2025.05.07/210855_/",
        #     "num_episodes": num_episodes,
        # },
        # {
        #     "model_type": "drqv2-mv-toddler-s3",
        #     "target_dir": "/home/user/landing/exp_local/2025.06.02/225452_seed=3/",
        #     "num_episodes": num_episodes,
        # },
        # {
        #     "model_type": "drqv2-mv-s0",
        #     "target_dir": "/home/user/landing/exp_local/2025.05.17/031234_replay_buffer_size=160000,seed=0/",
        #     "num_episodes": num_episodes,
        # },
        # {
        #     "model_type": "drqv2-mv-s1",
        #     "target_dir": "/home/user/landing/exp_local/2025.05.18/112032_replay_buffer_size=160000,seed=1/",
        #     "num_episodes": num_episodes,
        # },
        {
            "model_type": "drqv2-tdr-s1",
            "target_dir": "/home/user/landing/exp_local/2025.06.20/161619_/",
            "num_episodes": num_episodes,
        },
    ]

    return experiment_settings


if __name__ == "__main__":
    my_tests()
