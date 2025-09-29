import os
from pathlib import Path
import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
# from train_toddler import Workspace
# from train_gimbal_oracle import Workspace
from train_gimbal import Workspace
# from train import Workspace
from video import VideoRecorder  # for recording evaluation videos
import csv

def evaluate_workspace(workspace, model_type: str, num_episodes: int = 128, enable_video_recording: bool = True):
    """Run `num_episodes` episodes in evaluation mode.

    If `enable_video_recording` is True, videos are recorded.
    A video of the environment is recorded for every episode and
    saved if it ends in a failure (crash or timeout), with the outcome and model_type in the filename.
    """
    # Set up a dedicated folder for evaluation videos
    video_root = workspace.work_dir / "eval_video"
    if enable_video_recording:
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

        # Create recorder only if enabled
        video_recorder = None
        if enable_video_recording:
            video_recorder = VideoRecorder(video_root)
            video_recorder.init_with_frame(workspace.eval_env, enabled=True)

        while not time_step.last():
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
                action = workspace.agent.act(
                    time_step.observation,
                    workspace.global_step,
                    time_step.drone_state,
                    eval_mode=True,
                )
            time_step = workspace.eval_env.step(action)
            episode_reward += float(time_step.reward)
            step_count += 1

            if enable_video_recording:
                video_recorder.record_with_frame(workspace.eval_env, action)

        # Determine outcome flags
        landing_flag = bool(getattr(time_step, "landing_info", False))
        if landing_flag:
            crash_flag = False
            timeout_flag = False
        else:
            info = workspace.eval_env.env.env.env._computeInfo()
            crash_flag = bool(info.get("episode end flag", False))
            timeout_flag = not crash_flag

        # Save video only if enabled
        if enable_video_recording:
            if crash_flag or timeout_flag:
                outcome = "crash" if crash_flag else "timeout"
                filename = f"{model_type}_eval_ep{ep:04d}_{outcome}.mp4"
                video_recorder.save(filename)
            else:
                # Successful landing videos are saved too, but marked as success
                video_recorder.save(f"{model_type}_eval_ep{ep:04d}_success.mp4")

        # Accumulate statistics
        total_reward += episode_reward
        if landing_flag:
            landing_count += 1
        elif crash_flag:
            crash_count += 1
        else:
            timeout_count += 1

        # Record per-episode metrics
        episodes.append({
            "episode": ep,
            "reward": episode_reward,
            "landing": int(landing_flag),
            "crash": int(crash_flag),
            "timeout": int(timeout_flag),
            "length": step_count,
            "last_reward": float(time_step.reward) if hasattr(time_step, 'reward') else 0.0,
        })

        print(f"Episode {ep:3d} Reward: {episode_reward:.2f}, Length: {step_count:3d}")
        print(f"Episode {ep:3d} Flags: landed={landing_flag}, crashed={crash_flag}, timeout={timeout_flag}\n")

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
def my_tests(cfg):
    enable_video_recording = True
    # enable_video_recording = False
    # Prompt to confirm if video recording is not enabled
    if not enable_video_recording:
        response = input("Video recording is disabled. Do you want to proceed? (y/n): ")
        if response.lower() != 'y':
            print("Exiting the test script.")
            return

    exps = setup_exp()

    for exp in exps:
        model_type = exp["model_type"]
        print(f"\n\n======== Running Experiment: {model_type} ========")

        target_dir = exp["target_dir"]
        num_episodes = exp["num_episodes"]

        if "snapshot_name" in exp:
            snapshot_name = exp["snapshot_name"]
        else:
            snapshot_name = "snapshot.pt"
        snapshot_path = os.path.join(target_dir, snapshot_name)

        if not os.path.exists(target_dir):
            print(f"Directory {target_dir} does not exist. Skipping...")
            continue

        workspace = Workspace(cfg)
        workspace.load_snapshot(checkpoint_dir=snapshot_path)
        print(f"Loaded snapshot from {snapshot_path}")
        workspace.agent.eval()

        # Evaluate and save only failed episode videos, prefixed with model_type
        episodes, summary = evaluate_workspace(
            workspace,
            model_type,
            num_episodes=num_episodes,
            enable_video_recording=enable_video_recording,
        )

        # Save per-episode results (all episodes)
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
                    "last_reward",
                ],
            )
            writer.writeheader()
            for ep_data in episodes:
                writer.writerow(ep_data)

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
    num_episodes = 100

    experiment_settings = [
        # {
        #     "model_type": "drqv2-oracle-gimbal",
        #     "target_dir": "/home/user/landing/exp_local/2025.08.19/192658_seed=42/",
        #     "num_episodes": num_episodes,
        # },
        # {
        #     "model_type": "drqv2-oracle-gimbal",
        #     "target_dir": "/home/user/landing/exp_local/2025.08.18/223300_seed=1/",
        #     "num_episodes": num_episodes,
        #     "snapshot_name": "snapshot_bk_2601k.pt",
        # },
        # {
        #     "model_type": "drqv2-oracle-gimbal-1400k",
        #     "target_dir": "/home/user/landing/exp_local/2025.08.21/015303_seed=42/",
        #     "num_episodes": num_episodes,
        #     "snapshot_name": "snapshot_bk_1400k.pt",
        # },
        # {
        #     "model_type": "drqv2-oracle-gimbal-1848k",
        #     "target_dir": "/home/user/landing/exp_local/2025.08.21/015303_seed=42/",
        #     "num_episodes": num_episodes,
        #     "snapshot_name": "snapshot_bk_1848k.pt",
        # },
        # {
        #     "model_type": "drqv2-oracle-gimbal-2270k",
        #     "target_dir": "/home/user/landing/exp_local/2025.08.21/015303_seed=42/",
        #     "num_episodes": num_episodes,
        #     "snapshot_name": "snapshot_bk_2270k.pt",
        # },
        # {
        #     "model_type": "drqv2-base-250819",
        #     "target_dir": "/home/user/landing/exp_local/2025.08.19/192507_seed=42/",
        #     "num_episodes": num_episodes,
        #     "snapshot_name": "snapshot.pt",
        # },
        # {
        #     "model_type": "drqv2-oracle-gimbal-w-viz-rwd-1235k",
        #     "target_dir": "/home/user/landing/exp_local/2025.09.01/001943_/",
        #     "num_episodes": num_episodes,
        #     "snapshot_name": "snapshot_bk_1235k.pt",
        # },
        # {
        #     "model_type": "drqv2-oracle-gimbal-w-viz-rwd-1750k",
        #     "target_dir": "/home/user/landing/exp_local/2025.09.01/001943_/",
        #     "num_episodes": num_episodes,
        #     "snapshot_name": "snapshot_bk_1750k.pt",
        # },
        # {
        #     "model_type": "drqv2-oracle-gimbal-w-noise-3914k",
        #     "target_dir": "/home/user/landing/exp_local/2025.09.01/001723_env.eval.is_noisy_gimbal=True,env.train.is_noisy_gimbal=True/",
        #     "num_episodes": num_episodes,
        #     "snapshot_name": "snapshot_3914k_for_test.pt",
        # },
        # {
        #     "model_type": "drqv2--gimbal-7m",
        #     "target_dir": "/home/user/landing/exp_local/2025.08.30/111223_/",
        #     "num_episodes": num_episodes,
        #     "snapshot_name": "snapshot.pt",
        # },
        # {
        #     "model_type": "drqv2-gimbal-0913-7m",
        #     "target_dir": "/home/user/landing/exp_local/2025.09.13/002322_num_train_frames=7000000,stddev_schedule='linear(1.0, 0.01, 6200000)'/",
        #     "num_episodes": num_episodes,
        #     "snapshot_name": "snapshot.pt",
        # },
        # {
        #     "model_type": "drqv2-gimbal-0913-4m",
        #     "target_dir": "/home/user/landing/exp_local/2025.09.13/002250_num_train_frames=4000000,stddev_schedule='linear(1.0, 0.01, 3600000)'/",
        #     "num_episodes": num_episodes,
        #     "snapshot_name": "snapshot.pt",
        # },
        # {
        #     "model_type": "drqv2-gimbal-0913-3m",
        #     "target_dir": "/home/user/landing/exp_local/2025.09.13/002135_num_train_frames=3000000,stddev_schedule='linear(1.0, 0.01, 2700000)'/",
        #     "num_episodes": num_episodes,
        #     "snapshot_name": "snapshot.pt",
        # },
        # {
        #     "model_type": "drqv2-gimbal-0913-7m-hyprtune-5729k",
        #     "target_dir": "/home/user/landing/exp_local/2025.09.13/215446_agent.stddev_clip=0.42,batch_size=512,feature_dim=128,num_train_frames=7000000,replay_buffer_size=500000,stddev_schedule='linear(1.0, 0.01, 6250000)'/",
        #     "num_episodes": num_episodes,
        #     "snapshot_name": "snapshot_5729k.pt",
        # },
        # {
        #     "model_type": "drqv2-gimbal-0913-7m-hyprtune-6270k",
        #     "target_dir": "/home/user/landing/exp_local/2025.09.13/215446_agent.stddev_clip=0.42,batch_size=512,feature_dim=128,num_train_frames=7000000,replay_buffer_size=500000,stddev_schedule='linear(1.0, 0.01, 6250000)'/",
        #     "num_episodes": num_episodes,
        #     "snapshot_name": "snapshot_6270k.pt",
        # },
        # {
        #     "model_type": "drqv2-gimbal-0913-7m-hyprtune-6844k",
        #     "target_dir": "/home/user/landing/exp_local/2025.09.13/215446_agent.stddev_clip=0.42,batch_size=512,feature_dim=128,num_train_frames=7000000,replay_buffer_size=500000,stddev_schedule='linear(1.0, 0.01, 6250000)'/",
        #     "num_episodes": num_episodes,
        #     "snapshot_name": "snapshot_6844k.pt",
        # },
        # {
        #     "model_type": "drqv2-gimbal-0913-7m-hyprtune-6859k",
        #     "target_dir": "/home/user/landing/exp_local/2025.09.13/215446_agent.stddev_clip=0.42,batch_size=512,feature_dim=128,num_train_frames=7000000,replay_buffer_size=500000,stddev_schedule='linear(1.0, 0.01, 6250000)'/",
        #     "num_episodes": num_episodes,
        #     "snapshot_name": "snapshot_6859k.pt",
        # },
        # {
        #     "model_type": "drqv2-gimbal-0913-7m-hyprtune-7000k",
        #     "target_dir": "/home/user/landing/exp_local/2025.09.13/215446_agent.stddev_clip=0.42,batch_size=512,feature_dim=128,num_train_frames=7000000,replay_buffer_size=500000,stddev_schedule='linear(1.0, 0.01, 6250000)'/",
        #     "num_episodes": num_episodes,
        #     "snapshot_name": "snapshot.pt",
        # },
        {
            "model_type": "drqv2-gimbal-0924-7m-full",
            "target_dir": '/home/user/landing/exp_local/2025.09.24/222400_num_train_frames=7000000,seed=23,stddev_schedule="linear(1.0, 0.01, 6250000)"/',
            "num_episodes": num_episodes,
            "snapshot_name": "snapshot.pt",
        },
    ]

    return experiment_settings

if __name__ == "__main__":
    my_tests()