"""
Evaluation script to compare two trained models:
1. Drone-only model (3D actions) from train.py
2. Drone+Gimbal model (5D actions) from train_gimbal_curriculum.py

Both models are evaluated on the same LandingGimbalAviary environment
with identical episode initializations for fair comparison.
"""

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

import random
import numpy as np
import torch
from pathlib import Path

import dmc


# ============================================================================
# CONFIGURATION - Update these paths to your actual model checkpoints
# ============================================================================

# Model 1: Drone-only model (3D actions) from train.py
MODEL_3D_PATH = "/path/to/your/drone_only_model/snapshot.pt"  # UPDATE THIS

# Model 2: Drone+Gimbal model (5D actions) from train_gimbal_curriculum.py
MODEL_5D_PATH = "/path/to/your/gimbal_model/snapshot.pt"  # UPDATE THIS

# Evaluation settings
NUM_EPISODES = 100
FRAME_STACK = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPISODE_LEN_SEC = 18

# Environment name (registered gym environment)
ENV_NAME_3D = "landing-aviary-v0"  # For drone-only model
ENV_NAME_5D = "landing-aviary-v0-gimbal"  # For gimbal model

# ============================================================================


def set_all_seeds(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_model(checkpoint_path, device):
    """Load a model checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    with checkpoint_path.open('rb') as f:
        payload = torch.load(f, map_location=device)

    agent = payload['agent']
    agent.device = device

    # Move model components to device
    agent.encoder = agent.encoder.to(device)
    agent.actor = agent.actor.to(device)
    agent.critic = agent.critic.to(device)

    return agent


def run_episode(env, agent, is_5d_action, episode_seed):
    """Run a single episode and return results."""
    # Set seed before reset to ensure identical initialization
    set_all_seeds(episode_seed)

    time_step = env.reset()

    total_reward = 0
    step_count = 0

    while not time_step.last():
        # Get observation and drone state from time_step
        obs = time_step.observation
        drone_state = time_step.drone_state

        # Ensure proper types (float32)
        obs = np.asarray(obs, dtype=np.float32)
        drone_state = np.asarray(drone_state, dtype=np.float32)

        # Get action from agent
        with torch.no_grad():
            action = agent.act(
                obs,
                step=1000000,  # Use high step to get deterministic action
                drone_states=drone_state,
                eval_mode=True
            )

        # Handle action dimensions
        if is_5d_action:
            # 5D action: [vx, vy, vz, pitch, yaw]
            env_action = action
        else:
            # 3D action: [vx, vy, vz] -> pad with zeros for gimbal
            env_action = np.concatenate([action, np.array([0.0, 0.0], dtype=np.float32)])

        # Step environment
        time_step = env.step(env_action)

        total_reward += time_step.reward
        step_count += 1

    # Check if landing was successful
    success = time_step.landing_info
    position_error = time_step.position_error

    return {
        "success": success,
        "total_reward": total_reward,
        "steps": step_count,
        "x_error": position_error[0] if position_error else float('inf'),
        "y_error": position_error[1] if position_error else float('inf'),
    }


def evaluate_model(model_path, env_name, is_5d_action, num_episodes, base_seed=42):
    """Evaluate a model over multiple episodes."""
    print(f"\nLoading model from: {model_path}")
    agent = load_model(model_path, DEVICE)

    results = []
    successes = 0

    for ep in range(num_episodes):
        episode_seed = base_seed + ep

        # Set seed before creating environment
        set_all_seeds(episode_seed)

        # Create environment with proper wrapper for this model type
        if is_5d_action:
            env = dmc.make_with_gimbal(env_name, FRAME_STACK, 1, episode_seed)
        else:
            env = dmc.make(env_name, FRAME_STACK, 1, episode_seed)

        result = run_episode(env, agent, is_5d_action, episode_seed)
        results.append(result)

        # Close env after each episode
        env.close()

        if result["success"]:
            successes += 1

        # Progress update every 10 episodes
        if (ep + 1) % 10 == 0:
            current_sr = successes / (ep + 1)
            print(f"  Episode {ep + 1}/{num_episodes} - Current Success Rate: {current_sr:.2%}")

    # Compute statistics
    success_rate = successes / num_episodes
    avg_reward = np.mean([r["total_reward"] for r in results])
    avg_steps = np.mean([r["steps"] for r in results])

    # Compute average errors for successful landings
    successful_results = [r for r in results if r["success"]]
    if successful_results:
        avg_x_error = np.mean([r["x_error"] for r in successful_results])
        avg_y_error = np.mean([r["y_error"] for r in successful_results])
    else:
        avg_x_error = float('inf')
        avg_y_error = float('inf')

    return {
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "avg_steps": avg_steps,
        "avg_x_error": avg_x_error,
        "avg_y_error": avg_y_error,
        "successes": successes,
        "total_episodes": num_episodes,
        "results": results,
    }


def main():
    print("=" * 60)
    print("Model Comparison Evaluation")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Number of episodes per model: {NUM_EPISODES}")
    print(f"Frame stack: {FRAME_STACK}")
    print(f"Episode length: {EPISODE_LEN_SEC}s")
    print("=" * 60)

    # Base seed for reproducibility
    base_seed = 42

    # Evaluate 3D action model (drone-only)
    print("\n" + "=" * 60)
    print("Evaluating Model 1: Drone-only (3D actions)")
    print("=" * 60)
    results_3d = evaluate_model(
        MODEL_3D_PATH,
        ENV_NAME_3D,
        is_5d_action=False,
        num_episodes=NUM_EPISODES,
        base_seed=base_seed
    )

    # Evaluate 5D action model (drone+gimbal)
    print("\n" + "=" * 60)
    print("Evaluating Model 2: Drone+Gimbal (5D actions)")
    print("=" * 60)
    results_5d = evaluate_model(
        MODEL_5D_PATH,
        ENV_NAME_5D,
        is_5d_action=True,
        num_episodes=NUM_EPISODES,
        base_seed=base_seed
    )

    # Print comparison results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    print("\nModel 1: Drone-only (3D actions)")
    print(f"  Success Rate: {results_3d['success_rate']:.2%} ({results_3d['successes']}/{results_3d['total_episodes']})")
    print(f"  Average Reward: {results_3d['avg_reward']:.2f}")
    print(f"  Average Steps: {results_3d['avg_steps']:.1f}")
    print(f"  Average X Error (successful): {results_3d['avg_x_error']:.4f}")
    print(f"  Average Y Error (successful): {results_3d['avg_y_error']:.4f}")

    print("\nModel 2: Drone+Gimbal (5D actions)")
    print(f"  Success Rate: {results_5d['success_rate']:.2%} ({results_5d['successes']}/{results_5d['total_episodes']})")
    print(f"  Average Reward: {results_5d['avg_reward']:.2f}")
    print(f"  Average Steps: {results_5d['avg_steps']:.1f}")
    print(f"  Average X Error (successful): {results_5d['avg_x_error']:.4f}")
    print(f"  Average Y Error (successful): {results_5d['avg_y_error']:.4f}")

    # Per-episode comparison
    print("\n" + "=" * 60)
    print("PER-EPISODE COMPARISON")
    print("=" * 60)

    both_success = 0
    only_3d_success = 0
    only_5d_success = 0
    both_fail = 0

    for i in range(NUM_EPISODES):
        r3d = results_3d['results'][i]
        r5d = results_5d['results'][i]

        if r3d['success'] and r5d['success']:
            both_success += 1
        elif r3d['success'] and not r5d['success']:
            only_3d_success += 1
        elif not r3d['success'] and r5d['success']:
            only_5d_success += 1
        else:
            both_fail += 1

    print(f"Both succeeded: {both_success} episodes")
    print(f"Only 3D succeeded: {only_3d_success} episodes")
    print(f"Only 5D succeeded: {only_5d_success} episodes")
    print(f"Both failed: {both_fail} episodes")

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
