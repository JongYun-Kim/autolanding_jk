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
from collections import deque

from gym_pybullet_drones.envs.single_agent_rl.LandingAviary import LandingGimbalAviary
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType


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

# ============================================================================


class FrameStack:
    """Stack frames for observation."""
    def __init__(self, num_stack):
        self.num_stack = num_stack
        self._frames = deque([], maxlen=num_stack)

    def reset(self, obs):
        for _ in range(self.num_stack):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, obs):
        self._frames.append(obs)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate(list(self._frames), axis=0)


class DroneStateStack:
    """Stack drone states."""
    def __init__(self, num_stack, state_dim=11):
        self.num_stack = num_stack
        self.state_dim = state_dim
        self._states = deque([], maxlen=num_stack)

    def reset(self, state):
        for _ in range(self.num_stack):
            self._states.append(state)
        return self._get_state()

    def step(self, state):
        self._states.append(state)
        return self._get_state()

    def _get_state(self):
        return np.stack(list(self._states), axis=0)


def get_drone_state(env):
    """Extract drone state vector for the agent."""
    drone_state = env._getDroneStateVector(0)
    # velocity (3) + drone quaternion (4) + gimbal quaternion (4) = 11
    velocity = drone_state[10:13]
    drone_quat = drone_state[3:7]

    # Get gimbal quaternion
    if hasattr(env, 'gimbal_state_quat') and env.gimbal_state_quat is not None:
        gimbal_quat = env.gimbal_state_quat
    else:
        gimbal_quat = np.array([0, 0, 0, 1], dtype=np.float32)

    state = np.concatenate([velocity, drone_quat, gimbal_quat]).astype(np.float32)
    return state


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


def run_episode(env, agent, frame_stack, state_stack, is_5d_action, episode_seed):
    """Run a single episode and return results."""
    # Set seed before reset to ensure identical initialization
    set_all_seeds(episode_seed)

    obs = env.reset()
    obs_stacked = frame_stack.reset(obs)
    drone_state = get_drone_state(env)
    state_stacked = state_stack.reset(drone_state)

    total_reward = 0
    step_count = 0
    done = False

    while not done:
        # Get action from agent
        with torch.no_grad():
            action = agent.act(
                obs_stacked,
                step=1000000,  # Use high step to get deterministic action
                drone_states=state_stacked,
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
        obs, reward, done, info = env.step(env_action)

        # Update stacks
        obs_stacked = frame_stack.step(obs)
        drone_state = get_drone_state(env)
        state_stacked = state_stack.step(drone_state)

        total_reward += reward
        step_count += 1

    # Check if landing was successful
    success = info.get("landing", False)

    return {
        "success": success,
        "total_reward": total_reward,
        "steps": step_count,
        "x_error": info.get("x error", float('inf')),
        "y_error": info.get("y error", float('inf')),
    }


def evaluate_model(model_path, env, is_5d_action, num_episodes, base_seed=42):
    """Evaluate a model over multiple episodes."""
    print(f"\nLoading model from: {model_path}")
    agent = load_model(model_path, DEVICE)

    frame_stack = FrameStack(FRAME_STACK)
    state_stack = DroneStateStack(FRAME_STACK)

    results = []
    successes = 0

    for ep in range(num_episodes):
        episode_seed = base_seed + ep
        result = run_episode(env, agent, frame_stack, state_stack, is_5d_action, episode_seed)
        results.append(result)

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


def create_eval_environment():
    """Create the evaluation environment."""
    env = LandingGimbalAviary(
        drone_model=DroneModel.CF2X,
        initial_xyzs=None,
        initial_rpys=None,
        physics=Physics.PYB_GND_DRAG_DW,
        freq=240,
        aggregate_phy_steps=10,
        gui=False,
        record=False,
        obs=ObservationType.RGB,
        act=ActionType.VEL,
        episode_len_sec=EPISODE_LEN_SEC,
        gv_path_type="straight",
    )
    return env


def main():
    print("=" * 60)
    print("Model Comparison Evaluation")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Number of episodes per model: {NUM_EPISODES}")
    print(f"Frame stack: {FRAME_STACK}")
    print(f"Episode length: {EPISODE_LEN_SEC}s")
    print("=" * 60)

    # Create environment
    print("\nCreating evaluation environment...")
    env = create_eval_environment()

    # Base seed for reproducibility
    base_seed = 42

    # Evaluate 3D action model (drone-only)
    print("\n" + "=" * 60)
    print("Evaluating Model 1: Drone-only (3D actions)")
    print("=" * 60)
    results_3d = evaluate_model(
        MODEL_3D_PATH,
        env,
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
        env,
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

    # Close environment
    env.close()

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
