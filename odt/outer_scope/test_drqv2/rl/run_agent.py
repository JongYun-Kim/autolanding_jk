import os
import time
from typing import Optional

import torch

from agent import Agent, AgentConfig
import observation_generator as og


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        print(f"[env_int] Failed to parse env var '{name}' as int. Using default={default}.")
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        print(f"[env_float] Failed to parse env var '{name}' as float. Using default={default}.")
        return default


def env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def main():
    # Read config from env
    device = env_str("DEVICE", "")
    if device == "":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[run] Using device: {device}")

    cfg = AgentConfig(
        frame_stack=env_int("FRAME_STACK", 3),
        height=env_int("H", 84),
        width=env_int("W", 84),
        drone_state_dim=env_int("DRONE_STATE_DIM", 7),
        feature_dim=env_int("FEATURE_DIM", 128),
        hidden_dim=env_int("HIDDEN_DIM", 1024),
        device=device,
    )

    weights_path = env_str("WEIGHTS", "/app/weights/policy.pt")
    freq_hz = env_float("FREQ_HZ", 32.0)
    eval_batch = env_int("BATCH", 1)
    seed = env_int("SEED", 42)

    # Instantiate agent & (optionally) load weights
    agent = Agent(cfg)
    loaded = False
    if weights_path and os.path.isfile(weights_path):
        loaded = agent.load_weights(weights_path)
    else:
        print(f"[run] Weights not found at '{weights_path}'. Using random init.")

    print(agent.summary())
    print(f"[run] Weights loaded: {loaded}")

    # Build observation generator
    gen = og.ObservationGenerator(
        frame_stack=cfg.frame_stack,
        height=cfg.height,
        width=cfg.width,
        drone_state_dim=cfg.drone_state_dim,
        batch_size=eval_batch,
        device=cfg.device,
        seed=seed,
    )

    # Inference loop at fixed frequency
    period = 1.0 / max(freq_hz, 1e-6)
    step = 0
    print(f"[run] Starting inference loop at ~{freq_hz:.2f} Hz (period {period*1000:.1f} ms)")
    try:
        while True:
            t0 = time.time()

            obs_img, drone_states = gen.next()
            with torch.inference_mode():
                action = agent.act(obs_img, drone_states)
            # action shape: (B, 3)
            action_np = action.detach().cpu().numpy()

            step += 1
            print(f"[run][step={step}] action={action_np}")

            # sleep to maintain frequency
            elapsed = time.time() - t0

            if elapsed < period:
                sleep_s = period - elapsed
            else:
                sleep_s = 0.0
                print(f"[run][step={step}] Warning: Inference took longer ({elapsed*1000:.1f} ms) than period ({period*1000:.1f} ms).")
            time.sleep(sleep_s)
    except KeyboardInterrupt:
        print("[run] Inference loop stopped by user.")


if __name__ == "__main__":
    main()