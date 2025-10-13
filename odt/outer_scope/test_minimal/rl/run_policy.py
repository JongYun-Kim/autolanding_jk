import os
import torch
import numpy as np
from model import MinimalMLP, PolicyConfig

def load_policy(weights_path: str, cfg: PolicyConfig):
    policy = MinimalMLP(cfg)
    # state_dict 또는 TorchScript 둘 다 대응
    if weights_path.endswith(".pt") or weights_path.endswith(".pth"):
        state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            policy.load_state_dict(state["state_dict"])
        elif isinstance(state, dict):
            policy.load_state_dict(state)
        else:
            # 스크립트 모듈일 수도 있음
            try:
                return torch.jit.load(weights_path, map_location="cpu").eval()
            except Exception as _:
                raise RuntimeError("알 수 없는 weight 형식")
        policy.eval()
        return policy
    elif weights_path.endswith(".ts"):
        return torch.jit.load(weights_path, map_location="cpu").eval()
    else:
        raise ValueError("지원하지 않는 weight 확장자")

def main():
    # 환경 변수로 설정 (Docker에서 넘기기 쉬움)
    obs_dim = int(os.getenv("OBS_DIM", "16"))
    action_dim = int(os.getenv("ACTION_DIM", "2"))
    discrete = os.getenv("DISCRETE", "0") == "1"
    weights = os.getenv("WEIGHTS", "/app/weights/policy.pt")
    deterministic = os.getenv("DET", "1") == "1"

    cfg = PolicyConfig(obs_dim=obs_dim, action_dim=action_dim, discrete=discrete)
    policy = load_policy(weights, cfg)

    # 데모 입력(실전에서는 실제 관측을 넣으면 됨)
    obs = torch.from_numpy(np.random.randn(1, obs_dim).astype(np.float32))
    with torch.inference_mode():
        if hasattr(policy, "act"):
            act = policy.act(obs, deterministic=deterministic)
        else:
            # TorchScript 모듈인 경우 forward 사용
            act = policy(obs)
    print("action:", act.cpu().numpy())

if __name__ == "__main__":
    main()