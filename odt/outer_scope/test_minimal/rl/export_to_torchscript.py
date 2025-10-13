import os
import torch
from model import MinimalMLP, PolicyConfig

def main():
    obs_dim = int(os.getenv("OBS_DIM", "16"))
    action_dim = int(os.getenv("ACTION_DIM", "2"))
    discrete = os.getenv("DISCRETE", "0") == "1"
    out_path = os.getenv("OUT", "/app/weights/policy.ts")

    cfg = PolicyConfig(obs_dim=obs_dim, action_dim=action_dim, discrete=discrete)
    policy = MinimalMLP(cfg).eval()

    class Wrapper(torch.nn.Module):
        def __init__(self, p):
            super().__init__()
            self.p = p
        def forward(self, obs: torch.Tensor):
            # det action only
            return self.p.act(obs, deterministic=True)

    scripted = torch.jit.script(Wrapper(policy))
    scripted.save(out_path)
    print("saved:", out_path)

if __name__ == "__main__":
    main()