import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical

@dataclass
class PolicyConfig:
    obs_dim: int
    action_dim: int
    discrete: bool = False   # True면 이산 행동(로짓), False면 연속 행동(가우시안)
    hidden_dim: int = 64     # 작게 유지
    log_std_min: float = -5.0
    log_std_max: float = 2.0

class MinimalMLP(nn.Module):
    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_dim
        self.net = nn.Sequential(
            nn.Linear(cfg.obs_dim, h), nn.ReLU(inplace=True),
            nn.Linear(h, h), nn.ReLU(inplace=True),
        )
        if cfg.discrete:
            self.head = nn.Linear(h, cfg.action_dim)  # logits
        else:
            self.mu = nn.Linear(h, cfg.action_dim)
            self.log_std = nn.Parameter(torch.zeros(cfg.action_dim))
        self.discrete = cfg.discrete

        # 경량화: 초기화 간단하게
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """
        obs: [B, obs_dim]
        return: action tensor
        """
        x = self.net(obs)
        if self.discrete:
            logits = self.head(x)
            if deterministic:
                return torch.argmax(logits, dim=-1)
            dist = Categorical(logits=logits)
            return dist.sample()
        else:
            mu = self.mu(x)
            log_std = torch.clamp(self.log_std, self.cfg.log_std_min, self.cfg.log_std_max)
            std = log_std.exp()
            if deterministic:
                return mu
            dist = Normal(mu, std)
            return dist.sample()