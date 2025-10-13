import math
import os
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------
# Small helpers
# ----------------------

def conv2d_out_size(h: int, w: int, kernel_size: int, stride: int, padding: int = 0) -> Tuple[int, int]:
    """Compute output H, W for a single Conv2d layer without dilation."""
    h_out = math.floor((h + 2 * padding - kernel_size) / stride) + 1
    w_out = math.floor((w + 2 * padding - kernel_size) / stride) + 1
    return h_out, w_out


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


# ----------------------
# Model components (inference-only)
# ----------------------
class Encoder(nn.Module):
    """Image encoder with 4 conv layers mirroring drqv2.py, stride pattern (2,1,1,1)."""

    def __init__(self, in_channels: int, height: int, width: int):
        super().__init__()
        self.h0, self.w0 = height, width

        self.convnet = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(inplace=True),
        )

        # compute spatial size after conv stack (no padding used)
        h1, w1 = conv2d_out_size(self.h0, self.w0, 3, 2)
        h2, w2 = conv2d_out_size(h1, w1, 3, 1)
        h3, w3 = conv2d_out_size(h2, w2, 3, 1)
        h4, w4 = conv2d_out_size(h3, w3, 3, 1)
        self.spatial = (h4, w4)
        self.conv_feat_dim = 32 * h4 * w4

    def forward(self, obs_img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs_img: Float tensor in [0,255] or [0,1], shape (B, C, H, W)
        Returns:
            (B, conv_feat_dim) feature
        """
        x = obs_img
        if x.dtype != torch.float32:
            x = x.float()
        # Normalize to [0,1] if coming in [0,255]
        if x.max() > 1.5:
            x = x / 255.0
        h = self.convnet(x)
        h = h.view(h.shape[0], -1)
        return h


class Actor(nn.Module):
    def __init__(self, repr_dim: int, feature_dim: int, hidden_dim: int, action_dim: int = 3):
        super().__init__()
        # light trunk mirroring drqv2 (Linear->BN->Tanh)
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.Tanh(),
        )
        self.policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        h = self.trunk(features)
        mu = self.policy(h)
        # squash to [-1, 1]
        return torch.tanh(mu)


@dataclass
class AgentConfig:
    frame_stack: int = 3
    height: int = 84
    width: int = 84
    drone_state_dim: int = 7
    feature_dim: int = 128
    hidden_dim: int = 1024
    action_dim: int = 3  # fixed by requirement
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class Agent(nn.Module):
    """Inference-only agent: Encoder + (image features + stacked drone state) -> Actor -> action in [-1,1]^3."""

    def __init__(self, cfg: AgentConfig):
        super().__init__()
        self.cfg = cfg
        self.device_str = cfg.device

        self.encoder = Encoder(cfg.frame_stack, cfg.height, cfg.width)
        repr_dim = self.encoder.conv_feat_dim + cfg.drone_state_dim * cfg.frame_stack
        self.actor = Actor(repr_dim, cfg.feature_dim, cfg.hidden_dim, cfg.action_dim)
        self.to(self.device_str)
        self.eval()  # inference-only

    # Public API
    @torch.inference_mode()
    def act(self, obs_img: torch.Tensor, drone_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs_img: (B, C=frame_stack, H, W) float/uint8
            drone_states: (B, frame_stack, drone_state_dim)
        Returns:
            actions: (B, action_dim) in [-1,1]
        """
        obs_img = obs_img.to(self.device_str)
        drone_states = drone_states.to(self.device_str)

        img_feat = self.encoder(obs_img)
        B = drone_states.shape[0]
        flat_states = drone_states.reshape(B, -1)
        feat = torch.cat([img_feat, flat_states], dim=1)
        actions = self.actor(feat)
        return actions

    def load_weights(self, path: str) -> bool:
        """Load state dict. Accepts either a joint state_dict or a dict with 'encoder'/'actor'."""
        try:
            state = torch.load(path, map_location=self.device_str)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            # Two common layouts: flat keys or nested
            if any(k.startswith("encoder.") for k in state.keys()) or any(k.startswith("actor.") for k in state.keys()):
                self.load_state_dict(state, strict=False)
            elif "encoder" in state and "actor" in state:
                self.encoder.load_state_dict(state["encoder"], strict=False)
                self.actor.load_state_dict(state["actor"], strict=False)
            else:
                # try treating as actor-only (fine for random-encoder case)
                try:
                    self.actor.load_state_dict(state, strict=False)
                except Exception:
                    raise
            return True
        except Exception as e:
            print(f"[agent] Failed to load weights from '{path}': {e}\n[agent] Using randomly initialized weights instead.")
            return False

    # Introspection helpers
    def summary(self) -> str:
        enc_params = count_parameters(self.encoder)
        act_params = count_parameters(self.actor)
        total = enc_params + act_params
        h, w = self.encoder.spatial
        lines = [
            "Agent Summary:",
            f"  Device: {self.device_str}",
            f"  Input image: (C={self.cfg.frame_stack}, H={self.cfg.height}, W={self.cfg.width})",
            f"  Encoder spatial out: (C=32, H={h}, W={w}) -> conv_feat_dim={self.encoder.conv_feat_dim}",
            f"  Drone state: frame_stack*dim = {self.cfg.frame_stack}*{self.cfg.drone_state_dim} = {self.cfg.frame_stack * self.cfg.drone_state_dim}",
            f"  Actor feature_dim: {self.cfg.feature_dim}, hidden_dim: {self.cfg.hidden_dim}",
            f"  Action dim: {self.cfg.action_dim}",
            f"  Parameters: encoder={enc_params:,} | actor={act_params:,} | total={total:,}",
        ]
        return "\n".join(lines)