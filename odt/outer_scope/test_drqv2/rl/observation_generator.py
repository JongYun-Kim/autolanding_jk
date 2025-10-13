from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class ObservationGenerator:
    frame_stack: int
    height: int
    width: int
    drone_state_dim: int
    batch_size: int = 1
    device: str = "cpu"
    seed: int = 42

    def __post_init__(self):
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed)
        # Keep a CPU generator for reproducibility; tensors can be moved to target device each step
        self._cpu_gen = g

    def next(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            obs_img: (B, C=frame_stack, H, W) uint8 in [0,255]
            drone_states: (B, frame_stack, drone_state_dim) float32 ~ N(0,1)
        """
        B = self.batch_size
        img = torch.randint(
            low=0,
            high=256,
            size=(B, self.frame_stack, self.height, self.width),
            dtype=torch.uint8,
            generator=self._cpu_gen,
        )
        # Use a mild range for states to simulate sensible inputs
        states = torch.randn(
            (B, self.frame_stack, self.drone_state_dim),
            dtype=torch.float32,
            generator=self._cpu_gen,
        )
        # Move to requested device lazily per step
        return img.to(self.device), states.to(self.device)
