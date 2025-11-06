"""
Learning Rate Schedulers for RL Training

Supports interval-based learning rate scheduling with two types:
1. StepDecayScheduler: Discrete LR changes at specific steps
2. ExponentialDecayScheduler: Continuous exponential decay within intervals

Each scheduler can have multiple intervals with different parameters.
"""

import torch
from typing import List, Dict, Optional, Union
from omegaconf import DictConfig


class LRScheduler:
    """
    Base class for learning rate schedulers.

    Args:
        optimizer: PyTorch optimizer to schedule
        intervals: List of interval configurations
        name: Name of the optimizer (for logging)
    """

    def __init__(self, optimizer: torch.optim.Optimizer, intervals: List[Dict], name: str = "optimizer"):
        self.optimizer = optimizer
        self.intervals = self._parse_intervals(intervals)
        self.name = name
        self.current_lr = None

    def _parse_intervals(self, intervals: List[Dict]) -> List[Dict]:
        """Parse and validate interval configurations"""
        parsed = []
        for interval in intervals:
            parsed.append({
                'start': interval.get('start', 0),
                'end': interval.get('end', float('inf')),
                **interval
            })

        # Sort by start step
        parsed.sort(key=lambda x: x['start'])

        # Validate intervals don't overlap
        for i in range(len(parsed) - 1):
            if parsed[i]['end'] > parsed[i + 1]['start']:
                raise ValueError(f"Overlapping intervals: {parsed[i]} and {parsed[i + 1]}")

        return parsed

    def step(self, global_step: int) -> float:
        """
        Update learning rate based on current step.

        Args:
            global_step: Current training step

        Returns:
            Current learning rate
        """
        new_lr = self.get_lr(global_step)

        # Only update if LR changed
        if new_lr != self.current_lr:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            self.current_lr = new_lr

        return new_lr

    def get_lr(self, global_step: int) -> float:
        """Calculate learning rate for current step"""
        raise NotImplementedError

    def get_current_interval(self, global_step: int) -> Optional[Dict]:
        """Find the interval configuration for current step"""
        for interval in self.intervals:
            if interval['start'] <= global_step < interval['end']:
                return interval

        # If beyond all intervals, use the last one
        if self.intervals and global_step >= self.intervals[-1]['start']:
            return self.intervals[-1]

        return None


class StepDecayScheduler(LRScheduler):
    """
    Step-based learning rate decay.

    Learning rate changes to a fixed value at specific step boundaries.

    Example config:
        intervals:
          - {start: 0, end: 1000000, lr: 1.0e-4}
          - {start: 1000000, end: 3000000, lr: 5.0e-5}
          - {start: 3000000, end: 6000000, lr: 1.0e-5}
    """

    def get_lr(self, global_step: int) -> float:
        interval = self.get_current_interval(global_step)

        if interval is None:
            # No interval defined, return current LR or default
            if self.current_lr is not None:
                return self.current_lr
            # Get initial LR from optimizer
            return self.optimizer.param_groups[0]['lr']

        return interval['lr']


class ExponentialDecayScheduler(LRScheduler):
    """
    Exponential learning rate decay.

    Learning rate decays exponentially: lr = init_lr * (decay_rate ** num_decays)
    where num_decays = (step - interval_start) // decay_interval

    Example config:
        intervals:
          - {start: 0, end: 1000000, init_lr: 1.0e-4, decay_rate: 0.999, decay_interval: 1000}
          - {start: 1000000, end: 3000000, init_lr: 5.0e-5, decay_rate: 0.99, decay_interval: 5000}
    """

    def _parse_intervals(self, intervals: List[Dict]) -> List[Dict]:
        """Parse and validate exponential decay intervals"""
        parsed = super()._parse_intervals(intervals)

        # Validate required fields
        for interval in parsed:
            if 'init_lr' not in interval:
                raise ValueError(f"ExponentialDecayScheduler requires 'init_lr' in interval: {interval}")
            if 'decay_rate' not in interval:
                raise ValueError(f"ExponentialDecayScheduler requires 'decay_rate' in interval: {interval}")
            if 'decay_interval' not in interval:
                raise ValueError(f"ExponentialDecayScheduler requires 'decay_interval' in interval: {interval}")

        return parsed

    def get_lr(self, global_step: int) -> float:
        interval = self.get_current_interval(global_step)

        if interval is None:
            # No interval defined, return current LR or default
            if self.current_lr is not None:
                return self.current_lr
            return self.optimizer.param_groups[0]['lr']

        # Calculate number of decay steps within this interval
        steps_in_interval = global_step - interval['start']
        num_decays = steps_in_interval // interval['decay_interval']

        # Exponential decay
        lr = interval['init_lr'] * (interval['decay_rate'] ** num_decays)

        return lr


class LinearDecayScheduler(LRScheduler):
    """
    Linear learning rate decay.

    Learning rate interpolates linearly between init_lr and final_lr over the interval:
    lr = init_lr + (final_lr - init_lr) * progress
    where progress = (step - interval_start) / (interval_end - interval_start)

    Example config:
        intervals:
          - {start: 0, end: 2000000, init_lr: 1.0e-4, final_lr: 1.0e-5}
          - {start: 2000000, end: 4000000, init_lr: 1.0e-5, final_lr: 1.0e-6}
    """

    def _parse_intervals(self, intervals: List[Dict]) -> List[Dict]:
        """Parse and validate linear decay intervals"""
        parsed = super()._parse_intervals(intervals)

        # Validate required fields
        for interval in parsed:
            if 'init_lr' not in interval:
                raise ValueError(f"LinearDecayScheduler requires 'init_lr' in interval: {interval}")
            if 'final_lr' not in interval:
                raise ValueError(f"LinearDecayScheduler requires 'final_lr' in interval: {interval}")
            if interval['end'] == float('inf'):
                raise ValueError(f"LinearDecayScheduler requires finite 'end' in interval: {interval}")

        return parsed

    def get_lr(self, global_step: int) -> float:
        interval = self.get_current_interval(global_step)

        if interval is None:
            # No interval defined, return current LR or default
            if self.current_lr is not None:
                return self.current_lr
            return self.optimizer.param_groups[0]['lr']

        # Calculate linear interpolation progress
        interval_length = interval['end'] - interval['start']
        if interval_length == 0:
            return interval['init_lr']

        steps_in_interval = global_step - interval['start']
        progress = min(steps_in_interval / interval_length, 1.0)

        # Linear interpolation
        lr = interval['init_lr'] + (interval['final_lr'] - interval['init_lr']) * progress

        return lr


class ConstantLRScheduler(LRScheduler):
    """
    Constant learning rate (no scheduling).

    This is used for backward compatibility when no scheduling is configured.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, lr: float, name: str = "optimizer"):
        self.optimizer = optimizer
        self.lr = lr
        self.name = name
        self.current_lr = lr
        self.intervals = []

    def get_lr(self, global_step: int) -> float:
        return self.lr


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Optional[Union[DictConfig, Dict]],
    default_lr: float,
    name: str = "optimizer"
) -> LRScheduler:
    """
    Factory function to create appropriate LR scheduler from config.

    Args:
        optimizer: PyTorch optimizer
        config: Learning rate schedule configuration (can be None for constant LR)
        default_lr: Default learning rate if no scheduling
        name: Name of the optimizer (for logging)

    Returns:
        LRScheduler instance

    Example config:
        {
            'type': 'step_decay',
            'intervals': [
                {'start': 0, 'end': 1000000, 'lr': 1.0e-4},
                {'start': 1000000, 'end': 3000000, 'lr': 5.0e-5}
            ]
        }
    """

    # Backward compatibility: if no config, use constant LR
    if config is None or not config:
        return ConstantLRScheduler(optimizer, default_lr, name)

    # Convert DictConfig to dict if needed
    if isinstance(config, DictConfig):
        config = dict(config)

    # Get scheduler type
    scheduler_type = config.get('type', 'constant')
    intervals = config.get('intervals', [])

    if scheduler_type == 'step_decay':
        return StepDecayScheduler(optimizer, intervals, name)
    elif scheduler_type == 'exponential_decay':
        return ExponentialDecayScheduler(optimizer, intervals, name)
    elif scheduler_type == 'linear_decay':
        return LinearDecayScheduler(optimizer, intervals, name)
    elif scheduler_type == 'constant':
        return ConstantLRScheduler(optimizer, default_lr, name)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def create_multi_optimizer_schedulers(
    optimizers: Dict[str, torch.optim.Optimizer],
    lr_schedule_config: Optional[Union[DictConfig, Dict]],
    default_lr: float
) -> Dict[str, LRScheduler]:
    """
    Create schedulers for multiple optimizers (encoder, actor, critic).

    Args:
        optimizers: Dictionary of optimizers {name: optimizer}
        lr_schedule_config: LR schedule configuration with per-optimizer configs
        default_lr: Default learning rate

    Returns:
        Dictionary of schedulers {name: scheduler}

    Example config:
        {
            'encoder': {
                'type': 'step_decay',
                'intervals': [{'start': 0, 'end': 1000000, 'lr': 1.0e-4}]
            },
            'actor': {
                'type': 'exponential_decay',
                'intervals': [{'start': 0, 'end': 1000000, 'init_lr': 1.0e-4, 'decay_rate': 0.999, 'decay_interval': 1000}]
            },
            'critic': {
                'type': 'step_decay',
                'intervals': [{'start': 0, 'end': 1000000, 'lr': 1.0e-4}]
            }
        }
    """

    schedulers = {}

    for name, optimizer in optimizers.items():
        # Get config for this specific optimizer
        if lr_schedule_config and name in lr_schedule_config:
            opt_config = lr_schedule_config[name]
        else:
            # No config for this optimizer, use constant LR
            opt_config = None

        schedulers[name] = create_lr_scheduler(optimizer, opt_config, default_lr, name)

    return schedulers
