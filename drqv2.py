import torch
import torch.nn as nn
import torch.nn.functional as F
import utils


class RandomBrigthnessAug(nn.Module):
    def __init__(self, range, device):
        super().__init__()
        self.range = range
        self.device = device

    def forward(self, x):
        # x: [B, C, H, W], range [0,255] assumed
        n, c, h, w = x.size()
        factors = torch.empty((n, c, 1, 1), device=self.device).uniform_(1 - self.range, 1 + self.range)
        x = factors * x
        x = torch.clamp(x, 0, 255)
        return x


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift                                                 # [B, H, W, 2]
        return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False) # [B, C, H, W]


class AuxiliaryGimbalHead(nn.Module):
    """
    Auxiliary task head for predicting oracle gimbal angles.
    Takes encoder representation and predicts [pitch, yaw] in normalized space [-1, 1].

    Args:
        repr_dim: Dimension of encoder representation
        hidden_dim: Hidden dimension of the MLP (default: 128)
    """
    def __init__(self, repr_dim, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(repr_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)  # Output: [pitch, yaw]
        )
        self.apply(utils.weight_init)

    def forward(self, h):
        """
        Args:
            h: [B, repr_dim] encoder representation
        Returns:
            gimbal_pred: [B, 2] predicted normalized gimbal angles [pitch, yaw]
        """
        return self.mlp(h)


class EncoderBaseline(nn.Module):
    """
    기존 CNN + 상태 concat 방식.
    obs: [B, C(=num_stacks), H, W], 값 범위 [0,255]
    drone_states: [B, num_stacks*11] 또는 [B, num_stacks, 11]
    repr: [B, 32*35*35 + 11*num_stacks]
    """
    def __init__(self, obs_shape, num_stacks=3, drone_state_dim=11):
        super().__init__()
        assert len(obs_shape) == 3
        C, H, W = obs_shape
        self.num_stacks = num_stacks
        self.drone_state_dim = drone_state_dim

        self.repr_dim = 32 * 35 * 35 + (drone_state_dim * num_stacks)

        self.convnet = nn.Sequential(
            nn.Conv2d(C, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU()
        )
        self.apply(utils.weight_init)
        self.eval()

    def _reshape_states(self, drone_states):
        # input: [B, num_stacks*11] or [B, num_stacks, 11]
        if drone_states.dim() == 3 and drone_states.size(1) == self.num_stacks and drone_states.size(2) == self.drone_state_dim:
            return drone_states.view(drone_states.size(0), -1)  # [B, num_stacks*11]
        elif drone_states.dim() == 2 and drone_states.size(1) == self.num_stacks * self.drone_state_dim:
            return drone_states                                  # [B, num_stacks*11]
        else:
            raise ValueError(f"drone_states must be [B,{self.num_stacks},11] or [B,{self.num_stacks*11}]")

    def forward(self, obs, drone_states):
        """
        obs: [B, C, H, W], 0..255
        drone_states: [B, num_stacks*11] or [B, num_stacks, 11]
        return: h [B, repr_dim]
        """
        obs = obs / 255.0
        h = self.convnet(obs)                              # [B, 32, 35, 35]
        h = h.view(h.shape[0], -1)                         # [B, 32*35*35]
        drone_states = self._reshape_states(drone_states)  # [B, num_stacks*11]
        h = torch.cat((h, drone_states), dim=1)            # [B, repr_dim]
        return h



class Actor(nn.Module):
    """Original joint action actor"""
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.Tanh()
        )
        self.policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_shape[0])
        )
        self.apply(utils.weight_init)
        self.eval()

    def forward(self, obs, std):
        """
        obs: [B, repr_dim]
        std: scalar or tensor
        Returns: dist (TruncatedNormal)
        """
        h = self.trunk(obs)                # [B, feature_dim]
        mu = self.policy(h)                # [B, action_dim]
        mu = torch.tanh(mu)                # [-1,1]
        std = torch.ones_like(mu) * std
        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    """Original joint critic"""
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.Tanh()
        )
        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        """
        obs: [B, repr_dim]
        action: [B, action_dim]
        return: Q1,Q2 each [B,1]
        """
        h = self.trunk(obs)                    # [B, feature_dim]
        h_action = torch.cat([h, action], dim=-1)  # [B, feature_dim+action_dim]
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)
        return q1, q2


class DrQV2Agent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb,
                 frame_stack=3,
                 state_dim_per_frame: int = 11,
                 lr_schedule: dict = None,
                 auxiliary_task: dict = None,
                 # Legacy kwargs accepted for checkpoint compatibility
                 enc: str = 'org', enc_cfg: dict = None,
                 actor_type: str = 'original', actor_cfg: dict = None,
                 critic_type: str = 'original'):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.frame_stack = frame_stack
        self.state_dim_per_frame = state_dim_per_frame

        # Encoder
        self.encoder = EncoderBaseline(obs_shape, num_stacks=frame_stack,
                                       drone_state_dim=state_dim_per_frame).to(device)

        # Actor
        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim, hidden_dim).to(device)

        # Critic
        self.critic = Critic(self.encoder.repr_dim, action_shape, feature_dim, hidden_dim).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, action_shape, feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Auxiliary task setup (gimbal angle prediction)
        aux_cfg = auxiliary_task or {}
        self.aux_enabled = aux_cfg.get('enable', False)
        self.aux_weight = aux_cfg.get('weight', 0.1)
        self.aux_hidden_dim = aux_cfg.get('hidden_dim', 128)

        if self.aux_enabled:
            self.auxiliary_head = AuxiliaryGimbalHead(
                self.encoder.repr_dim,
                hidden_dim=self.aux_hidden_dim
            ).to(device)
        else:
            self.auxiliary_head = None

        # Optimizers
        encoder_params = list(self.encoder.parameters())
        if self.aux_enabled:
            encoder_params += list(self.auxiliary_head.parameters())
        self.encoder_opt = torch.optim.Adam(encoder_params, lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Learning rate schedulers
        from lr_scheduler import create_multi_optimizer_schedulers
        optimizers = {
            'encoder': self.encoder_opt,
            'actor': self.actor_opt,
            'critic': self.critic_opt
        }
        self.lr_schedulers = create_multi_optimizer_schedulers(optimizers, lr_schedule, lr)

        # Data Augmentations
        self.aug = RandomShiftsAug(pad=4)
        self.brightaug = RandomBrigthnessAug(range=0.5, device=device)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)
        if self.aux_enabled:
            self.auxiliary_head.train(training)

    def eval(self, training=False):
        self.training = training
        self.encoder.eval()
        self.actor.eval()
        self.critic.eval()
        if self.aux_enabled:
            self.auxiliary_head.eval()

    def act(self, obs, step, drone_states, eval_mode):
        """
        obs: numpy or tensor [C(=frame_stack), H, W]
        drone_states: numpy or tensor [frame_stack, 11]
        return: action np.ndarray [action_dim], [-1,1]
        """
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)                    # [1,C,H,W]
        drone_states = torch.as_tensor(drone_states, device=self.device).unsqueeze(0)  # [1,T,11]

        # 인코더 전처리/전달
        with torch.no_grad():  # in the train script, act() is already in no_grad mode; this is redundant but harmless!
            h = self.encoder(obs, drone_states)  # [1, repr_dim]
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(h, stddev)
            if eval_mode:
                action = dist.mean
            else:
                action = dist.sample(clip=None)
                if step < self.num_expl_steps:
                    action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step, oracle_gimbal=None):
        metrics = dict()
        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        # Compute auxiliary loss if enabled and oracle_gimbal is available
        total_loss = critic_loss
        if self.aux_enabled and oracle_gimbal is not None:
            # Predict gimbal angles from encoder representation
            gimbal_pred = self.auxiliary_head(obs)  # [B, 2]

            # Compute MSE loss between predicted and oracle gimbal angles
            aux_loss = F.mse_loss(gimbal_pred, oracle_gimbal)

            # Add weighted auxiliary loss to total loss
            total_loss = critic_loss + self.aux_weight * aux_loss

            if self.use_tb:
                metrics['aux_loss'] = aux_loss.item()
                metrics['aux_gimbal_error'] = (gimbal_pred - oracle_gimbal).abs().mean().item()

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()
            if self.aux_enabled and oracle_gimbal is not None:
                metrics['total_loss'] = total_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        total_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()
        return metrics

    def update_actor(self, obs, step):
        metrics = dict()
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)
        actor_loss = -Q.mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
        return metrics

    def update(self, replay_iter, step, difficulty=None):
        metrics = dict()
        if step % self.update_every_steps != 0:
            return metrics

        # Update learning rates
        for name, scheduler in self.lr_schedulers.items():
            current_lr = scheduler.step(step)
            if self.use_tb:
                metrics[f'lr_{name}'] = current_lr

        batch = next(replay_iter)
        # batch unpack
        # obs: [B,C,H,W], action: [B,A], reward:[B,1 or rewards-dim], discount:[B,1]
        # next_obs:[B,C,H,W], drone_state: [B,T,11], next_drone_state: [B,T,11], oracle_gimbal: [B,2]
        obs, action, reward, discount, next_obs, drone_state, next_drone_state, oracle_gimbal = utils.to_torch(batch, self.device)

        # Handles curriculum conditional reward
        if difficulty is not None:
            assert reward.shape[1] == 3.                 # (B, rewards-dim)
            reward = reward[:, difficulty-1:difficulty]  # (B, 1)

        # augment
        obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float())
        obs = self.brightaug(obs.float())
        next_obs = self.brightaug(next_obs.float())

        # encode
        h_obs = self.encoder(obs, drone_state)                    # [B, repr_dim]
        with torch.no_grad():
            h_next = self.encoder(next_obs, next_drone_state)     # [B, repr_dim]

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(self.update_critic(h_obs, action, reward, discount, h_next, step, oracle_gimbal))

        # update actor (stop gradient through critic)
        metrics.update(self.update_actor(h_obs.detach(), step))

        # update target
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)
        return metrics
