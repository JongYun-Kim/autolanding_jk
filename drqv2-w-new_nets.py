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


# Time-aware Transformer Utilities
class LNFFN(nn.Module):
    """Pre-LN FFN block"""
    def __init__(self, d_model: int, mlp_ratio: float = 4.0, p_drop: float = 0.05):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        self.drop = nn.Dropout(p_drop) if p_drop != 0.0 else nn.Identity()

    def forward(self, x):
        # x: [B, N, d]
        h = self.ln(x)
        h = self.ff(h)
        return x + self.drop(h)


class LNMultiheadAttention(nn.Module):
    """Pre-LN + residual wrapper for nn.MultiheadAttention"""
    def __init__(self, d_model: int, nhead: int = 4, p_drop: float = 0.05, cross: bool = False):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model) if cross else None
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=p_drop, batch_first=True)
        self.drop = nn.Dropout(p_drop) if p_drop != 0.0 else nn.Identity()
        self.cross = cross

    def forward(self, q, k=None, v=None, attn_mask=None, key_padding_mask=None):
        # q: [B, Nq, d], k,v: [B, Nk, d]
        qn = self.ln_q(q)
        if self.cross:
            kn = self.ln_kv(k)
            vn = self.ln_kv(v)
            out, _ = self.attn(qn, kn, vn, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        else:
            out, _ = self.attn(qn, qn, qn, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        return q + self.drop(out)


class TimeEmbedding(nn.Module):
    """Learnable time embedding for t=0..T-1"""
    def __init__(self, T: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(T, d_model)

    def forward(self, t_idx: torch.Tensor):
        # t_idx: [B, T]
        return self.emb(t_idx)               # [B, T, d]


class StateTokenizerPerFrame(nn.Module):
    """t별로 [VEL_t, DRONEQ_t, GIMBALQ_t] → 3개 상태 토큰"""
    def __init__(self, d_model: int):
        super().__init__()
        self.proj_vel = nn.Linear(3, d_model)
        self.proj_dq  = nn.Linear(4, d_model)
        self.proj_gq  = nn.Linear(4, d_model)

    def forward(self, vel, qd, qg):
        """
        vel: [B, T, 3], qd: [B, T, 4], qg: [B, T, 4]
        return: [B, T, 3, d]
        """
        vel_tok = self.proj_vel(vel)   # [B,T,d]
        dq_tok  = self.proj_dq(qd)     # [B,T,d]
        gq_tok  = self.proj_gq(qg)     # [B,T,d]
        tok = torch.stack([vel_tok, dq_tok, gq_tok], dim=2)  # [B,T,3,d]
        return tok


# util for A with time
class ImageGlobalTokenPerFrame(nn.Module):
    """각 프레임을 별도 CNN 처리 → GAP → per-frame global image token"""
    def __init__(self, d_model: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 8, stride=4, padding=2), nn.ReLU(inplace=True),  # [B,32,21,21]
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True), # [B,64,11,11]
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True) # [B,128,6,6]
        )
        self.proj = nn.Linear(128, d_model)

    def forward(self, x_img_stacked: torch.Tensor):
        """
        x_img_stacked: [B, T, H, W] 또는 [B, C=T, H, W]
        return: img_tok_t [B, T, d]
        """
        if x_img_stacked.dim() == 4:
            # [B, C=T, H, W] 가정
            B, T, H, W = x_img_stacked.shape
            frames = x_img_stacked.unbind(dim=1)   # T * [B, H, W]
            frames = [f.unsqueeze(1) for f in frames]  # T * [B,1,H,W]
            x_bt1hw = torch.cat(frames, dim=0)     # [B*T,1,H,W]
        elif x_img_stacked.dim() == 5:
            # [B, T, H, W] (존재한다면)
            B, T, H, W = x_img_stacked.shape
            frames = x_img_stacked.unbind(dim=1)
            frames = [f.unsqueeze(1) for f in frames]  # [B,1,H,W]
            x_bt1hw = torch.cat(frames, dim=0)         # [B*T,1,H,W]
        else:
            raise ValueError("x_img_stacked must be [B,T,H,W] or [B,T=channels,H,W]")

        h = self.stem(x_bt1hw)             # [B*T,128,6,6]
        h = h.mean(dim=[2, 3])             # [B*T,128]
        tok = self.proj(h)                 # [B*T,d]
        tok = tok.view(B, T, -1)           # [B,T,d]
        return tok


# lightweight encoder (time-aware; with global image token)
class EncoderA_TimeAware(nn.Module):
    """
    (A_time) 경량 시간 인지 인코더:
    - 각 t: 이미지 글로벌 토큰 1개 + 상태 3토큰
    - time embedding 추가
    - 프레임별 cross-attn(Q=state_t, K/V=image_t) × cross_depth
    - (옵션) temporal self-attn × self_depth
    출력 z: [B, d_model], repr_dim=d_model
    """
    def __init__(self, T, d_model=256, nhead=4, p_drop=0.05, cross_depth=1, self_depth=1):
        super().__init__()
        self.T = T
        self.repr_dim = d_model

        self.img_tok   = ImageGlobalTokenPerFrame(d_model)
        self.state_tok = StateTokenizerPerFrame(d_model)
        self.time_emb  = TimeEmbedding(T, d_model)

        self.cross_layers = nn.ModuleList([LNMultiheadAttention(d_model, nhead, p_drop, cross=True) for _ in range(cross_depth)])
        self.cross_ffn    = nn.ModuleList([LNFFN(d_model, p_drop=p_drop) for _ in range(cross_depth)])

        self.temporal_layers = nn.ModuleList([LNMultiheadAttention(d_model, nhead, p_drop, cross=False) for _ in range(self_depth)])
        self.temporal_ffn    = nn.ModuleList([LNFFN(d_model, p_drop=p_drop) for _ in range(self_depth)])

        self.drop = nn.Dropout(p_drop) if p_drop != 0.0 else nn.Identity()
        self.apply(utils.weight_init)
        self.eval()

    @staticmethod
    def _split_state(drone_states, T):
        """
        drone_states: [B, T*11] 또는 [B, T, 11]
        return: vel:[B,T,3], qd:[B,T,4], qg:[B,T,4]
        """
        if drone_states.dim() == 2:
            B = drone_states.size(0)
            assert drone_states.size(1) % 11 == 0
            T_in = drone_states.size(1) // 11
            assert T_in == T, f"T mismatch: got {T_in}, expected {T}"
            ds = drone_states.view(B, T, 11)
        elif drone_states.dim() == 3:
            B, T_in, D = drone_states.shape
            assert D == 11 and T_in == T, f"Expected [B,{T},11], got {drone_states.shape}"
            ds = drone_states
        else:
            raise ValueError("drone_states must be [B,T*11] or [B,T,11]")

        vel = ds[..., 0:3]     # [B,T,3]
        qd  = ds[..., 3:7]     # [B,T,4]
        qg  = ds[..., 7:11]    # [B,T,4]
        return vel, qd, qg

    def forward(self, obs, drone_states):
        """
        obs: [B, C=T, H, W], 값 범위 [0,255]
        drone_states: [B, T*11] or [B, T, 11]
        return: z [B, d_model]
        """
        B, C, H, W = obs.shape
        assert C == self.T, f"obs channels must equal T ({self.T}), got C={C}"
        obs = obs / 255.0

        # per-frame tokens
        img_tok_t = self.img_tok(obs)               # [B,T,d]
        vel, qd, qg = self._split_state(drone_states, self.T)
        state_tok_t3 = self.state_tok(vel, qd, qg)  # [B,T,3,d]

        # time emb
        t_idx = torch.arange(self.T, device=obs.device).unsqueeze(0).expand(B, self.T)  # [B,T]
        tau = self.time_emb(t_idx)                            # [B,T,d]
        img_tok_t    = img_tok_t + tau                        # [B,T,d]
        state_tok_t3 = state_tok_t3 + tau.unsqueeze(2)        # [B,T,3,d]

        # frame-wise cross-attn
        fused_state_t3 = []
        for t in range(self.T):
            img_t = img_tok_t[:, t:t+1, :]                   # [B,1,d]
            st_t  = state_tok_t3[:, t, :, :]                 # [B,3,d]
            for attn, ffn in zip(self.cross_layers, self.cross_ffn):
                st_t = attn(st_t, k=img_t, v=img_t)          # [B,3,d]
                st_t = ffn(st_t)                             # [B,3,d]
            fused_state_t3.append(st_t)
        fused = torch.stack(fused_state_t3, dim=1)           # [B,T,3,d]

        # temporal self-attn (옵션)
        tokens = fused.view(B, self.T * 3, -1)               # [B, T*3, d]
        for attn, ffn in zip(self.temporal_layers, self.temporal_ffn):
            tokens = attn(tokens)                            # [B, T*3, d]
            tokens = ffn(tokens)                             # [B, T*3, d]

        z = tokens.mean(dim=1)                               # [B, d]
        return self.drop(z)


# util for B: patch and positional encoding
class PatchEmbed(nn.Module):
    """Conv(kernel=stride=patch)로 패치 임베딩"""
    def __init__(self, img_size=84, patch=12, in_ch=1, d_model=256):
        super().__init__()
        assert img_size % patch == 0, "img_size must be divisible by patch"
        self.grid = img_size // patch
        self.proj = nn.Conv2d(in_ch, d_model, kernel_size=patch, stride=patch)

    def forward(self, x):
        # x: [B,1,H,W]
        h = self.proj(x)                   # [B,d,G,G]
        h = h.flatten(2).transpose(1, 2)   # [B, Np=G*G, d]
        return h


class PositionalEncoding2D(nn.Module):
    def __init__(self, n_tokens: int, d_model: int):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        return x + self.pos                # [B, Np, d]


# encoder B (time-aware; patch-based)
class EncoderB_TimeAware(nn.Module):
    """
    (B_time) 패치 기반 시간 인지 인코더 (가성비: frame-local → temporal)
    1) 각 프레임 t: patch_t(Np)에 2D pos + time emb, state_t(3)에 time emb
    2) 프레임-로컬 인코더 depth_local층: [state_t(3), patch_t(Np)]
    3) 프레임 요약: patch 평균 → img_sum_t(1)
    4) Temporal 인코더 depth_temp층: concat([state_t(3), img_sum_t(1)] across t)
    5) 평균 풀링 → z
    repr_dim = d_model
    """
    def __init__(self, T, d_model=256, nhead=4, p_drop=0.05, img_size=84, patch=12, depth_local=2, depth_temp=1):
        super().__init__()
        self.T = T
        self.repr_dim = d_model

        self.patch = PatchEmbed(img_size=img_size, patch=patch, in_ch=1, d_model=d_model)
        self.Np = (img_size // patch) ** 2
        self.pos = PositionalEncoding2D(self.Np, d_model)

        self.state_tok = StateTokenizerPerFrame(d_model)
        self.time_emb  = TimeEmbedding(T, d_model)

        self.local_attn = nn.ModuleList([LNMultiheadAttention(d_model, nhead, p_drop, cross=False) for _ in range(depth_local)])
        self.local_ffn  = nn.ModuleList([LNFFN(d_model, p_drop=p_drop) for _ in range(depth_local)])

        self.temp_attn  = nn.ModuleList([LNMultiheadAttention(d_model, nhead, p_drop, cross=False) for _ in range(depth_temp)])
        self.temp_ffn   = nn.ModuleList([LNFFN(d_model, p_drop=p_drop) for _ in range(depth_temp)])

        self.drop = nn.Dropout(p_drop) if p_drop != 0.0 else nn.Identity()
        self.apply(utils.weight_init)
        self.eval()

    @staticmethod
    def _split_state(drone_states, T):
        if drone_states.dim() == 2:
            B = drone_states.size(0)
            assert drone_states.size(1) % 11 == 0
            T_in = drone_states.size(1) // 11
            assert T_in == T, f"T mismatch: got {T_in}, expected {T}"
            ds = drone_states.view(B, T, 11)
        elif drone_states.dim() == 3:
            B, T_in, D = drone_states.shape
            assert D == 11 and T_in == T, f"Expected [B,{T},11], got {drone_states.shape}"
            ds = drone_states
        else:
            raise ValueError("drone_states must be [B,T*11] or [B,T,11]")
        vel = ds[..., 0:3]  # [B,T,3]
        qd  = ds[..., 3:7]  # [B,T,4]
        qg  = ds[..., 7:11] # [B,T,4]
        return vel, qd, qg

    def forward(self, obs, drone_states):
        """
        obs: [B, C=T, H, W], 값 범위 [0,255]
        drone_states: [B, T*11] or [B, T, 11]
        return: z [B, d_model]
        """
        B, C, H, W = obs.shape
        assert C == self.T, f"obs channels must equal T ({self.T}), got C={C}"
        obs = obs / 255.0

        # (B*T,1,H,W)로 펼쳐 프레임별 패치 임베딩
        frames = obs.unbind(dim=1)                  # T * [B,H,W]
        frames = [f.unsqueeze(1) for f in frames]   # T * [B,1,H,W]
        x_bt1hw = torch.cat(frames, dim=0)          # [B*T,1,H,W]

        patch_tok = self.patch(x_bt1hw)             # [B*T, Np, d]
        patch_tok = self.pos(patch_tok)             # [B*T, Np, d]
        patch_tok = patch_tok.view(B, self.T, self.Np, -1)  # [B,T,Np,d]

        vel, qd, qg = self._split_state(drone_states, self.T)
        state_tok = self.state_tok(vel, qd, qg)     # [B,T,3,d]

        # time emb
        t_idx = torch.arange(self.T, device=obs.device).unsqueeze(0).expand(B, self.T)  # [B,T]
        tau = self.time_emb(t_idx)                   # [B,T,d]
        patch_tok = patch_tok + tau.unsqueeze(2)     # [B,T,Np,d]
        state_tok = state_tok + tau.unsqueeze(2)     # [B,T,3,d]

        # frame-local encoder
        tokens_local = torch.cat([state_tok, patch_tok], dim=2)   # [B,T, 3+Np, d]
        tokens_local = tokens_local.view(B * self.T, 3 + self.Np, -1)  # [B*T, Nf, d]
        for attn, ffn in zip(self.local_attn, self.local_ffn):
            tokens_local = attn(tokens_local)                     # [B*T, Nf, d]
            tokens_local = ffn(tokens_local)                      # [B*T, Nf, d]
        tokens_local = tokens_local.view(B, self.T, 3 + self.Np, -1)    # [B,T,Nf,d]

        # frame summary
        state_upd = tokens_local[:, :, :3, :]                     # [B,T,3,d]
        patch_upd = tokens_local[:, :, 3:, :]                     # [B,T,Np,d]
        img_sum   = patch_upd.mean(dim=2, keepdim=True)           # [B,T,1,d]

        # temporal encoder over [state_t(3), img_sum_t(1)]
        temp_seq = torch.cat([state_upd, img_sum], dim=2)         # [B,T,4,d]
        temp_seq = temp_seq.view(B, self.T * 4, -1)               # [B, 4T, d]
        for attn, ffn in zip(self.temp_attn, self.temp_ffn):
            temp_seq = attn(temp_seq)                             # [B, 4T, d]
            temp_seq = ffn(temp_seq)                              # [B, 4T, d]

        z = temp_seq.mean(dim=1)                                  # [B,d]
        return self.drop(z)


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


class AutoregressiveActorGimbalFirst(nn.Module):
    """
    Autoregressive actor: Gimbal → Drone
    Motivation: "Point-then-move" strategy
    - First decide where to point the camera
    - Then decide drone movement given camera orientation
    """
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()
        assert action_shape[0] == 5, f"Expected action_shape[0]=5, got {action_shape[0]}"

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.Tanh()
        )

        # Gimbal policy (unconditional)
        self.gimbal_policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2)  # [pitch, yaw]
        )

        # Drone policy (conditioned on gimbal)
        self.drone_policy = nn.Sequential(
            nn.Linear(feature_dim + 2, hidden_dim),  # +2 for gimbal actions
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3)  # [vx, vy, vz]
        )

        self.apply(utils.weight_init)
        self.eval()

    def forward(self, obs, std):
        """
        obs: [B, repr_dim]
        std: scalar or tensor
        Returns: dist (TruncatedNormal over joint action)
        """
        h = self.trunk(obs)  # [B, feature_dim]

        # Step 1: Gimbal action mean
        gimbal_mu = self.gimbal_policy(h)  # [B, 2]
        gimbal_mu = torch.tanh(gimbal_mu)

        # Step 2: Drone action mean (conditioned on gimbal)
        h_drone = torch.cat([h, gimbal_mu], dim=-1)  # [B, feature_dim + 2]
        drone_mu = self.drone_policy(h_drone)  # [B, 3]
        drone_mu = torch.tanh(drone_mu)

        # Joint action mean
        mu = torch.cat([drone_mu, gimbal_mu], dim=-1)  # [B, 5]
        std = torch.ones_like(mu) * std
        dist = utils.TruncatedNormal(mu, std)
        return dist


class AutoregressiveActorDroneFirst(nn.Module):
    """
    Autoregressive actor: Drone → Gimbal
    Motivation: "Move-then-point" strategy
    - First decide where the drone should go
    - Then adjust gimbal to keep target in view
    """
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()
        assert action_shape[0] == 5, f"Expected action_shape[0]=5, got {action_shape[0]}"

        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.Tanh()
        )

        # Drone policy (unconditional)
        self.drone_policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3)  # [vx, vy, vz]
        )

        # Gimbal policy (conditioned on drone)
        self.gimbal_policy = nn.Sequential(
            nn.Linear(feature_dim + 3, hidden_dim),  # +3 for drone actions
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2)  # [pitch, yaw]
        )

        self.apply(utils.weight_init)
        self.eval()

    def forward(self, obs, std):
        """
        obs: [B, repr_dim]
        std: scalar or tensor
        Returns: dist (TruncatedNormal over joint action)
        """
        h = self.trunk(obs)

        # Step 1: Drone action mean
        drone_mu = self.drone_policy(h)  # [B, 3]
        drone_mu = torch.tanh(drone_mu)

        # Step 2: Gimbal action mean (conditioned on drone)
        h_gimbal = torch.cat([h, drone_mu], dim=-1)  # [B, feature_dim + 3]
        gimbal_mu = self.gimbal_policy(h_gimbal)  # [B, 2]
        gimbal_mu = torch.tanh(gimbal_mu)

        # Joint action mean
        mu = torch.cat([drone_mu, gimbal_mu], dim=-1)  # [B, 5]
        std = torch.ones_like(mu) * std
        dist = utils.TruncatedNormal(mu, std)
        return dist


class MultiHeadAttentionActor(nn.Module):
    """
    Multi-head attention actor: Parallel but contextually-aware actions
    Motivation: Both drone and gimbal policies see each other's intent
    - Cross-attention allows coordination without sequential dependency
    - Most flexible for learning coordination strategy
    """
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, nhead=4):
        super().__init__()
        assert action_shape[0] == 5, f"Expected action_shape[0]=5, got {action_shape[0]}"

        self.feature_dim = feature_dim
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.Tanh()
        )

        # Learnable query embeddings for drone and gimbal
        self.drone_query = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.gimbal_query = nn.Parameter(torch.randn(1, 1, feature_dim))

        # Cross-attention between observation and queries
        self.cross_attn = nn.MultiheadAttention(
            feature_dim, nhead, batch_first=True
        )

        # Self-attention for coordination between drone and gimbal
        self.self_attn = nn.MultiheadAttention(
            feature_dim, nhead, batch_first=True
        )

        # Layer norms
        self.ln1 = nn.LayerNorm(feature_dim)
        self.ln2 = nn.LayerNorm(feature_dim)

        # Separate policy heads
        self.drone_policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3)  # [vx, vy, vz]
        )

        self.gimbal_policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2)  # [pitch, yaw]
        )

        self.apply(utils.weight_init)
        self.eval()

    def forward(self, obs, std):
        """
        obs: [B, repr_dim]
        std: scalar or tensor
        Returns: dist (TruncatedNormal over joint action)
        """
        B = obs.shape[0]
        h = self.trunk(obs)  # [B, feature_dim]

        # Expand queries for batch
        drone_q = self.drone_query.expand(B, -1, -1)   # [B, 1, feature_dim]
        gimbal_q = self.gimbal_query.expand(B, -1, -1) # [B, 1, feature_dim]

        # Stack queries
        queries = torch.cat([drone_q, gimbal_q], dim=1)  # [B, 2, feature_dim]

        # Use observation as key/value
        h_kv = h.unsqueeze(1)  # [B, 1, feature_dim]

        # Cross-attention: queries attend to observation
        attn_out, _ = self.cross_attn(queries, h_kv, h_kv)  # [B, 2, feature_dim]
        queries = self.ln1(queries + attn_out)  # Residual connection

        # Self-attention: drone and gimbal queries attend to each other
        attn_out2, _ = self.self_attn(queries, queries, queries)  # [B, 2, feature_dim]
        h_coordinated = self.ln2(queries + attn_out2)  # [B, 2, feature_dim]

        # Extract drone and gimbal features
        h_drone = h_coordinated[:, 0, :]   # [B, feature_dim]
        h_gimbal = h_coordinated[:, 1, :]  # [B, feature_dim]

        # Generate actions
        drone_mu = torch.tanh(self.drone_policy(h_drone))    # [B, 3]
        gimbal_mu = torch.tanh(self.gimbal_policy(h_gimbal)) # [B, 2]

        # Joint action mean
        mu = torch.cat([drone_mu, gimbal_mu], dim=-1)  # [B, 5]
        std = torch.ones_like(mu) * std
        dist = utils.TruncatedNormal(mu, std)
        return dist


class FactoredCritic(nn.Module):
    """
    Factored critic for autoregressive actors
    Q(s, a_drone, a_gimbal) = Q_drone(s, a_drone) + Q_gimbal(s, a_drone, a_gimbal)
    Allows separate evaluation of:
    - Base value of drone actions
    - Additional value from gimbal coordination
    """
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()
        assert action_shape[0] == 5, f"Expected action_shape[0]=5, got {action_shape[0]}"

        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.Tanh()
        )

        # Q1: drone component + joint component
        self.Q1_drone = nn.Sequential(
            nn.Linear(feature_dim + 3, hidden_dim),  # repr + drone_action
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        self.Q1_joint = nn.Sequential(
            nn.Linear(feature_dim + 5, hidden_dim),  # repr + full_action
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

        # Q2: drone component + joint component
        self.Q2_drone = nn.Sequential(
            nn.Linear(feature_dim + 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        self.Q2_joint = nn.Sequential(
            nn.Linear(feature_dim + 5, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        """
        obs: [B, repr_dim]
        action: [B, 5] = [drone(3), gimbal(2)]
        return: Q1, Q2 each [B, 1]
        """
        h = self.trunk(obs)  # [B, feature_dim]
        drone_action = action[:, :3]  # [B, 3]

        # Q1 = Q1_drone(s, a_drone) + Q1_joint(s, a_full)
        q1_d = self.Q1_drone(torch.cat([h, drone_action], dim=-1))
        q1_j = self.Q1_joint(torch.cat([h, action], dim=-1))
        q1 = q1_d + q1_j

        # Q2 = Q2_drone(s, a_drone) + Q2_joint(s, a_full)
        q2_d = self.Q2_drone(torch.cat([h, drone_action], dim=-1))
        q2_j = self.Q2_joint(torch.cat([h, action], dim=-1))
        q2 = q2_d + q2_j

        return q1, q2


class DrQV2Agent:
    """
    Public API with extended selection options:
      - enc (encoder_type): 'org' | 'A' | 'B'
      - enc_cfg: dict (encoder hyperparameters)
      - actor_type: 'original' | 'autoregressive_gimbal_first' | 'autoregressive_drone_first' | 'multihead_attention'
      - actor_cfg: dict (actor hyperparameters)
      - critic_type: 'original' | 'factored'
      - state_dim_per_frame: default 11 (vx,vy,vz, qd(4), qg(4))
    """
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb,
                 frame_stack=3,
                 state_dim_per_frame: int = 11,
                 enc: str = 'not_received',
                 enc_cfg: dict = None,
                 actor_type: str = 'original',
                 actor_cfg: dict = None,
                 critic_type: str = 'original',
                 lr_schedule: dict = None,
                 auxiliary_task: dict = None):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.frame_stack = frame_stack
        self.encoder_type = enc
        self.encoder_cfg = enc_cfg or {}
        self.actor_type = actor_type
        self.actor_cfg = actor_cfg or {}
        self.critic_type = critic_type
        self.state_dim_per_frame = state_dim_per_frame

        # Encoder 선택/생성
        C, H, W = obs_shape
        if enc == 'not_received':
            enc = 'org'
            for _ in range(8): print("[DrQV2Agent] encoder_type not received; defaulting to 'org' (baseline encoder).")
        if enc == 'org':
            self.encoder = EncoderBaseline(obs_shape, num_stacks=frame_stack,
                                           drone_state_dim=state_dim_per_frame).to(device)
        elif enc == 'A':
            d_model = self.encoder_cfg.get('d_model', 256)
            nhead   = self.encoder_cfg.get('nhead', 4)
            p_drop  = self.encoder_cfg.get('p_drop', 0.05)
            cross_depth = self.encoder_cfg.get('cross_depth', 1)
            self_depth  = self.encoder_cfg.get('self_depth', 1)
            self.encoder = EncoderA_TimeAware(T=frame_stack, d_model=d_model, nhead=nhead, p_drop=p_drop,
                                              cross_depth=cross_depth, self_depth=self_depth).to(device)
        elif enc == 'B':
            d_model = self.encoder_cfg.get('d_model', 256)
            nhead   = self.encoder_cfg.get('nhead', 4)
            p_drop  = self.encoder_cfg.get('p_drop', 0.05)
            patch   = self.encoder_cfg.get('patch', 12)       # → Np=49; pixel size of each patch
            img_sz  = self.encoder_cfg.get('img_size', H)     # 기본 obs H 사용
            depth_local = self.encoder_cfg.get('depth_local', 2)
            depth_temp  = self.encoder_cfg.get('depth_temp', 1)
            self.encoder = EncoderB_TimeAware(T=frame_stack, d_model=d_model, nhead=nhead, p_drop=p_drop,
                                              img_size=img_sz, patch=patch,
                                              depth_local=depth_local, depth_temp=depth_temp).to(device)
        else:
            raise ValueError(f"Unknown encoder_type: {enc}")

        # Actor 선택/생성
        print(f"[DrQV2Agent] Creating actor with type: {actor_type}")
        if actor_type == 'original':
            self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim, hidden_dim).to(device)
        elif actor_type == 'autoregressive_gimbal_first':
            self.actor = AutoregressiveActorGimbalFirst(
                self.encoder.repr_dim, action_shape, feature_dim, hidden_dim
            ).to(device)
        elif actor_type == 'autoregressive_drone_first':
            self.actor = AutoregressiveActorDroneFirst(
                self.encoder.repr_dim, action_shape, feature_dim, hidden_dim
            ).to(device)
        elif actor_type == 'multihead_attention':
            nhead = self.actor_cfg.get('nhead', 4)
            self.actor = MultiHeadAttentionActor(
                self.encoder.repr_dim, action_shape, feature_dim, hidden_dim, nhead=nhead
            ).to(device)
        else:
            raise ValueError(f"Unknown actor_type: {actor_type}")

        # Critic 선택/생성
        print(f"[DrQV2Agent] Creating critic with type: {critic_type}")
        if critic_type == 'original':
            self.critic = Critic(self.encoder.repr_dim, action_shape, feature_dim, hidden_dim).to(device)
            self.critic_target = Critic(self.encoder.repr_dim, action_shape, feature_dim, hidden_dim).to(device)
        elif critic_type == 'factored':
            self.critic = FactoredCritic(self.encoder.repr_dim, action_shape, feature_dim, hidden_dim).to(device)
            self.critic_target = FactoredCritic(self.encoder.repr_dim, action_shape, feature_dim, hidden_dim).to(device)
        else:
            raise ValueError(f"Unknown critic_type: {critic_type}")

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
            print(f"[DrQV2Agent] Auxiliary gimbal task ENABLED (weight={self.aux_weight})")
        else:
            self.auxiliary_head = None
            print("[DrQV2Agent] Auxiliary gimbal task DISABLED")

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
