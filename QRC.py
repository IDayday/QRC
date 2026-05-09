"""
QRC: Quasimetric Reachability Closure for online goal-conditioned RL.

本文件实现 QRC 文档中的主算法：
  1) 单一 bounded reachability critic Z_theta(s,a,g)；
  2) supported TD-Z backbone；
  3) same-trajectory direct evidence supervision；
  4) sampled replay closure 的单侧、非递归蒸馏；
  5) 可选 stitched evidence edge；
  6) actor 直接最大化 Z_theta。

依赖：仅依赖 PyTorch / NumPy；训练入口负责提供 episode-level replay buffer。
"""
from __future__ import annotations

import math
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# 网络组件
# -----------------------------------------------------------------------------


class SquashedGaussianActor(nn.Module):
    """连续动作 actor，输出 tanh-squashed Gaussian。

    注意：QRC 的 actor loss 是最大化 Z(s, a, g)，不是传统 Q value。
    这里保留 stochastic actor 是为了训练期探索；eval 时默认使用 tanh(mean)。
    """

    def __init__(self, state_dim: int, goal_dim: int, action_dim: int, hidden_dims=(256, 256)):
        super().__init__()
        dims = [state_dim + goal_dim] + list(hidden_dims)
        layers = []
        for din, dout in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(din, dout), nn.ReLU()]
        self.backbone = nn.Sequential(*layers)
        self.mean = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std = nn.Linear(hidden_dims[-1], action_dim)
        self.LOG_STD_MIN = -20.0
        self.LOG_STD_MAX = 2.0

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.distributions.Normal:
        h = self.backbone(torch.cat([state, goal], dim=-1))
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return torch.distributions.Normal(mean, log_std.exp())

    def sample(self, state: torch.Tensor, goal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.forward(state, goal)
        raw = dist.rsample()
        action = torch.tanh(raw)
        # tanh 变换后的 log_prob，用于可选熵/诊断；QRC 默认不用 entropy objective。
        log_prob = dist.log_prob(raw) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        mean_action = torch.tanh(dist.mean)
        return action, log_prob, mean_action


class SubgoalProposal(nn.Module):
    """轻量 proposal 网络。

    它只用于 planner_projected / mixed candidate source：raw proposal 必须先投影到 replay states，
    不能直接作为 QRC 主方法的 value witness。raw_planner 模式仅作为 failure diagnostic。
    """

    def __init__(self, state_dim: int, goal_dim: int, hidden_dims=(256, 256)):
        super().__init__()
        dims = [state_dim + goal_dim] + list(hidden_dims)
        layers = []
        for din, dout in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(din, dout), nn.ReLU()]
        self.backbone = nn.Sequential(*layers)
        self.loc = nn.Linear(hidden_dims[-1], state_dim)
        self.log_scale = nn.Linear(hidden_dims[-1], state_dim)
        self.LOG_SCALE_MIN = -8.0
        self.LOG_SCALE_MAX = 2.0

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.distributions.Laplace:
        h = self.backbone(torch.cat([state, goal], dim=-1))
        loc = self.loc(h)
        scale = self.log_scale(h).clamp(self.LOG_SCALE_MIN, self.LOG_SCALE_MAX).exp()
        return torch.distributions.Laplace(loc, scale)


class ReachabilityHead(nn.Module):
    """单个 reachability critic head。

    支持两种参数化：
      - sigmoid_z：直接输出 Z in (0,1)；工程起步最稳。
      - distance：输出 D=softplus(f)>=0，再取 Z=exp(-D)；更贴近 quasimetric 表达。
    """

    def __init__(
        self,
        state_dim: int,
        goal_dim: int,
        action_dim: int,
        hidden_dims=(256, 256),
        parameterization: str = "sigmoid_z",
        init_z: float = 0.05,
    ):
        super().__init__()
        if parameterization not in ("sigmoid_z", "distance"):
            raise ValueError("parameterization must be 'sigmoid_z' or 'distance'")
        self.parameterization = parameterization
        dims = [state_dim + action_dim + goal_dim] + list(hidden_dims)
        layers = []
        for din, dout in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(din, dout), nn.ReLU()]
        layers += [nn.Linear(hidden_dims[-1], 1)]
        self.net = nn.Sequential(*layers)

        # 中文注释：critic 初始过乐观会导致 actor 追逐幻觉目标；这里用较小 init_z 做保守初始化。
        init_z = float(np.clip(init_z, 1e-4, 1.0 - 1e-4))
        if parameterization == "sigmoid_z":
            init_bias = math.log(init_z / (1.0 - init_z))
        else:
            # softplus(bias) ≈ -log(init_z)
            init_d = -math.log(init_z)
            init_bias = math.log(math.exp(init_d) - 1.0) if init_d < 20 else init_d
        nn.init.constant_(self.net[-1].bias, init_bias)

    def raw(self, state: torch.Tensor, action: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action, goal], dim=-1))

    def z(self, state: torch.Tensor, action: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        out = self.raw(state, action, goal)
        if self.parameterization == "sigmoid_z":
            return torch.sigmoid(out).clamp(1e-6, 1.0 - 1e-6)
        d = F.softplus(out).clamp(max=80.0)
        return torch.exp(-d).clamp(1e-8, 1.0)

    def distance(self, state: torch.Tensor, action: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        out = self.raw(state, action, goal)
        if self.parameterization == "distance":
            return F.softplus(out).clamp(max=80.0)
        z = torch.sigmoid(out).clamp(1e-8, 1.0)
        return -torch.log(z)


class EnsembleReachabilityCritic(nn.Module):
    """Z critic ensemble，用于 TD 目标和 closure certificate。"""

    def __init__(
        self,
        state_dim: int,
        goal_dim: int,
        action_dim: int,
        n_heads: int = 4,
        hidden_dims=(256, 256),
        parameterization: str = "sigmoid_z",
        init_z: float = 0.05,
    ):
        super().__init__()
        self.heads = nn.ModuleList([
            ReachabilityHead(
                state_dim=state_dim,
                goal_dim=goal_dim,
                action_dim=action_dim,
                hidden_dims=hidden_dims,
                parameterization=parameterization,
                init_z=init_z,
            )
            for _ in range(int(n_heads))
        ])
        self.n_heads = int(n_heads)
        self.parameterization = parameterization

    def forward_z(self, state: torch.Tensor, action: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return torch.cat([head.z(state, action, goal) for head in self.heads], dim=-1)

    def forward_d(self, state: torch.Tensor, action: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return torch.cat([head.distance(state, action, goal) for head in self.heads], dim=-1)


# -----------------------------------------------------------------------------
# QRC Agent
# -----------------------------------------------------------------------------


class QRCAgent:
    def __init__(
        self,
        state_dim: int,
        goal_dim: int,
        action_dim: int,
        device: torch.device,
        gamma: float = 0.98,
        tau: float = 0.005,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        proposal_lr: float = 1e-4,
        hidden_dim: int = 256,
        n_heads: int = 4,
        parameterization: str = "sigmoid_z",
        init_z: float = 0.05,
        policy_delay: int = 2,
        actor_agg: str = "mean",
        action_l2: float = 1e-4,
        lambda_dir: float = 1.0,
        lambda_clo: float = 0.05,
        lambda_stitch: float = 0.0,
        lambda_proposal: float = 0.05,
        direct_batch_size: int = 256,
        closure_batch_size: int = 128,
        stitch_batch_size: int = 128,
        h_relab: int = 32,
        closure_candidates: int = 8,
        closure_source: str = "none",
        projection_pool_size: int = 64,
        closure_start_updates: int = 5000,
        closure_margin_z: float = 0.0,
        beta_init: float = 2.0,
        cert_sigma_floor: float = 0.01,
        calib_freq: int = 5000,
        calib_size: int = 512,
        calib_quantile: float = 0.90,
        calib_ema: float = 0.10,
        anti_degenerate_min_dist: float = 0.25,
        distance_mode: str = "xy",
        allow_raw_planner_witness: bool = False,
        use_proposal: bool = True,
        use_stitch: bool = False,
        stitch_h1_max: int = 16,
        stitch_h2_max: int = 16,
        stitch_knn_candidates: int = 32,
        stitch_join_sigma: float = 0.5,
        stitch_join_radius: float = 0.75,
        stitch_min_conf: float = 0.25,
        writer=None,
        logger=None,
    ):
        if closure_source not in ("none", "random", "planner_raw", "planner_projected", "mixed"):
            raise ValueError("closure_source must be none/random/planner_raw/planner_projected/mixed")
        if actor_agg not in ("mean", "min"):
            raise ValueError("actor_agg must be mean or min")
        if distance_mode not in ("xy", "full"):
            raise ValueError("distance_mode must be xy or full")

        self.state_dim = int(state_dim)
        self.goal_dim = int(goal_dim)
        self.action_dim = int(action_dim)
        self.device = device
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.policy_delay = max(1, int(policy_delay))
        self.actor_agg = actor_agg
        self.action_l2 = float(action_l2)

        self.lambda_dir = float(lambda_dir)
        self.lambda_clo = float(lambda_clo)
        self.lambda_stitch = float(lambda_stitch)
        self.lambda_proposal = float(lambda_proposal)
        self.direct_batch_size = int(direct_batch_size)
        self.closure_batch_size = int(closure_batch_size)
        self.stitch_batch_size = int(stitch_batch_size)
        self.h_relab = int(h_relab)
        self.closure_candidates = int(closure_candidates)
        self.closure_source = closure_source
        self.projection_pool_size = int(projection_pool_size)
        self.closure_start_updates = int(closure_start_updates)
        self.closure_margin_z = float(closure_margin_z)
        self.beta = float(beta_init)
        self.cert_sigma_floor = float(cert_sigma_floor)
        self.calib_freq = int(calib_freq)
        self.calib_size = int(calib_size)
        self.calib_quantile = float(calib_quantile)
        self.calib_ema = float(calib_ema)
        self.anti_degenerate_min_dist = float(anti_degenerate_min_dist)
        self.distance_mode = str(distance_mode)
        self.allow_raw_planner_witness = bool(allow_raw_planner_witness)
        self.use_proposal = bool(use_proposal)
        self.use_stitch = bool(use_stitch)
        self.stitch_h1_max = int(stitch_h1_max)
        self.stitch_h2_max = int(stitch_h2_max)
        self.stitch_knn_candidates = int(stitch_knn_candidates)
        self.stitch_join_sigma = float(stitch_join_sigma)
        self.stitch_join_radius = float(stitch_join_radius)
        self.stitch_min_conf = float(stitch_min_conf)
        self.writer = writer
        self.logger = logger
        self.total_it = 0

        hidden = (int(hidden_dim), int(hidden_dim))
        self.actor = SquashedGaussianActor(state_dim, goal_dim, action_dim, hidden).to(device)
        self.actor_target = SquashedGaussianActor(state_dim, goal_dim, action_dim, hidden).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = EnsembleReachabilityCritic(
            state_dim, goal_dim, action_dim, n_heads=n_heads, hidden_dims=hidden,
            parameterization=parameterization, init_z=init_z,
        ).to(device)
        self.critic_target = EnsembleReachabilityCritic(
            state_dim, goal_dim, action_dim, n_heads=n_heads, hidden_dims=hidden,
            parameterization=parameterization, init_z=init_z,
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.proposal = SubgoalProposal(state_dim, goal_dim, hidden).to(device)
        self.proposal_opt = torch.optim.Adam(self.proposal.parameters(), lr=proposal_lr)

    # ------------------------------------------------------------------
    # 保存 / 加载
    # ------------------------------------------------------------------
    def save(self, folder: str, save_optims: bool = False):
        os.makedirs(folder, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(folder, "qrc_actor.pth"))
        torch.save(self.actor_target.state_dict(), os.path.join(folder, "qrc_actor_target.pth"))
        torch.save(self.critic.state_dict(), os.path.join(folder, "qrc_critic.pth"))
        torch.save(self.critic_target.state_dict(), os.path.join(folder, "qrc_critic_target.pth"))
        torch.save(self.proposal.state_dict(), os.path.join(folder, "qrc_proposal.pth"))
        torch.save({"total_it": self.total_it, "beta": self.beta}, os.path.join(folder, "qrc_meta.pth"))
        if save_optims:
            torch.save(self.actor_opt.state_dict(), os.path.join(folder, "qrc_actor_opt.pth"))
            torch.save(self.critic_opt.state_dict(), os.path.join(folder, "qrc_critic_opt.pth"))
            torch.save(self.proposal_opt.state_dict(), os.path.join(folder, "qrc_proposal_opt.pth"))

    def load(self, folder: str):
        self.actor.load_state_dict(torch.load(os.path.join(folder, "qrc_actor.pth"), map_location=self.device))
        self.critic.load_state_dict(torch.load(os.path.join(folder, "qrc_critic.pth"), map_location=self.device))
        proposal_path = os.path.join(folder, "qrc_proposal.pth")
        if os.path.exists(proposal_path):
            self.proposal.load_state_dict(torch.load(proposal_path, map_location=self.device))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        meta_path = os.path.join(folder, "qrc_meta.pth")
        if os.path.exists(meta_path):
            meta = torch.load(meta_path, map_location=self.device)
            self.total_it = int(meta.get("total_it", self.total_it))
            self.beta = float(meta.get("beta", self.beta))

    @torch.no_grad()
    def select_action(self, state, goal, deterministic: bool = False):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).view(1, -1)
        goal_t = torch.as_tensor(goal, dtype=torch.float32, device=self.device).view(1, -1)
        action, _, mean_action = self.actor.sample(state_t, goal_t)
        out = mean_action if deterministic else action
        return out.cpu().numpy().reshape(-1)

    # ------------------------------------------------------------------
    # 基础 helper
    # ------------------------------------------------------------------
    def _state_to_goal(self, state: torch.Tensor) -> torch.Tensor:
        """把 replay full state 映射成 goal 表示。

        Ant 代码通常 state_dim == goal_dim；若后续环境 goal 只取前若干维，这里会自动裁剪。
        """
        if state.shape[-1] == self.goal_dim:
            return state
        if state.shape[-1] > self.goal_dim:
            return state[..., : self.goal_dim]
        pad = self.goal_dim - state.shape[-1]
        return F.pad(state, (0, pad))

    def _goal_xy_proxy(self, goal: torch.Tensor, state_like: Optional[torch.Tensor] = None) -> torch.Tensor:
        if goal.shape[-1] >= 2:
            return goal[..., :2]
        if state_like is None:
            return goal
        return self._state_to_goal(state_like)[..., : goal.shape[-1]]

    def _dist_state_state(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.distance_mode == "xy" and x.shape[-1] >= 2 and y.shape[-1] >= 2:
            return torch.norm(x[..., :2] - y[..., :2], p=2, dim=-1, keepdim=True)
        return torch.norm(x - y, p=2, dim=-1, keepdim=True)

    def _dist_state_goal(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        if self.distance_mode == "xy" and state.shape[-1] >= 2 and goal.shape[-1] >= 2:
            return torch.norm(state[..., :2] - goal[..., :2], p=2, dim=-1, keepdim=True)
        goal_proxy = self._state_to_goal(state)
        common = min(goal_proxy.shape[-1], goal.shape[-1])
        return torch.norm(goal_proxy[..., :common] - goal[..., :common], p=2, dim=-1, keepdim=True)

    def _hit(self, next_state: torch.Tensor, goal: torch.Tensor, distance_threshold: float) -> torch.Tensor:
        return (self._dist_state_goal(next_state, goal) <= float(distance_threshold)).float()

    def _actor_action(self, actor: SquashedGaussianActor, state: torch.Tensor, goal: torch.Tensor, deterministic: bool) -> torch.Tensor:
        action, _, mean_action = actor.sample(state, goal)
        return mean_action if deterministic else action

    @torch.no_grad()
    def _target_z_heads(self, state: torch.Tensor, goal: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        action = self._actor_action(self.actor_target, state, goal, deterministic=deterministic)
        return self.critic_target.forward_z(state, action, goal)

    @torch.no_grad()
    def _target_z_mean_std(self, state: torch.Tensor, goal: torch.Tensor, deterministic: bool = True):
        heads = self._target_z_heads(state, goal, deterministic=deterministic)
        mu = heads.mean(dim=-1, keepdim=True)
        std = heads.std(dim=-1, keepdim=True, unbiased=False) if heads.shape[-1] > 1 else torch.zeros_like(mu)
        return mu, std, heads

    @staticmethod
    def _float_metrics(metrics: Dict[str, object]) -> Dict[str, float]:
        out = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                out[k] = float(v.detach().mean().cpu())
            else:
                out[k] = float(v)
        return out

    def _write_metrics(self, metrics: Dict[str, float]):
        if self.writer is None:
            return
        tag_map = {
            "Z/td_loss": "z_td_loss",
            "Z/direct_loss": "z_direct_loss",
            "Z/direct_pred": "z_direct_pred",
            "Z/direct_target": "z_direct_target",
            "Z/mean": "z_mean",
            "Z/min": "z_min",
            "Z/max": "z_max",
            "D/mean": "d_mean",
            "D/saturation_rate": "d_saturation_rate",
            "QRC/closure_loss": "qrc_closure_loss",
            "QRC/closure_uplift": "qrc_closure_uplift",
            "QRC/best_Z_cert": "qrc_best_z_cert",
            "QRC/best_D_cert": "qrc_best_d_cert",
            "QRC/witness_hit_rate": "qrc_witness_hit_rate",
            "QRC/candidate_M": "qrc_candidate_m",
            "QRC/nondegenerate_rate": "qrc_nondegenerate_rate",
            "QRC/random_vs_projected_ratio": "qrc_random_vs_projected_ratio",
            "QRC/closure_accept_rate": "qrc_closure_accept_rate",
            "QRC/closure_gap": "qrc_closure_gap",
            "QRC/raw_planner_failure_rate": "qrc_raw_planner_failure_rate",
            "QRC/projected_distance": "qrc_projected_distance",
            "QRC/triangle_violation_rate": "qrc_triangle_violation_rate",
            "QRC/closure_target_outlier_rate": "qrc_closure_target_outlier_rate",
            "Evidence/direct_edge_count": "evidence_direct_edge_count",
            "Evidence/stitch_coverage_rate": "evidence_stitch_coverage_rate",
            "Evidence/join_dist": "evidence_join_dist",
            "Evidence/join_conf": "evidence_join_conf",
            "Evidence/h1": "evidence_h1",
            "Evidence/h2": "evidence_h2",
            "Evidence/target_z": "evidence_target_z",
            "Actor/loss": "actor_loss",
            "Actor/Z_action_mean": "actor_z_action_mean",
            "Proposal/loss": "proposal_loss",
            "Calib/beta": "calib_beta",
            "Calib/beta_new": "calib_beta_new",
        }
        for tag, key in tag_map.items():
            if key in metrics:
                self.writer.add_scalar(tag, float(metrics[key]), self.total_it)

    # ------------------------------------------------------------------
    # Replay sampling helpers
    # ------------------------------------------------------------------
    def _sample_episode_nonterminal_indices(self, replay_buffer, batch_size: int):
        active = replay_buffer._active_episode_slots()
        if active.size == 0:
            return None
        lengths = replay_buffer._episode_lengths[active].astype(np.int64)
        valid = lengths >= 1
        active = active[valid]
        lengths = lengths[valid]
        if active.size == 0:
            return None
        probs = np.maximum(lengths, 1).astype(np.float64)
        probs = probs / probs.sum()
        ep = replay_buffer._rng.choice(active, size=batch_size, replace=True, p=probs)
        chosen_lengths = replay_buffer._episode_lengths[ep].astype(np.int64)
        step = np.array([replay_buffer._rng.integers(0, max(int(L), 1)) for L in chosen_lengths], dtype=np.int64)
        return ep.astype(np.int64), step.astype(np.int64)

    def _sample_replay_states_with_indices(self, replay_buffer, batch_size: int):
        ep, step = replay_buffer._sample_episode_time_indices(batch_size)
        obs_key = replay_buffer.observation_key
        states_np = replay_buffer._obs[obs_key][ep, step]
        states = torch.as_tensor(states_np, dtype=torch.float32, device=self.device)
        return states, ep.astype(np.int64), step.astype(np.int64)

    def _sample_future_pairs(self, replay_buffer, batch_size: int, horizon: int):
        sample = self._sample_episode_nonterminal_indices(replay_buffer, batch_size)
        if sample is None:
            return None
        ep, start = sample
        lengths = replay_buffer._episode_lengths[ep].astype(np.int64)
        # future_step=start+h-1，对应 h 步 transition 后的 next_obs。
        max_h = np.minimum(int(horizon), lengths - start)
        max_h = np.maximum(max_h, 1)
        h = np.array([replay_buffer._rng.integers(1, int(mh) + 1) for mh in max_h], dtype=np.int64)
        future_step = start + h - 1
        return ep, start, h, future_step

    # ------------------------------------------------------------------
    # TD-Z backbone
    # ------------------------------------------------------------------
    def _td_z_loss(self, state, action, next_state, goal, distance_threshold: float):
        with torch.no_grad():
            hit = self._hit(next_state, goal, distance_threshold)
            next_z = self._target_z_heads(next_state, goal, deterministic=True).mean(dim=-1, keepdim=True)
            # 文档公式：Y_TD = gamma * [hit + (1-hit) Z_target(next_state,g)]
            y = self.gamma * (hit + (1.0 - hit) * next_z)
            y = y.clamp(0.0, 1.0)
        pred_heads = self.critic.forward_z(state, action, goal)
        td_loss = F.mse_loss(pred_heads, y.expand_as(pred_heads))
        with torch.no_grad():
            d_heads = -torch.log(pred_heads.clamp_min(1e-8))
            metrics = self._float_metrics({
                "z_td_loss": td_loss,
                "z_mean": pred_heads.mean(),
                "z_min": pred_heads.min(),
                "z_max": pred_heads.max(),
                "d_mean": d_heads.mean(),
                "d_saturation_rate": ((pred_heads < 1e-3) | (pred_heads > 1.0 - 1e-3)).float().mean(),
            })
        return td_loss, metrics

    # ------------------------------------------------------------------
    # Direct same-trajectory evidence loss
    # ------------------------------------------------------------------
    def _direct_evidence_loss(self, replay_buffer):
        zero = torch.zeros((), dtype=torch.float32, device=self.device)
        zero_m = {
            "z_direct_loss": 0.0,
            "z_direct_pred": 0.0,
            "z_direct_target": 0.0,
            "evidence_direct_edge_count": 0.0,
        }
        if replay_buffer is None or replay_buffer.num_steps_can_sample() < 2 or self.lambda_dir <= 0.0:
            return zero, zero_m
        sample = self._sample_future_pairs(replay_buffer, self.direct_batch_size, self.h_relab)
        if sample is None:
            return zero, zero_m
        ep, start, h, future_step = sample
        obs_key = replay_buffer.observation_key
        ag_key = replay_buffer.achieved_goal_key
        state_np = replay_buffer._obs[obs_key][ep, start]
        action_np = replay_buffer._actions[ep, start]
        if ag_key in replay_buffer._next_obs:
            goal_np = replay_buffer._next_obs[ag_key][ep, future_step]
        else:
            goal_np = replay_buffer._next_obs[obs_key][ep, future_step][..., : self.goal_dim]
        state = torch.as_tensor(state_np, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(action_np, dtype=torch.float32, device=self.device)
        goal = torch.as_tensor(goal_np, dtype=torch.float32, device=self.device)
        if goal.shape[-1] != self.goal_dim:
            goal = self._state_to_goal(goal)
        h_t = torch.as_tensor(h, dtype=torch.float32, device=self.device).view(-1, 1)
        target_z = torch.pow(torch.tensor(self.gamma, dtype=torch.float32, device=self.device), h_t).clamp(0.0, 1.0)
        pred_heads = self.critic.forward_z(state, action, goal)
        loss = F.mse_loss(pred_heads, target_z.expand_as(pred_heads))
        return loss, self._float_metrics({
            "z_direct_loss": loss,
            "z_direct_pred": pred_heads.mean(),
            "z_direct_target": target_z.mean(),
            "evidence_direct_edge_count": float(state.shape[0]),
        })

    # ------------------------------------------------------------------
    # Proposal training: supervised future-state predictor，供 projected candidate 使用
    # ------------------------------------------------------------------
    def _proposal_loss(self, replay_buffer):
        zero = torch.zeros((), dtype=torch.float32, device=self.device)
        if (not self.use_proposal) or self.lambda_proposal <= 0.0 or replay_buffer is None or replay_buffer.num_steps_can_sample() < 2:
            return zero, {"proposal_loss": 0.0}
        sample = self._sample_future_pairs(replay_buffer, min(self.direct_batch_size, 256), self.h_relab)
        if sample is None:
            return zero, {"proposal_loss": 0.0}
        ep, start, _, future_step = sample
        obs_key = replay_buffer.observation_key
        ag_key = replay_buffer.achieved_goal_key
        state_np = replay_buffer._obs[obs_key][ep, start]
        target_state_np = replay_buffer._next_obs[obs_key][ep, future_step]
        if ag_key in replay_buffer._next_obs:
            goal_np = replay_buffer._next_obs[ag_key][ep, future_step]
        else:
            goal_np = target_state_np[..., : self.goal_dim]
        state = torch.as_tensor(state_np, dtype=torch.float32, device=self.device)
        target_state = torch.as_tensor(target_state_np, dtype=torch.float32, device=self.device)
        goal = torch.as_tensor(goal_np, dtype=torch.float32, device=self.device)
        if goal.shape[-1] != self.goal_dim:
            goal = self._state_to_goal(goal)
        dist = self.proposal(state, goal)
        loss = F.smooth_l1_loss(dist.loc, target_state)
        return loss, self._float_metrics({"proposal_loss": loss})

    # ------------------------------------------------------------------
    # Closure candidate sampling and certificate
    # ------------------------------------------------------------------
    def _sample_bridge_anchors(self, replay_buffer, batch_size: int):
        state, _, _ = self._sample_replay_states_with_indices(replay_buffer, batch_size)
        goal_state, _, _ = self._sample_replay_states_with_indices(replay_buffer, batch_size)
        goal = self._state_to_goal(goal_state)
        return state, goal

    @torch.no_grad()
    def _sample_closure_candidates(self, state: torch.Tensor, goal: torch.Tensor, replay_buffer, M: int):
        B = state.shape[0]
        M = int(M)
        source = self.closure_source
        if source == "none":
            cand, _, _ = self._sample_replay_states_with_indices(replay_buffer, B * M)
            return cand.view(B, M, self.state_dim), {"qrc_projected_distance": 0.0, "qrc_random_vs_projected_ratio": 1.0}

        if source == "random":
            cand, _, _ = self._sample_replay_states_with_indices(replay_buffer, B * M)
            return cand.view(B, M, self.state_dim), {"qrc_projected_distance": 0.0, "qrc_random_vs_projected_ratio": 1.0}

        if source == "planner_raw":
            # 中文注释：raw planner witness 违反 QRC 的 evidence-first 主原则，仅用于失败诊断。
            if not self.allow_raw_planner_witness:
                raise RuntimeError(
                    "closure_source=planner_raw requires --allow_raw_planner_witness. "
                    "QRC main method should use planner_projected instead."
                )
            dist = self.proposal(state, goal)
            raw = dist.loc[:, None, :]
            if M > 1:
                samples = dist.rsample((M - 1,)).transpose(0, 1)
                cand = torch.cat([raw, samples], dim=1)
            else:
                cand = raw
            return cand, {"qrc_projected_distance": -1.0, "qrc_random_vs_projected_ratio": 0.0}

        def projected(num_out: int):
            raw = self.proposal(state, goal).loc.detach()
            pool_size = max(int(self.projection_pool_size), int(num_out))
            pool, _, _ = self._sample_replay_states_with_indices(replay_buffer, B * pool_size)
            pool = pool.view(B, pool_size, self.state_dim)
            if self.distance_mode == "xy" and self.state_dim >= 2:
                dist = torch.norm(pool[..., :2] - raw[:, None, :2], p=2, dim=-1)
            else:
                dist = torch.norm(pool - raw[:, None, :], p=2, dim=-1)
            topk = torch.topk(-dist, k=num_out, dim=1).indices
            row = torch.arange(B, device=self.device)[:, None].expand(-1, num_out)
            proj = pool[row, topk]
            return proj, dist[row, topk].mean()

        if source == "planner_projected":
            proj, proj_dist = projected(M)
            return proj, {"qrc_projected_distance": float(proj_dist.cpu()), "qrc_random_vs_projected_ratio": 0.0}

        # mixed: 一半 projected，一半 random replay states。
        m_proj = max(1, M // 2)
        m_rand = M - m_proj
        proj, proj_dist = projected(m_proj)
        if m_rand > 0:
            rand, _, _ = self._sample_replay_states_with_indices(replay_buffer, B * m_rand)
            cand = torch.cat([proj, rand.view(B, m_rand, self.state_dim)], dim=1)
        else:
            cand = proj
        ratio_random = float(m_rand) / float(max(M, 1))
        return cand, {"qrc_projected_distance": float(proj_dist.cpu()), "qrc_random_vs_projected_ratio": ratio_random}

    @torch.no_grad()
    def _closure_certificate(self, state: torch.Tensor, goal: torch.Tensor, cand: torch.Tensor):
        B, M, D = cand.shape
        s_flat = state[:, None, :].expand(-1, M, -1).reshape(B * M, D)
        m_flat = cand.reshape(B * M, D)
        m_goal_flat = self._state_to_goal(m_flat)
        g_flat = goal[:, None, :].expand(-1, M, -1).reshape(B * M, self.goal_dim)

        z_sm = self._target_z_heads(s_flat, m_goal_flat, deterministic=True)
        z_mg = self._target_z_heads(m_flat, g_flat, deterministic=True)
        mu_sm = z_sm.mean(dim=-1, keepdim=True)
        mu_mg = z_mg.mean(dim=-1, keepdim=True)
        sig_sm = z_sm.std(dim=-1, keepdim=True, unbiased=False) if z_sm.shape[-1] > 1 else torch.zeros_like(mu_sm)
        sig_mg = z_mg.std(dim=-1, keepdim=True, unbiased=False) if z_mg.shape[-1] > 1 else torch.zeros_like(mu_mg)
        lcb_sm = (mu_sm - self.beta * (sig_sm + self.cert_sigma_floor)).clamp(0.0, 1.0)
        lcb_mg = (mu_mg - self.beta * (sig_mg + self.cert_sigma_floor)).clamp(0.0, 1.0)
        z_cert = (lcb_sm * lcb_mg).view(B, M, 1).clamp(0.0, 1.0)

        d_sm = self._dist_state_state(state[:, None, :].expand(-1, M, -1), cand)
        d_mg = self._dist_state_goal(cand.reshape(B * M, D), g_flat).view(B, M, 1)
        nondeg = ((d_sm >= self.anti_degenerate_min_dist) & (d_mg >= self.anti_degenerate_min_dist)).float()
        z_cert = z_cert * nondeg

        best_idx = z_cert.squeeze(-1).argmax(dim=1)
        row = torch.arange(B, device=self.device)
        best_z = z_cert[row, best_idx]
        best_m = cand[row, best_idx]
        best_non = nondeg[row, best_idx]
        return best_z, best_m, best_non, {
            "nondeg_all": nondeg,
            "d_sm": d_sm[row, best_idx],
            "d_mg": d_mg[row, best_idx],
        }

    def _closure_loss(self, replay_buffer):
        zero = torch.zeros((), dtype=torch.float32, device=self.device)
        zero_m = {
            "qrc_closure_loss": 0.0,
            "qrc_closure_uplift": 0.0,
            "qrc_best_z_cert": 0.0,
            "qrc_best_d_cert": 0.0,
            "qrc_witness_hit_rate": 0.0,
            "qrc_candidate_m": float(self.closure_candidates),
            "qrc_nondegenerate_rate": 0.0,
            "qrc_random_vs_projected_ratio": 0.0,
            "qrc_closure_accept_rate": 0.0,
            "qrc_closure_gap": 0.0,
            "qrc_raw_planner_failure_rate": 0.0,
            "qrc_projected_distance": 0.0,
            "qrc_triangle_violation_rate": 0.0,
            "qrc_closure_target_outlier_rate": 0.0,
        }
        if (
            self.lambda_clo <= 0.0
            or self.closure_source == "none"
            or replay_buffer is None
            or self.total_it < self.closure_start_updates
            or replay_buffer.num_steps_can_sample() < max(2, self.closure_batch_size)
        ):
            return zero, zero_m

        state, goal = self._sample_bridge_anchors(replay_buffer, self.closure_batch_size)
        cand, cand_info = self._sample_closure_candidates(state, goal, replay_buffer, self.closure_candidates)
        best_z, _, best_non, cert_info = self._closure_certificate(state, goal, cand)

        # 中文注释：closure loss 只更新 critic。actor action stop-gradient，避免 closure 直接牵动 actor。
        with torch.no_grad():
            a_bar = self._actor_action(self.actor, state, goal, deterministic=True)
        pred_heads = self.critic.forward_z(state, a_bar, goal)
        pred_mean = pred_heads.mean(dim=-1, keepdim=True)
        uplift = F.relu(best_z.detach() - pred_heads - self.closure_margin_z)
        loss = F.smooth_l1_loss(uplift, torch.zeros_like(uplift), reduction="mean")
        gap = best_z.detach() - pred_mean.detach()
        accept = ((gap > self.closure_margin_z) & (best_non > 0.5)).float()
        best_d = -torch.log(best_z.clamp_min(1e-8))
        metrics = self._float_metrics({
            "qrc_closure_loss": loss,
            "qrc_closure_uplift": uplift.mean(),
            "qrc_best_z_cert": best_z.mean(),
            "qrc_best_d_cert": best_d.mean(),
            "qrc_witness_hit_rate": accept.mean(),
            "qrc_candidate_m": float(self.closure_candidates),
            "qrc_nondegenerate_rate": cert_info["nondeg_all"].mean(),
            "qrc_closure_accept_rate": accept.mean(),
            "qrc_closure_gap": gap.mean(),
            "qrc_raw_planner_failure_rate": 1.0 - best_non.mean(),
            "qrc_triangle_violation_rate": accept.mean(),
            "qrc_closure_target_outlier_rate": (best_z > 0.98).float().mean(),
        })
        metrics.update({k: float(v) for k, v in cand_info.items()})
        return loss, metrics

    # ------------------------------------------------------------------
    # Stitched evidence edge loss
    # ------------------------------------------------------------------
    def _stitch_loss(self, replay_buffer):
        zero = torch.zeros((), dtype=torch.float32, device=self.device)
        zero_m = {
            "evidence_stitch_coverage_rate": 0.0,
            "evidence_join_dist": 0.0,
            "evidence_join_conf": 0.0,
            "evidence_h1": 0.0,
            "evidence_h2": 0.0,
            "evidence_target_z": 0.0,
            "stitch_loss": 0.0,
        }
        if (not self.use_stitch) or self.lambda_stitch <= 0.0 or replay_buffer is None or replay_buffer.num_steps_can_sample() < 4:
            return zero, zero_m
        sample = self._sample_future_pairs(replay_buffer, self.stitch_batch_size, horizon=self.stitch_h1_max)
        if sample is None:
            return zero, zero_m
        ep1, start1, h1, m_step = sample
        B = int(ep1.shape[0])
        obs_key = replay_buffer.observation_key
        ag_key = replay_buffer.achieved_goal_key
        s_np = replay_buffer._obs[obs_key][ep1, start1]
        a_np = replay_buffer._actions[ep1, start1]
        m_state_np = replay_buffer._next_obs[obs_key][ep1, m_step]

        # 候选第二段起点 m'：从 replay 中采样若干 nonterminal state，选离 m 最近的。
        cand_sample = self._sample_episode_nonterminal_indices(replay_buffer, B * self.stitch_knn_candidates)
        if cand_sample is None:
            return zero, zero_m
        ep2_all, start2_all = cand_sample
        cand_start_np = replay_buffer._obs[obs_key][ep2_all, start2_all].reshape(B, self.stitch_knn_candidates, self.state_dim)
        m_state = torch.as_tensor(m_state_np, dtype=torch.float32, device=self.device)
        cand_start = torch.as_tensor(cand_start_np, dtype=torch.float32, device=self.device)
        if self.distance_mode == "xy" and self.state_dim >= 2:
            dist = torch.norm(cand_start[..., :2] - m_state[:, None, :2], p=2, dim=-1)
        else:
            dist = torch.norm(cand_start - m_state[:, None, :], p=2, dim=-1)
        best = dist.argmin(dim=1)
        row_np = np.arange(B)
        best_ep2 = ep2_all.reshape(B, self.stitch_knn_candidates)[row_np, best.detach().cpu().numpy()]
        best_start2 = start2_all.reshape(B, self.stitch_knn_candidates)[row_np, best.detach().cpu().numpy()]
        best_dist = dist[torch.arange(B, device=self.device), best].view(B, 1)
        join_conf = torch.exp(-best_dist.pow(2) / (2.0 * max(self.stitch_join_sigma, 1e-6) ** 2))
        accept = ((best_dist <= self.stitch_join_radius) & (join_conf >= self.stitch_min_conf)).float()

        lengths2 = replay_buffer._episode_lengths[best_ep2].astype(np.int64)
        max_h2 = np.minimum(self.stitch_h2_max, lengths2 - best_start2)
        max_h2 = np.maximum(max_h2, 1)
        h2 = np.array([replay_buffer._rng.integers(1, int(mh) + 1) for mh in max_h2], dtype=np.int64)
        g_step2 = best_start2 + h2 - 1
        if ag_key in replay_buffer._next_obs:
            g_np = replay_buffer._next_obs[ag_key][best_ep2, g_step2]
        else:
            g_np = replay_buffer._next_obs[obs_key][best_ep2, g_step2][..., : self.goal_dim]

        state = torch.as_tensor(s_np, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(a_np, dtype=torch.float32, device=self.device)
        goal = torch.as_tensor(g_np, dtype=torch.float32, device=self.device)
        if goal.shape[-1] != self.goal_dim:
            goal = self._state_to_goal(goal)
        h_total = torch.as_tensor(h1 + h2, dtype=torch.float32, device=self.device).view(B, 1)
        target_z = torch.pow(torch.tensor(self.gamma, dtype=torch.float32, device=self.device), h_total) * join_conf
        pred_heads = self.critic.forward_z(state, action, goal)
        uplift = F.relu(target_z.detach() - pred_heads)
        loss = (accept.detach() * F.smooth_l1_loss(uplift, torch.zeros_like(uplift), reduction="none")).mean()
        return loss, self._float_metrics({
            "stitch_loss": loss,
            "evidence_stitch_coverage_rate": accept.mean(),
            "evidence_join_dist": best_dist.mean(),
            "evidence_join_conf": join_conf.mean(),
            "evidence_h1": float(np.mean(h1)),
            "evidence_h2": float(np.mean(h2)),
            "evidence_target_z": target_z.mean(),
        })

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    @torch.no_grad()
    def update_beta(self, replay_buffer):
        if self.calib_freq <= 0 or replay_buffer is None or self.total_it % self.calib_freq != 0 or replay_buffer.num_steps_can_sample() < 2:
            return {"calib_beta": self.beta, "calib_beta_new": self.beta}
        sample = self._sample_future_pairs(replay_buffer, self.calib_size, self.h_relab)
        if sample is None:
            return {"calib_beta": self.beta, "calib_beta_new": self.beta}
        ep, start, h, future_step = sample
        obs_key = replay_buffer.observation_key
        ag_key = replay_buffer.achieved_goal_key
        state_np = replay_buffer._obs[obs_key][ep, start]
        action_np = replay_buffer._actions[ep, start]
        if ag_key in replay_buffer._next_obs:
            goal_np = replay_buffer._next_obs[ag_key][ep, future_step]
        else:
            goal_np = replay_buffer._next_obs[obs_key][ep, future_step][..., : self.goal_dim]
        state = torch.as_tensor(state_np, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(action_np, dtype=torch.float32, device=self.device)
        goal = torch.as_tensor(goal_np, dtype=torch.float32, device=self.device)
        if goal.shape[-1] != self.goal_dim:
            goal = self._state_to_goal(goal)
        h_t = torch.as_tensor(h, dtype=torch.float32, device=self.device).view(-1, 1)
        target_z = torch.pow(torch.tensor(self.gamma, dtype=torch.float32, device=self.device), h_t).clamp(0.0, 1.0)
        heads = self.critic_target.forward_z(state, action, goal)
        mu = heads.mean(dim=-1, keepdim=True)
        std = heads.std(dim=-1, keepdim=True, unbiased=False) if heads.shape[-1] > 1 else torch.zeros_like(mu)
        z_score = (target_z - mu).abs() / (std + self.cert_sigma_floor + 1e-6)
        finite = torch.isfinite(z_score.reshape(-1))
        beta_new = self.beta
        if finite.any():
            beta_new = float(torch.quantile(z_score.reshape(-1)[finite].clamp(0.0, 20.0), self.calib_quantile).cpu())
            beta_new = float(np.clip(beta_new, 0.5, 10.0))
            self.beta = (1.0 - self.calib_ema) * self.beta + self.calib_ema * beta_new
            self.beta = float(np.clip(self.beta, 0.5, 10.0))
        return {"calib_beta": self.beta, "calib_beta_new": beta_new}

    # ------------------------------------------------------------------
    # Main train step
    # ------------------------------------------------------------------
    def train(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        goal: torch.Tensor,
        replay_buffer=None,
        distance_threshold: float = 0.5,
    ) -> Dict[str, float]:
        td_loss, td_metrics = self._td_z_loss(state, action, next_state, goal, distance_threshold)
        direct_loss, direct_metrics = self._direct_evidence_loss(replay_buffer)
        closure_loss, closure_metrics = self._closure_loss(replay_buffer)
        stitch_loss, stitch_metrics = self._stitch_loss(replay_buffer)

        critic_loss = (
            td_loss
            + self.lambda_dir * direct_loss
            + self.lambda_clo * closure_loss
            + self.lambda_stitch * stitch_loss
        )
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=20.0)
        self.critic_opt.step()

        # Proposal 网络单独训练，避免它的梯度混入 critic/actor。
        prop_loss, prop_metrics = self._proposal_loss(replay_buffer)
        if self.use_proposal and prop_loss.requires_grad and self.lambda_proposal > 0.0:
            self.proposal_opt.zero_grad(set_to_none=True)
            (self.lambda_proposal * prop_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.proposal.parameters(), max_norm=10.0)
            self.proposal_opt.step()

        actor_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        actor_z = torch.zeros((), dtype=torch.float32, device=self.device)
        if self.total_it % self.policy_delay == 0:
            a_pi, _, _ = self.actor.sample(state, goal)
            z_heads = self.critic.forward_z(state, a_pi, goal)
            z_for_actor = z_heads.mean(dim=-1, keepdim=True) if self.actor_agg == "mean" else z_heads.min(dim=-1, keepdim=True)[0]
            actor_z = z_for_actor.mean()
            actor_loss = -actor_z + self.action_l2 * a_pi.pow(2).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
            self.actor_opt.step()

        beta_metrics = self.update_beta(replay_buffer)
        self.total_it += 1
        self._soft_update_targets()

        metrics = self._float_metrics({
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "actor_z_action_mean": actor_z,
        })
        metrics.update(td_metrics)
        metrics.update(direct_metrics)
        metrics.update(closure_metrics)
        metrics.update(stitch_metrics)
        metrics.update(prop_metrics)
        metrics.update(beta_metrics)
        self._write_metrics(metrics)
        if self.logger is not None:
            try:
                self.logger.store(**metrics)
            except Exception:
                pass
        return metrics

    def _soft_update_targets(self):
        with torch.no_grad():
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
