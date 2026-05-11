from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import time
from typing import Dict, List, Tuple

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from multiworld.envs.mujoco import register_custom_envs as register_mujoco_envs

from HER_adaptive_backup import HERReplayBuffer, PathBuilder, GOAL_TYPE_FUTURE, GOAL_TYPE_ROLLOUT, GOAL_TYPE_REPLAY
from QRC import QRCAgent


# -----------------------------------------------------------------------------
# CSV 诊断记录器
# -----------------------------------------------------------------------------


CSV_FIELDS = [
    "row_type", "wall_time", "elapsed_min", "env_step", "update", "episode", "episode_step",
    "exp_name", "env_name", "seed", "replay_size", "steps_per_second",
    "eval_train_success", "eval_train_distance", "eval_test_success", "eval_test_distance",
    "train_episode_distance", "train_orig_goal_ratio", "train_future_goal_ratio", "train_replay_goal_ratio", "train_actor_goal_ratio",
    "critic_loss", "actor_loss", "actor_loss_step", "actor_loss_ema",
    "actor_z_action_mean", "actor_z_action_mean_step", "actor_z_action_mean_ema", "actor_update_flag", "actor_update_count", "actor_batch_fraction",
    "z_td_loss", "z_td_target_mean", "z_td_pred_mean", "z_td_hit_rate", "z_direct_loss", "z_direct_pred", "z_direct_target", "z_direct_gap", "z_mean", "z_min", "z_max",
    "z_replay_goal_mean", "z_replay_goal_td_target", "z_actor_goal_mean",
    "d_mean", "d_saturation_rate",
    "qrc_td_closure_override_rate", "qrc_td_direct_next_z_mean", "qrc_td_closure_next_z_mean",
    "qrc_td_closure_gap_mean", "qrc_td_best_z_raw", "qrc_td_best_z_lcb",
    "qrc_closure_loss", "qrc_closure_uplift", "qrc_best_z_cert", "qrc_best_z_lcb", "qrc_best_z_raw", "qrc_best_z_raw_at_lcb", "qrc_best_d_cert",
    "qrc_witness_hit_rate", "qrc_candidate_m", "qrc_nondegenerate_rate", "qrc_random_vs_projected_ratio",
    "qrc_closure_accept_rate", "qrc_raw_accept_rate", "qrc_closure_gap", "qrc_closure_gap_lcb", "qrc_closure_gap_raw",
    "qrc_pred_z_mean", "qrc_cert_suppression", "qrc_cert_mu_sm", "qrc_cert_mu_mg", "qrc_cert_sig_sm", "qrc_cert_sig_mg",
    "qrc_raw_planner_failure_rate", "qrc_projected_distance", "qrc_triangle_violation_rate", "qrc_closure_target_outlier_rate",
    "evidence_direct_edge_count", "evidence_stitch_coverage_rate", "evidence_join_dist", "evidence_join_conf",
    "evidence_h1", "evidence_h2", "evidence_target_z", "stitch_loss",
    "proposal_loss", "calib_beta", "calib_beta_new",
]


class CSVMetricLogger:
    """每个 run 单独写 diagnostics.csv，便于 pandas 直接读取。"""

    def __init__(self, path: str, fields: List[str]):
        self.path = path
        self.fields = list(fields)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.file = open(path, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fields, extrasaction="ignore")
        self.writer.writeheader()
        self.file.flush()

    def write(self, row: Dict[str, object]):
        clean = {k: row.get(k, "") for k in self.fields}
        self.writer.writerow(clean)
        self.file.flush()

    def close(self):
        self.file.flush()
        self.file.close()


# -----------------------------------------------------------------------------
# 环境 / replay helper
# -----------------------------------------------------------------------------


def _as_float_dist(status, key: str = "xy-distance") -> float:
    if status is None:
        return float("nan")
    v = status.get(key, None)
    if v is None and key == "xy-distance":
        v = status.get("xy_distance", None)
    if v is None:
        return float("nan")
    arr = np.asarray(v).reshape(-1)
    return float(arr[0])


def infer_dims(env) -> Tuple[int, int, int]:
    obs = env.reset()
    state_dim = int(np.asarray(obs["observation"]).reshape(-1).shape[0])
    goal_dim = int(np.asarray(obs["desired_goal"]).reshape(-1).shape[0])
    action_dim = int(env.action_space.shape[0])
    return state_dim, goal_dim, action_dim


def seed_env(env, seed: int):
    """兼容旧 gym / multiworld 的环境播种。"""
    try:
        env.seed(int(seed))
    except Exception:
        pass
    try:
        env.action_space.seed(int(seed))
    except Exception:
        pass


def goal_from_state_np(state_np: np.ndarray, goal_dim: int) -> np.ndarray:
    if state_np.shape[-1] == goal_dim:
        return state_np
    if state_np.shape[-1] > goal_dim:
        return state_np[..., :goal_dim]
    pad = goal_dim - state_np.shape[-1]
    return np.pad(state_np, [(0, 0)] * (state_np.ndim - 1) + [(0, pad)], mode="constant")


def sample_control_batch(
    replay_buffer: HERReplayBuffer,
    batch_size: int,
    p_orig: float,
    p_future: float,
    p_replay_goal: float,
    h_relab: int,
    goal_dim: int,
    device: torch.device,
):
    """采样 QRC control batch。

    目标混合：original goal + same-episode future goal + replay/random achieved goal。
    replay/random goal 只用于 TD-Z 约束 critic，不用于 direct evidence，也默认不用于 actor 更新。
    """
    ep_slots, step_indices = replay_buffer._sample_episode_time_indices(batch_size)
    lengths = replay_buffer._episode_lengths[ep_slots].astype(np.int64)

    obs_key = replay_buffer.observation_key
    dg_key = replay_buffer.desired_goal_key
    ag_key = replay_buffer.achieved_goal_key

    states_np = replay_buffer._obs[obs_key][ep_slots, step_indices].copy()
    actions_np = replay_buffer._actions[ep_slots, step_indices].copy()
    next_states_np = replay_buffer._next_obs[obs_key][ep_slots, step_indices].copy()

    # 原始 goal：使用 replay 中 transition 对应的 desired_goal。
    if dg_key in replay_buffer._next_obs:
        goals_np = replay_buffer._next_obs[dg_key][ep_slots, step_indices].copy()
    else:
        goals_np = goal_from_state_np(replay_buffer._next_obs[obs_key][ep_slots, step_indices].copy(), goal_dim)
    if goals_np.shape[-1] != goal_dim:
        goals_np = goal_from_state_np(goals_np, goal_dim)

    # 中文注释：用显式三路 goal mix，而不是只有 orig/future。
    # replay/random goal 为 critic 提供更多低可达性 TD 约束，抑制 direct evidence 带来的过乐观。
    probs = np.asarray([float(p_orig), float(p_future), float(p_replay_goal)], dtype=np.float64)
    if probs.sum() <= 0:
        probs = np.asarray([0.25, 0.50, 0.25], dtype=np.float64)
    probs = np.maximum(probs, 0.0)
    probs = probs / probs.sum()
    u = replay_buffer._rng.random(batch_size)
    use_orig = u < probs[0]
    use_future = (u >= probs[0]) & (u < probs[0] + probs[1])
    use_replay = ~(use_orig | use_future)

    goal_source_type = np.full(batch_size, GOAL_TYPE_ROLLOUT, dtype=np.int64)
    goal_source_episode_slot = np.full(batch_size, -1, dtype=np.int64)
    goal_source_index = np.full(batch_size, -1, dtype=np.int64)

    if np.any(use_future):
        idx = np.where(use_future)[0]
        # future_step=start+h-1，允许 h=1，即下一状态 achieved goal。
        max_future = np.minimum(lengths[idx] - 1, step_indices[idx] + int(h_relab) - 1)
        max_future = np.maximum(max_future, step_indices[idx])
        span = np.maximum(max_future - step_indices[idx] + 1, 1).astype(np.int64)
        future_steps = step_indices[idx] + np.floor(replay_buffer._rng.random(idx.shape[0]) * span).astype(np.int64)
        if ag_key in replay_buffer._next_obs:
            future_goals = replay_buffer._next_obs[ag_key][ep_slots[idx], future_steps]
        else:
            future_goals = goal_from_state_np(replay_buffer._next_obs[obs_key][ep_slots[idx], future_steps], goal_dim)
        if future_goals.shape[-1] != goal_dim:
            future_goals = goal_from_state_np(future_goals, goal_dim)
        goals_np[idx] = future_goals
        goal_source_type[idx] = GOAL_TYPE_FUTURE
        goal_source_episode_slot[idx] = ep_slots[idx]
        goal_source_index[idx] = future_steps

    if np.any(use_replay):
        idx = np.where(use_replay)[0]
        replay_ep, replay_step = replay_buffer._sample_episode_time_indices(idx.shape[0])
        if ag_key in replay_buffer._next_obs:
            replay_goals = replay_buffer._next_obs[ag_key][replay_ep, replay_step]
        else:
            replay_goals = goal_from_state_np(replay_buffer._next_obs[obs_key][replay_ep, replay_step], goal_dim)
        if replay_goals.shape[-1] != goal_dim:
            replay_goals = goal_from_state_np(replay_goals, goal_dim)
        goals_np[idx] = replay_goals
        goal_source_type[idx] = GOAL_TYPE_REPLAY
        goal_source_episode_slot[idx] = replay_ep
        goal_source_index[idx] = replay_step

    states = torch.as_tensor(states_np, dtype=torch.float32, device=device)
    actions = torch.as_tensor(actions_np, dtype=torch.float32, device=device)
    next_states = torch.as_tensor(next_states_np, dtype=torch.float32, device=device)
    goals = torch.as_tensor(goals_np, dtype=torch.float32, device=device)
    batch_info = {
        "episode_slots": ep_slots.reshape(-1).astype(np.int64),
        "step_indices": step_indices.reshape(-1).astype(np.int64),
        "goal_source_type": goal_source_type.reshape(-1).astype(np.int64),
        "goal_source_episode_slot": goal_source_episode_slot.reshape(-1).astype(np.int64),
        "goal_source_index": goal_source_index.reshape(-1).astype(np.int64),
        "orig_goal_ratio": float(np.mean(use_orig.astype(np.float32))),
        "future_goal_ratio": float(np.mean(use_future.astype(np.float32))),
        "replay_goal_ratio": float(np.mean(use_replay.astype(np.float32))),
        "actor_goal_ratio": float(np.mean((~use_replay).astype(np.float32))),
    }
    return states, actions, next_states, goals, batch_info


@torch.no_grad()
def eval_policy(policy: QRCAgent, env, n_eval: int, max_episode_length: int, distance_threshold: float, deterministic: bool = True):
    distances = []
    successes = []
    for _ in range(int(n_eval)):
        obs = env.reset()
        state = obs["observation"]
        goal = obs["desired_goal"]
        t = 0
        while True:
            action = policy.select_action(state, goal, deterministic=deterministic)
            next_obs, _, _, status = env.step(action)
            state = next_obs["observation"]
            dist = _as_float_dist(status)
            t += 1
            done = dist < distance_threshold or t >= max_episode_length
            if done:
                distances.append(float(dist))
                successes.append(float(dist < distance_threshold))
                break
    return float(np.mean(distances)), float(np.mean(successes))


def save_config(folder: str, args, train_env_name: str, test_env_name: str):
    payload = vars(args).copy()
    payload["train_env_name"] = train_env_name
    payload["test_env_name"] = test_env_name
    path = os.path.join(folder, "config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def make_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--env_name", default="AntU", choices=["AntU", "AntFb", "AntMaze", "AntFg"])
    p.add_argument("--distance_threshold", default=0.5, type=float)
    p.add_argument("--start_timesteps", default=10000, type=int)
    p.add_argument("--eval_freq", default=5000, type=int)
    p.add_argument("--max_timesteps", default=1000000, type=int)
    p.add_argument("--max_episode_length", default=600, type=int)
    p.add_argument("--batch_size", default=1024, type=int)
    p.add_argument("--updates_per_step", default=1, type=int)
    p.add_argument("--replay_buffer_size", default=1000000, type=int)
    p.add_argument("--episode_slot_multiplier", default=4.0, type=float)
    p.add_argument("--n_eval", default=50, type=int)
    p.add_argument("--n_eval_test", default=10, type=int)
    p.add_argument("--device", default="cuda")
    p.add_argument("--cuda_visible_devices", default=None, type=str)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--exp_name", default="qrc")
    p.add_argument("--log_root", default="results_qrc")
    p.add_argument("--save_freq", default=50000, type=int)
    p.add_argument("--csv_train_log_freq", default=101, type=int, help="默认用 101 避免与 policy_delay=2 发生日志采样别名；代码也记录 last/EMA actor 指标")

    # QRC core hyperparameters
    p.add_argument("--gamma", default=0.98, type=float, help="reachability Z=gamma^d 的折扣，Ant 推荐 0.98")
    p.add_argument("--tau", default=0.005, type=float, help="target network soft-update 系数")
    p.add_argument("--actor_lr", default=3e-4, type=float)
    p.add_argument("--critic_lr", default=3e-4, type=float)
    p.add_argument("--proposal_lr", default=1e-4, type=float)
    p.add_argument("--hidden_dim", default=256, type=int)
    p.add_argument("--n_heads", default=4, type=int, help="Z critic ensemble head 数，用于 LCB certificate")
    p.add_argument("--parameterization", default="sigmoid_z", choices=["sigmoid_z", "distance"])
    p.add_argument("--init_z", default=0.05, type=float, help="critic 初始 reachability，越小越保守")
    p.add_argument("--policy_delay", default=2, type=int)
    p.add_argument("--actor_agg", default="mean", choices=["mean", "min"])
    p.add_argument("--action_l2", default=1e-4, type=float)
    p.add_argument("--deterministic_actor_update", dest="deterministic_actor_update", action="store_true",
                   help="actor 更新使用 tanh(mean)；更贴近 QRC 文档中的 μ_phi")
    p.add_argument("--no-deterministic_actor_update", dest="deterministic_actor_update", action="store_false")
    p.set_defaults(deterministic_actor_update=True)
    p.add_argument("--exploration_noise", default=0.10, type=float, help="环境交互时加到 mean action 上的 Gaussian 噪声；eval 不使用")
    p.add_argument("--torch_num_threads", default=1, type=int, help="限制每个实验进程 CPU 线程，便于多卡并行")

    # Evidence / closure weights
    p.add_argument("--p_orig", default=0.25, type=float, help="control batch 中原始目标比例")
    p.add_argument("--p_future", default=0.50, type=float, help="control batch 中 same-episode future goal 比例；actor 可使用")
    p.add_argument("--p_replay_goal", default=0.25, type=float, help="control batch 中 replay/random achieved goal 比例；只约束 critic TD，默认不更新 actor")
    p.add_argument("--h_relab", default=32, type=int, help="future-pair evidence window")
    p.add_argument("--lambda_dir", default=1.0, type=float, help="direct same-trajectory evidence loss 权重")
    p.add_argument("--lambda_clo", default=0.05, type=float, help="closure one-sided distillation loss 权重")
    p.add_argument("--lambda_stitch", default=0.0, type=float, help="stitched evidence loss 权重；默认关闭")
    p.add_argument("--lambda_proposal", default=0.0, type=float, help="proposal 监督损失权重；direct/random 阶段必须为 0，projected/mixed 阶段可设 0.05")
    p.add_argument("--direct_batch_size", default=256, type=int)
    p.add_argument("--closure_batch_size", default=128, type=int)
    p.add_argument("--stitch_batch_size", default=128, type=int)
    p.add_argument("--closure_candidates", default=8, type=int)
    p.add_argument("--closure_source", default="none", choices=["none", "random", "planner_raw", "planner_projected", "mixed"])
    p.add_argument("--projection_pool_size", default=64, type=int)
    p.add_argument("--closure_start_updates", default=5000, type=int)
    p.add_argument("--closure_margin_z", default=0.0, type=float)
    p.add_argument("--closure_loss_target", default="lcb", choices=["lcb", "raw"],
                   help="one-sided closure 的 target：lcb 为主方法，raw 仅用于诊断 LCB 是否过保守")
    p.add_argument("--td_closure_mode", default="none", choices=["none", "recursive_raw", "recursive_lcb"],
                   help="不安全 recursive TD closure 对照；主方法必须保持 none")
    p.add_argument("--td_closure_start_updates", default=0, type=int,
                   help="recursive TD closure 对照从第几个 update 开始")
    p.add_argument("--beta_init", default=2.0, type=float)
    p.add_argument("--beta_mode", default="fixed", choices=["fixed", "dynamic", "diagnostic"], help="fixed 默认不更新 beta；dynamic 才用校准 EMA；diagnostic 只记录 beta_new")
    p.add_argument("--cert_sigma_floor", default=0.01, type=float)
    p.add_argument("--calib_freq", default=5000, type=int)
    p.add_argument("--calib_size", default=512, type=int)
    p.add_argument("--calib_quantile", default=0.90, type=float)
    p.add_argument("--calib_ema", default=0.10, type=float)
    p.add_argument("--anti_degenerate_min_dist", default=0.25, type=float)
    p.add_argument("--distance_mode", default="xy", choices=["xy", "full"])
    p.add_argument("--allow_raw_planner_witness", action="store_true")
    p.add_argument("--use_proposal", dest="use_proposal", action="store_true")
    p.add_argument("--no-use_proposal", dest="use_proposal", action="store_false")
    p.set_defaults(use_proposal=True)

    # Stitched evidence
    p.add_argument("--use_stitch", dest="use_stitch", action="store_true")
    p.add_argument("--no-use_stitch", dest="use_stitch", action="store_false")
    p.set_defaults(use_stitch=False)
    p.add_argument("--stitch_h1_max", default=16, type=int)
    p.add_argument("--stitch_h2_max", default=16, type=int)
    p.add_argument("--stitch_knn_candidates", default=32, type=int)
    p.add_argument("--stitch_join_sigma", default=0.5, type=float)
    p.add_argument("--stitch_join_radius", default=0.75, type=float)
    p.add_argument("--stitch_min_conf", default=0.25, type=float)

    p.add_argument("--eval_deterministic", dest="eval_deterministic", action="store_true")
    p.add_argument("--no-eval_deterministic", dest="eval_deterministic", action="store_false")
    p.set_defaults(eval_deterministic=True)
    return p


def main():
    args = make_parser().parse_args()
    if args.torch_num_threads > 0:
        torch.set_num_threads(int(args.torch_num_threads))
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    env_map = {
        "AntU": ("AntULongTrainEnv-v0", "AntULongTestEnv-v0"),
        "AntFb": ("AntFbMedTrainEnv-v1", "AntFbMedTestEnv-v1"),
        "AntMaze": ("AntMazeMedTrainEnv-v1", "AntMazeMedTestEnv-v1"),
        "AntFg": ("AntFgMedTrainEnv-v1", "AntFgMedTestEnv-v1"),
    }
    train_env_name, test_env_name = env_map[args.env_name]
    print(args)
    print("Environments:", train_env_name, test_env_name)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    register_mujoco_envs()
    env = gym.make(train_env_name)
    train_eval_env = gym.make(train_env_name)
    test_eval_env = gym.make(test_env_name)
    seed_env(env, args.seed)
    seed_env(train_eval_env, args.seed + 1000)
    seed_env(test_eval_env, args.seed + 2000)
    state_dim, goal_dim, action_dim = infer_dims(env)
    print(f"Dims: state_dim={state_dim}, goal_dim={goal_dim}, action_dim={action_dim}")

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    ex_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    folder = os.path.join(args.log_root, train_env_name, "QRC", args.exp_name, f"seed{args.seed}_{ex_time}")
    os.makedirs(folder, exist_ok=True)
    print("Run folder:", folder)
    print("Saved config to:", save_config(folder, args, train_env_name, test_env_name))

    for file in ["QRC.py", "train_ant_qrc.py", "HER_adaptive_backup.py"]:
        if os.path.exists(file):
            shutil.copy(file, os.path.join(folder, os.path.basename(file)))

    writer = SummaryWriter(folder)
    csv_logger = CSVMetricLogger(os.path.join(folder, "diagnostics.csv"), CSV_FIELDS)

    policy = QRCAgent(
        state_dim=state_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        device=device,
        gamma=args.gamma,
        tau=args.tau,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        proposal_lr=args.proposal_lr,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        parameterization=args.parameterization,
        init_z=args.init_z,
        policy_delay=args.policy_delay,
        actor_agg=args.actor_agg,
        action_l2=args.action_l2,
        deterministic_actor_update=args.deterministic_actor_update,
        exploration_noise=args.exploration_noise,
        lambda_dir=args.lambda_dir,
        lambda_clo=args.lambda_clo,
        lambda_stitch=args.lambda_stitch,
        lambda_proposal=args.lambda_proposal,
        direct_batch_size=args.direct_batch_size,
        closure_batch_size=args.closure_batch_size,
        stitch_batch_size=args.stitch_batch_size,
        h_relab=args.h_relab,
        closure_candidates=args.closure_candidates,
        closure_source=args.closure_source,
        projection_pool_size=args.projection_pool_size,
        closure_start_updates=args.closure_start_updates,
        closure_margin_z=args.closure_margin_z,
        closure_loss_target=args.closure_loss_target,
        td_closure_mode=args.td_closure_mode,
        td_closure_start_updates=args.td_closure_start_updates,
        beta_init=args.beta_init,
        beta_mode=args.beta_mode,
        cert_sigma_floor=args.cert_sigma_floor,
        calib_freq=args.calib_freq,
        calib_size=args.calib_size,
        calib_quantile=args.calib_quantile,
        calib_ema=args.calib_ema,
        anti_degenerate_min_dist=args.anti_degenerate_min_dist,
        distance_mode=args.distance_mode,
        allow_raw_planner_witness=args.allow_raw_planner_witness,
        use_proposal=args.use_proposal,
        use_stitch=args.use_stitch,
        stitch_h1_max=args.stitch_h1_max,
        stitch_h2_max=args.stitch_h2_max,
        stitch_knn_candidates=args.stitch_knn_candidates,
        stitch_join_sigma=args.stitch_join_sigma,
        stitch_join_radius=args.stitch_join_radius,
        stitch_min_conf=args.stitch_min_conf,
        writer=writer,
    )

    replay_buffer = HERReplayBuffer(
        max_size=args.replay_buffer_size,
        max_episode_length=args.max_episode_length,
        episode_slot_multiplier=args.episode_slot_multiplier,
        env=env,
        fraction_goals_are_rollout_goals=0.2,
        fraction_resampled_goals_are_env_goals=0.0,
        fraction_resampled_goals_are_replay_buffer_goals=0.5,
        ob_keys_to_save=["state_achieved_goal", "state_desired_goal"],
        desired_goal_keys=["desired_goal", "state_desired_goal"],
        observation_key="observation",
        desired_goal_key="desired_goal",
        achieved_goal_key="achieved_goal",
        vectorized=True,
    )

    path_builder = PathBuilder()
    obs = env.reset()
    state = obs["observation"]
    goal = obs["desired_goal"]
    episode_timesteps = 0
    episode_num = 0
    train_start_time = time.time()
    last_eval_time = train_start_time
    last_eval_step = 0
    last_episode_distance = float("nan")
    last_orig_ratio = 0.0
    last_future_ratio = 0.0
    last_replay_ratio = 0.0
    last_actor_goal_ratio = 0.0

    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.select_action(state, goal, deterministic=False)
        next_obs, _, _, status = env.step(action)
        next_state = next_obs["observation"]
        dist = _as_float_dist(status)
        done = dist < args.distance_threshold

        path_builder.add_all(
            observations=obs,
            actions=action,
            next_observations=next_obs,
            terminals=[1.0 * done],
        )
        state = next_state
        obs = next_obs
        last_episode_distance = dist

        if t >= args.start_timesteps and replay_buffer.num_steps_can_sample() >= max(args.batch_size, 2):
            for _ in range(int(args.updates_per_step)):
                state_b, action_b, next_state_b, goal_b, batch_info = sample_control_batch(
                    replay_buffer,
                    batch_size=args.batch_size,
                    p_orig=args.p_orig,
                    p_future=args.p_future,
                    p_replay_goal=args.p_replay_goal,
                    h_relab=args.h_relab,
                    goal_dim=goal_dim,
                    device=device,
                )
                last_orig_ratio = float(batch_info.get("orig_goal_ratio", 0.0))
                last_future_ratio = float(batch_info.get("future_goal_ratio", 0.0))
                last_replay_ratio = float(batch_info.get("replay_goal_ratio", 0.0))
                last_actor_goal_ratio = float(batch_info.get("actor_goal_ratio", 0.0))
                goal_source = np.asarray(batch_info["goal_source_type"]).reshape(-1)
                # actor 不追 replay/random goal；这些 goal 只通过 TD-Z 给 critic 提供低可达性约束。
                actor_mask = torch.as_tensor(goal_source != GOAL_TYPE_REPLAY, dtype=torch.bool, device=device)
                replay_goal_mask = torch.as_tensor(goal_source == GOAL_TYPE_REPLAY, dtype=torch.bool, device=device)
                metrics = policy.train(
                    state_b,
                    action_b,
                    next_state_b,
                    goal_b,
                    replay_buffer=replay_buffer,
                    actor_mask=actor_mask,
                    replay_goal_mask=replay_goal_mask,
                    distance_threshold=args.distance_threshold,
                )
                if policy.total_it % args.csv_train_log_freq == 0:
                    elapsed = time.time() - train_start_time
                    row = {
                        "row_type": "train",
                        "wall_time": time.time(),
                        "elapsed_min": elapsed / 60.0,
                        "env_step": t + 1,
                        "update": policy.total_it,
                        "episode": episode_num,
                        "episode_step": episode_timesteps,
                        "exp_name": args.exp_name,
                        "env_name": args.env_name,
                        "seed": args.seed,
                        "replay_size": replay_buffer.num_steps_can_sample(),
                        "train_episode_distance": last_episode_distance,
                        "train_orig_goal_ratio": last_orig_ratio,
                        "train_future_goal_ratio": last_future_ratio,
                        "train_replay_goal_ratio": last_replay_ratio,
                        "train_actor_goal_ratio": last_actor_goal_ratio,
                    }
                    row.update(metrics)
                    csv_logger.write(row)

        if done or episode_timesteps >= args.max_episode_length:
            replay_buffer.add_path(path_builder.get_all_stacked())
            path_builder = PathBuilder()
            writer.add_scalar("Train/episode_final_distance", last_episode_distance, t + 1)
            writer.add_scalar("Train/replay_size", replay_buffer.num_steps_can_sample(), t + 1)
            obs = env.reset()
            state = obs["observation"]
            goal = obs["desired_goal"]
            episode_timesteps = 0
            episode_num += 1

        if (t + 1) % args.eval_freq == 0 and t >= args.start_timesteps:
            now = time.time()
            interval = now - last_eval_time
            sps = float((t + 1) - last_eval_step) / max(interval, 1e-8)
            last_eval_time = now
            last_eval_step = t + 1
            replay_size = replay_buffer.num_steps_can_sample()
            print(f"[Eval {t + 1}] sps={sps:.2f} replay={replay_size} updates={policy.total_it}")

            train_d, train_s = eval_policy(policy, train_eval_env, args.n_eval, args.max_episode_length, args.distance_threshold, args.eval_deterministic)
            test_d, test_s = eval_policy(policy, test_eval_env, args.n_eval_test, args.max_episode_length, args.distance_threshold, args.eval_deterministic)
            writer.add_scalar("Eval/train_success", train_s, t + 1)
            writer.add_scalar("Eval/train_distance", train_d, t + 1)
            writer.add_scalar("Eval/test_success", test_s, t + 1)
            writer.add_scalar("Eval/test_distance", test_d, t + 1)
            writer.add_scalar("Train/steps_per_second", sps, t + 1)
            print(f"QRC | train_success={train_s:.3f} test_success={test_s:.3f} train_d={train_d:.3f} test_d={test_d:.3f}")

            csv_logger.write({
                "row_type": "eval",
                "wall_time": time.time(),
                "elapsed_min": (time.time() - train_start_time) / 60.0,
                "env_step": t + 1,
                "update": policy.total_it,
                "episode": episode_num,
                "episode_step": episode_timesteps,
                "exp_name": args.exp_name,
                "env_name": args.env_name,
                "seed": args.seed,
                "replay_size": replay_size,
                "steps_per_second": sps,
                "eval_train_success": train_s,
                "eval_train_distance": train_d,
                "eval_test_success": test_s,
                "eval_test_distance": test_d,
                "train_episode_distance": last_episode_distance,
                "train_orig_goal_ratio": last_orig_ratio,
                "train_future_goal_ratio": last_future_ratio,
                "train_replay_goal_ratio": last_replay_ratio,
                "train_actor_goal_ratio": last_actor_goal_ratio,
                "calib_beta": policy.beta,
            })
            policy.save(folder)

        if args.save_freq > 0 and (t + 1) % args.save_freq == 0 and t >= args.start_timesteps:
            policy.save(folder)

    policy.save(folder)
    writer.close()
    csv_logger.close()


if __name__ == "__main__":
    main()
