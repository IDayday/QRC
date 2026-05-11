# QRC / SCOPE-HER 决策型实验计划 v3

## 目标

本轮不再做大矩阵堆模块，而是用最少实验回答三个关键问题：

1. 单一 bounded reachability critic 是否能在 balanced goal mix 下学出可控的 Z landscape？
2. 安全 one-sided closure 是否能产生非零且合理的训练信号？
3. 如果只有 recursive closure 有收益，是否说明在线安全 closure 的核心瓶颈是“证据强度不足 / 注入强度不足”？

这套实验用于快速决定：QRC 继续作为主线，还是降级为 SCOPE-HER/EBFB 的诊断分支。

## 已实现的关键修改

- `actor_loss` / `actor_z_action_mean` 记录 last 与 EMA，避免 `policy_delay` 与日志频率采样别名。
- control batch 支持 `p_orig / p_future / p_replay_goal` 三路目标混合。replay/random goal 只进 TD-Z，不进 actor。
- proposal 网络只在 projected / mixed / raw planner 诊断中训练。
- `beta_mode=fixed|diagnostic|dynamic`，避免早期动态 beta 把 certificate 压死。
- 新增 `closure_loss_target=lcb|raw`：
  - `lcb` 是安全主方法；
  - `raw` 是非递归诊断，用于判断 LCB 是否过保守。
- 新增 `td_closure_mode=none|recursive_raw|recursive_lcb`：
  - `none` 是 QRC 主方法；
  - `recursive_*` 是不安全诊断，用来判断收益是否依赖 TD backbone 污染。

## Phase A：balanced direct-only sanity

运行：

```bash
GPUS_STR="0 1 2 3" SEEDS_STR="52 58 66" \
nohup ./run_qrc_phaseA_direct_screen.sh > qrc_A.log 2>&1 &
```

判据：

- `abs(z_direct_gap) < 0.05~0.08`
- `z_replay_goal_mean` 不长期大于 `0.5`
- `actor_z_action_mean_ema` 真实非零并上升
- train distance 或 success 有改善趋势

如果 Phase A 失败，不要跑 closure；先修 actor 或 critic goal mix。

## Phase B：closure activation

运行：

```bash
GPUS_STR="0 1 2 3" SEEDS_STR="52 58 66" \
nohup ./run_qrc_phaseB_closure_activation.sh > qrc_B.log 2>&1 &
```

判据：

- `qrc_raw_accept_rate > 0`
- `qrc_closure_accept_rate` 最好在 `0.01~0.10`
- `qrc_closure_uplift` 非零但不要很大，通常 `<0.05`
- 若 raw 有 accept 而 lcb 没有，说明 LCB/beta 过保守。
- 若 raw 也没有 accept，说明 critic 仍过乐观或 random candidate 命中率太低。

## Phase C：recursive-vs-safe 关键安全对照

运行：

```bash
GPUS_STR="0 1 2 3" SEEDS_STR="52 58 66" \
nohup ./run_qrc_phaseC_recursive_safety.sh > qrc_C.log 2>&1 &
```

解释：

- 如果 `recursive_*` 明显有效，而 `one-sided_*` 无效：收益可能依赖高风险 recursive propagation；QRC 安全版本需要新的 evidence-strengthening 机制。
- 如果 `one-sided_lcb/raw` 也有效：QRC 主线保留。
- 如果二者都无效：closure candidate / critic calibration / actor coupling 仍有根本问题。

## Phase D：proposal-witness separation

运行：

```bash
GPUS_STR="0 1 2 3" SEEDS_STR="52 58 66" \
nohup ./run_qrc_phaseD_proposal_witness.sh > qrc_D.log 2>&1 &
```

只在 Phase B/C 有信号后运行。raw planner 只作为失败诊断，不作为主方法。

## Phase E：AntFb confirmation

运行：

```bash
GPUS_STR="0 1 2 3" SEEDS_STR="52 58 66" \
BEST_TARGET=lcb BEST_BETA=2.0 BEST_M=8 \
nohup ./run_qrc_phaseE_antfb_confirm.sh > qrc_E.log 2>&1 &
```

只在 AntU 上出现清晰机制信号后运行。

## 快速汇总

```bash
python analyze_qrc_results.py \
  --root results_qrc_research \
  --out qrc_summary_latest
```

重点文件：

- `qrc_summary_latest/run_train_tail_mean.csv`
- `qrc_summary_latest/group_train_tail_mean.csv`
- `qrc_summary_latest/run_eval_last_and_best.csv`
- `qrc_summary_latest/group_eval_summary.csv`
- `qrc_summary_latest/decision_flags_by_run.csv`

## Go / no-go 标准

继续 QRC 主线的条件：

- Phase A direct-only 稳定；
- Phase B safe closure 有非零 accept/uplift，且不破坏 direct critic；
- Phase C 中 safe one-sided 至少接近 recursive diagnostic 的一部分收益；
- Phase D 中 projected/mixed 至少不弱于 random，raw planner 诊断不成为主方法证据。

降级 QRC 的条件：

- direct critic 学稳但 actor 不动；
- safe closure 长期 accept≈0/uplift≈0；
- 只有 recursive closure 有收益；
- AntFb/Test generalization 持续为 0。

若降级，下一步转向 SCOPE-HER 的 SGR/SSH：同轨迹 supported propagation 进入主 backbone，跨轨迹只保留 CBI 诊断。
