#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# QRC Phase-1: random replay closure attribution
# 目的：验证 random replay bridge 是否可解释为 sampled closure estimator。
# 问题：M=1/8/16 是否随 witness hit probability 提升而改善？
# 记录重点：QRC/closure_accept_rate、best_Z_cert、nondegenerate_rate、eval success。
# ============================================================

PROJECT_ROOT="${PROJECT_ROOT:-/root/remote/project/GCRL}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_ant_qrc.py}"
ENV_NAME="${ENV_NAME:-AntU}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs_qrc_phase1}"
mkdir -p "${LOG_ROOT}"

MAX_TIMESTEPS="${MAX_TIMESTEPS:-500000}"
START_TIMESTEPS="${START_TIMESTEPS:-10000}"
EVAL_FREQ="${EVAL_FREQ:-5000}"
N_EVAL="${N_EVAL:-50}"
N_EVAL_TEST="${N_EVAL_TEST:-10}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
REPLAY_BUFFER_SIZE="${REPLAY_BUFFER_SIZE:-1000000}"
MAX_EPISODE_LENGTH="${MAX_EPISODE_LENGTH:-600}"
DISTANCE_THRESHOLD="${DISTANCE_THRESHOLD:-0.5}"
SEEDS_STR="${SEEDS:-52 58 66}"
read -r -a SEEDS_ARR <<< "${SEEDS_STR}"
GPU_G0="${GPU_G0:-0}"; GPU_G1="${GPU_G1:-1}"; GPU_G2="${GPU_G2:-2}"; GPU_G3="${GPU_G3:-3}"

COMMON_ARGS=(
  --env_name "${ENV_NAME}"
  --distance_threshold "${DISTANCE_THRESHOLD}"
  --start_timesteps "${START_TIMESTEPS}"
  --eval_freq "${EVAL_FREQ}"
  --max_timesteps "${MAX_TIMESTEPS}"
  --max_episode_length "${MAX_EPISODE_LENGTH}"
  --batch_size "${BATCH_SIZE}"
  --replay_buffer_size "${REPLAY_BUFFER_SIZE}"
  --n_eval "${N_EVAL}"
  --n_eval_test "${N_EVAL_TEST}"
  --gamma "${GAMMA:-0.98}"
  --tau "${TAU:-0.005}"
  --n_heads "${N_HEADS:-4}"
  --parameterization "${PARAMETERIZATION:-sigmoid_z}"
  --actor_lr "${ACTOR_LR:-3e-4}"
  --critic_lr "${CRITIC_LR:-3e-4}"
  --p_orig "${P_ORIG:-0.25}"
  --h_relab "${H_RELAB:-32}"
  --lambda_dir "${LAMBDA_DIR:-1.0}"
  --lambda_clo "${LAMBDA_CLO:-0.05}"
  --closure_batch_size "${CLOSURE_BATCH_SIZE:-128}"
  --closure_start_updates "${CLOSURE_START_UPDATES:-5000}"
  --anti_degenerate_min_dist "${ANTI_DEGENERATE_MIN_DIST:-0.25}"
  --calib_freq "${CALIB_FREQ:-5000}"
  --no-use_stitch
  --log_root "${PROJECT_ROOT}/results_qrc"
)

run_group () {
  local gpu="$1"; local exp_name="$2"; shift 2; local extra=("$@")
  (
    cd "${PROJECT_ROOT}"
    for seed in "${SEEDS_ARR[@]}"; do
      log_file="${LOG_ROOT}/${exp_name}_${ENV_NAME}_seed${seed}.log"
      echo "[START] gpu=${gpu} exp=${exp_name} seed=${seed} log=${log_file}"
      CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
        "${COMMON_ARGS[@]}" --seed "${seed}" --exp_name "${exp_name}" "${extra[@]}" \
        > "${log_file}" 2>&1
      echo "[DONE] gpu=${gpu} exp=${exp_name} seed=${seed}"
    done
  ) &
}

run_group "${GPU_G0}" "p1_a0_direct_only" --closure_source none --lambda_clo 0.0 --closure_candidates 1
run_group "${GPU_G1}" "p1_a1_random_m1" --closure_source random --closure_candidates 1
run_group "${GPU_G2}" "p1_a2_random_m8" --closure_source random --closure_candidates 8
run_group "${GPU_G3}" "p1_a3_random_m16" --closure_source random --closure_candidates 16

echo "QRC Phase-1 launched. Logs: ${LOG_ROOT}"
wait
echo "QRC Phase-1 finished."
