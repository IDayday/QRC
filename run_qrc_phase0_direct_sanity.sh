#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# QRC Phase-0: direct-only sanity check
# 目的：首先验证单一 bounded reachability critic 是否稳定。
# 问题：TD-Z + direct future-pair evidence 能否在不启用 closure 的情况下稳定学习？
# 记录重点：Z/direct_pred vs target、Z saturation、D mean、eval success、SPS。
# ============================================================

PROJECT_ROOT="${PROJECT_ROOT:-/root/remote/project/GCRL}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_ant_qrc.py}"
ENV_NAME="${ENV_NAME:-AntU}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs_qrc_phase0}"
mkdir -p "${LOG_ROOT}"

MAX_TIMESTEPS="${MAX_TIMESTEPS:-200000}"
START_TIMESTEPS="${START_TIMESTEPS:-10000}"
EVAL_FREQ="${EVAL_FREQ:-5000}"
N_EVAL="${N_EVAL:-20}"
N_EVAL_TEST="${N_EVAL_TEST:-5}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
REPLAY_BUFFER_SIZE="${REPLAY_BUFFER_SIZE:-1000000}"
MAX_EPISODE_LENGTH="${MAX_EPISODE_LENGTH:-600}"
DISTANCE_THRESHOLD="${DISTANCE_THRESHOLD:-0.5}"

GPU_G0="${GPU_G0:-0}"   # sigmoid_z direct only
GPU_G1="${GPU_G1:-1}"   # distance direct only
SEEDS_STR="${SEEDS:-52 58 66}"
read -r -a SEEDS_ARR <<< "${SEEDS_STR}"

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
  --actor_lr "${ACTOR_LR:-3e-4}"
  --critic_lr "${CRITIC_LR:-3e-4}"
  --proposal_lr "${PROPOSAL_LR:-1e-4}"
  --p_orig "${P_ORIG:-0.25}"
  --h_relab "${H_RELAB:-32}"
  --lambda_dir "${LAMBDA_DIR:-1.0}"
  --lambda_clo 0.0
  --closure_source none
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

run_group "${GPU_G0}" "p0_qrc_sigmoid_direct" --parameterization sigmoid_z --init_z "${INIT_Z:-0.05}"
run_group "${GPU_G1}" "p0_qrc_distance_direct" --parameterization distance --init_z "${INIT_Z:-0.05}"

echo "QRC Phase-0 launched. Logs: ${LOG_ROOT}"
wait
echo "QRC Phase-0 finished."
