#!/usr/bin/env bash
# Shared launcher helpers for QRC decision experiments.
# Source this file from phase scripts; do not run it directly.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_ant_qrc.py}"
ENV_NAME="${ENV_NAME:-AntU}"
RUN_TAG="${RUN_TAG:-qrc_research_$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/results_qrc_research}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs_qrc_research/${RUN_TAG}}"
mkdir -p "${LOG_ROOT}"

GPUS_STR="${GPUS_STR:-0 1 2 3}"
read -r -a GPUS <<< "${GPUS_STR}"
NUM_GPUS="${#GPUS[@]}"
MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-2}"
MIN_FREE_MEM_PER_JOB_MB="${MIN_FREE_MEM_PER_JOB_MB:-6000}"
LAUNCH_SLEEP_SEC="${LAUNCH_SLEEP_SEC:-1}"
SLOT_CHECK_SLEEP_SEC="${SLOT_CHECK_SLEEP_SEC:-10}"
DRY_RUN="${DRY_RUN:-0}"

SEEDS_STR="${SEEDS_STR:-52 58 66}"
read -r -a SEEDS <<< "${SEEDS_STR}"

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
ulimit -n 65535 || true

if [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "[ERROR] GPUS_STR is empty." >&2
  exit 1
fi
if [[ ! -f "${PROJECT_ROOT}/${TRAIN_SCRIPT}" ]]; then
  echo "[ERROR] Missing train script: ${PROJECT_ROOT}/${TRAIN_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${PROJECT_ROOT}/QRC.py" ]]; then
  echo "[ERROR] Missing ${PROJECT_ROOT}/QRC.py" >&2
  exit 1
fi

MANIFEST="${LOG_ROOT}/manifest.csv"
PID_FILE="${LOG_ROOT}/pids.txt"
: > "${PID_FILE}"
echo "run_tag,phase,exp_id,env,seed,gpu,log_file,args" > "${MANIFEST}"

declare -A GPU_PIDS
ALL_PIDS=()
JOB_INDEX=0
FAIL_COUNT=0

# Common defaults; group-specific arguments appended later can override these.
COMMON_ARGS=(
  --env_name "${ENV_NAME}"
  --distance_threshold "${DISTANCE_THRESHOLD:-0.5}"
  --start_timesteps "${START_TIMESTEPS:-10000}"
  --eval_freq "${EVAL_FREQ:-5000}"
  --max_timesteps "${MAX_TIMESTEPS:-300000}"
  --max_episode_length "${MAX_EPISODE_LENGTH:-600}"
  --batch_size "${BATCH_SIZE:-1024}"
  --updates_per_step "${UPDATES_PER_STEP:-1}"
  --replay_buffer_size "${REPLAY_BUFFER_SIZE:-1000000}"
  --episode_slot_multiplier "${EPISODE_SLOT_MULTIPLIER:-4.0}"
  --n_eval "${N_EVAL:-20}"
  --n_eval_test "${N_EVAL_TEST:-5}"
  --device cuda
  --log_root "${RESULT_ROOT}"
  --save_freq "${SAVE_FREQ:-50000}"
  --csv_train_log_freq "${CSV_TRAIN_LOG_FREQ:-101}"
  --gamma "${GAMMA:-0.98}"
  --tau "${TAU:-0.005}"
  --actor_lr "${ACTOR_LR:-3e-4}"
  --critic_lr "${CRITIC_LR:-3e-4}"
  --proposal_lr "${PROPOSAL_LR:-1e-4}"
  --hidden_dim "${HIDDEN_DIM:-256}"
  --n_heads "${N_HEADS:-4}"
  --parameterization "${PARAMETERIZATION:-sigmoid_z}"
  --init_z "${INIT_Z:-0.05}"
  --policy_delay "${POLICY_DELAY:-2}"
  --actor_agg "${ACTOR_AGG:-mean}"
  --action_l2 "${ACTION_L2:-1e-4}"
  --exploration_noise "${EXPLORATION_NOISE:-0.10}"
  --torch_num_threads "${TORCH_NUM_THREADS_PER_JOB:-1}"
  --p_orig "${P_ORIG:-0.25}"
  --p_future "${P_FUTURE:-0.50}"
  --p_replay_goal "${P_REPLAY_GOAL:-0.25}"
  --h_relab "${H_RELAB:-32}"
  --lambda_dir "${LAMBDA_DIR:-1.0}"
  --lambda_clo "${LAMBDA_CLO:-0.0}"
  --lambda_stitch "${LAMBDA_STITCH:-0.0}"
  --lambda_proposal "${LAMBDA_PROPOSAL:-0.0}"
  --direct_batch_size "${DIRECT_BATCH_SIZE:-256}"
  --closure_batch_size "${CLOSURE_BATCH_SIZE:-128}"
  --stitch_batch_size "${STITCH_BATCH_SIZE:-128}"
  --closure_candidates "${CLOSURE_CANDIDATES:-8}"
  --closure_source "${CLOSURE_SOURCE:-none}"
  --projection_pool_size "${PROJECTION_POOL_SIZE:-64}"
  --closure_start_updates "${CLOSURE_START_UPDATES:-20000}"
  --closure_margin_z "${CLOSURE_MARGIN_Z:-0.0}"
  --closure_loss_target "${CLOSURE_LOSS_TARGET:-lcb}"
  --td_closure_mode "${TD_CLOSURE_MODE:-none}"
  --td_closure_start_updates "${TD_CLOSURE_START_UPDATES:-0}"
  --beta_init "${BETA_INIT:-2.0}"
  --beta_mode "${BETA_MODE:-fixed}"
  --cert_sigma_floor "${CERT_SIGMA_FLOOR:-0.01}"
  --calib_freq "${CALIB_FREQ:-5000}"
  --calib_size "${CALIB_SIZE:-512}"
  --calib_quantile "${CALIB_QUANTILE:-0.90}"
  --calib_ema "${CALIB_EMA:-0.10}"
  --anti_degenerate_min_dist "${ANTI_DEGENERATE_MIN_DIST:-0.25}"
  --distance_mode "${DISTANCE_MODE:-xy}"
  --no-use_stitch
  --eval_deterministic
  --deterministic_actor_update
)

prune_gpu_pids() {
  local gpu="$1"
  local new_list=""
  local pid
  for pid in ${GPU_PIDS[${gpu}]:-}; do
    if kill -0 "${pid}" 2>/dev/null; then
      new_list+="${pid} "
    fi
  done
  GPU_PIDS[${gpu}]="${new_list}"
}

count_gpu_pids() {
  local gpu="$1"
  prune_gpu_pids "${gpu}"
  local tmp=( ${GPU_PIDS[${gpu}]:-} )
  echo "${#tmp[@]}"
}

gpu_free_mem_mb() {
  local gpu="$1"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo 999999
    return 0
  fi
  local free_mem
  free_mem="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu}" 2>/dev/null | head -n 1 | tr -d ' ')"
  if [[ -z "${free_mem}" ]]; then echo 999999; else echo "${free_mem}"; fi
}

wait_for_gpu_slot() {
  local gpu="$1"
  local n free_mem
  while true; do
    n="$(count_gpu_pids "${gpu}")"
    free_mem="$(gpu_free_mem_mb "${gpu}")"
    if [[ "${n}" -lt "${MAX_JOBS_PER_GPU}" ]]; then
      if [[ "${MIN_FREE_MEM_PER_JOB_MB}" -le 0 || "${free_mem}" -ge "${MIN_FREE_MEM_PER_JOB_MB}" ]]; then
        break
      fi
    fi
    echo "[THROTTLE] gpu=${gpu} jobs=${n}/${MAX_JOBS_PER_GPU} free_mem=${free_mem}MB; waiting..."
    sleep "${SLOT_CHECK_SLEEP_SEC}"
  done
}

run_job() {
  local gpu="$1"; local phase="$2"; local exp_id="$3"; local seed="$4"
  shift 4
  local extra_args=("$@")
  local exp_name="${RUN_TAG}_${phase}_${exp_id}"
  local log_file="${LOG_ROOT}/${phase}_${exp_id}_${ENV_NAME}_seed${seed}_gpu${gpu}.log"
  local cmd=("${PYTHON_BIN}" "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}" --seed "${seed}" --exp_name "${exp_name}" "${extra_args[@]}")
  echo "${RUN_TAG},${phase},${exp_id},${ENV_NAME},${seed},${gpu},${log_file},${cmd[*]}" >> "${MANIFEST}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY] gpu=${gpu} phase=${phase} exp=${exp_id} seed=${seed}"
    echo "      ${cmd[*]}"
    return 0
  fi
  wait_for_gpu_slot "${gpu}"
  echo "[START] gpu=${gpu} phase=${phase} exp=${exp_id} seed=${seed} log=${log_file}"
  (
    cd "${PROJECT_ROOT}"
    export CUDA_VISIBLE_DEVICES="${gpu}"
    "${cmd[@]}"
  ) > "${log_file}" 2>&1 &
  local pid="$!"
  GPU_PIDS[${gpu}]="${GPU_PIDS[${gpu}]:-}${pid} "
  ALL_PIDS+=("${pid}")
  echo "${pid} ${gpu} ${phase} ${exp_id} ${seed} ${log_file}" >> "${PID_FILE}"
  sleep "${LAUNCH_SLEEP_SEC}"
}

launch_group() {
  local phase="$1"; local exp_id="$2"
  shift 2
  local extra_args=("$@")
  local seed gpu
  for seed in "${SEEDS[@]}"; do
    gpu="${GPUS[$((JOB_INDEX % NUM_GPUS))]}"
    JOB_INDEX=$((JOB_INDEX + 1))
    run_job "${gpu}" "${phase}" "${exp_id}" "${seed}" "${extra_args[@]}"
  done
}

qrc_wait_all() {
  echo
  echo "Launched jobs: ${JOB_INDEX}"
  echo "Run tag: ${RUN_TAG}"
  echo "Results: ${RESULT_ROOT}"
  echo "Logs: ${LOG_ROOT}"
  echo "Manifest: ${MANIFEST}"
  echo "PIDs: ${PID_FILE}"
  echo
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN=1, no jobs launched."
    return 0
  fi
  set +e
  local pid rc
  for pid in "${ALL_PIDS[@]}"; do
    wait "${pid}"
    rc="$?"
    if [[ "${rc}" -ne 0 ]]; then
      FAIL_COUNT=$((FAIL_COUNT + 1))
      echo "[FAIL] pid=${pid} rc=${rc}"
    fi
  done
  set -e
  echo "All launched jobs finished. Failures: ${FAIL_COUNT}"
  if [[ "${FAIL_COUNT}" -ne 0 ]]; then exit 1; fi
}
