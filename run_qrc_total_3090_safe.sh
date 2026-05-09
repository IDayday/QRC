#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# QRC total launcher for multi-RTX-3090 servers.
# 默认不会同时塞满所有任务；每张卡按队列限流，并用 nvidia-smi 空闲显存做保护。
#
# 推荐先跑：run_qrc_phase0_direct_sanity.sh
# 再跑：run_qrc_phase1_random_closure.sh
# 最后根据诊断选择是否跑本大矩阵。
# ============================================================

PROJECT_ROOT="${PROJECT_ROOT:-/root/remote/project/GCRL}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_ant_qrc.py}"
GPUS_STR="${GPUS_STR:-0 1 2 3 4}"
read -r -a GPUS <<< "${GPUS_STR}"
NUM_GPUS="${#GPUS[@]}"
MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-4}"
MIN_FREE_MEM_PER_JOB_MB="${MIN_FREE_MEM_PER_JOB_MB:-4500}"
SLOT_CHECK_SLEEP_SEC="${SLOT_CHECK_SLEEP_SEC:-20}"
LAUNCH_SLEEP_SEC="${LAUNCH_SLEEP_SEC:-1}"
DRY_RUN="${DRY_RUN:-0}"

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
ulimit -n 65535 || true

RUN_TAG="${RUN_TAG:-qrc_total_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs_qrc_total/${RUN_TAG}}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/results_qrc_total}"
mkdir -p "${LOG_ROOT}"

SEEDS_STR="${SEEDS:-52 58 66}"
read -r -a SEEDS_ARR <<< "${SEEDS_STR}"
ENV_EASY="${ENV_EASY:-AntU}"
ENV_MED="${ENV_MED:-AntFb}"
ENV_HARD="${ENV_HARD:-AntMaze}"

COMMON_ARGS=(
  --distance_threshold "${DISTANCE_THRESHOLD:-0.5}"
  --start_timesteps "${START_TIMESTEPS:-10000}"
  --eval_freq "${EVAL_FREQ:-5000}"
  --max_timesteps "${MAX_TIMESTEPS:-1000000}"
  --max_episode_length "${MAX_EPISODE_LENGTH:-600}"
  --batch_size "${BATCH_SIZE:-1024}"
  --replay_buffer_size "${REPLAY_BUFFER_SIZE:-1000000}"
  --n_eval "${N_EVAL:-50}"
  --n_eval_test "${N_EVAL_TEST:-10}"
  --device cuda
  --log_root "${RESULT_ROOT}"
  --gamma "${GAMMA:-0.98}"
  --tau "${TAU:-0.005}"
  --n_heads "${N_HEADS:-4}"
  --actor_lr "${ACTOR_LR:-3e-4}"
  --critic_lr "${CRITIC_LR:-3e-4}"
  --proposal_lr "${PROPOSAL_LR:-1e-4}"
  --hidden_dim "${HIDDEN_DIM:-256}"
  --policy_delay "${POLICY_DELAY:-2}"
  --p_orig "${P_ORIG:-0.25}"
  --h_relab "${H_RELAB:-32}"
  --lambda_dir "${LAMBDA_DIR:-1.0}"
  --lambda_clo "${LAMBDA_CLO:-0.05}"
  --closure_batch_size "${CLOSURE_BATCH_SIZE:-128}"
  --direct_batch_size "${DIRECT_BATCH_SIZE:-256}"
  --projection_pool_size "${PROJECTION_POOL_SIZE:-64}"
  --closure_start_updates "${CLOSURE_START_UPDATES:-5000}"
  --anti_degenerate_min_dist "${ANTI_DEGENERATE_MIN_DIST:-0.25}"
  --calib_freq "${CALIB_FREQ:-5000}"
  --calib_quantile "${CALIB_QUANTILE:-0.90}"
  --calib_ema "${CALIB_EMA:-0.10}"
  --csv_train_log_freq "${CSV_TRAIN_LOG_FREQ:-100}"
)

if [[ ! -f "${PROJECT_ROOT}/${TRAIN_SCRIPT}" ]]; then
  echo "[ERROR] Missing ${PROJECT_ROOT}/${TRAIN_SCRIPT}. Copy QRC.py and train_ant_qrc.py into PROJECT_ROOT first." >&2
  exit 1
fi
if [[ ! -f "${PROJECT_ROOT}/QRC.py" ]]; then
  echo "[ERROR] Missing ${PROJECT_ROOT}/QRC.py" >&2
  exit 1
fi

MANIFEST="${LOG_ROOT}/manifest.csv"
PID_FILE="${LOG_ROOT}/pids.txt"
: > "${PID_FILE}"
echo "run_tag,exp_id,purpose,env,seed,gpu,log_file,args" > "${MANIFEST}"

declare -A GPU_PIDS
ALL_PIDS=()
JOB_INDEX=0
FAIL_COUNT=0

prune_gpu_pids() {
  local gpu="$1"; local new=""; local pid
  for pid in ${GPU_PIDS[${gpu}]:-}; do
    if kill -0 "${pid}" 2>/dev/null; then new+="${pid} "; fi
  done
  GPU_PIDS[${gpu}]="${new}"
}
count_gpu_pids() { local gpu="$1"; prune_gpu_pids "${gpu}"; local arr=( ${GPU_PIDS[${gpu}]:-} ); echo "${#arr[@]}"; }
gpu_free_mem_mb() {
  local gpu="$1"
  if ! command -v nvidia-smi >/dev/null 2>&1; then echo 999999; return; fi
  local m; m="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu}" 2>/dev/null | head -n 1 | tr -d ' ')"
  [[ -z "${m}" ]] && echo 999999 || echo "${m}"
}
wait_for_gpu_slot() {
  local gpu="$1"; local n mem
  while true; do
    n="$(count_gpu_pids "${gpu}")"; mem="$(gpu_free_mem_mb "${gpu}")"
    if [[ "${n}" -lt "${MAX_JOBS_PER_GPU}" && ( "${MIN_FREE_MEM_PER_JOB_MB}" -le 0 || "${mem}" -ge "${MIN_FREE_MEM_PER_JOB_MB}" ) ]]; then
      break
    fi
    echo "[THROTTLE] gpu=${gpu} jobs=${n}/${MAX_JOBS_PER_GPU} free_mem=${mem}MB"
    sleep "${SLOT_CHECK_SLEEP_SEC}"
  done
}

run_job() {
  local gpu="$1"; local exp_id="$2"; local purpose="$3"; local env_name="$4"; local seed="$5"; shift 5
  local extra=("$@")
  local exp_name="${RUN_TAG}_${exp_id}"
  local log_file="${LOG_ROOT}/${exp_id}_${env_name}_seed${seed}_gpu${gpu}.log"
  local cmd=("${PYTHON_BIN}" "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}" --env_name "${env_name}" --seed "${seed}" --exp_name "${exp_name}" "${extra[@]}")
  echo "${RUN_TAG},${exp_id},${purpose//,/;},${env_name},${seed},${gpu},${log_file},${cmd[*]}" >> "${MANIFEST}"
  if [[ "${DRY_RUN}" == "1" ]]; then echo "[DRY] ${cmd[*]}"; return; fi
  wait_for_gpu_slot "${gpu}"
  echo "[START] gpu=${gpu} exp=${exp_id} env=${env_name} seed=${seed} log=${log_file}"
  (
    cd "${PROJECT_ROOT}"
    export CUDA_VISIBLE_DEVICES="${gpu}"
    "${cmd[@]}"
  ) > "${log_file}" 2>&1 &
  local pid="$!"
  GPU_PIDS[${gpu}]="${GPU_PIDS[${gpu}]:-}${pid} "
  ALL_PIDS+=("${pid}")
  echo "${pid} ${gpu} ${exp_id} ${env_name} ${seed} ${log_file}" >> "${PID_FILE}"
  sleep "${LAUNCH_SLEEP_SEC}"
}

launch_exp() {
  local exp_id="$1"; local purpose="$2"; local env_name="$3"; shift 3; local extra=("$@")
  local seed gpu
  for seed in "${SEEDS_ARR[@]}"; do
    gpu="${GPUS[$((JOB_INDEX % NUM_GPUS))]}"; JOB_INDEX=$((JOB_INDEX + 1))
    run_job "${gpu}" "${exp_id}" "${purpose}" "${env_name}" "${seed}" "${extra[@]}"
  done
}

# P0: 表征 sanity。
launch_exp "P0_AntU_sigmoid_direct" "QRC-Z sigmoid direct-only sanity" "${ENV_EASY}" \
  --parameterization sigmoid_z --closure_source none --lambda_clo 0.0 --no-use_stitch
launch_exp "P0_AntU_distance_direct" "QRC-D distance direct-only sanity" "${ENV_EASY}" \
  --parameterization distance --closure_source none --lambda_clo 0.0 --no-use_stitch

# P1: random sampled closure。
launch_exp "P1_AntU_randomM1" "Random replay closure M=1" "${ENV_EASY}" \
  --parameterization sigmoid_z --closure_source random --closure_candidates 1 --no-use_stitch
launch_exp "P1_AntU_randomM8" "Random replay closure M=8" "${ENV_EASY}" \
  --parameterization sigmoid_z --closure_source random --closure_candidates 8 --no-use_stitch
launch_exp "P1_AntU_randomM16" "Random replay closure M=16" "${ENV_EASY}" \
  --parameterization sigmoid_z --closure_source random --closure_candidates 16 --no-use_stitch

# P2: proposal-witness separation。
launch_exp "P2_AntU_rawPlannerDiag" "Raw planner witness diagnostic; expected weak/unstable" "${ENV_EASY}" \
  --parameterization sigmoid_z --closure_source planner_raw --allow_raw_planner_witness --closure_candidates 8 --no-use_stitch
launch_exp "P2_AntU_projected" "Planner proposal projected to replay manifold" "${ENV_EASY}" \
  --parameterization sigmoid_z --closure_source planner_projected --closure_candidates 8 --no-use_stitch
launch_exp "P2_AntU_mixed" "Mixed random/projected closure" "${ENV_EASY}" \
  --parameterization sigmoid_z --closure_source mixed --closure_candidates 8 --no-use_stitch

# P3: medium/hard topology and stitching。
launch_exp "P3_Med_direct" "AntFb direct-only baseline" "${ENV_MED}" \
  --parameterization sigmoid_z --closure_source none --lambda_clo 0.0 --no-use_stitch
launch_exp "P3_Med_mixed" "AntFb mixed closure" "${ENV_MED}" \
  --parameterization sigmoid_z --closure_source mixed --closure_candidates 8 --no-use_stitch
launch_exp "P3_Med_mixedStitch" "AntFb mixed closure plus stitched evidence" "${ENV_MED}" \
  --parameterization sigmoid_z --closure_source mixed --closure_candidates 8 --use_stitch --lambda_stitch "${LAMBDA_STITCH:-0.05}"
launch_exp "P3_Hard_direct" "Hard topology direct-only baseline" "${ENV_HARD}" \
  --parameterization sigmoid_z --closure_source none --lambda_clo 0.0 --no-use_stitch
launch_exp "P3_Hard_mixedStitch" "Hard topology mixed closure plus stitched evidence" "${ENV_HARD}" \
  --parameterization sigmoid_z --closure_source mixed --closure_candidates 16 --use_stitch --lambda_stitch "${LAMBDA_STITCH:-0.05}"

# P4: proposal ablation。
launch_exp "P4_AntU_randomM8_noProposal" "Random closure without proposal training" "${ENV_EASY}" \
  --parameterization sigmoid_z --closure_source random --closure_candidates 8 --no-use_proposal --no-use_stitch
launch_exp "P4_AntU_projected_noProposal" "Projected source with untrained proposal diagnostic" "${ENV_EASY}" \
  --parameterization sigmoid_z --closure_source planner_projected --closure_candidates 8 --no-use_proposal --no-use_stitch

echo
echo "Launched job count: ${JOB_INDEX}"
echo "Run tag: ${RUN_TAG}"
echo "Logs: ${LOG_ROOT}"
echo "Manifest: ${MANIFEST}"
echo "PIDs: ${PID_FILE}"
echo

if [[ "${DRY_RUN}" == "1" ]]; then exit 0; fi
set +e
for pid in "${ALL_PIDS[@]}"; do
  wait "${pid}"; rc="$?"
  if [[ "${rc}" -ne 0 ]]; then FAIL_COUNT=$((FAIL_COUNT + 1)); echo "[FAIL] pid=${pid} rc=${rc}"; fi
done
set -e

echo "All QRC jobs finished. Failures: ${FAIL_COUNT}"
[[ "${FAIL_COUNT}" -eq 0 ]]
