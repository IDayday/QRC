#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" train_ant_qrc.py \
  --env_name "${ENV_NAME:-AntU}" \
  --max_timesteps "${MAX_TIMESTEPS:-20000}" \
  --eval_freq "${EVAL_FREQ:-10000}" \
  --start_timesteps "${START_TIMESTEPS:-1000}" \
  --batch_size "${BATCH_SIZE:-256}" \
  --n_eval 2 --n_eval_test 1 \
  --log_root "${RESULT_ROOT:-${PROJECT_ROOT}/results_qrc_research}" \
  --exp_name "${EXP_NAME:-smoke_qrc_v3}" \
  --seed "${SEED:-52}" \
  --p_orig 0.25 --p_future 0.50 --p_replay_goal 0.25 \
  --lambda_dir 1.0 --lambda_clo 0.0 --closure_source none \
  --td_closure_mode none --closure_loss_target lcb \
  --csv_train_log_freq 101 --torch_num_threads 1
