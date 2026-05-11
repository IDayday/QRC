#!/usr/bin/env bash
set -euo pipefail
# Phase A: balanced direct-only sanity. Do not run closure here.
# Goal: determine whether single bounded reachability critic + actor is viable.
ENV_NAME="${ENV_NAME:-AntU}"
MAX_TIMESTEPS="${MAX_TIMESTEPS:-300000}"
EVAL_FREQ="${EVAL_FREQ:-5000}"
LAMBDA_CLO="0.0"
CLOSURE_SOURCE="none"
TD_CLOSURE_MODE="none"
CLOSURE_LOSS_TARGET="lcb"
source "${PROJECT_ROOT:-$(pwd)}/qrc_launcher_utils.sh"

launch_group "A" "A0_sigmoid_mean_lr3e4" \
  --parameterization sigmoid_z --actor_lr 3e-4 --actor_agg mean --lambda_clo 0.0 --closure_source none --td_closure_mode none --lambda_proposal 0.0

launch_group "A" "A1_sigmoid_min_lr1e4" \
  --parameterization sigmoid_z --actor_lr 1e-4 --actor_agg min --lambda_clo 0.0 --closure_source none --td_closure_mode none --lambda_proposal 0.0

if [[ "${RUN_DISTANCE:-0}" == "1" ]]; then
  launch_group "A" "A2_distance_min_lr1e4" \
    --parameterization distance --actor_lr 1e-4 --actor_agg min --init_z 0.05 --lambda_clo 0.0 --closure_source none --td_closure_mode none --lambda_proposal 0.0
fi

qrc_wait_all
