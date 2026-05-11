#!/usr/bin/env bash
set -euo pipefail
# Phase C: unsafe recursive-vs-safe one-sided contrast.
# Goal: determine whether apparent gains require writing closure into TD bootstrap.
ENV_NAME="${ENV_NAME:-AntU}"
MAX_TIMESTEPS="${MAX_TIMESTEPS:-300000}"
EVAL_FREQ="${EVAL_FREQ:-5000}"
PARAMETERIZATION="${PARAMETERIZATION:-sigmoid_z}"
ACTOR_LR="${ACTOR_LR:-1e-4}"
ACTOR_AGG="${ACTOR_AGG:-min}"
CLOSURE_START_UPDATES="${CLOSURE_START_UPDATES:-20000}"
source "${PROJECT_ROOT:-$(pwd)}/qrc_launcher_utils.sh"

launch_group "C" "C0_direct" \
  --lambda_clo 0.0 --closure_source none --td_closure_mode none --lambda_proposal 0.0

launch_group "C" "C1_safe_oneSidedLCB" \
  --lambda_clo 0.02 --closure_source random --closure_candidates 8 --closure_loss_target lcb --beta_init 2.0 --td_closure_mode none --lambda_proposal 0.0

launch_group "C" "C2_safe_oneSidedRAW" \
  --lambda_clo 0.02 --closure_source random --closure_candidates 8 --closure_loss_target raw --beta_init 2.0 --td_closure_mode none --lambda_proposal 0.0

launch_group "C" "C3_unsafe_recursiveLCB" \
  --lambda_clo 0.0 --closure_source random --closure_candidates 8 --td_closure_mode recursive_lcb --td_closure_start_updates 0 --beta_init 2.0 --lambda_proposal 0.0

launch_group "C" "C4_unsafe_recursiveRAW" \
  --lambda_clo 0.0 --closure_source random --closure_candidates 8 --td_closure_mode recursive_raw --td_closure_start_updates 0 --beta_init 2.0 --lambda_proposal 0.0

qrc_wait_all
