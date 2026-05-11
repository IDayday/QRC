#!/usr/bin/env bash
set -euo pipefail
# Phase D: proposal-witness separation. Only run after B/C show meaningful safe closure signal.
ENV_NAME="${ENV_NAME:-AntU}"
MAX_TIMESTEPS="${MAX_TIMESTEPS:-300000}"
EVAL_FREQ="${EVAL_FREQ:-5000}"
PARAMETERIZATION="${PARAMETERIZATION:-sigmoid_z}"
ACTOR_LR="${ACTOR_LR:-1e-4}"
ACTOR_AGG="${ACTOR_AGG:-min}"
CLOSURE_START_UPDATES="${CLOSURE_START_UPDATES:-20000}"
source "${PROJECT_ROOT:-$(pwd)}/qrc_launcher_utils.sh"

launch_group "D" "D0_random_oneSidedLCB" \
  --lambda_clo 0.02 --closure_source random --closure_candidates 8 --closure_loss_target lcb --beta_init 2.0 --td_closure_mode none --lambda_proposal 0.0

launch_group "D" "D1_projected_oneSidedLCB" \
  --lambda_clo 0.02 --closure_source planner_projected --closure_candidates 8 --closure_loss_target lcb --beta_init 2.0 --td_closure_mode none --lambda_proposal 0.05 --use_proposal

launch_group "D" "D2_mixed_oneSidedLCB" \
  --lambda_clo 0.02 --closure_source mixed --closure_candidates 8 --closure_loss_target lcb --beta_init 2.0 --td_closure_mode none --lambda_proposal 0.05 --use_proposal

launch_group "D" "D3_rawPlanner_diag" \
  --lambda_clo 0.02 --closure_source planner_raw --closure_candidates 1 --closure_loss_target raw --beta_init 2.0 --td_closure_mode none --lambda_proposal 0.05 --use_proposal --allow_raw_planner_witness

qrc_wait_all
