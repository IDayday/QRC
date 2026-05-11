#!/usr/bin/env bash
set -euo pipefail
# Phase B: closure activation test. Only run after Phase A is healthy.
# Goal: distinguish candidate failure, LCB over-conservatism, and one-sided signal strength.
ENV_NAME="${ENV_NAME:-AntU}"
MAX_TIMESTEPS="${MAX_TIMESTEPS:-300000}"
EVAL_FREQ="${EVAL_FREQ:-5000}"
PARAMETERIZATION="${PARAMETERIZATION:-sigmoid_z}"
ACTOR_LR="${ACTOR_LR:-1e-4}"
ACTOR_AGG="${ACTOR_AGG:-min}"
CLOSURE_START_UPDATES="${CLOSURE_START_UPDATES:-20000}"
BETA_MODE="fixed"
source "${PROJECT_ROOT:-$(pwd)}/qrc_launcher_utils.sh"

launch_group "B" "B0_direct" \
  --lambda_clo 0.0 --closure_source none --td_closure_mode none --lambda_proposal 0.0

launch_group "B" "B1_randomM8_oneSidedLCB_beta2" \
  --lambda_clo 0.02 --closure_source random --closure_candidates 8 --closure_loss_target lcb --beta_init 2.0 --td_closure_mode none --lambda_proposal 0.0

launch_group "B" "B2_randomM8_oneSidedRAW" \
  --lambda_clo 0.02 --closure_source random --closure_candidates 8 --closure_loss_target raw --beta_init 2.0 --td_closure_mode none --lambda_proposal 0.0

launch_group "B" "B3_randomM16_oneSidedLCB_beta2" \
  --lambda_clo 0.02 --closure_source random --closure_candidates 16 --closure_loss_target lcb --beta_init 2.0 --td_closure_mode none --lambda_proposal 0.0

launch_group "B" "B4_randomM8_oneSidedLCB_beta3" \
  --lambda_clo 0.02 --closure_source random --closure_candidates 8 --closure_loss_target lcb --beta_init 3.0 --td_closure_mode none --lambda_proposal 0.0

qrc_wait_all
