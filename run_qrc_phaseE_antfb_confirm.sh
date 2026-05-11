#!/usr/bin/env bash
set -euo pipefail
# Phase E: AntFb confirmation. Run only after AntU decision phases identify a viable setting.
ENV_NAME="${ENV_NAME:-AntFb}"
MAX_TIMESTEPS="${MAX_TIMESTEPS:-500000}"
EVAL_FREQ="${EVAL_FREQ:-5000}"
PARAMETERIZATION="${PARAMETERIZATION:-sigmoid_z}"
ACTOR_LR="${ACTOR_LR:-1e-4}"
ACTOR_AGG="${ACTOR_AGG:-min}"
CLOSURE_START_UPDATES="${CLOSURE_START_UPDATES:-20000}"
source "${PROJECT_ROOT:-$(pwd)}/qrc_launcher_utils.sh"

launch_group "E" "E0_direct" \
  --lambda_clo 0.0 --closure_source none --td_closure_mode none --lambda_proposal 0.0

launch_group "E" "E1_best_random_safe" \
  --lambda_clo "${BEST_LAMBDA_CLO:-0.02}" --closure_source random --closure_candidates "${BEST_M:-8}" --closure_loss_target "${BEST_TARGET:-lcb}" --beta_init "${BEST_BETA:-2.0}" --td_closure_mode none --lambda_proposal 0.0

launch_group "E" "E2_best_mixed_safe" \
  --lambda_clo "${BEST_LAMBDA_CLO:-0.02}" --closure_source mixed --closure_candidates "${BEST_M:-8}" --closure_loss_target "${BEST_TARGET:-lcb}" --beta_init "${BEST_BETA:-2.0}" --td_closure_mode none --lambda_proposal 0.05 --use_proposal

qrc_wait_all
