#!/usr/bin/env bash
set -euo pipefail
pkill -TERM -f train_ant_qrc.py || true
pkill -TERM -f run_qrc_phase || true
sleep 10
if pgrep -af 'train_ant_qrc.py|run_qrc_phase' >/dev/null 2>&1; then
  echo '[INFO] Some QRC processes still alive; sending KILL.'
  pkill -KILL -f train_ant_qrc.py || true
  pkill -KILL -f run_qrc_phase || true
fi
pgrep -af 'train_ant_qrc.py|run_qrc_phase' || true
