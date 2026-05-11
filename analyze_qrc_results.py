#!/usr/bin/env python3
"""Summarize QRC decision experiments.

This script reads diagnostics.csv files and writes compact run/group summaries plus
simple go/no-go flags for Phase A/B/C.

Usage:
  python analyze_qrc_results.py --root results_qrc_research --out qrc_summary_A
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

KEY_TRAIN_COLS = [
    'z_direct_pred','z_direct_target','z_direct_gap','z_td_loss','z_td_target_mean','z_td_pred_mean','z_td_hit_rate',
    'z_replay_goal_mean','z_replay_goal_td_target','z_actor_goal_mean',
    'actor_loss','actor_loss_step','actor_loss_ema','actor_z_action_mean','actor_z_action_mean_step',
    'actor_z_action_mean_ema','actor_update_flag','actor_update_count','actor_batch_fraction',
    'qrc_closure_accept_rate','qrc_raw_accept_rate','qrc_closure_uplift',
    'qrc_best_z_raw','qrc_best_z_lcb','qrc_pred_z_mean','qrc_closure_gap_raw','qrc_closure_gap_lcb',
    'qrc_cert_suppression','qrc_closure_uses_raw','qrc_td_closure_override_rate','qrc_td_direct_next_z_mean',
    'qrc_td_closure_next_z_mean','qrc_td_closure_gap_mean','qrc_td_best_z_raw','qrc_td_best_z_lcb',
    'calib_beta','calib_beta_new','train_orig_goal_ratio','train_future_goal_ratio','train_replay_goal_ratio'
]
KEY_EVAL_COLS = ['eval_train_success','eval_train_distance','eval_test_success','eval_test_distance']
ID_COLS = ['exp_name','env_name','seed','diagnostics_file']


def safe_cols(df, cols):
    return [c for c in cols if c in df.columns]


def _num(s):
    return pd.to_numeric(s, errors='coerce')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='results_qrc_research')
    ap.add_argument('--out', default='qrc_summary')
    ap.add_argument('--tail-train-rows', default=20, type=int)
    args = ap.parse_args()
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(root.rglob('diagnostics.csv'))
    if not files:
        print(f'No diagnostics.csv under {root}')
        return

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f'[skip] {f}: {e}')
            continue
        df['diagnostics_file'] = str(f)
        frames.append(df)
    if not frames:
        print('No readable diagnostics.csv')
        return
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out / 'all_diagnostics_collected.csv', index=False)

    train = df[df.get('row_type','') == 'train'].copy()
    eval_df = df[df.get('row_type','') == 'eval'].copy()

    if not train.empty:
        for c in safe_cols(train, ['update','env_step'] + KEY_TRAIN_COLS):
            train[c] = _num(train[c])
        train_last = train.sort_values(['diagnostics_file','update']).groupby('diagnostics_file', as_index=False).tail(1)
        train_tail = train.sort_values(['diagnostics_file','update']).groupby('diagnostics_file', as_index=False).tail(args.tail_train_rows)
        tail_means = train_tail.groupby('diagnostics_file')[safe_cols(train_tail, KEY_TRAIN_COLS)].mean(numeric_only=True).reset_index()
        id_last = train_last[safe_cols(train_last, ID_COLS + ['env_step','update'])]
        train_run = id_last.merge(tail_means, on='diagnostics_file', how='left')
        train_run.to_csv(out / 'run_train_tail_mean.csv', index=False)

        group_cols = [c for c in ['exp_name','env_name'] if c in train_run.columns]
        if group_cols:
            agg_cols = safe_cols(train_run, KEY_TRAIN_COLS + ['env_step','update'])
            group_train = train_run.groupby(group_cols)[agg_cols].agg(['mean','std','min','max']).reset_index()
            group_train.to_csv(out / 'group_train_tail_mean.csv', index=False)

        # Phase-specific decision flags.
        flags = []
        for _, r in train_run.iterrows():
            direct_gap = abs(float(r.get('z_direct_gap', np.nan)))
            actor_z = float(r.get('actor_z_action_mean_ema', np.nan))
            replay_z = float(r.get('z_replay_goal_mean', np.nan))
            closure_accept = float(r.get('qrc_closure_accept_rate', np.nan))
            closure_uplift = float(r.get('qrc_closure_uplift', np.nan))
            raw_accept = float(r.get('qrc_raw_accept_rate', np.nan))
            td_override = float(r.get('qrc_td_closure_override_rate', np.nan))
            flags.append({
                'diagnostics_file': r.get('diagnostics_file',''),
                'exp_name': r.get('exp_name',''),
                'env_name': r.get('env_name',''),
                'seed': r.get('seed',''),
                'phaseA_direct_gap_ok': bool(np.isfinite(direct_gap) and direct_gap <= 0.08),
                'phaseA_actor_log_ok': bool(np.isfinite(actor_z) and actor_z > 0.02),
                'phaseA_replay_goal_not_overoptimistic': bool((not np.isfinite(replay_z)) or replay_z <= 0.50),
                'phaseB_lcb_accept_nonzero': bool(np.isfinite(closure_accept) and closure_accept > 0.005),
                'phaseB_uplift_reasonable': bool(np.isfinite(closure_uplift) and 0.0 < closure_uplift < 0.05),
                'phaseB_raw_accept_nonzero': bool(np.isfinite(raw_accept) and raw_accept > 0.005),
                'phaseC_recursive_active': bool(np.isfinite(td_override) and td_override > 0.005),
            })
        pd.DataFrame(flags).to_csv(out / 'decision_flags_by_run.csv', index=False)

    if not eval_df.empty:
        for c in safe_cols(eval_df, ['update','env_step'] + KEY_EVAL_COLS):
            eval_df[c] = _num(eval_df[c])
        eval_last = eval_df.sort_values(['diagnostics_file','env_step']).groupby('diagnostics_file', as_index=False).tail(1)
        eval_best = eval_df.groupby('diagnostics_file')[safe_cols(eval_df, KEY_EVAL_COLS)].max(numeric_only=True).reset_index()
        id_last = eval_last[safe_cols(eval_last, ID_COLS + ['env_step','update'] + KEY_EVAL_COLS)]
        eval_run = id_last.merge(eval_best, on='diagnostics_file', suffixes=('_last','_best'), how='left')
        eval_run.to_csv(out / 'run_eval_last_and_best.csv', index=False)
        group_cols = [c for c in ['exp_name','env_name'] if c in eval_run.columns]
        if group_cols:
            agg_cols = [c for c in eval_run.columns if c.startswith('eval_')]
            group_eval = eval_run.groupby(group_cols)[agg_cols].agg(['mean','std','min','max']).reset_index()
            group_eval.to_csv(out / 'group_eval_summary.csv', index=False)

    print(f'Wrote QRC decision summary to {out}')


if __name__ == '__main__':
    main()
