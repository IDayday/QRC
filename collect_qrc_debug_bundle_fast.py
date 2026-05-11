#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_qrc_debug_bundle_fast.py

QRC 实验诊断数据极速打包脚本。

相对旧版的核心改动：
1. 默认不解析 TensorBoard event，不复制完整 diagnostics.csv，不读取完整大日志。
2. diagnostics.csv 默认只解析 head/tail 抽样；如需完整统计，可显式加 --scan-full-diagnostics。
3. 日志 head/tail 用 seek 从文件头尾读取，不再一次性 read_text 整个文件。
4. run 目录和 log 文件处理使用多进程并行。
5. 默认只记录 checkpoint 元数据，不哈希完整权重。

典型用法：
  cd /root/remote/project/QRC
  python collect_qrc_debug_bundle_fast.py \
    --project-root /root/remote/project/QRC \
    --results-root /root/remote/project/QRC/results_qrc_total \
    --logs-root /root/remote/project/QRC/logs_qrc_total \
    --extra-log /root/remote/project/QRC/qrc_total.log \
    --out qrc_debug_actual_fast.zip \
    --max-mb 95 \
    --workers 16

若仍觉得慢：
  --diag-tail-rows 300 --log-tail-lines 300 --workers 12 --skip-tree

若需要更完整但会慢一些：
  --scan-full-diagnostics --scan-full-log-errors
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import fnmatch
import hashlib
import json
import math
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SKIP_DIR_NAMES = {
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".cache",
    "wandb", "node_modules", ".vscode", ".idea", "htmlcov",
}
MODEL_SUFFIXES = {".pth", ".pt", ".ckpt", ".pkl", ".pickle"}
EVENT_PREFIX = "events.out.tfevents"

ERROR_PAT = re.compile(
    r"(Traceback|RuntimeError|CUDA out of memory|out of memory|Killed|NAN|NaN|nan|inf|Inf|overflow|assert|ValueError|Exception)",
    re.IGNORECASE,
)

# 这些列足以判断 QRC 是否是 critic 学不进去、closure 过强、actor 梯度失败、还是 eval 无效。
KEY_METRIC_CANDIDATES = [
    "row_type", "env_step", "update", "total_it", "elapsed_sec", "steps_per_second", "replay_size",
    "eval_train_success", "eval_test_success", "eval_train_distance", "eval_test_distance",
    "TrainEvalSuccess", "TestEvalSuccess", "TrainEvalDistance", "TestEvalDistance", "success_rate", "eval_distance",
    "z_td_loss", "z_direct_loss", "z_direct_pred", "z_direct_target", "z_mean", "z_min", "z_max",
    "d_mean", "d_saturation_rate", "actor_loss", "actor_z_action_mean",
    "qrc_closure_loss", "qrc_closure_uplift", "qrc_best_z_cert", "qrc_best_d_cert",
    "qrc_witness_hit_rate", "qrc_closure_accept_rate", "qrc_closure_gap", "qrc_nondegenerate_rate",
    "qrc_raw_planner_failure_rate", "qrc_projected_distance", "qrc_triangle_violation_rate",
    "qrc_closure_target_outlier_rate", "qrc_candidate_m",
    "evidence_direct_edge_count", "evidence_stitch_coverage_rate", "evidence_join_dist",
    "evidence_join_conf", "evidence_h1", "evidence_h2", "evidence_target_z",
    "critic_loss", "critic_ctrl_loss", "D_KL", "q_target_mean", "q_pred_mean",
]

SOURCE_FILES = [
    "QRC.py", "train_ant_qrc.py", "HER_adaptive_backup.py", "Models.py",
    "QRC_README_实验说明.md", "README.md", "project_chat.md", "QRC_algorithm_design.md",
    "run_qrc_phase0_direct_sanity.sh", "run_qrc_phase1_random_closure.sh",
    "run_qrc_phase2_projected_stitch.sh", "run_qrc_total_3090_safe.sh",
]

CONFIG_KEYS = [
    "env_name", "train_env_name", "test_env_name", "seed", "exp_name", "max_timesteps", "start_timesteps",
    "eval_freq", "batch_size", "gamma", "parameterization", "closure_source", "closure_candidates",
    "lambda_dir", "lambda_clo", "lambda_stitch", "closure_start_updates", "p_orig", "h_relab",
    "actor_lr", "critic_lr", "q_lr", "pi_lr", "n_heads", "n_Q", "n_Z", "distance_threshold",
    "deterministic_actor_update", "exploration_noise", "policy_delay", "init_z", "cert_beta", "cert_sigma_floor",
]


def now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except OSError:
        return 0


def human_size(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    v = float(n)
    for u in units:
        if v < 1024 or u == units[-1]:
            return f"{v:.1f}{u}"
        v /= 1024.0
    return f"{n}B"


def safe_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def safe_name(s: str, max_len: int = 180) -> str:
    out = re.sub(r"[^A-Za-z0-9_.=-]+", "_", s)
    return out[-max_len:]


def run_cmd(cmd: Sequence[str], timeout: int = 15) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=timeout)
        return out.decode("utf-8", errors="replace")
    except Exception as e:
        return f"[command failed] {' '.join(cmd)}\n{type(e).__name__}: {e}\n"


def sha256_file(path: Path, max_bytes: Optional[int] = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        remaining = max_bytes
        while True:
            if remaining is None:
                chunk = f.read(1024 * 1024)
            else:
                if remaining <= 0:
                    break
                chunk = f.read(min(1024 * 1024, remaining))
                remaining -= len(chunk)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def read_head_lines(path: Path, n: int, max_bytes: int = 2_000_000) -> List[str]:
    if n <= 0:
        return []
    try:
        with path.open("rb") as f:
            data = f.read(max_bytes)
        text = data.decode("utf-8", errors="replace")
        return text.splitlines()[:n]
    except Exception:
        return []


def read_tail_lines(path: Path, n: int, max_bytes: int = 4_000_000, block_size: int = 64 * 1024) -> List[str]:
    """从文件尾部读取最后 n 行。不会把整个文件读入内存。"""
    if n <= 0:
        return []
    try:
        size = file_size(path)
        if size <= 0:
            return []
        data = b""
        with path.open("rb") as f:
            pos = size
            while pos > 0 and data.count(b"\n") <= n and len(data) < max_bytes:
                read_size = min(block_size, pos, max_bytes - len(data))
                if read_size <= 0:
                    break
                pos -= read_size
                f.seek(pos)
                data = f.read(read_size) + data
        # 如果不是从文件头开始读，第一行可能是截断残片，丢弃。
        lines_b = data.splitlines()
        if pos > 0 and lines_b:
            lines_b = lines_b[1:]
        lines = [x.decode("utf-8", errors="replace") for x in lines_b]
        return lines[-n:]
    except Exception:
        return []


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8", errors="replace")


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Optional[List[str]] = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        keys: List[str] = []
        seen = set()
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        fieldnames = keys if keys else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def to_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        if isinstance(x, str) and x.strip() == "":
            return None
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def parse_csv_lines_with_header(header_line: str, data_lines: List[str], sample_part: str) -> List[Dict[str, str]]:
    """用 diagnostics 的 header 解析抽样行。遇到损坏行则尽量跳过。"""
    if not header_line:
        return []
    # 过滤空行和重复 header。
    cleaned = [ln for ln in data_lines if ln.strip() and ln.strip() != header_line.strip()]
    if not cleaned:
        return []
    text = header_line + "\n" + "\n".join(cleaned) + "\n"
    out: List[Dict[str, str]] = []
    try:
        reader = csv.DictReader(text.splitlines())
        for row in reader:
            if row is None:
                continue
            # DictReader 遇到列数不匹配时可能放 None key。
            if None in row:
                row.pop(None, None)
            row["__sample_part"] = sample_part
            out.append(row)
    except Exception:
        return []
    return out


def get_first_existing(row: Dict[str, object], keys: Sequence[str]) -> Optional[object]:
    for k in keys:
        if k in row and str(row.get(k, "")).strip() != "":
            return row[k]
    return None


def is_eval_row(row: Dict[str, str]) -> bool:
    rt = str(row.get("row_type", "")).lower()
    if rt == "eval":
        return True
    return any(str(row.get(k, "")).strip() != "" for k in [
        "eval_train_success", "eval_test_success", "TrainEvalSuccess", "TestEvalSuccess", "success_rate",
    ])


def summarize_sampled_rows(rows: List[Dict[str, str]], fieldnames: List[str]) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "summary_scope": "head_tail_sample_only",
        "sample_rows": len(rows),
        "sample_columns": ";".join(fieldnames[:200]),
    }
    if not rows:
        summary["status"] = "empty_sample"
        return summary
    summary["status"] = "ok"
    last = rows[-1]
    eval_rows = [r for r in rows if is_eval_row(r)]
    summary["sample_eval_rows"] = len(eval_rows)
    summary["last_row_type"] = last.get("row_type", "")
    summary["last_env_step"] = get_first_existing(last, ["env_step", "t", "step", "global_step"])
    summary["last_update"] = get_first_existing(last, ["update", "total_it", "grad_step"])

    for key in KEY_METRIC_CANDIDATES:
        if key not in fieldnames and key not in last:
            continue
        vals = [to_float(r.get(key)) for r in rows]
        finite = [v for v in vals if v is not None]
        bad_count = sum(1 for r in rows if str(r.get(key, "")).lower() in {"nan", "inf", "-inf"})
        if finite:
            summary[f"{key}__last"] = finite[-1]
            summary[f"{key}__min"] = min(finite)
            summary[f"{key}__max"] = max(finite)
            summary[f"{key}__mean"] = sum(finite) / len(finite)
        if bad_count:
            summary[f"{key}__bad_count"] = bad_count

    for keyset_name, keys in {
        "train_success": ["eval_train_success", "TrainEvalSuccess", "success_rate"],
        "test_success": ["eval_test_success", "TestEvalSuccess"],
        "train_distance": ["eval_train_distance", "TrainEvalDistance", "eval_distance"],
        "test_distance": ["eval_test_distance", "TestEvalDistance"],
    }.items():
        vals: List[float] = []
        for r in eval_rows:
            fv = to_float(get_first_existing(r, keys))
            if fv is not None:
                vals.append(fv)
        if vals:
            summary[f"eval_{keyset_name}_last"] = vals[-1]
            summary[f"eval_{keyset_name}_best"] = max(vals) if "success" in keyset_name else min(vals)
            summary[f"eval_{keyset_name}_mean"] = sum(vals) / len(vals)
    return summary


def incremental_full_csv_summary(path: Path, max_eval_rows: int = 2000) -> Tuple[Dict[str, object], List[Dict[str, str]]]:
    """可选慢路径：完整扫描 diagnostics，流式统计关键指标，不把全表放入内存。"""
    summary: Dict[str, object] = {"summary_scope": "full_stream_scan"}
    eval_rows: List[Dict[str, str]] = []
    stats: Dict[str, Dict[str, float]] = {}
    bad_counts: Dict[str, int] = {}
    row_count = 0
    train_count = 0
    eval_count = 0
    last_row: Dict[str, str] = {}
    fieldnames: List[str] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            active_keys = [k for k in KEY_METRIC_CANDIDATES if k in fieldnames]
            for row in reader:
                row_count += 1
                last_row = row
                if is_eval_row(row):
                    eval_count += 1
                    if len(eval_rows) < max_eval_rows:
                        eval_rows.append(row)
                    elif max_eval_rows > 0:
                        # 保留最近 eval：简单滚动。
                        eval_rows.pop(0)
                        eval_rows.append(row)
                else:
                    train_count += 1
                for key in active_keys:
                    fv = to_float(row.get(key))
                    if fv is None:
                        if str(row.get(key, "")).lower() in {"nan", "inf", "-inf"}:
                            bad_counts[key] = bad_counts.get(key, 0) + 1
                        continue
                    st = stats.setdefault(key, {"count": 0.0, "sum": 0.0, "min": fv, "max": fv, "last": fv})
                    st["count"] += 1.0
                    st["sum"] += fv
                    st["min"] = min(st["min"], fv)
                    st["max"] = max(st["max"], fv)
                    st["last"] = fv
        summary.update({
            "status": "ok",
            "diagnostics_rows": row_count,
            "train_rows": train_count,
            "eval_rows": eval_count,
            "diagnostics_columns": ";".join(fieldnames[:200]),
            "last_row_type": last_row.get("row_type", "") if last_row else "",
            "last_env_step": get_first_existing(last_row, ["env_step", "t", "step", "global_step"]) if last_row else "",
            "last_update": get_first_existing(last_row, ["update", "total_it", "grad_step"]) if last_row else "",
        })
        for key, st in stats.items():
            c = max(st["count"], 1.0)
            summary[f"{key}__last"] = st["last"]
            summary[f"{key}__min"] = st["min"]
            summary[f"{key}__max"] = st["max"]
            summary[f"{key}__mean"] = st["sum"] / c
        for key, cnt in bad_counts.items():
            summary[f"{key}__bad_count"] = cnt
        return summary, eval_rows
    except Exception as e:
        return {"summary_scope": "full_stream_scan", "status": "scan_failed", "error": f"{type(e).__name__}: {e}"}, []


def parse_config(config_path: Path, run_dir: Path, project_root: Path) -> Dict[str, object]:
    out: Dict[str, object] = {
        "run_dir": safe_relpath(run_dir, project_root),
        "config_path": safe_relpath(config_path, project_root),
    }
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8", errors="replace"))
        for k in CONFIG_KEYS:
            if k in cfg:
                out[k] = cfg[k]
    except Exception as e:
        out["config_error"] = f"{type(e).__name__}: {e}"
    return out


def copy_small_text(src: Path, dst: Path, max_bytes: int = 1_000_000) -> None:
    ensure_dir(dst.parent)
    try:
        if file_size(src) <= max_bytes:
            shutil.copy2(src, dst)
        else:
            head = read_head_lines(src, 200, max_bytes=max_bytes // 2)
            tail = read_tail_lines(src, 400, max_bytes=max_bytes // 2)
            write_text(dst, "\n".join([f"# TRUNCATED: {src}", "# ---- HEAD ----", *head, "# ---- TAIL ----", *tail]) + "\n")
    except Exception as e:
        write_text(dst, f"copy failed: {type(e).__name__}: {e}\n")


def process_run_dir_worker(args_tuple) -> Dict[str, object]:
    (
        run_dir_s, project_root_s, staging_s, diag_head_rows, diag_tail_rows, diag_tail_max_bytes,
        scan_full_diagnostics, max_eval_rows, hash_metadata_bytes,
    ) = args_tuple
    run_dir = Path(run_dir_s)
    project_root = Path(project_root_s)
    staging = Path(staging_s)
    rel_dir = safe_relpath(run_dir, project_root)
    run_key = safe_name(rel_dir)

    result: Dict[str, object] = {
        "run_dir": rel_dir,
        "run_key": run_key,
        "has_config": False,
        "has_diagnostics": False,
    }
    config_row: Dict[str, object] = {"run_dir": rel_dir}

    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        result["has_config"] = True
        config_row = parse_config(cfg_path, run_dir, project_root)
        copy_small_text(cfg_path, staging / "configs" / run_key / "config.json", max_bytes=2_000_000)
        result.update({k: v for k, v in config_row.items() if k not in {"config_path", "run_dir"}})

    diag_path = run_dir / "diagnostics.csv"
    diag_summary: Dict[str, object] = {"run_dir": rel_dir}
    if diag_path.exists():
        result["has_diagnostics"] = True
        result["diagnostics_size_bytes"] = file_size(diag_path)
        result["diagnostics_size_human"] = human_size(file_size(diag_path))
        head_lines = read_head_lines(diag_path, max(diag_head_rows + 1, 1), max_bytes=1_000_000)
        header = head_lines[0] if head_lines else ""
        head_data = head_lines[1: 1 + max(diag_head_rows, 0)] if len(head_lines) > 1 else []
        tail_lines = read_tail_lines(diag_path, diag_tail_rows, max_bytes=diag_tail_max_bytes)
        fieldnames: List[str] = []
        if header:
            try:
                fieldnames = next(csv.reader([header]))
            except Exception:
                fieldnames = []
        sample_rows = []
        sample_rows.extend(parse_csv_lines_with_header(header, head_data, "head"))
        sample_rows.extend(parse_csv_lines_with_header(header, tail_lines, "tail"))

        if sample_rows:
            write_csv(staging / "diagnostics_samples" / run_key / "diagnostics_head_tail.csv", sample_rows)
            eval_sample = [r for r in sample_rows if is_eval_row(r)]
            if eval_sample:
                write_csv(staging / "diagnostics_eval_sample" / run_key / "diagnostics_eval_sample_rows.csv", eval_sample)

        diag_summary = summarize_sampled_rows(sample_rows, fieldnames)
        diag_summary.update({
            "run_dir": rel_dir,
            "diagnostics_path": safe_relpath(diag_path, project_root),
            "diagnostics_size_bytes": file_size(diag_path),
            "diagnostics_size_human": human_size(file_size(diag_path)),
        })

        if scan_full_diagnostics:
            full_summary, eval_rows = incremental_full_csv_summary(diag_path, max_eval_rows=max_eval_rows)
            # full_summary 的关键统计覆盖 tail_summary；tail 的抽样样本仍保留。
            diag_summary.update(full_summary)
            if eval_rows:
                write_csv(staging / "diagnostics_eval_fullscan" / run_key / "diagnostics_eval_rows.csv", eval_rows)

        # runs_index 常用列。
        for k in [
            "diagnostics_rows", "train_rows", "eval_rows", "sample_rows", "sample_eval_rows",
            "last_env_step", "last_update", "eval_train_success_last", "eval_train_success_best",
            "eval_test_success_last", "eval_test_success_best", "z_direct_pred__last",
            "z_direct_target__last", "d_saturation_rate__last", "qrc_closure_accept_rate__last",
            "steps_per_second__last", "summary_scope", "status",
        ]:
            if k in diag_summary:
                result[k] = diag_summary[k]

    # 只列 run 目录第一层 checkpoint，避免遍历 replay/子目录。
    ckpt_rows: List[Dict[str, object]] = []
    try:
        for p in run_dir.iterdir():
            if not p.is_file():
                continue
            if p.suffix.lower() in MODEL_SUFFIXES or p.name.endswith(".pth"):
                row: Dict[str, object] = {
                    "run_dir": rel_dir,
                    "file": safe_relpath(p, project_root),
                    "size_bytes": file_size(p),
                    "size_human": human_size(file_size(p)),
                    "mtime": _dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
                    "included_in_zip": False,
                }
                if hash_metadata_bytes and hash_metadata_bytes > 0:
                    try:
                        row[f"sha256_first_{hash_metadata_bytes}_bytes"] = sha256_file(p, max_bytes=hash_metadata_bytes)
                    except Exception as e:
                        row["hash_error"] = f"{type(e).__name__}: {e}"
                ckpt_rows.append(row)
    except Exception as e:
        ckpt_rows.append({"run_dir": rel_dir, "error": f"{type(e).__name__}: {e}"})

    # worker 写本地小 JSON，主进程聚合；避免大对象过多返回。
    out_payload = {
        "run_index": result,
        "config_summary": config_row,
        "diag_summary": diag_summary,
        "checkpoint_metadata": ckpt_rows,
    }
    tmp_json = staging / "_worker_runs" / f"{run_key}.json"
    ensure_dir(tmp_json.parent)
    tmp_json.write_text(json.dumps(out_payload, ensure_ascii=False), encoding="utf-8")
    return {"run_dir": rel_dir, "worker_json": str(tmp_json), "ok": True}


def process_log_worker(args_tuple) -> Dict[str, object]:
    (
        log_path_s, project_root_s, staging_s, index, log_head_lines, log_tail_lines,
        log_head_max_bytes, log_tail_max_bytes, scan_full_log_errors, max_error_hits,
    ) = args_tuple
    log_path = Path(log_path_s)
    project_root = Path(project_root_s)
    staging = Path(staging_s)
    rel = safe_relpath(log_path, project_root)
    name = f"{int(index):04d}_{safe_name(rel, 220)}.tail.txt"
    dst = staging / "logs_tail" / name
    ensure_dir(dst.parent)
    head = read_head_lines(log_path, log_head_lines, max_bytes=log_head_max_bytes)
    tail = read_tail_lines(log_path, log_tail_lines, max_bytes=log_tail_max_bytes)
    text = "\n".join([
        f"# Source: {log_path}",
        f"# Original size: {human_size(file_size(log_path))}",
        "# ---- HEAD ----",
        *head,
        "# ---- TAIL ----",
        *tail,
        "",
    ])
    dst.write_text(text, encoding="utf-8", errors="replace")

    hits: List[Dict[str, object]] = []
    if scan_full_log_errors:
        try:
            with log_path.open("r", encoding="utf-8", errors="replace") as f:
                for line_no, line in enumerate(f, 1):
                    if ERROR_PAT.search(line):
                        hits.append({"log_file": rel, "line": line_no, "text": line.rstrip()[:1200]})
                        if len(hits) >= max_error_hits:
                            break
        except Exception as e:
            hits.append({"log_file": rel, "line": -1, "text": f"full scan failed: {type(e).__name__}: {e}"})
    else:
        for section_name, lines in [("head", head), ("tail", tail)]:
            for i, line in enumerate(lines, 1):
                if ERROR_PAT.search(line):
                    hits.append({"log_file": rel, "line": f"{section_name}:{i}", "text": line.rstrip()[:1200]})
                    if len(hits) >= max_error_hits:
                        break
            if len(hits) >= max_error_hits:
                break

    return {
        "log_file": rel,
        "tail_file": str(dst.relative_to(staging)),
        "size_bytes": file_size(log_path),
        "size_human": human_size(file_size(log_path)),
        "error_hits": hits,
    }


def is_probably_run_dir(p: Path) -> bool:
    try:
        return (p / "diagnostics.csv").exists() or (p / "config.json").exists()
    except Exception:
        return False


def walk_limited(root: Path, max_depth: int):
    root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        p = Path(dirpath)
        try:
            depth = len(p.relative_to(root).parts)
        except Exception:
            depth = 0
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
        if depth >= max_depth:
            dirnames[:] = []
        yield p, dirnames, filenames, depth


def find_run_dirs(roots: Sequence[Path], max_depth: int = 8, run_filter: Optional[str] = None) -> List[Path]:
    runs: List[Path] = []
    seen = set()
    pat = re.compile(run_filter) if run_filter else None
    for root in roots:
        if not root.exists():
            continue
        root = root.resolve()
        if is_probably_run_dir(root):
            key = str(root)
            if key not in seen and (pat is None or pat.search(key)):
                runs.append(root); seen.add(key)
            continue
        for p, dirnames, filenames, _depth in walk_limited(root, max_depth=max_depth):
            if "diagnostics.csv" in filenames or "config.json" in filenames:
                key = str(p.resolve())
                if key not in seen and (pat is None or pat.search(key)):
                    runs.append(p.resolve())
                    seen.add(key)
                # run 目录通常没有需要继续向下找的子 run，停止下钻可显著省时。
                dirnames[:] = []
    return sorted(runs, key=lambda x: str(x))


def find_log_files(project_root: Path, log_roots: Sequence[Path], extra_logs: Sequence[Path], max_depth: int = 4) -> List[Path]:
    files: List[Path] = []
    seen = set()
    for log in extra_logs:
        p = log if log.is_absolute() else project_root / log
        if p.exists() and p.is_file():
            key = str(p.resolve())
            if key not in seen:
                files.append(p.resolve()); seen.add(key)
    for root in log_roots:
        if not root.exists():
            continue
        if root.is_file():
            key = str(root.resolve())
            if key not in seen:
                files.append(root.resolve()); seen.add(key)
            continue
        for p, _dirnames, filenames, _depth in walk_limited(root, max_depth=max_depth):
            for fn in filenames:
                fp = p / fn
                # 训练日志通常是 .log；launcher stdout 可能无后缀或 .txt。
                if fp.suffix in {".log", ".txt", ""} or fn.endswith(".out"):
                    key = str(fp.resolve())
                    if key not in seen:
                        files.append(fp.resolve()); seen.add(key)
    return sorted(files, key=lambda x: str(x))


def write_tree_fast(roots: Sequence[Path], dst: Path, max_depth: int = 5, max_entries: int = 5000) -> None:
    lines: List[str] = []
    count = 0
    for root in roots:
        if not root.exists():
            lines.append(f"[MISSING] {root}")
            continue
        root = root.resolve()
        lines.append(f"\n# TREE {root}")
        for p, dirnames, filenames, depth in walk_limited(root, max_depth=max_depth):
            indent = "  " * depth
            lines.append(f"{indent}{p.name}/")
            count += 1
            for fn in sorted(filenames):
                fp = p / fn
                mark = " [SKIP_MODEL]" if fp.suffix.lower() in MODEL_SUFFIXES else ""
                lines.append(f"{indent}  {fn}  {human_size(file_size(fp))}{mark}")
                count += 1
                if count >= max_entries:
                    lines.append("[TREE TRUNCATED]")
                    write_text(dst, "\n".join(lines) + "\n")
                    return
    write_text(dst, "\n".join(lines) + "\n")


def collect_code_unique(project_root: Path, run_dirs: Sequence[Path], staging: Path, include_run_snapshots: bool = True) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    candidates: List[Path] = []
    for name in SOURCE_FILES:
        p = project_root / name
        if p.exists() and p.is_file():
            candidates.append(p)
    if include_run_snapshots:
        for rd in run_dirs:
            for name in ["QRC.py", "train_ant_qrc.py", "HER_adaptive_backup.py", "Models.py", "QRC_README_实验说明.md"]:
                p = rd / name
                if p.exists() and p.is_file():
                    candidates.append(p)
    unique: Dict[str, Path] = {}
    seen_paths = set()
    for p in sorted(candidates, key=lambda x: str(x)):
        if str(p.resolve()) in seen_paths:
            continue
        seen_paths.add(str(p.resolve()))
        try:
            h = sha256_file(p)
            row: Dict[str, object] = {
                "path": safe_relpath(p, project_root),
                "sha256": h,
                "size_bytes": file_size(p),
                "mtime": _dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
            }
            if h not in unique:
                unique[h] = p
                dst_name = f"{safe_name(p.name, 80)}.{h[:10]}{p.suffix if p.suffix and not p.name.endswith(p.suffix) else ''}"
                dst = staging / "code_unique" / dst_name
                ensure_dir(dst.parent)
                shutil.copy2(p, dst)
                row["copied_as"] = str(dst.relative_to(staging))
            else:
                row["duplicate_of"] = safe_relpath(unique[h], project_root)
            rows.append(row)
        except Exception as e:
            rows.append({"path": str(p), "error": f"{type(e).__name__}: {e}"})
    return rows


def collect_tensorboard_scalars(run_dirs: Sequence[Path], project_root: Path, max_events: int = 80, max_scalars: int = 2000) -> List[Dict[str, object]]:
    """默认不用。开启后只保留 event scalar 摘要，仍可能较慢。"""
    rows: List[Dict[str, object]] = []
    try:
        from tensorboard.backend.event_processing import event_accumulator  # type: ignore
    except Exception as e:
        return [{"status": "tensorboard_not_available", "detail": f"{type(e).__name__}: {e}"}]
    event_files: List[Path] = []
    for rd in run_dirs:
        event_files.extend(sorted(rd.glob("events.out.tfevents*")))
    for ef in event_files[:max_events]:
        try:
            ea = event_accumulator.EventAccumulator(
                str(ef),
                size_guidance={
                    event_accumulator.SCALARS: max_scalars,
                    event_accumulator.HISTOGRAMS: 0,
                    event_accumulator.IMAGES: 0,
                    event_accumulator.COMPRESSED_HISTOGRAMS: 0,
                    event_accumulator.TENSORS: 0,
                },
            )
            ea.Reload()
            for tag in ea.Tags().get("scalars", []):
                vals = ea.Scalars(tag)
                y = [float(v.value) for v in vals if math.isfinite(float(v.value))]
                if not y:
                    continue
                rows.append({
                    "run_dir": safe_relpath(ef.parent, project_root),
                    "event_file": safe_relpath(ef, project_root),
                    "tag": tag,
                    "count_loaded": len(vals),
                    "first_step": vals[0].step,
                    "last_step": vals[-1].step,
                    "last_value": y[-1],
                    "min_value": min(y),
                    "max_value": max(y),
                    "mean_value": sum(y) / len(y),
                })
        except Exception as e:
            rows.append({
                "run_dir": safe_relpath(ef.parent, project_root),
                "event_file": safe_relpath(ef, project_root),
                "tag": "__parse_error__",
                "detail": f"{type(e).__name__}: {e}",
            })
    return rows


def zip_dir(src_dir: Path, zip_path: Path, compresslevel: int = 1) -> int:
    if zip_path.exists():
        zip_path.unlink()
    ensure_dir(zip_path.parent)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=compresslevel) as zf:
        for p in sorted(src_dir.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(src_dir))
    return file_size(zip_path)


def remove_dir_if_exists(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def load_worker_jsons(paths: List[str]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    run_rows: List[Dict[str, object]] = []
    cfg_rows: List[Dict[str, object]] = []
    diag_rows: List[Dict[str, object]] = []
    ckpt_rows: List[Dict[str, object]] = []
    for p_s in paths:
        try:
            payload = json.loads(Path(p_s).read_text(encoding="utf-8"))
            run_rows.append(payload.get("run_index", {}))
            cfg = payload.get("config_summary", {})
            if cfg:
                cfg_rows.append(cfg)
            diag = payload.get("diag_summary", {})
            if diag:
                diag_rows.append(diag)
            ckpt_rows.extend(payload.get("checkpoint_metadata", []))
        except Exception as e:
            run_rows.append({"worker_json": p_s, "error": f"{type(e).__name__}: {e}"})
    return run_rows, cfg_rows, diag_rows, ckpt_rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fast QRC debug bundle collector. Avoids full reads of large CSV/log/event files by default.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project-root", default=os.getcwd())
    parser.add_argument("--results-root", action="append", default=None)
    parser.add_argument("--logs-root", action="append", default=None)
    parser.add_argument("--extra-log", action="append", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-mb", type=float, default=95.0)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 8) // 2)))
    parser.add_argument("--jobs", type=int, default=None, help="--workers 的别名；指定后覆盖 --workers。")
    parser.add_argument("--max-walk-depth", type=int, default=8)
    parser.add_argument("--max-log-walk-depth", type=int, default=4)
    parser.add_argument("--run-filter", default=None, help="只收集路径匹配该正则的 run。")
    parser.add_argument("--limit-run-dirs", type=int, default=0)
    parser.add_argument("--limit-logs", type=int, default=0)

    parser.add_argument("--diag-head-rows", type=int, default=20)
    parser.add_argument("--diag-tail-rows", type=int, default=800)
    parser.add_argument("--diag-tail-max-bytes", type=int, default=4_000_000)
    parser.add_argument("--scan-full-diagnostics", action="store_true", help="完整流式扫描 diagnostics.csv，统计更完整但会慢。")
    parser.add_argument("--max-eval-rows", type=int, default=2000)

    parser.add_argument("--log-head-lines", type=int, default=40)
    parser.add_argument("--log-tail-lines", type=int, default=800)
    parser.add_argument("--log-head-max-bytes", type=int, default=500_000)
    parser.add_argument("--log-tail-max-bytes", type=int, default=2_000_000)
    parser.add_argument("--scan-full-log-errors", action="store_true", help="完整扫描日志错误，默认只扫 head/tail。")
    parser.add_argument("--max-error-hits-per-log", type=int, default=60)

    parser.add_argument("--tensorboard", action="store_true", help="解析 TensorBoard scalars；默认关闭以提速。")
    parser.add_argument("--no-tensorboard", action="store_true", help="兼容旧脚本参数；fast 版默认已关闭。")
    parser.add_argument("--max-tensorboard-events", type=int, default=80)
    parser.add_argument("--max-tensorboard-scalars", type=int, default=2000)

    # 兼容旧脚本参数：fast 版默认不复制完整 diagnostics。
    parser.add_argument("--max-full-diagnostics", type=int, default=0)
    parser.add_argument("--full-diagnostics-per-file-mb", type=float, default=0.0)
    parser.add_argument("--max-log-copy-bytes", type=int, default=0, help="兼容旧脚本；fast 版使用 --log-tail-max-bytes。")

    parser.add_argument("--hash-metadata-bytes", type=int, default=0, help="checkpoint 元数据 hash 前 N 字节；0 表示不 hash，最快。")
    parser.add_argument("--zip-compresslevel", type=int, default=1)
    parser.add_argument("--tree-depth", type=int, default=5)
    parser.add_argument("--skip-tree", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--progress-every", type=int, default=5)
    args = parser.parse_args()
    if args.jobs is not None:
        args.workers = int(args.jobs)

    t0 = time.time()
    project_root = Path(args.project_root).expanduser().resolve()
    result_args = args.results_root if args.results_root is not None else ["results_qrc_total"]
    log_args = args.logs_root if args.logs_root is not None else ["logs_qrc_total"]
    extra_args = args.extra_log if args.extra_log is not None else ["qrc_total.log"]
    result_roots = [(Path(p) if Path(p).is_absolute() else project_root / p).resolve() for p in result_args]
    log_roots = [(Path(p) if Path(p).is_absolute() else project_root / p).resolve() for p in log_args]
    extra_logs = [Path(p) for p in extra_args]

    print(f"[INFO] project_root={project_root}", flush=True)
    print(f"[INFO] finding run dirs...", flush=True)
    run_dirs = find_run_dirs(result_roots, max_depth=args.max_walk_depth, run_filter=args.run_filter)
    if args.limit_run_dirs and args.limit_run_dirs > 0:
        run_dirs = run_dirs[: args.limit_run_dirs]
    print(f"[INFO] Found run dirs: {len(run_dirs)}", flush=True)
    for rd in run_dirs[:50]:
        print(f"  RUN {rd}", flush=True)

    print(f"[INFO] finding log files...", flush=True)
    log_files = find_log_files(project_root, log_roots, extra_logs, max_depth=args.max_log_walk_depth)
    if args.limit_logs and args.limit_logs > 0:
        log_files = log_files[: args.limit_logs]
    print(f"[INFO] Found log files: {len(log_files)}", flush=True)
    for lf in log_files[:50]:
        print(f"  LOG {lf}", flush=True)

    if args.dry_run:
        print("[DRY-RUN] no zip will be created.", flush=True)
        return 0

    out_path = Path(args.out).expanduser().resolve() if args.out else (project_root / f"qrc_debug_fast_{now_stamp()}.zip")

    with tempfile.TemporaryDirectory(prefix="qrc_debug_fast_") as td:
        staging = Path(td) / "bundle"
        ensure_dir(staging)

        manifest = {
            "created_at": _dt.datetime.now().isoformat(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "project_root": str(project_root),
            "result_roots": [str(p) for p in result_roots],
            "log_roots": [str(p) for p in log_roots],
            "run_dir_count": len(run_dirs),
            "log_file_count": len(log_files),
            "workers": args.workers,
            "fast_mode": True,
            "scan_full_diagnostics": bool(args.scan_full_diagnostics),
            "scan_full_log_errors": bool(args.scan_full_log_errors),
            "tensorboard": bool(args.tensorboard),
            "note": "Fast mode reads only head/tail of diagnostics/logs unless full scan flags are enabled. Check summaries' summary_scope column.",
        }
        write_text(staging / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        write_text(staging / "README_诊断包说明.md", textwrap.dedent(f"""
        # QRC fast debug bundle

        这是由 `collect_qrc_debug_bundle_fast.py` 生成的极速诊断包。

        ## 默认策略

        - 不包含 `.pth/.pt/.ckpt/.pkl` 权重本体。
        - 不包含原始 TensorBoard event 文件。
        - `diagnostics.csv` 默认只保留 head/tail 抽样；`diagnostics_summary.csv` 的 `summary_scope=head_tail_sample_only` 表示统计只来自抽样行。
        - 若使用了 `--scan-full-diagnostics`，则会完整流式扫描 diagnostics 并记录 `summary_scope=full_stream_scan`。
        - 日志默认只保留 head/tail，并只在 head/tail 中扫描错误；若使用 `--scan-full-log-errors`，才完整扫描日志错误。

        ## 生成参数

        ```json
        {json.dumps(manifest, ensure_ascii=False, indent=2)}
        ```
        """))

        # 系统快照：避免 du -sh 递归扫大目录。
        write_text(staging / "process_snapshot.txt", run_cmd(["bash", "-lc", "date; echo; ps -eo pid,ppid,stat,etime,pcpu,pmem,args | grep -E 'train_ant_qrc|run_qrc|python|CUDA_VISIBLE_DEVICES' | grep -v grep"], timeout=15))
        write_text(staging / "nvidia_smi_snapshot.txt", run_cmd(["bash", "-lc", "nvidia-smi || true"], timeout=15))
        write_text(staging / "disk_snapshot.txt", run_cmd(["bash", "-lc", "df -h .; echo; pwd"], timeout=10))

        if not args.skip_tree:
            print("[INFO] writing shallow tree...", flush=True)
            write_tree_fast([p for p in [project_root] + result_roots + log_roots if p.exists()], staging / "tree.txt", max_depth=args.tree_depth, max_entries=5000)

        # 代码快照先做，较小。
        print("[INFO] collecting code snapshots...", flush=True)
        code_rows = collect_code_unique(project_root, run_dirs, staging, include_run_snapshots=True)
        write_csv(staging / "code_hashes.csv", code_rows)

        # run dirs 多进程。
        print(f"[INFO] processing run dirs with {args.workers} workers...", flush=True)
        worker_jsons: List[str] = []
        if run_dirs:
            tasks = [
                (
                    str(rd), str(project_root), str(staging), args.diag_head_rows, args.diag_tail_rows,
                    args.diag_tail_max_bytes, bool(args.scan_full_diagnostics), args.max_eval_rows,
                    args.hash_metadata_bytes,
                )
                for rd in run_dirs
            ]
            with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
                futures = [ex.submit(process_run_dir_worker, t) for t in tasks]
                for i, fut in enumerate(as_completed(futures), 1):
                    try:
                        res = fut.result()
                        if res.get("worker_json"):
                            worker_jsons.append(str(res["worker_json"]))
                    except Exception as e:
                        err_path = staging / "_worker_errors" / f"run_error_{i}.txt"
                        write_text(err_path, f"{type(e).__name__}: {e}\n")
                    if i % max(1, args.progress_every) == 0 or i == len(futures):
                        print(f"[INFO] processed run dirs {i}/{len(futures)}", flush=True)

        run_rows, cfg_rows, diag_rows, ckpt_rows = load_worker_jsons(worker_jsons)
        write_csv(staging / "runs_index.csv", run_rows)
        write_csv(staging / "configs_summary.csv", cfg_rows)
        write_csv(staging / "diagnostics_summary.csv", diag_rows)
        write_csv(staging / "checkpoint_metadata.csv", ckpt_rows)

        # logs 多进程。
        print(f"[INFO] processing logs with {args.workers} workers...", flush=True)
        log_rows: List[Dict[str, object]] = []
        log_error_rows: List[Dict[str, object]] = []
        if log_files:
            tasks = [
                (
                    str(lf), str(project_root), str(staging), idx, args.log_head_lines, args.log_tail_lines,
                    args.log_head_max_bytes, args.log_tail_max_bytes, bool(args.scan_full_log_errors),
                    args.max_error_hits_per_log,
                )
                for idx, lf in enumerate(log_files)
            ]
            with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
                futures = [ex.submit(process_log_worker, t) for t in tasks]
                for i, fut in enumerate(as_completed(futures), 1):
                    try:
                        res = fut.result()
                        log_rows.append({k: v for k, v in res.items() if k != "error_hits"})
                        log_error_rows.extend(res.get("error_hits", []))
                    except Exception as e:
                        log_error_rows.append({"log_file": "__worker_error__", "line": -1, "text": f"{type(e).__name__}: {e}"})
                    if i % max(1, args.progress_every) == 0 or i == len(futures):
                        print(f"[INFO] processed logs {i}/{len(futures)}", flush=True)
        write_csv(staging / "logs_index.csv", log_rows)
        write_csv(staging / "log_error_scan.csv", log_error_rows)

        if args.tensorboard and not args.no_tensorboard:
            print("[INFO] parsing TensorBoard scalars; this may take time...", flush=True)
            tb_rows = collect_tensorboard_scalars(run_dirs, project_root, max_events=args.max_tensorboard_events, max_scalars=args.max_tensorboard_scalars)
            write_csv(staging / "tensorboard_scalar_summary.csv", tb_rows)
        else:
            write_csv(staging / "tensorboard_scalar_summary.csv", [{"status": "skipped_fast_mode", "hint": "rerun with --tensorboard if needed"}])

        # zip + cap。
        print("[INFO] zipping bundle...", flush=True)
        cap = int(args.max_mb * 1024 * 1024)
        size = zip_dir(staging, out_path, compresslevel=max(0, min(9, args.zip_compresslevel)))
        shrink_notes: List[str] = []

        if size > cap:
            # 第一轮瘦身：日志只保留索引和错误扫描。
            remove_dir_if_exists(staging / "logs_tail")
            shrink_notes.append("Removed logs_tail because zip exceeded cap.")
            size = zip_dir(staging, out_path, compresslevel=max(0, min(9, args.zip_compresslevel)))

        if size > cap:
            # 第二轮瘦身：diagnostics 抽样去掉，只保留 summary/eval/config。
            remove_dir_if_exists(staging / "diagnostics_samples")
            shrink_notes.append("Removed diagnostics_samples because zip exceeded cap.")
            size = zip_dir(staging, out_path, compresslevel=max(0, min(9, args.zip_compresslevel)))

        if shrink_notes:
            write_text(staging / "SHRINK_NOTES.txt", "\n".join(shrink_notes) + f"\nfinal_size={human_size(size)}\n")
            size = zip_dir(staging, out_path, compresslevel=max(0, min(9, args.zip_compresslevel)))

        elapsed = time.time() - t0
        print(f"[OK] wrote {out_path}", flush=True)
        print(f"[OK] zip size: {human_size(size)}", flush=True)
        print(f"[OK] run dirs: {len(run_dirs)}, logs: {len(log_files)}, elapsed: {elapsed:.1f}s", flush=True)
        if size > cap:
            print(f"[WARN] zip above requested cap {args.max_mb}MB. Rerun with --diag-tail-rows 200 --log-tail-lines 200 --skip-tree", flush=True)
            return 2
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
