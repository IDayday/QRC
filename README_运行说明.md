# QRC 研究系统 v3：运行说明

## 安装

```bash
cd /root/remote/project/QRC
unzip /path/to/qrc_research_system_v3.zip -d /tmp/qrc_v3
cp /tmp/qrc_v3/* .
chmod +x run_qrc_*.sh qrc_launcher_utils.sh analyze_qrc_results.py collect_qrc_debug_bundle_fast.py
```

## 0. 停止旧实验

```bash
./run_qrc_stop_all.sh
```

## 1. Smoke test

```bash
GPU=0 ./run_qrc_smoke_single.sh
```

## 2. Phase A：direct-only

```bash
GPUS_STR="0 1 2 3" SEEDS_STR="52 58 66" \
nohup ./run_qrc_phaseA_direct_screen.sh > qrc_A.log 2>&1 &
```

查看日志：

```bash
tail -f logs_qrc_research/*/*.log
watch -n 2 nvidia-smi
```

汇总：

```bash
python analyze_qrc_results.py --root results_qrc_research --out qrc_summary_A
```

## 3. Phase B：closure activation

只在 Phase A 指标正常后运行：

```bash
GPUS_STR="0 1 2 3" SEEDS_STR="52 58 66" \
nohup ./run_qrc_phaseB_closure_activation.sh > qrc_B.log 2>&1 &
```

## 4. Phase C：recursive-vs-safe

```bash
GPUS_STR="0 1 2 3" SEEDS_STR="52 58 66" \
nohup ./run_qrc_phaseC_recursive_safety.sh > qrc_C.log 2>&1 &
```

## 5. Phase D/E

```bash
GPUS_STR="0 1 2 3" SEEDS_STR="52 58 66" \
nohup ./run_qrc_phaseD_proposal_witness.sh > qrc_D.log 2>&1 &

GPUS_STR="0 1 2 3" SEEDS_STR="52 58 66" \
BEST_TARGET=lcb BEST_BETA=2.0 BEST_M=8 \
nohup ./run_qrc_phaseE_antfb_confirm.sh > qrc_E.log 2>&1 &
```

## 6. 收集小于 100MB 的诊断包

```bash
python collect_qrc_debug_bundle_fast.py \
  --project-root /root/remote/project/QRC \
  --results-root /root/remote/project/QRC/results_qrc_research \
  --logs-root /root/remote/project/QRC/logs_qrc_research \
  --extra-log /root/remote/project/QRC/qrc_A.log \
  --out qrc_v3_debug_fast.zip \
  --max-mb 95 \
  --jobs 16 \
  --diag-tail-rows 1000 \
  --log-tail-lines 1000 \
  --no-tensorboard \
  --skip-tree
```
