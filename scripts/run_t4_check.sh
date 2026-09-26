#!/bin/bash
# T4（LSTM）补测（2026-09-26 夜，按项目铁律：nohup + 断点续跑 + 详尽日志 + 独立监控）
#
# 目的：本轮所有结论只覆盖 T2（LightGBM）—— 我构建的全部缓存与 70 月回测都带
#   `--skip-t4`，而 **T4 在生产路径的 Step2 参与选股**（T2+T4 Rank 融合）。
#   本脚本在**与 T2 完全相同的口径**下补测 T4，复核记忆里对 T4 的测量：
#   Fold3（2025 全年）+0.0712 ｜ Fold4（2025 Q2-Q4）+0.1007 ｜ Fold5 −0.2584 ｜ Fold6 −0.0909
#
# 口径（与 T2 的便宜版三项完全一致，保证可比）：
#   · 月份 2025-01 ~ 2026-06（18 个月，覆盖 Fold3/Fold4）
#   · purge 25 交易日（默认，去泄漏）；特征视图 lastday_agg
#   · **不传 --skip-t4** ← 本次的核心
#   · workers=2：LSTM 内部 TF 单 op 16 线程/进程 → 2×16=32 ≤ 36 核，避免过载
#
# 断点续跑：build_prediction_cache 每完成一个月写 month_*.json；崩溃后**重跑同命令**即跳过已完成月。
#
# 用法：
#   nohup bash scripts/run_t4_check.sh > logs/t4_check_wrapper.log 2>&1 &
# 进度查看：
#   tail -f logs/t4_test_18m.log          # 主日志
#   cat  logs/t4_check_progress.log       # 15 分钟一次的进度快照
#   ls output/backtest_v2/.cache_tmp_.t4_test_18m/month_*.json | wc -l   # 已完成月数
set -u
PROJ=/public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x
PY=/home/zhulei/anaconda3/envs/zhulei_py312/bin/python
cd "$PROJ" || exit 1
LOG=logs/t4_test_18m.log
OUT=output/backtest_v2/.t4_test_18m.json
TMP=output/backtest_v2/.cache_tmp_.t4_test_18m
PROG=logs/t4_check_progress.log

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$PROG"; }

# TF 必须清掉 KMP_AFFINITY（项目铁律：否则锁核、单核跑整晚）
export V4_BT_WORKERS=2

# ── 独立监控：每 15 分钟写一次进度快照（进程挂了也能从时间戳看出）──
(
  while true; do
    sleep 900
    D=$(ls "$TMP"/month_*.json 2>/dev/null | wc -l)
    E=$(ls "$TMP"/month_*.error 2>/dev/null | wc -l)
    echo "[$(date '+%F %T')] 进度 $D/18 个月，失败 $E ｜ 日志尾: $(tail -1 "$LOG" 2>/dev/null | cut -c1-110)" >> "$PROG"
    if [ "$D" -ge 18 ]; then break; fi
  done
) &
MON=$!

log "═══ T4 补测启动（18 个月，purge 25，lastday_agg，**含 T4**，workers=2）═══"
log "  启动配置：$(nproc) 核 ｜ TF intraop=$(grep -m1 lstm_tf_intraop_threads sequoia_x/model_selection_v2/config.py | grep -oE '[0-9]+')"

T0=$(date +%s)
env -u KMP_AFFINITY -u OMP_NUM_THREADS "$PY" -u scripts/build_prediction_cache.py \
  --start-month 2025-01 --end-month 2026-06 \
  --feature-view lastday_agg \
  --output "$OUT" >> "$LOG" 2>&1
RC=$?
T1=$(date +%s)
kill $MON 2>/dev/null

D=$(ls "$TMP"/month_*.json 2>/dev/null | wc -l)
log "═══ 结束 exit=$RC ｜ 耗时 $(( (T1-T0)/60 ))min ｜ 完成 $D/18 个月 ═══"
[ -f "$OUT" ] && log "  输出: $OUT" || log "  ⚠️ 未产出输出文件"
