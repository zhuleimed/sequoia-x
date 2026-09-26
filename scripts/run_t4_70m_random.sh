#!/bin/bash
# 70 月（2020-09~2026-06）**含 T4** 的 purge 缓存（2026-09-27）
#
# 为什么建它
# ----------
# 到目前为止 **所有 70 月缓存都带 `--skip-t4`**（t4 字段全 0），所以：
#   · T4 的评估只有 18 个月（.t4_test_18m / 公平版 .t4_test_18m_random）
#   · T2 与 T4 从未在 70 月口径上做过公平对比
#   · 今天发现的"中性化后 T2 显著"（ic_by_horizon.py）也无法对 T4 复现
# 本脚本补齐这一块，跑完即可对 T2 / T4 做同口径（同窗口、同样本量、同 purge）的
# 多周期 + 中性化分析。
#
# 口径（与 `run_t4_check_random.sh` 的公平版**完全一致**，便于互相印证）：
#   2020-09 ~ 2026-06 ｜ purge 25 ｜ random 抽样（种子 42）｜ lastday_agg ｜ workers=2 ｜ 含 T4
#
# 两点设计说明
# ------------
# 1) **排队**：先等 `run_t4_check_random.sh`（18 月）跑完再开始。理由：每个 worker 的 TF
#    用 16 个 intraop 线程，两个任务同时跑 = 4×16 = 64 线程挤 36 核，两边一起变慢，
#    总吞吐不增反降。
# 2) **checkpoint 复用**：2025-01~2026-06 这 18 个月，18 月任务已建好的
#    `t4_checkpoint_cache_{月}.keras` 与本任务**同窗口/同数据/同种子 ⇒ 模型完全相同**，
#    直接复用（省约 1/4 时间）。但 2020-2024 段若存在同名 checkpoint 说明是**别的口径**
#    的遗留（deep_lstm.py:574 会静默 load 并跳过训练）→ 首次启动即中止。
#
# 断点续跑：重跑同命令即跳过已完成月（month_*.json）
# 用法：
#   nohup bash scripts/run_t4_70m_random.sh > logs/t4_70m_random_wrapper.log 2>&1 &
set -u
PROJ=/public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x
PY=/home/zhulei/anaconda3/envs/zhulei_py312/bin/python
cd "$PROJ" || exit 1
LOG=logs/t4_70m_random.log
OUT=output/backtest_v2/.t4_70m_random.json
TMP=output/backtest_v2/.cache_tmp_.t4_70m_random
PROG=logs/t4_70m_random_progress.log
CKPT=data/models/v2_selection
TOTAL=70

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$PROG"; }

export V4_BT_WORKERS=2
export V2_TRAIN_SAMPLE_MODE=random
export V2_TRAIN_PURGE_DAYS=25

# ── 1) 排队：等 18 月任务结束 ──
while pgrep -f "run_t4_check_random.sh" > /dev/null 2>&1; do
  log "⏳ 等待 18 月补测结束中（避免 4×16 线程过载）…"
  sleep 300
done

# ── 2) 前置自检（仅首次启动）──
if [ ! -f "$LOG" ]; then
  BAD=$(ls "$CKPT"/t4_checkpoint_cache_202[0-4]-*.keras 2>/dev/null | wc -l)
  if [ "$BAD" -gt 0 ]; then
    log "❌ 中止：$CKPT 下有 $BAD 个 2020-2024 段的 t4_checkpoint_cache_*.keras"
    log "   （deep_lstm.py:574 会直接 load 并跳过训练 → 会静默复用别的口径的模型）"
    exit 1
  fi
fi

# ── 3) 独立监控：每 15 分钟一次进度快照 ──
(
  while true; do
    sleep 900
    D=$(ls "$TMP"/month_*.json 2>/dev/null | wc -l)
    E=$(ls "$TMP"/month_*.error 2>/dev/null | wc -l)
    echo "[$(date '+%F %T')] 进度 $D/$TOTAL 个月，失败 $E ｜ 日志尾: $(tail -1 "$LOG" 2>/dev/null | cut -c1-110)" >> "$PROG"
    if [ "$D" -ge $TOTAL ]; then break; fi
  done
) &
MON=$!

log "═══ 70 月含 T4 补测启动（2020-09~2026-06，purge 25，random，lastday_agg，workers=2）═══"
T0=$(date +%s)
env -u KMP_AFFINITY -u OMP_NUM_THREADS "$PY" -u scripts/build_prediction_cache.py \
  --start-month 2020-09 --end-month 2026-06 \
  --feature-view lastday_agg \
  --output "$OUT" >> "$LOG" 2>&1
RC=$?
T1=$(date +%s)
kill $MON 2>/dev/null

D=$(ls "$TMP"/month_*.json 2>/dev/null | wc -l)
log "═══ 结束 exit=$RC ｜ 耗时 $(( (T1-T0)/60 ))min ｜ 完成 $D/$TOTAL 个月 ═══"

# ── 4) 自动分析：T2 / T4 各一套（多周期 + 中性化）──
if [ -f "$OUT" ]; then
  log "── 自动分析 ──"
  "$PY" -u experiments/attribution_4x/ic_by_horizon.py --cache="$OUT" --tag=t2_70m_random --field=t2 >> "$PROG" 2>&1
  "$PY" -u experiments/attribution_4x/ic_by_horizon.py --cache="$OUT" --tag=t4_70m_random --field=t4 >> "$PROG" 2>&1
  "$PY" -u experiments/attribution_4x/t4_ic.py --cache="$OUT" --tag=70m_random >> "$PROG" 2>&1
  log "  分析完成 → experiments/attribution_4x/out/{ic_by_horizon_t2_70m_random,ic_by_horizon_t4_70m_random,t4_ic_70m_random}.csv"
fi
