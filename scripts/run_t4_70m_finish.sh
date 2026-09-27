#!/bin/bash
# 70 月含 T4 缓存：补完 + 分析（2026-09-27 无人值守修复）
#
# 为什么需要它
# ------------
# `run_t4_70m_random.sh`（4 workers × 8 线程）在 10:46 结束时只有 **57/70 个月**：
# 12 个月（2025-01~2026-06 段）在 Step5 特征构建时崩了，日志报
#     [mutex.cc : 2442] RAW: Check w->waitp->cond == nullptr failed
#     BrokenProcessPool: A process in the process pool was terminated abruptly
# 这是 **TF/absl 在 `fork()` 子进程时的经典崩溃**：worker 已经加载并跑过 TensorFlow
# （T4 训练），随后 `ProcessPoolExecutor`（默认 fork）再派生 8 个特征构建子进程，
# 子进程继承了父进程里处于中间状态的 absl 互斥量 → 断言失败 → 子进程猝死。
# 并发越高越容易撞上：18 月任务 2 workers 时 1/18 失败；本任务 4 workers 时 12/52。
#
# 本脚本做什么
# ------------
# **循环重跑同一条构建命令直到 70 个月齐全**（脚本是断点续跑，已完成月自动跳过）。
# 唯一调整：`V4_BT_WORKERS` 4 → 2（**调度参数，不改变任何模型数值**，只为降低
# 并发 fork×TF 的碰撞概率）。
#
# ⚠️ 可比性参数一律**不动**（用户明确要求 + 项目铁律）：
#    V2_TRAIN_SAMPLE_MODE=random ｜ V2_TRAIN_PURGE_DAYS=25 ｜ TF_NUM_INTRAOP_THREADS=8
#    ｜ --feature-view lastday_agg ｜ 月份范围 2020-09~2026-06
#    这样新补的 13 个月与已建的 57 个月**口径完全一致**。
#
# 长期建议（不在本脚本内做，需人工评估）：把特征构建的 `ProcessPoolExecutor` 换成
# `mp_context=multiprocessing.get_context("spawn")` —— 子进程不再继承 TF 状态，
# 可从根上消除该崩溃。但那是**生产代码改动**（月度重训链共用），且 spawn 需 pickle
# 任务参数、可能变慢，故只记录、不擅自改。
#
# 用法：nohup bash scripts/run_t4_70m_finish.sh > logs/t4_70m_finish_wrapper.log 2>&1 &
set -u
PROJ=/public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x
PY=/home/zhulei/anaconda3/envs/zhulei_py312/bin/python
cd "$PROJ" || exit 1
LOG=logs/t4_70m_random.log
OUT=output/backtest_v2/.t4_70m_random.json
PROG=logs/t4_70m_finish_progress.log
TOTAL=70
MAX_ROUNDS=6

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$PROG"; }
count_months(){ "$PY" -c "
import json,sys
try:
    d=json.load(open('$OUT')); print(len(d))
except Exception: print(0)"; }

# 口径参数（与主任务一致，不得改动）
export V4_BT_WORKERS=2          # ← 仅此一项与主任务不同（调度参数）
export TF_NUM_INTRAOP_THREADS=8
export TF_NUM_INTEROP_THREADS=4
export OMP_NUM_THREADS=8
export V2_TRAIN_SAMPLE_MODE=random
export V2_TRAIN_PURGE_DAYS=25

log "═══ 70 月 T4 缓存补完启动（workers=2，其余口径不变）═══"
PREV=-1
for R in $(seq 1 $MAX_ROUNDS); do
  N=$(count_months)
  log "第 $R 轮开始：当前 $N/$TOTAL 个月"
  if [ "$N" -ge $TOTAL ]; then log "✅ 已齐全，跳过构建"; break; fi
  if [ "$N" -le "$PREV" ] && [ "$R" -gt 1 ]; then
    log "⚠️ 本轮无进展（$PREV → $N）→ 停止重试，留待人工处理"
    break
  fi
  PREV=$N
  env -u KMP_AFFINITY \
      TF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=4 OMP_NUM_THREADS=8 \
      V4_BT_WORKERS=2 V2_TRAIN_SAMPLE_MODE=random V2_TRAIN_PURGE_DAYS=25 \
      "$PY" -u scripts/build_prediction_cache.py \
    --start-month 2020-09 --end-month 2026-06 \
    --feature-view lastday_agg --output "$OUT" >> "$LOG" 2>&1
  log "  第 $R 轮结束 exit=$? ｜ 现有 $(count_months)/$TOTAL 个月"
done

N=$(count_months)
log "═══ 构建阶段收尾：$N/$TOTAL 个月 ═══"

# ── 分析（无论是否 70 个月都跑；若是 57~69 个月，报告里会注明覆盖月数）──
if [ "$N" -ge 60 ]; then
  log "── 自动分析（T2 / T4 多周期+中性化 + T4 vs T2）──"
  "$PY" -u experiments/attribution_4x/ic_by_horizon.py --cache="$OUT" --tag=t2_70m_random --field=t2 >> "$PROG" 2>&1
  "$PY" -u experiments/attribution_4x/ic_by_horizon.py --cache="$OUT" --tag=t4_70m_random --field=t4 >> "$PROG" 2>&1
  "$PY" -u experiments/attribution_4x/t4_ic.py --cache="$OUT" --tag=70m_random >> "$PROG" 2>&1
  log "  分析完成（覆盖月数 $N）"
else
  log "⚠️ 仅 $N 个月，未达分析门槛（60）"
fi
