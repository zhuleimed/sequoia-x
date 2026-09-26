#!/bin/bash
# T4 补测·公平版（2026-09-27 日）：把训练样本选择从 tail 换成 random
#
# 为什么再测一次
# --------------
# 09-26 夜的 T4 补测（run_t4_check.sh）走默认 `V2_TRAIN_SAMPLE_MODE=tail`：
#   训练窗口 = 最近 12 个月（build_prediction_cache.py:537-541），purge 25 后约
#   7.0 万样本 / 约 25 个采样日（146 采样日共 412,486 样本 ≈ 2,825 条/日），
#   而 tail 只取**时间上最后的 5000 条 ≈ 最后 1.8 个采样日**（约两个市场截面）。
#   让 LSTM 在 ~2 个截面上学序列模式基本是退化的 —— 它几乎没有"多样本"可学。
#   所以那次测得的 T4 IC≈0 **不能当作"LSTM 没能力"的证据**（口径不公平）。
#
# 本脚本只改一件事：`V2_TRAIN_SAMPLE_MODE=random`（**同样的 12 个月窗口、同样 5000 条**，
# 只把"取最后 5000 条"换成"窗口内随机抽 5000 条"，固定种子 42）。
# → 这正是泄漏排查时用过的"同窗口同样本量、只换取样规则"对照，才与路径 B 可比。
# purge 25 照旧默认开启（去泄漏不变），因此不是"为了好看而放开泄漏"。
#
# 前置：必须先把 data/models/v2_selection/t4_checkpoint_cache_*.keras 里的旧
#   checkpoint 挪走 —— `deep_lstm.py:574` 只要文件存在就 load_model 并**直接 return**
#   （断点续跑机制），否则本次会原样复用 09-26 那一批 tail 模型 = 空跑。
#   本脚本启动前会自检该目录为空，非空则中止。
#
# 口径：2025-01 ~ 2026-06（18 个月）｜purge 25｜lastday_agg｜含 T4｜workers=2
# 断点续跑：崩溃后**重跑同命令**即跳过已完成月（month_*.json 已存）
#
# 用法：
#   nohup bash scripts/run_t4_check_random.sh > logs/t4_check_random_wrapper.log 2>&1 &
# 进度：
#   tail -f logs/t4_test_18m_random.log
#   cat  logs/t4_check_random_progress.log
set -u
PROJ=/public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x
PY=/home/zhulei/anaconda3/envs/zhulei_py312/bin/python
cd "$PROJ" || exit 1
LOG=logs/t4_test_18m_random.log
OUT=output/backtest_v2/.t4_test_18m_random.json
TMP=output/backtest_v2/.cache_tmp_.t4_test_18m_random
PROG=logs/t4_check_random_progress.log
CKPT=data/models/v2_selection

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$PROG"; }

export V4_BT_WORKERS=2
export V2_TRAIN_SAMPLE_MODE=random      # ← 本次唯一变量（同样 5000 条，只换抽法）
export V2_TRAIN_PURGE_DAYS=25           # 显式写死：去泄漏，不因换抽样而放开

# ── 前置自检：只在**首次启动**时拦截"别的口径遗留的 checkpoint" ──
# 若 $LOG 已存在，说明本脚本先前跑过，这些 checkpoint 是它自己产出的 →
# 属于正常断点续跑（铁律二：同命令恢复），放行。
if [ ! -f "$LOG" ]; then
  LEFT=$(ls "$CKPT"/t4_checkpoint_cache_2025-*.keras \
            "$CKPT"/t4_checkpoint_cache_2026-0[1-6].keras 2>/dev/null | wc -l)
  if [ "$LEFT" -gt 0 ]; then
    log "❌ 中止：$CKPT 下仍有 $LEFT 个 2025-01~2026-06 的 t4_checkpoint_cache_*.keras"
    log "   deep_lstm.py:574 会直接 load 它们并跳过训练 → 本次会复用旧模型（空跑）"
    log "   请先挪走（例：mv 到 _t4_ckpt_backup_<日期>/），再重跑本命令。"
    exit 1
  fi
fi

# ── 独立监控：每 15 分钟写一次进度快照 ──
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

log "═══ T4 公平版补测启动（18 个月，purge 25，lastday_agg，**random 抽样**，workers=2）═══"
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

# ── 自动分析（缓存建好后立即出 IC，省一次手工步骤）──
if [ -f "$OUT" ]; then
  log "── 自动分析 ──"
  "$PY" -u experiments/attribution_4x/t4_ic.py --cache="$OUT" --tag=random >> "$PROG" 2>&1
  log "  分析完成 → experiments/attribution_4x/out/t4_ic_random.csv"
fi
