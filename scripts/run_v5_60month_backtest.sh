#!/bin/bash
# V5 129维 70个月回测（2026-09-26）：与 V4 同口径对照，回答「V5 到底值多少」
#
# 与 run_v4_60month_backtest.sh 的三处关键差异（照抄会出错）：
#   1) **不设 V4_SAMPLE_END_FIX=2026-08-19** —— 那是 V4 用来命中旧 129 维缓存的；
#      V5 必须用 v5 缓存 3ccb10244903（hash 由 cfg.sample_end=2026-09-24 决定）。
#      这里显式钉成 2026-09-24（= DB 最后交易日 = v5 缓存构建时的取值），保证可复现。
#   2) 阶段2 必须带 **--period full** —— 记忆「V4回测70月口径铁律」记载：不带它
#      run_shared_backtest 会用 PERIODS（只有 11 个月），出现「缓存70月≠回测70月」。
#      老脚本只传了 --all，正是 8/20 那次踩的坑。
#   3) 输出换名，不覆盖任何既有结果。
#
# 用法: nohup bash scripts/run_v5_60month_backtest.sh > logs/v5_60m_bt.log 2>&1 &
PROJ=$(cd "$(dirname "$0")/.." && pwd)
PY=/home/zhulei/anaconda3/envs/zhulei_py312/bin/python
# 2026-09-26 参数化：支持切换特征视图（默认 full = 原样，OUT 路径不变）
#   用法：FEATURE_VIEW=lastday_agg VIEW_TAG=v5_387 [OUTDIR=...] bash scripts/run_v5_60month_backtest.sh
FEATURE_VIEW="${FEATURE_VIEW:-full}"
VIEW_TAG="${VIEW_TAG:-v5}"
OUT="$PROJ/output/backtest_v2/prediction_cache_${VIEW_TAG}_60m.json"
TMPDIR="$PROJ/output/backtest_v2/.cache_tmp_prediction_cache_${VIEW_TAG}_60m"
OUTDIR="${OUTDIR:-$PROJ/output/backtest_v5}"
cd "$PROJ"
log(){ echo "[$(date '+%F %T')] $1"; }
notify(){ "$PY" scripts/notify_wechat.py "$1" 2>/dev/null || true; }

# v5 缓存对应的 sample_end（= DB 最后交易日；9/25 中秋休市故仍为 09-24）
export V4_SAMPLE_END_FIX="2026-09-24"

MONITOR_PID=""
if [ "${SKIP_PROGRESS:-0}" != "1" ]; then
    (
        TOTAL=70
        LAST_PUSH_HOUR=$(date +%-H)
        while true; do
            H=$(date +%-H)
            PUSH=0
            if [ "$H" -ge 7 ] && [ "$H" -le 23 ]; then PUSH=1; fi
            if [ "$H" -eq 6 ]; then PUSH=2; fi
            if [ "$PUSH" != "0" ] && [ "$H" != "$LAST_PUSH_HOUR" ]; then
                DONE=$(ls "$TMPDIR"/month_*.json 2>/dev/null | wc -l)
                ERR=$(ls "$TMPDIR"/month_*.error 2>/dev/null | wc -l)
                pct=$(( DONE * 100 / TOTAL ))
                if [ "$PUSH" = "2" ]; then
                    notify "🌙 V5 70月回测【夜间汇总】: $DONE/$TOTAL 个月 ($pct%)，失败 $ERR。"
                else
                    notify "⏳ V5 70月回测【${H}:00】: $DONE/$TOTAL 个月 ($pct%)，失败 $ERR。"
                fi
                LAST_PUSH_HOUR="$H"
            fi
            if [ "$DONE" -ge "$TOTAL" ]; then break; fi
            sleep 60
        done
    ) &
    MONITOR_PID=$!
    log "进度监控器启动 PID=$MONITOR_PID"
fi

notify "🚀 V5 70个月回测启动（2020-09~2026-06）特征视图=${FEATURE_VIEW}。"

# ── 阶段1: 70 个月 V5 预测缓存（断点续跑）──
log "阶段1: 构建 70 个月 V5 预测缓存 -> $OUT"
START_T=$(date +%s)
PYTHONPATH=$PROJ "$PY" -u scripts/build_prediction_cache.py \
  --start-month 2020-09 --end-month 2026-06 --skip-t4 \
  --feature-view "$FEATURE_VIEW" \
  --output "$OUT" >> "logs/v5_60m_bt_${VIEW_TAG}.log" 2>&1
RC=$?
ELAPSED=$(( ($(date +%s) - START_T) / 60 ))
if [ $MONITOR_PID ]; then kill $MONITOR_PID 2>/dev/null; fi
log "阶段1 预测缓存: exit=$RC, 耗时 ${ELAPSED}min"
if [ $RC -ne 0 ] || [ ! -f "$OUT" ]; then
    notify "❌ V5 70月预测缓存构建失败(exit=$RC,${ELAPSED}min)。重跑同命令即断点续跑。"
    exit 1
fi
notify "✅ V5 预测缓存完成(${ELAPSED}min)。开始 70 月并行回测..."

# ── 阶段2: 并行回测（**必须 --period full**）──
log "阶段2: 并行回测（--period full = 2020-09~2026-06）..."
"$PY" -u scripts/run_shared_backtest.py --all --period full --cache "$OUT" \
  --output-dir "$OUTDIR" \
  >> "logs/v5_60m_bt_${VIEW_TAG}.log" 2>&1
RC2=$?
log "阶段2 回测: exit=$RC2"
if [ $RC2 -eq 0 ]; then
    notify "✅ V5 129维 70个月回测完成。结果在 output/backtest_v2/（与 summary_all.csv 的 V4 基准对照）。"
else
    notify "⚠️ V5 70月回测 exit=$RC2（部分配置可能已完成）。"
fi
