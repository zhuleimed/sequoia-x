#!/bin/bash
# V4 129维 70个月回测：构建预测缓存 → 并行回测（nohup 脱离会话，断点续跑，进度微信推送）
# 2026-08-20: 验证 129维 vs 旧121维 收益不劣化（同 2020-09~2026-06 口径公平对比）
# 用法: nohup bash scripts/run_v4_60month_backtest.sh > logs/v4_60m_bt.log 2>&1 &
PROJ=$(cd "$(dirname "$0")/.." && pwd)
PY=/home/zhulei/anaconda3/envs/zhulei_py312/bin/python
OUT="$PROJ/output/backtest_v2/prediction_cache_v4_60m.json"
TMPDIR="$PROJ/output/backtest_v2/.cache_tmp_prediction_cache_v4_60m"
cd "$PROJ"
log(){ echo "[$(date '+%F %T')] $1"; }
notify(){ "$PY" scripts/notify_wechat.py "$1" 2>/dev/null || true; }

# ── 进度监控器（后台）: 每 15min 统计已完成月份 temp 文件数 → 推送 ──
MONITOR_PID=""
if [ "${SKIP_PROGRESS:-0}" != "1" ]; then
    (
        TOTAL=70   # 2020-09~2026-06 = 70个月, 与旧体系最终最优配置同口径对比
        while true; do
            sleep 900   # 15min
            DONE=$(ls "$TMPDIR"/month_*.json 2>/dev/null | wc -l)
            ERR=$(ls "$TMPDIR"/month_*.error 2>/dev/null | wc -l)
            if [ "$DONE" -ge "$TOTAL" ]; then break; fi
            pct=$(( DONE * 100 / TOTAL ))
            notify "⏳ V4 70月回测进度: 已构建 $DONE/$TOTAL 个月 ($pct%)，失败月份 $ERR。（断点续跑，继续中）"
            # 避免推送过密——若已无进展则不重复推（脚本退出后结束）
        done
    ) &
    MONITOR_PID=$!
    log "进度监控器启动 PID=$MONITOR_PID"
fi

notify "🚀 V4 129维 70个月回测启动（2020-09~2026-06，与旧体系同口径）。后端预测缓存构建+并行回测，每15分钟推进度，完成后推最终结果。"

# ── 阶段1: 构建 70 个月 V4 预测缓存（多进程24/断点续跑）──
log "阶段1: 构建 70 个月 V4 预测缓存 -> $OUT"
START_T=$(date +%s)
PYTHONPATH=$PROJ "$PY" -u scripts/build_prediction_cache.py \
  --start-month 2020-09 --end-month 2026-06 --skip-t4 \
  --output "$OUT" >> logs/v4_60m_bt.log 2>&1
RC=$?
ELAPSED=$(( ($(date +%s) - START_T) / 60 ))
if [ $MONITOR_PID ]; then kill $MONITOR_PID 2>/dev/null; fi  # 结束进度监控
log "阶段1 预测缓存: exit=$RC, 耗时 ${ELAPSED}min"
if [ $RC -ne 0 ] || [ ! -f "$OUT" ]; then
    notify "❌ V4 60月预测缓存构建失败(exit=$RC, ${ELAPSED}min)。断点续跑: 重跑同命令即跳过已完成月。"
    exit 1
fi
notify "✅ V4 60月预测缓存构建完成(${ELAPSED}min, $(ls "$TMPDIR"/month_*.json 2>/dev/null | wc -l)个月)。开始并行回测..."

# ── 阶段2: 并行回测（共享 V4 预测缓存, 72组配置）──
log "阶段2: 并行回测（2020-09~2026-06, 多风控/TOP_N 配置）..."
"$PY" -u scripts/run_shared_backtest.py --all --cache "$OUT" \
  >> logs/v4_60m_bt.log 2>&1
RC2=$?
log "阶段2 回测: exit=$RC2"
if [ $RC2 -eq 0 ]; then
    notify "✅ V4 129维 70个月回测完成。结果见 output/backtest_v2/，对比旧121维评估收益是否不劣化。"
else
    notify "⚠️ V4 70月回测 exit=$RC2（部分配置可能完成，见回测 CSV/JSON）。"
fi
log "═══ V4 70月回测链结束 ═══"
