#!/bin/bash
# ════════════════════════════════════════════════════════════
#  完整自动管线监督（2026-08-01 重写，三阶段）：
#
#  阶段1: 等待当前 T2/T1/T3 构建进程退出（处理 2023-01~2025-07）
#  阶段2: 补建 2025-08~2026-06 的 T2/T1/T3（旧11个月用旧缓存，已删除需重建）
#  阶段3: 全部完成 → 启动 T4 71 个月并行训练
#
#  用法：
#    nohup bash scripts/pipeline_t4_auto_start.sh > logs/pipeline_t4_auto.log 2>&1 &
# ════════════════════════════════════════════════════════════

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CACHE_FILE="$PROJECT_DIR/output/backtest_v2/prediction_cache.json"
LAUNCHER="$SCRIPT_DIR/launch_t4_parallel.sh"
BUILD_SCRIPT="$SCRIPT_DIR/build_prediction_cache.py"

TARGET_MONTH="2026-06"    # 全部完成标志
POLL_INTERVAL=120          # 每 2 分钟
MAX_WAIT_HOURS=24

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

wait_for_month() {
    # 等待 TARGET_MONTH 出现在缓存（且含 t2）
    local start_ts=$(date +%s)
    while :; do
        local has_month="no"
        if [[ -f "$CACHE_FILE" ]]; then
            has_month=$(python - "$CACHE_FILE" "$TARGET_MONTH" << 'PYEOF'
import json, sys
try:
    c = json.load(open(sys.argv[1]))
    m = sys.argv[2]
    print("yes" if m in c and "t2" in c[m] else "no")
except Exception:
    print("no")
PYEOF
)
        fi
        if [[ "$has_month" == "yes" ]]; then
            return 0
        fi
        local now_ts=$(date +%s)
        local elapsed_h=$(( (now_ts - start_ts) / 3600 ))
        if [[ $elapsed_h -ge $MAX_WAIT_HOURS ]]; then
            log "❌ 等待超时（${MAX_WAIT_HOURS}h）"
            return 1
        fi
        log "等待中... ($elapsed_h h, 缓存尚无 $TARGET_MONTH)"
        sleep $POLL_INTERVAL
    done
}

wait_for_pid() {
    # 等待指定进程退出
    local pid="$1"
    local name="$2"
    log "等待 $name (PID=$pid) 退出..."
    while kill -0 "$pid" 2>/dev/null; do
        sleep $POLL_INTERVAL
    done
    log "✅ $name (PID=$pid) 已退出"
}

# ════════════════════════════════════════════════════════════
#  阶段 1：等当前 T2/T1/T3 构建完成
# ════════════════════════════════════════════════════════════
CUR_BUILD_PID="${1:-}"
if [[ -n "$CUR_BUILD_PID" ]] && kill -0 "$CUR_BUILD_PID" 2>/dev/null; then
    wait_for_pid "$CUR_BUILD_PID" "T2/T1/T3 主构建(2023-01~2025-07)"
else
    log "当前无运行中的 T2/T1/T3 构建（或已退出），跳过阶段1"
fi

# ════════════════════════════════════════════════════════════
#  阶段 2：补建 2025-08 ~ 2026-06 的 T2/T1/T3
# ════════════════════════════════════════════════════════════
if [[ "$(python - "$CACHE_FILE" << 'PYEOF'
import json, sys
try:
    c = json.load(open(sys.argv[1]))
    print("yes" if "2025-08" in c else "no")
except Exception:
    print("no")
PYEOF
)" == "yes" ]]; then
    log "2025-08 已在缓存中，跳过补建"
else
    log "🚀 阶段2: 补建 2025-08~2026-06 的 T2/T1/T3 (skip-t4)..."
    nohup python -u "$BUILD_SCRIPT" --start-month 2025-08 --end-month 2026-06 --skip-t4 \
        > "$PROJECT_DIR/logs/pred_cache_backfill_11m.log" 2>&1 &
    BUILD_PID=$!
    log "补建进程: PID=$BUILD_PID"
    wait_for_pid "$BUILD_PID" "补建(2025-08~2026-06)"
fi

# ════════════════════════════════════════════════════════════
#  阶段 3：确认全部完成 → 启动 T4
# ════════════════════════════════════════════════════════════
log "🚀 阶段3: 确认 $TARGET_MONTH 完成..."
if wait_for_month; then
    log "✅ 全部 T2/T1/T3 完成，启动 T4 71 个月训练..."
    bash "$LAUNCHER" start
    log "✅ T4 已启动，监督脚本完成使命退出"
    exit 0
else
    log "❌ 阶段3 超时，监督退出（可手动检查后重跑）"
    exit 1
fi
