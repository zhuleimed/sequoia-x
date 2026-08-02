#!/bin/bash
# ════════════════════════════════════════════════════════════════
#  T4 LSTM 并行月度训练启动器
#
#  功能：
#    - 并行启动 11 个月份的 T4 Worker（每月一个独立进程）
#    - 使用 nohup 解绑终端，关闭 SSH 后继续运行
#    - 支持断点续跑：已完成月份自动跳过
#    - 完成后自动合并结果
#
#  用法：
#    bash scripts/launch_t4_parallel.sh          # 启动
#    bash scripts/launch_t4_parallel.sh --status # 查看进度
#    bash scripts/launch_t4_parallel.sh --merge  # 手动合并
#    bash scripts/launch_t4_parallel.sh --kill   # 终止所有
# ════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WORKER="$SCRIPT_DIR/t4_monthly_worker.py"
LOG_DIR="$PROJECT_DIR/output/backtest_v2/.t4_logs"
TMP_DIR="$PROJECT_DIR/output/backtest_v2/.t4_tmp"
PID_DIR="$PROJECT_DIR/output/backtest_v2/.t4_pids"

# 月份列表：动态生成 2020-08 ~ 2026-06（71 个月）
MONTHS=()
_y=2020; _m=8
while [[ $_y -lt 2026 || ($_y -eq 2026 && $_m -le 6) ]]; do
    MONTHS+=("$(printf '%04d-%02d' $_y $_m)")
    _m=$((_m + 1))
    if [[ $_m -gt 12 ]]; then _m=1; _y=$((_y + 1)); fi
done

# 并发上限：36核 / (2+1+2线程) ≈ 7；实测每 worker ~1.3 核 → 12 并发安全（与 build 16 进程共用 36 核）
MAX_PARALLEL=12

CMD="${1:-start}"

# ── 辅助函数 ──

status_all() {
    echo "=============================================="
    echo "  T4 并行训练状态 ($(date '+%Y-%m-%d %H:%M:%S'))"
    echo "=============================================="
    local total=0 ok=0 running=0 pending=0 failed=0
    for m in "${MONTHS[@]}"; do
        total=$((total + 1))
        if [[ -f "$TMP_DIR/t4_${m}.json" ]]; then
            ok=$((ok + 1))
            printf "  [✓] %s  已完成\n" "$m"
        elif [[ -f "$PID_DIR/t4_${m}.pid" ]]; then
            local pid
            pid=$(cat "$PID_DIR/t4_${m}.pid")
            if kill -0 "$pid" 2>/dev/null; then
                running=$((running + 1))
                printf "  [▶] %s  运行中 (PID=%s)\n" "$m" "$pid"
            else
                failed=$((failed + 1))
                printf "  [✗] %s  进程已退出 (曾PID=%s)\n" "$m" "$pid"
            fi
        else
            pending=$((pending + 1))
            printf "  [ ] %s  待启动\n" "$m"
        fi
    done
    echo "----------------------------------------------"
    echo "  总计=$total 已完成=$ok 运行中=$running 待启动=$pending 失败=$failed"
    echo "=============================================="

    # 显示最近日志摘要
    if [[ $running -gt 0 ]] || [[ $ok -gt 0 ]]; then
        echo ""
        echo "最近训练日志:"
        for m in "${MONTHS[@]}"; do
            local log="$LOG_DIR/t4_${m}.log"
            if [[ -f "$log" ]]; then
                local last
                last=$(tail -1 "$log" 2>/dev/null | cut -c1-120)
                [[ -n "$last" ]] && printf "  %s: %s\n" "$m" "$last"
            fi
        done
    fi
}

kill_all() {
    echo "终止所有 T4 Worker..."
    for m in "${MONTHS[@]}"; do
        if [[ -f "$PID_DIR/t4_${m}.pid" ]]; then
            local pid
            pid=$(cat "$PID_DIR/t4_${m}.pid")
            if kill "$pid" 2>/dev/null; then
                echo "  已终止 $m (PID=$pid)"
            fi
            rm -f "$PID_DIR/t4_${m}.pid"
        fi
    done
    echo "完成"
}

merge_results() {
    echo "合并 T4 预测结果到 prediction_cache.json..."
    cd "$PROJECT_DIR"
    python -c "
import json
from pathlib import Path

cache_path = Path('output/backtest_v2/prediction_cache.json')
tmp_dir = Path('output/backtest_v2/.t4_tmp')
cache = json.loads(cache_path.read_text())

merged = 0
skipped = 0
for tf in sorted(tmp_dir.glob('t4_*.json')):
    m = tf.stem[len('t4_'):]
    entry = json.loads(tf.read_text())
    if m in cache:
        # 只更新 t4 字段（按主缓存 symbols 对齐），保留 t2/t1/t3
        t4_map = dict(zip(entry.get('symbols', []), entry.get('t4', [])))
        cache[m]['t4'] = [t4_map.get(s, 0.0) for s in cache[m]['symbols']]
        merged += 1
        print(f'  [✓] {m}')
    else:
        skipped += 1
        print(f'  [!] {m}: 主缓存无此月（build 未完成），跳过——全部完成后重跑 --merge')

cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
print(f'合并完成: {merged} merged, {skipped} skipped')
"
}

launch() {
    # 创建目录
    mkdir -p "$LOG_DIR" "$TMP_DIR" "$PID_DIR"

    echo "=============================================="
    echo "  T4 LSTM 并行月度训练启动"
    echo "  启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  并行进程: ${#MONTHS[@]} 个月份"
    echo "=============================================="

    # 验证 Worker 脚本存在
    if [[ ! -f "$WORKER" ]]; then
        echo "错误: Worker 脚本不存在: $WORKER"
        exit 1
    fi

    local launched=0 skipped=0
    for m in "${MONTHS[@]}"; do
        # 断点续跑：跳过已完成月份
        if [[ -f "$TMP_DIR/t4_${m}.json" ]]; then
            echo "  [skip] $m (已完成)"
            skipped=$((skipped + 1))
            continue
        fi

        # 清理旧 PID 文件（如果进程已死）
        if [[ -f "$PID_DIR/t4_${m}.pid" ]]; then
            local old_pid
            old_pid=$(cat "$PID_DIR/t4_${m}.pid")
            if ! kill -0 "$old_pid" 2>/dev/null; then
                rm -f "$PID_DIR/t4_${m}.pid"
            else
                echo "  [skip] $m (已在运行, PID=$old_pid)"
                continue
            fi
        fi

        # 并发控制：等待当前运行数 < MAX_PARALLEL（防止超 36 核）
        while :; do
            local running_count=0
            for pid_file in "$PID_DIR"/*.pid; do
                [[ -e "$pid_file" ]] || continue
                local p
                p=$(cat "$pid_file")
                kill -0 "$p" 2>/dev/null && running_count=$((running_count + 1))
            done
            if [[ $running_count -lt $MAX_PARALLEL ]]; then
                break
            fi
            echo "  [wait] 并发已达 $MAX_PARALLEL，等待空位..."
            sleep 60
        done

        # 启动 Worker（nohup 解绑；env -u KMP_AFFINITY 防 .bashrc 锁核，见 CLAUDE.md 铁律）
        local log_file="$LOG_DIR/launcher_${m}.log"
        env -u KMP_AFFINITY -u OMP_NUM_THREADS \
            nohup python -u "$WORKER" --month "$m" \
            > "$log_file" 2>&1 &
        local pid=$!
        echo "$pid" > "$PID_DIR/t4_${m}.pid"

        echo "  [start] $m (PID=$pid)"
        launched=$((launched + 1))
    done

    echo "----------------------------------------------"
    echo "  启动=$launched 跳过=$skipped"
    echo "  PID 目录: $PID_DIR"
    echo "  日志目录: $LOG_DIR"
    echo "=============================================="
    echo ""
    echo "监控命令:"
    echo "  bash scripts/launch_t4_parallel.sh --status"
    echo "  tail -f $LOG_DIR/t4_*.log"
    echo ""
    echo "合并命令（所有完成后执行）:"
    echo "  bash scripts/launch_t4_parallel.sh --merge"
}

# ── 主逻辑 ──

cd "$PROJECT_DIR"

case "$CMD" in
    start|launch)
        launch
        ;;
    --status|-s|status)
        status_all
        ;;
    --merge|-m|merge)
        merge_results
        ;;
    --kill|-k|kill|stop)
        kill_all
        ;;
    *)
        echo "用法: bash scripts/launch_t4_parallel.sh [start|--status|--merge|--kill]"
        echo ""
        echo "  start    启动并行训练（默认）"
        echo "  --status 查看进度"
        echo "  --merge  合并结果到 prediction_cache.json"
        echo "  --kill   终止所有 Worker"
        exit 1
        ;;
esac
