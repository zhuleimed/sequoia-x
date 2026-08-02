#!/bin/bash
# ════════════════════════════════════════════════════════════
#  72 组全量回测并行启动器（2026-08-02）
#
#  组定义：4 TOP_N × 3 时段 × 6 风控 = 72 组
#    TOP_N: 10/15/20/25
#    时段:  全周期(2020-09~2026-06) / 2025年(2025-01~12) / 2026年(2026-01~06)
#    风控:  M0~M5
#  并行:   24 进程（每组 ~1-2 核，36 核机器充裕；内存 24×2GB=48GB）
#  用法:   bash scripts/launch_backtest_parallel.sh
#  结果:   output/backtest_v2/details_{mode}_top{n}_{period}.json + 汇总脚本
# ════════════════════════════════════════════════════════════
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# 可用环境变量覆盖：TOP_N_OVERRIDE / MODES_OVERRIDE / PERIODS_OVERRIDE（;分隔）
TOP_N_LIST=(${TOP_N_OVERRIDE:-10 15 20 25})
# 格式: 名称|起始月|结束月
PERIODS=(${PERIODS_OVERRIDE:-"全周期|2020-09|2026-06" "2021-2026|2021-01|2026-06" "2020年|2020-09|2020-12" "2021年|2021-01|2021-12" "2022年|2022-01|2022-12" "2023年|2023-01|2023-12" "2024年|2024-01|2024-12" "2025年|2025-01|2025-12" "2026年|2026-01|2026-06"})
MODES=(${MODES_OVERRIDE:-M0 M1 M2 M3 M4 M5})
FUSION="${1:-pred_std}"    # 可传 ic_weighted 对比用
MAX_PARALLEL="${2:-24}"

LOG_DIR="logs/bt_parallel"
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "  72 组并行回测启动 | $(date '+%Y-%m-%d %H:%M:%S')"
echo "  融合方法: $FUSION | 并行: $MAX_PARALLEL"
echo "=============================================="

total=0
for top_n in "${TOP_N_LIST[@]}"; do
    for pdef in "${PERIODS[@]}"; do
        IFS='|' read -r pname pstart pend <<< "$pdef"
        for mode in "${MODES[@]}"; do
            total=$((total + 1))
            # 并发控制
            while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
                sleep 3
            done
            nohup /home/zhulei/anaconda3/envs/zhulei_py312/bin/python -u scripts/run_comprehensive_backtest.py \
                --period "$pname" --top-n "$top_n" --mode "$mode" \
                --start-month "$pstart" --end-month "$pend" \
                --fusion-method "$FUSION" \
                > "$LOG_DIR/bt_${mode}_top${top_n}_${pname}.log" 2>&1 &
            echo "  [start] $mode top$top_n $pname (PID=$!)"
        done
    done
done

echo "  共 $total 组，等待全部完成..."
wait
echo "=============================================="
echo "  全部完成: $total 组 | $(date '+%H:%M:%S')"
echo "  日志: $LOG_DIR/"
echo "=============================================="
