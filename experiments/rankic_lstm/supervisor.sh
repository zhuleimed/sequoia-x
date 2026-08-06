#!/bin/bash
# V3 方向三无人值守监督链（2026-08-06）
# 阶段1: 等待单月验证完成（.tmp/rankic_lstm/2026-06.json）
# 阶段2: 校验单月结果有效性（预测存在 + 无 NaN + std>0）→ 启动 70 个月全量
# 阶段3: 等待全量完成（70 个文件）→ 自动跑 --analyze IC 对照
# 全程 nohup 解绑（PPID=1），退出 Claude Code 不影响
# 使用: nohup bash experiments/rankic_lstm/supervisor.sh > /dev/null 2>&1 &

set -u
PY=/home/zhulei/anaconda3/envs/zhulei_py312/bin/python
ROOT=/public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x
cd "$ROOT" || exit 1
LOG="$ROOT/logs/exp_rankic_supervisor_20260806.log"
TMP="$ROOT/.tmp/rankic_lstm"

echo "[supervisor] ═══ 启动 $(date) ═══" >> "$LOG"

# ── 阶段 1：等待单月验证完成（最长 3h）──
for i in $(seq 1 180); do
    [ -f "$TMP/2026-06.json" ] && break
    sleep 60
done

if [ ! -f "$TMP/2026-06.json" ]; then
    echo "[supervisor] ❌ 单月验证 3h 超时未完成 → 终止 $(date)" >> "$LOG"
    exit 1
fi
echo "[supervisor] ✅ 单月验证文件已生成 $(date)" >> "$LOG"

# ── 阶段 2：校验单月结果有效性（铁律五：输出验证，坏结果不让通过）──
if ! "$PY" -c "
import json, numpy as np
d = json.load(open('$TMP/2026-06.json'))
assert d.get('rankic') and d.get('huber'), '缺少预测字段'
r = np.array(d['rankic']); h = np.array(d['huber'])
assert not np.isnan(r).any() and not np.isnan(h).any(), '预测含 NaN'
assert np.std(r) > 1e-7 and np.std(h) > 1e-7, '预测无方差'
print(f'单月验证有效: n={d[\"n\"]} rankic_std={np.std(r):.4f} huber_std={np.std(h):.4f}')
" >> "$LOG" 2>&1; then
    echo "[supervisor] ❌ 单月验证数据异常 → 终止（需人工检查）$(date)" >> "$LOG"
    exit 1
fi

# 等单月验证进程彻底退出（防文件竞争）
sleep 30

# ── 阶段 3：启动 70 个月全量（16 workers）；已在跑则跳过（2026-08-07 防重复启动）──
if pgrep -f "experiment_rankic.py" > /dev/null; then
    echo "[supervisor] 全量实验已在运行，跳过启动 $(date)" >> "$LOG"
else
    echo "[supervisor] 🚀 启动 70 个月全量 (16 workers) $(date)" >> "$LOG"
    env -u KMP_AFFINITY -u OMP_NUM_THREADS nohup "$PY" experiments/rankic_lstm/experiment_rankic.py --workers 16 \
        >> "$ROOT/logs/exp_rankic_20260807_v2.log" 2>&1 &
    FULL_PID=$!
    echo "[supervisor] 全量 PID=$FULL_PID" >> "$LOG"
fi

# ── 阶段 4：等待全量完成（70 个文件，最长 14h）──
for i in $(seq 1 840); do
    cnt=$(ls "$TMP"/*.json 2>/dev/null | wc -l)
    [ "$cnt" -ge 70 ] && break
    if [ $((i % 10)) -eq 0 ]; then
        echo "[supervisor] 等待中... 已完成 $cnt/70 ($(date))" >> "$LOG"
    fi
    sleep 60
done

cnt=$(ls "$TMP"/*.json 2>/dev/null | wc -l)
echo "[supervisor] 全量结束: $cnt/70 个月 $(date)" >> "$LOG"
if [ "$cnt" -lt 70 ]; then
    echo "[supervisor] ⚠️ 未满 70——检查断点续跑可补" >> "$LOG"
fi

# ── 阶段 5：自动 IC 对照分析 ──
echo "[supervisor] 📊 开始 IC 对照分析 $(date)" >> "$LOG"
env -u KMP_AFFINITY -u OMP_NUM_THREADS "$PY" experiments/rankic_lstm/experiment_rankic.py --analyze >> "$LOG" 2>&1
echo "[supervisor] ✅ 分析完成 $(date) — 结果见 $ROOT/output/backtest_v2/experiments/rankic_lstm_ic_report.csv" >> "$LOG"
echo "[supervisor] ═══ 监督链结束 $(date) ═══" >> "$LOG"
