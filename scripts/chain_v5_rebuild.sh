#!/bin/bash
# V5 缓存重建链（2026-09-25）——等 amount 回填结束后自动接续
#
# 顺序（每一步都验证上一步，失败即停）：
#   0. 等 amount 真值回填进程退出 → 打印其 summary
#   1. 残差兜底：真值取不到的 NULL 用 close×volume 计算填充（用户 2026-09-25 确认）
#   2. 重建 v5 缓存（feature_version=5）
#   3. 新缓存体检：全零列数应从 35 降到个位
#   4. 干跑验证（隔离输出，不入库不发信号）
#
# ⚠️ 本链**不碰模拟盘**：不调用 v2_monthly_retrain.py（它无条件写买入信号）。
#    真正的选股/写信号交给 10/1 03:00 的 cron 月度重训。
set -u
PROJ=/public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x
PY=/home/zhulei/anaconda3/envs/zhulei_py312/bin/python
cd "$PROJ" || exit 1
log(){ echo "[$(date '+%F %T')] $*"; }

# ── 0. 等回填结束 ──
# ⚠️ 2026-09-25 踩坑记录（`pgrep -f` 自匹配陷阱）：
#   初版写的是 `while pgrep -f "fill_amount_gap_20260925.py"; do sleep 30; done`，
#   结果**永远等不出来** —— `pgrep -f` 匹配的是"完整命令行含该子串"的**任何**进程，
#   连"我用来看进度的诊断命令"都会命中（它命令行里也带着这个文件名），于是自我维持死循环。
#   连改成全路径 "python -u scripts/xxx.py" 也不行：**写这条检查命令的动作本身**
#   就把该字符串放进了自己进程的命令行。
#   项目脚本 run_post_rebuild_verify.sh 早有同款警告：
#     "⚠️ 用 pgrep -f 精确匹配 + grep -v 排除自身；勿用裸 grep"
#   ⇒ 若今后确需等待，用**标记文件**（进程结束时 touch）或 `flock`，
#     不要用 `pgrep -f <文件名字符串>`。
#
# 本次 amount 回填已于 2026-09-25 17:35 完成（剩余缺失 575.6万 → 5.4万行），故无需等待。
log "amount 回填已结束，直接进入后续步骤"
cat data/backup_amount_fill_20260925/summary.json 2>/dev/null || log "⚠️ 未找到 summary.json"

# ── 1. 残差兜底 ──
log "① 残差兜底（close×volume 计算填充）..."
$PY -u scripts/fill_amount_gap_20260925.py --fill-computed 2>&1 | \
  grep -vE "数据库初始化|INFO     sequoia" | tail -14

# ── 2. 重建 v5 ──
log "② 重建 v5 缓存（~3.1h，workers=16）..."
T0=$(date +%s)
$PY -u scripts/rebuild_dataset_cache.py --workers 16 > logs/rebuild_v5_20260925.log 2>&1
RC=$?
T1=$(date +%s)
if [ $RC -ne 0 ]; then
  log "❌ 重建失败 exit=$RC 耗时=$(( (T1-T0)/60 ))min —— 链条中止"
  exit 1
fi
log "✅ 重建完成，耗时 $(( (T1-T0)/60 ))min"

# ── 3. 新缓存体检：全零列数 ──
log "③ 新缓存体检（全零列）..."
$PY - <<'PYEOF'
import glob, json, os
import numpy as np
best = None
for d in glob.glob('data/cache/v2_dataset/*/'):
    mp = os.path.join(d, 'metadata.json')
    if not os.path.exists(mp):
        continue
    m = json.load(open(mp))
    if m.get('params', {}).get('feature_version') != 5:
        continue
    if best is None or m.get('created', '') > best[1]:
        best = (d, m.get('created', ''), m)
if best is None:
    print('  ⚠️ 未找到 feature_version=5 的缓存')
    raise SystemExit(1)
d, created, m = best
X = np.load(os.path.join(d, 'X.npy'), mmap_mode='r')
n = X.shape[0]
blk = np.concatenate([np.asarray(X[i:i+1500]) for i in np.linspace(0, n-1500, 8).astype(int)])
z = (blk == 0.0).mean(axis=(0, 1))
zero_cols = [int(i) for i in np.where(z > 0.999)[0]]
print(f'  目录: {d}')
print(f'  created={created}  X_shape={X.shape}  n_samples={m.get("n_samples")}')
print(f'  全零列 {len(zero_cols)} 个（v4 为 35 个）-> {zero_cols}')
print(f'  col16(turnover_rate) 非零样本占比: {(blk[:,:,16]!=0).any(axis=1).mean()*100:.1f}%')
print(f'  col17(amount_ratio)  非零样本占比: {(blk[:,:,17]!=0).any(axis=1).mean()*100:.1f}%')
print(f'  col37-52(大盘+市场)  非零样本占比: {(blk[:,:,37:53]!=0).any(axis=(1,2)).mean()*100:.1f}%')
PYEOF

# ── 4. 干跑验证 ──
log "④ 干跑验证（隔离输出，不入库不发信号）..."
$PY -u scripts/build_prediction_cache.py --start-month 2026-09 --end-month 2026-09 \
    --skip-t4 --output output/backtest_v2/.dryrun_cache.json \
    > logs/dryrun_v5_20260925.log 2>&1
RC=$?
if [ $RC -ne 0 ]; then
  log "❌ 干跑失败 exit=$RC —— 见 logs/dryrun_v5_20260925.log"
  exit 1
fi
log "✅ 干跑完成 —— 见 logs/dryrun_v5_20260925.log"
log "═══ V5 链完成 ═══"
