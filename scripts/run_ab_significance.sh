#!/bin/bash
# A/B 显著性检验（2026-09-27，用户选 b）：tail vs random 的"钱"差异到底显不显著？
#
# 背景：09-27 的 20 组合 A/B 只产出 summary_all.csv（**月度收益序列从未落盘**，
#   因为 runner 组装汇总行时漏带 `_monthly_labels/_monthly_returns` —— 已修），
#   所以当时只能按 σ 估算"33pp ≈ 0.45σ"，做不了配对检验。
#
# 本脚本：用修好的 runner 重跑**单组合**（M4 × TOP10 × --period full = 69 个月，
#   约 3-5 分钟/个，而不是 20 组合的 60 分钟），让月度序列落盘 → 做配对检验。
#
# 口径：与 A/B 完全一致（相同缓存、相同 M4/TOP10、相同 69 月）
#   顺带验证两个 runner 修复：① `--period full` 在单组合分支不再被静默忽略（应报"全周期70月/69月"）
#                              ② `monthly_returns.csv` 应生成
set -u
PROJ=/public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x
PY=/home/zhulei/anaconda3/envs/zhulei_py312/bin/python
cd "$PROJ" || exit 1
PROG=logs/ab_significance.log
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$PROG"; }

run_one(){   # $1=tag  $2=cache
  log "重跑 $1（单组合 M4/TOP10/--period full）..."
  "$PY" -u scripts/run_shared_backtest.py --top-n 10 --mode M4 --period full \
    --cache "$2" --output-dir "output/ab_sig_$1" > "logs/ab_sig_$1.log" 2>&1
  local rc=$?
  local f="output/ab_sig_$1/monthly_returns.csv"
  log "  $1 exit=$rc ｜ 月度序列: $([ -f "$f" ] && echo "✅ 已生成" || echo "❌ 缺失")"
  # 验证 full 修复：时段名应为"全周期70月"
  grep -m1 "时段:" "logs/ab_sig_$1.log" | sed 's/INFO.*-//' | xargs -I{} log "  {}"
}

run_one tail   output/backtest_v2/.purged_70m.json
run_one random output/backtest_v2/.t2_70m_random.json

log "── 配对检验 ──"
"$PY" - <<'PYEOF' 2>&1 | tee -a "$PROG"
import csv, numpy as np
from scipy import stats

def load(tag):
    with open(f"output/ab_sig_{tag}/monthly_returns.csv") as f:
        rows = list(csv.reader(f))
    hdr = rows[0]
    key = [h for h in hdr if "M4" in h and "T10" in h][0]
    j = hdr.index(key)
    out = {}
    for r in rows[1:]:
        if len(r) > j and r[j].strip(): out[r[0]] = float(r[j])
    return out

A, B = load("tail"), load("random")            # A=tail（现行口径）, B=random
ms = sorted(set(A) & set(B))
a = np.array([A[m] for m in ms]); b = np.array([B[m] for m in ms])
print(f"共同月份: {len(ms)} 个（{ms[0]} ~ {ms[-1]}）")
print(f"  月均收益: A(tail) {a.mean()*100:+.3f}%  vs  B(random) {b.mean()*100:+.3f}%  ｜ 差 {(a.mean()-b.mean())*100:+.3f}pp/月")
print(f"  月波动  : A {a.std(ddof=1)*100:.2f}%  vs  B {b.std(ddof=1)*100:.2f}%")
t, p = stats.ttest_rel(a, b)
print(f"  配对 t 检验（同月配对，抵消市场共同波动）: t={t:+.2f}  p={p:.3f}  "
      f"{'✅ 显著' if p<0.05 else '❌ 不显著'}")
d = a - b
boot = [np.random.RandomState(s).choice(d, len(d), replace=True).mean()
        for s in range(2000)]
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"  月均差的自助 95% CI: [{lo*100:+.3f}%, {hi*100:+.3f}%]  "
      f"{'（跨 0 ⇒ 不显著）' if lo < 0 < hi else '（不跨 0 ⇒ 显著）'}")
print(f"  A 更好的月份: {(d>0).sum()}/{len(d)}")
PYEOF
log "═══ 完成 ═══"
