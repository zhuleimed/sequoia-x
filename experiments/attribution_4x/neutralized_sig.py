#!/usr/bin/env python3
"""组合层中性化的**显著性检验**（补 (b) 的"单路径"缺口，2026-09-27）

背景：实验 (b) 显示"组合层中性化让 E1 从 +87.6% 提到 +179.7%、夏普 0.42→0.68"，
但那是**单条路径**——今天已被审计教训过：单路径不能当结论。
本脚本取两条路径的**月度收益序列**做配对检验 + i.i.d./块自助 + 刀切。

用法（断点续跑：结果已存在即跳过；--force 重算）
    env -u KMP_AFFINITY python experiments/attribution_4x/neutralized_sig.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

os.environ.pop("KMP_AFFINITY", None)
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent / "out"
ORIG = ROOT / "output/backtest_v2/.purged_70m.json"
NEUT = ROOT / "output/backtest_v2/.t2_neutralized.json"
RES = OUT / "neutralized_sig.json"


def run(p: Path) -> tuple[np.ndarray, list[str]]:
    from sequoia_x.core.config import Settings
    from sequoia_x.data.engine import DataEngine
    from sequoia_x.model_selection_v2.backtest.monthly_engine import MonthlyBacktestEngine
    from sequoia_x.model_selection_v2.config import get_config
    c = json.load(open(p))
    ks = sorted(c)
    bt = MonthlyBacktestEngine(cfg=get_config(), engine=DataEngine(Settings()), top_n=10,
                               risk_mode="M4", initial_capital=500_000.0, use_real_t4=True,
                               prediction_cache=c, keep_survivors=False,
                               intra_exit_policy="hard_stop_only", hard_stop_pct=-0.12)
    m = bt.run(ks[0], ks[-1])
    return np.array(m["monthly_returns"], float), m["_monthly_labels"]


def main() -> int:
    if RES.exists() and "--force" not in sys.argv:
        print(f"已有结果 {RES}（--force 重算）")
        return 0
    a, labels = run(ORIG)
    b, _ = run(NEUT)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    d = b - a
    t, p = stats.ttest_rel(b, a)
    print(f"月数 {n} ｜ 中性化 − 原始：月均差 {d.mean()*100:+.3f}pp  t={t:+.2f}  p={p:.4f} "
          f"{'显著' if p < 0.05 else '不显著'} ｜ 中性化更好的月数 {(d>0).sum()}/{n}")
    out = {"n_months": n, "mean_diff": float(d.mean()), "paired_t": float(t), "paired_p": float(p),
           "share_better": float((d > 0).mean()), "ci": {}}
    for L in (1, 3, 6, 12):
        means = []
        for s in range(5000):
            if L == 1:
                idx = np.random.RandomState(s).choice(n, n, replace=True)
            else:
                nb = int(np.ceil(n / L))
                st = np.random.RandomState(s).randint(0, n, nb)
                idx = np.concatenate([(np.arange(x, x + L) % n) for x in st])[:n]
            means.append(d[idx].mean())
        lo, hi = np.percentile(means, [2.5, 97.5])
        out["ci"][f"L{L}"] = [float(lo), float(hi)]
        print(f"  {'i.i.d.' if L == 1 else f'块自助 L={L:<2d}'} 95%CI "
              f"[{lo*100:+.3f}, {hi*100:+.3f}] {'✓ 不跨 0（显著）' if lo > 0 else '✗ 跨 0'}")
    rest = np.delete(d, np.argsort(-d)[:3])
    out["jackknife_mean"] = float(rest.mean())
    print(f"  剔最好 3 个月：{d.mean()*100:+.3f} → {rest.mean()*100:+.3f}pp"
          f"（t={rest.mean()/(rest.std(ddof=1)/np.sqrt(len(rest))):+.2f}）")
    out["labels"] = labels[:n]
    out["monthly_returns_orig"] = a.tolist()
    out["monthly_returns_neut"] = b.tolist()
    RES.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n产物: {RES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
