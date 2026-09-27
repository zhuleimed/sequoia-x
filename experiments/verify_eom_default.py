#!/usr/bin/env python3
"""核验「回测月末清仓价默认已翻转为生产口径(close)」（2026-09-27 夜，无人值守）

背景
----
审计实验 (a) 查出：回测用**末日开盘价**清仓、生产用**收盘价** ⇒ 回测系统性低估收益
（E1 +87.6%→+167.2%、B +152.1%→+244.7%）。据用户指示**立即把回测默认改为 close**。
本脚本做**自动化回归核验**（给 07:33 的自动检查读）：

  ① 默认（不传 eom_sell_price）时，三个臂必须 **逐位命中 (a) 实验的 close 列**
  ② 显式传 eom_sell_price="open" 时，必须仍复现旧口径（参数没坏）
  ③ 生产模拟盘路径未被影响（静态检查：SimEngine 自己用收盘价清仓，不读回测参数）

退出码：0=全部 PASS；1=有 FAIL（供无人值守时判定"要不要处置"）
用法：
    env -u KMP_AFFINITY python experiments/verify_eom_default.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.pop("KMP_AFFINITY", None)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
AB = ROOT / "experiments/attribution_4x/out/eom_price_ab.json"
OUT = ROOT / "experiments/attribution_4x/out/verify_eom_default.json"
TOL = 0.01          # 总收益容差 1pp

# (policy, hard_stop, 标签, 期望 close 值)
ARMS = [("all", -0.12, "A12全规则", 0.441),
        ("hard_stop_only", -0.12, "E1只留硬止损-12%", 0.1672),
        ("none", None, "B纯持有", 0.2447)]
EXPECT_OPEN_E1 = 0.0876     # 显式 open 时的 E1


def run(policy, hsp, eom=None):
    from sequoia_x.core.config import Settings
    from sequoia_x.data.engine import DataEngine
    from sequoia_x.model_selection_v2.backtest.monthly_engine import MonthlyBacktestEngine
    from sequoia_x.model_selection_v2.config import get_config
    cache = json.load(open(ROOT / "output/backtest_v2/.purged_70m.json"))
    ks = sorted(cache)
    kw = {}
    if eom is not None:
        kw["eom_sell_price"] = eom
    bt = MonthlyBacktestEngine(cfg=get_config(), engine=DataEngine(Settings()), top_n=10,
                              risk_mode="M4", initial_capital=500_000.0, use_real_t4=True,
                              prediction_cache=cache, keep_survivors=False,
                              intra_exit_policy=policy, hard_stop_pct=hsp, **kw)
    m = bt.run(ks[0], ks[-1])
    return m["total_return"], m["sharpe"], m["max_drawdown"]


def main() -> int:
    res, ok = {}, True
    print("① 默认（应命中 close = 生产口径）", flush=True)
    for policy, hsp, label, exp in ARMS:
        tr, sh, mdd = run(policy, hsp)
        good = abs(tr - exp) < TOL
        ok &= good
        res[label] = {"default_total": tr, "expect_close": exp, "pass": good,
                      "sharpe": sh, "max_drawdown": mdd}
        print(f"   {label:22s} {tr:+.1%}（期望 {exp:+.1%}）{'✅' if good else '❌ FAIL'}", flush=True)

    print("② 显式 open（应复现旧口径）", flush=True)
    tr, _, _ = run("hard_stop_only", -0.12, eom="open")
    good = abs(tr - EXPECT_OPEN_E1) < TOL
    ok &= good
    res["E1|open"] = {"total": tr, "expect": EXPECT_OPEN_E1, "pass": good}
    print(f"   E1|open {tr:+.1%}（期望 {EXPECT_OPEN_E1:+.1%}）{'✅' if good else '❌ FAIL'}", flush=True)

    print("③ 生产路径未被影响（静态检查）", flush=True)
    eng = (ROOT / "sequoia_x/simulation/engine.py").read_text()
    good = "liquidate_all_at_close" in eng and "eom_sell_price" not in eng
    ok &= good
    res["prod_untouched"] = {"pass": good}
    print(f"   生产用 liquidate_all_at_close 且不读回测参数 → {'✅' if good else '❌ FAIL'}", flush=True)

    if AB.exists():
        res["source_artifact"] = str(AB)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"\n{'='*60}\n总判定: {'✅ 全部 PASS' if ok else '❌ 存在 FAIL（需处置）'}\n产物: {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
