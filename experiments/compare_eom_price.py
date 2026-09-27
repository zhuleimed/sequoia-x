#!/usr/bin/env python3
"""审计实验 (a)：月末清仓价的"末日开盘 vs 末日收盘"（2026-09-27）

问题
----
回测 `monthly_engine._sell_all_positions` 用**末日开盘价**清仓；
生产 `simulation/engine.py::liquidate_all_at_close` 用**末日收盘价**（注释自陈"回测用开盘价"）。
而"末日开盘→收盘"= **+0.41%/月**（69 月复利 ≈ +34%）⇒ 这是**第 5 处回测↔生产口径差**，
也是审计指出的"#17 的 0.6pp/月缺口"的一部分（执行时点，与规则无关）。

做法
----
同一份干净缓存（`.purged_70m.json`）、同一组关键臂，各跑两遍：
  eom_sell_price="open"（默认，应**逐位复现**既有数字：A12 +40.4% / E1 +87.6% / B +152.1%）
  eom_sell_price="close"（= 生产口径）
⇒ 直接量出该口径差的代价，并验证"open"这条路径未被本次改动影响（回归检查）。

断点续跑：结果 JSON 已含某 (臂, 设置) 即跳过；`--force` 重算。
用法：
    env -u KMP_AFFINITY python experiments/compare_eom_price.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.pop("KMP_AFFINITY", None)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CACHE = ROOT / "output/backtest_v2/.purged_70m.json"
RES = ROOT / "experiments/attribution_4x/out/eom_price_ab.json"
# (policy, hard_stop_pct, 标签)
ARMS = [("all", -0.12, "A12全规则"),
        ("hard_stop_only", -0.12, "E1只留硬止损-12%"),
        ("none", None, "B纯持有")]
SETTINGS = [("open", "回测历史口径"), ("close", "生产口径")]
EXPECT_OPEN = {"A12全规则": 0.404, "E1只留硬止损-12%": 0.876, "B纯持有": 1.521}   # 回归检查用
FORCE = "--force" in sys.argv


def main() -> int:
    from sequoia_x.core.config import Settings
    from sequoia_x.data.engine import DataEngine
    from sequoia_x.model_selection_v2.backtest.monthly_engine import MonthlyBacktestEngine
    from sequoia_x.model_selection_v2.config import get_config

    cache = json.load(open(CACHE))
    months = sorted(cache)
    res = json.load(open(RES)) if RES.exists() and not FORCE else {}
    for policy, hsp, label in ARMS:
        for setting, desc in SETTINGS:
            key = f"{label}|{setting}"
            if key in res:
                print(f"跳过 {key}（已有）", flush=True)
                continue
            bt = MonthlyBacktestEngine(cfg=get_config(), engine=DataEngine(Settings()), top_n=10,
                                       risk_mode="M4", initial_capital=500_000.0, use_real_t4=True,
                                       prediction_cache=cache, keep_survivors=False,
                                       intra_exit_policy=policy, hard_stop_pct=hsp,
                                       eom_sell_price=setting)
            m = bt.run(months[0], months[-1])
            res[key] = {"policy": policy, "hard_stop_pct": hsp, "eom_sell_price": setting,
                        "total_return": m["total_return"], "annual_return": m["annual_return"],
                        "sharpe": m["sharpe"], "max_drawdown": m["max_drawdown"],
                        "n_trades": m.get("n_trades")}
            RES.write_text(json.dumps(res, ensure_ascii=False, indent=2))
            print(f"  [{key}] 总收益 {m['total_return']:+.2%} 夏普 {m['sharpe']:.2f} "
                  f"回撤 {m['max_drawdown']:+.2%}", flush=True)

    print("\n" + "=" * 96)
    print(f"{'臂':22s} {'口径':16s} {'总收益':>9s} {'年化':>8s} {'夏普':>6s} {'回撤':>8s} {'交易':>6s}")
    for policy, hsp, label in ARMS:
        for setting, desc in SETTINGS:
            k = f"{label}|{setting}"
            if k not in res:
                continue
            r = res[k]
            print(f"{label:22s} {setting + '(' + desc + ')':16s} {r['total_return']:>+8.1%} "
                  f"{r['annual_return']:>+7.1%} {r['sharpe']:>6.2f} {r['max_drawdown']:>+7.1%} "
                  f"{r['n_trades']:>6}")
    print("\n【回归检查】open 口径应复现既有数字：")
    for policy, hsp, label in ARMS:
        k = f"{label}|open"
        if k in res:
            got, exp = res[k]["total_return"], EXPECT_OPEN.get(label)
            ok = exp is not None and abs(got - exp) < 0.01
            print(f"  {label:22s} {got:+.1%} vs 期望 {exp:+.1%} {'✅' if ok else '⚠️ 不一致'}")
    print("\n【口径差】close − open（= 末日开盘→收盘效应）：")
    for _, _, label in ARMS:
        a, b = res.get(f"{label}|open"), res.get(f"{label}|close")
        if a and b:
            print(f"  {label:22s} 总收益 {b['total_return']-a['total_return']:+.1%} ｜ "
                  f"夏普 {b['sharpe']-a['sharpe']:+.2f} ｜ 回撤 {b['max_drawdown']-a['max_drawdown']:+.1%}")
    print(f"\n产物: {RES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
