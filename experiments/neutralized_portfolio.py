#!/usr/bin/env python3
"""审计实验 (b)：组合层中性化 —— "中性化 IC 能否变成钱"（2026-09-27）

问题
----
分析发现：**中性化后 IC 显著**（T2 0.0152→**0.0256**，p=0.002），但那是"事后评估"——
把 y 残差化。**没人验证过：在组合层做中性化（把预测分数对风格残差化再选股）能否改善实盘收益。**
若不能改善 ⇒ 待办 #3（组合层中性化）应撤销。

做法（不重训，纯离线）
------
1. 每月（买入日）算池内每只的（行业, 成交额五分位, 波动五分位）——与 ic_by_horizon 同口径
2. 把当月**预测分数 t2** 对上述哑变量做 OLS，取**残差**作为新分数（= 去掉风格暴露后的选股打分）
3. 写出新缓存 `.t2_neutralized.json`（t2=残差，t4=0 ⇒ 融合退化为纯 T2；
   t1/t3 沿用原缓存，与中性化无关）
4. 同一套引擎参数（E1：hard_stop_only + −12%）跑**原缓存 vs 中性化缓存**，对比收益/夏普/回撤
5. 另报持有期口径的 TOP10/TOP30 超额（与 selection_alpha 同口径）

断点续跑：缓存/结果已存在即跳过；`--force` 重算。
用法：
    env -u KMP_AFFINITY python experiments/neutralized_portfolio.py
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.pop("KMP_AFFINITY", None)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CACHE = ROOT / "output/backtest_v2/.purged_70m.json"
NCACHE = ROOT / "output/backtest_v2/.t2_neutralized.json"
RES = ROOT / "experiments/attribution_4x/out/neutralized_portfolio.json"
POLICY, HSP = "hard_stop_only", -0.12
FORCE = "--force" in sys.argv


def build_neutralized_cache() -> None:
    """把每月 t2 对（行业+成交额分位+波动分位）残差化，写出新缓存。"""
    from experiments.attribution_4x.selection_alpha import engine_buys
    cx = sqlite3.connect(ROOT / "data/sequoia_v2.db")
    px = pd.read_sql("SELECT symbol,date,close,amount FROM stock_daily "
                     "WHERE date>='2020-09-01' AND close>0 ORDER BY symbol,date", cx)
    ind = pd.read_sql("SELECT symbol, industry_l1 FROM tdx_stock_industry", cx)
    cal = np.array([r[0] for r in cx.execute(
        "SELECT DISTINCT date FROM index_daily WHERE symbol='sh.000300' "
        "AND date>='2020-09-01' ORDER BY date")])
    cx.close()
    ind_map = dict(zip(ind["symbol"], ind["industry_l1"]))
    feats = {s: (g["date"].to_numpy(), g["close"].to_numpy(float), g["amount"].to_numpy(float))
             for s, g in px.groupby("symbol", sort=False)}
    cache = json.load(open(CACHE))
    buy_dates = {dt: None for dt in sorted(engine_buys(CACHE)["date"].unique())}
    out, ndiag = {}, []
    for dt in buy_dates:
        month = dt[:7]
        if month not in cache:
            continue
        syms = cache[month]["symbols"]
        p2 = np.array(cache[month]["t2"], float)
        rows = []
        for s, p in zip(syms, p2):
            a = feats.get(s)
            if a is None:
                continue
            dts, cls, amts = a
            j0 = int(np.searchsorted(dts, dt))
            if j0 >= len(dts) or dts[j0] != dt or j0 < 20:
                continue
            amt = float(np.nanmean(amts[j0 - 20:j0 + 1]))
            r = np.diff(cls[j0 - 20:j0 + 1]) / cls[j0 - 20:j0]
            rows.append({"symbol": s, "p": p, "amt": np.log(max(amt, 1.0)),
                         "vol": float(np.std(r)) if len(r) > 1 else 0.0,
                         "ind": ind_map.get(s, "未知")})
        d = pd.DataFrame(rows)
        if len(d) < 200:
            continue
        d["amt_q"] = pd.qcut(d["amt"], 5, labels=False, duplicates="drop")
        d["vol_q"] = pd.qcut(d["vol"], 5, labels=False, duplicates="drop")
        D = pd.get_dummies(d[["ind", "amt_q", "vol_q"]].astype(str), drop_first=True)
        X = np.column_stack([np.ones(len(d)), D.to_numpy(float)])
        beta, *_ = np.linalg.lstsq(X, d["p"].to_numpy(), rcond=None)
        resid = d["p"].to_numpy() - X @ beta
        # 新分数 = 残差（原缓存其余字段沿用；未匹配到的股票给很小的分数 → 不会被选）
        new = dict(cache[month])
        new["t2"] = [float(v) for v in resid]
        new["symbols"] = d["symbol"].tolist()
        keep = {s: i for i, s in enumerate(syms)}
        # t1/t3 与 symbols 对齐
        for f in ("t1", "t3"):
            if f in cache[month]:
                new[f] = [cache[month][f][keep[s]] for s in new["symbols"]]
        new["t4"] = [0.0] * len(new["symbols"])
        out[month] = new
        ndiag.append({"month": month, "n": len(d),
                      "corr_raw_resid": float(pd.Series(p2).corr(pd.Series(d["p"]), method="spearman"))})
    NCACHE.write_text(json.dumps(out, ensure_ascii=False))
    print(f"中性化缓存已写出: {NCACHE}（{len(out)} 个月）", flush=True)


def run_engine(cache_path: Path) -> dict:
    from sequoia_x.core.config import Settings
    from sequoia_x.data.engine import DataEngine
    from sequoia_x.model_selection_v2.backtest.monthly_engine import MonthlyBacktestEngine
    from sequoia_x.model_selection_v2.config import get_config
    cache = json.load(open(cache_path))
    months = sorted(cache)
    bt = MonthlyBacktestEngine(cfg=get_config(), engine=DataEngine(Settings()), top_n=10,
                               risk_mode="M4", initial_capital=500_000.0, use_real_t4=True,
                               prediction_cache=cache, keep_survivors=False,
                               intra_exit_policy=POLICY, hard_stop_pct=HSP)
    m = bt.run(months[0], months[-1])
    return {k: m[k] for k in ("total_return", "annual_return", "sharpe", "max_drawdown")}


def main() -> int:
    res = json.load(open(RES)) if RES.exists() and not FORCE else {}
    if not NCACHE.exists() or FORCE:
        build_neutralized_cache()
    for tag, path in [("原始", CACHE), ("中性化", NCACHE)]:
        if tag in res:
            print(f"跳过 {tag}（已有）", flush=True)
            continue
        res[tag] = run_engine(path)
        RES.write_text(json.dumps(res, ensure_ascii=False, indent=2))
        print(f"  [{tag}] {res[tag]['total_return']:+.2%} 夏普 {res[tag]['sharpe']:.2f} "
              f"回撤 {res[tag]['max_drawdown']:+.2%}", flush=True)

    print("\n" + "=" * 78)
    print(f"{'口径':10s} {'总收益':>9s} {'年化':>8s} {'夏普':>6s} {'回撤':>8s}")
    for tag in ("原始", "中性化"):
        if tag in res:
            r = res[tag]
            print(f"{tag:10s} {r['total_return']:>+8.1%} {r['annual_return']:>+7.1%} "
                  f"{r['sharpe']:>6.2f} {r['max_drawdown']:>+7.1%}")
    if "原始" in res and "中性化" in res:
        d = res["中性化"]["total_return"] - res["原始"]["total_return"]
        ds = res["中性化"]["sharpe"] - res["原始"]["sharpe"]
        print(f"\n【结论】组合层中性化：总收益 {d:+.1%}，夏普 {ds:+.2f} ⇒ "
              f"{'✅ 有改善（值得做 #3）' if d > 0.05 else '❌ 未改善 ⇒ 待办 #3 应撤销/降级'}")
    print(f"\n产物: {RES} ｜ 缓存: {NCACHE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
