#!/usr/bin/env python3
"""选股 alpha 三项检验 + 块自助（2026-09-27 入库版）

为什么有这个脚本
----------------
09-27 做的三项关键分析（买入日效应 / 持有期 alpha / 组合宽度+β）当时是用 **/tmp 下的临时脚本**
跑的，没有入库 ⇒ 独立审计判定"**结论不可一键复现**"，违反项目自己的引用纪律。
本脚本把它们合并为**唯一的、可复现的实现**，并修正审计查出的两个口径问题：

  · 原 #17 写"模型 TOP10"，实际是**引擎实际成交名单**（停牌/涨停未成交且不补位）
    ⇒ 本脚本显式运行引擎取真实买单（policy/阈值**写死**，避免回落实盘参数）。
  · 原 #17b 的"块自助 95%CI"实现有误（审计复算 TOP30 其实**不跨 0**）
    ⇒ 本脚本用标准的**循环移动块自助**（Politis-Romano circular MBB），并同时给出
      i.i.d. 自助（L=1）作对照，另报"剔除最好 3 个月"的刀切结果。

口径（与 ic_by_horizon.py / t4_ic.py 一致）
------------------------------------------
  · ref/buy_date = 该月第一个交易日（引擎实际买入日）；month_end = 该月最后一个交易日
  · 持有期收益 = close(month_end)/close(buy_date) − 1（已剥离买入日当天）
  · 整段收益   = close(month_end)/open(buy_date) − 1（含买入日）
  · IC = 逐月 Spearman 再平均（**非池化**）；配对检验均按"同月配对"

用法
----
    env -u KMP_AFFINITY python experiments/attribution_4x/selection_alpha.py \
        --cache=output/backtest_v2/.purged_70m.json --tag=tail_t2
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

# KMP_AFFINITY 清除（项目铁律：.bashrc 绑核会让数值库抢同一核心）
os.environ.pop("KMP_AFFINITY", None)

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
# 仓库根加入 sys.path（本脚本在 experiments/attribution_4x/ 下，需能 import sequoia_x）
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent / "out"
CACHE = ROOT / "output/backtest_v2/.purged_70m.json"
TAG = "tail_t2"
# ⚠️ 写死（勿回落实盘参数）：这三项分析的历史口径是 A 臂 = 月内全规则 + 硬止损 −8%
ENGINE_POLICY, ENGINE_HSP = "all", -0.08
for _a in sys.argv[1:]:
    if _a.startswith("--cache="):
        CACHE = Path(_a[len("--cache="):])
    elif _a.startswith("--tag="):
        TAG = _a[len("--tag="):]

WIDTHS = [10, 30, 50, 100]
BLOCK_LS = [1, 3, 6, 12]      # 块自助的块长（L=1 即 i.i.d. 自助）


# ══════════════════════════════════════════════════════════════
#  块自助（Politis-Romano 循环移动块）
# ══════════════════════════════════════════════════════════════
def block_bootstrap_mean(x: np.ndarray, L: int, n_boot: int = 5000,
                         seed: int = 0) -> tuple[float, float]:
    """对"均值的抽样分布"做块自助，返回 95% CI。

    为什么必须分块：月度超额收益序列有自相关（相邻月共享市场行情、持仓重叠）。
    i.i.d. 自助（L=1）会**低估**方差 ⇒ CI 偏窄、假显著。
    循环移动块：从环上随机取 ⌈n/L⌉ 个长度 L 的块拼成 n 个观测（Politis & Romano 1994）。
    """
    n = len(x)
    if L <= 1:
        boots = [np.random.RandomState(seed + s).choice(x, n, replace=True).mean()
                 for s in range(n_boot)]
    else:
        n_blocks = int(np.ceil(n / L))
        boots = []
        for s in range(n_boot):
            starts = np.random.RandomState(seed + s).randint(0, n, n_blocks)
            idx = np.concatenate([(np.arange(st, st + L) % n) for st in starts])[:n]
            boots.append(x[idx].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def engine_buys(cache_path: Path) -> pd.DataFrame:
    """运行引擎（口径写死）取**实际成交**买单 —— 复现 #17 必须用它，而非"预测 TOP10"。"""
    from sequoia_x.core.config import Settings
    from sequoia_x.data.engine import DataEngine
    from sequoia_x.model_selection_v2.backtest.monthly_engine import MonthlyBacktestEngine
    from sequoia_x.model_selection_v2.config import get_config
    cache = json.load(open(cache_path))
    months = sorted(cache)
    bt = MonthlyBacktestEngine(cfg=get_config(), engine=DataEngine(Settings()), top_n=10,
                               risk_mode="M4", initial_capital=500_000.0, use_real_t4=True,
                               prediction_cache=cache, keep_survivors=False,
                               intra_exit_policy=ENGINE_POLICY, hard_stop_pct=ENGINE_HSP)
    bt.run(months[0], months[-1])
    buys = [{"date": t.date, "symbol": t.symbol} for t in bt.trades if t.trade_type == "buy"]
    return pd.DataFrame(buys)


def main() -> int:
    cx = sqlite3.connect(ROOT / "data/sequoia_v2.db")
    print("加载行情/行业…", flush=True)
    px = pd.read_sql("SELECT symbol,date,open,close,amount FROM stock_daily "
                     "WHERE date>='2020-09-01' AND close>0 ORDER BY symbol,date", cx)
    cal = [r[0] for r in cx.execute(
        "SELECT DISTINCT date FROM index_daily WHERE symbol='sh.000300' "
        "AND date>='2020-09-01' ORDER BY date")]
    ind = pd.read_sql("SELECT symbol, industry_l1 FROM tdx_stock_industry", cx)
    pool = set(json.load(open(ROOT / "output/backtest_v2/.stock_pool.json")))
    cx.close()
    ind_map = dict(zip(ind["symbol"], ind["industry_l1"]))
    cal = np.array(cal)

    print(f"取引擎实际买单（policy={ENGINE_POLICY}, hard_stop={ENGINE_HSP}）…", flush=True)
    buys = engine_buys(CACHE)
    buy_dates = sorted(buys["date"].unique())
    print(f"  {len(buys)} 笔 / {len(buy_dates)} 个买入日")

    # 每月的月末交易日
    month_end = {dt: [x for x in cal if x[:7] == dt[:7]][-1] for dt in buy_dates}
    dates = sorted(set(list(month_end.keys()) + list(month_end.values())))
    ph = ",".join("?" * len(dates))
    px2 = px[px["date"].isin(dates)]
    pc = px2.pivot_table(index="symbol", columns="date", values="close")
    po = px2.pivot_table(index="symbol", columns="date", values="open")
    pool_idx = set(pc.index) & pool

    # ── ① 买入日效应（模型 vs 随机 vs 全池） + 日历剖面 ──
    rng = np.random.RandomState(42)
    rows1 = []
    for dt, de in month_end.items():
        if dt not in pc.columns or de not in pc.columns:
            continue
        sub = pd.DataFrame({"c0": pc[dt], "c1": pc[de], "o0": po[dt]}).dropna()
        sub = sub[sub.index.isin(pool_idx)]
        sub = sub[sub["o0"] > 0]
        if len(sub) < 200:
            continue
        picks = [s for s in buys[buys["date"] == dt]["symbol"] if s in sub.index]
        if len(picks) < 3:
            continue
        m = sub.loc[picks]
        rnd = sub.sample(n=min(300, len(sub)), random_state=rng.randint(1 << 31))
        rows1.append({"date": dt,
                      # 买入日**当天**的开盘→收盘（用 c0 = 买入日收盘；2026-09-27 修：
                      #   初版误写成 c1/o0（=整段），导致"买入日"那一行印的其实是整段值）
                      "model_intra": (m["c0"] / m["o0"] - 1).mean(),
                      "random_intra": (rnd["c0"] / rnd["o0"] - 1).mean(),
                      "pool_intra": (sub["c0"] / sub["o0"] - 1).mean(),
                      # 持有期（剥离买入日）
                      "model_hold": (m["c1"] / m["c0"] - 1).mean(),
                      "random_hold": (rnd["c1"] / rnd["c0"] - 1).mean(),
                      "pool_hold": (sub["c1"] / sub["c0"] - 1).mean(),
                      # 整段（含买入日）
                      "model_full": (m["c1"] / m["o0"] - 1).mean(),
                      "random_full": (rnd["c1"] / rnd["o0"] - 1).mean(),
                      "pool_full": (sub["c1"] / sub["o0"] - 1).mean()})
    df1 = pd.DataFrame(rows1)
    df1.to_csv(OUT / f"buyday_effect_{TAG}.csv", index=False, encoding="utf-8-sig")

    def paired(a, b, name):
        d = (df1[a] - df1[b]).dropna()
        t, p = stats.ttest_rel(df1[a].dropna(), df1[b].dropna())
        print(f"  {name:34s} 差 {d.mean()*100:+.4f}pp  t={t:+.2f}  p={p:.4f} "
              f"{'✅' if p < 0.05 else '❌不显著'}  更好 {(d>0).sum()}/{len(d)}")

    print(f"\n【① 买入日 / 持有期：{len(df1)} 个月】")
    print(f"  模型 买入日开盘→收盘 {df1['model_intra'].mean()*100:+.3f}% ｜ "
          f"随机 {df1['random_intra'].mean()*100:+.3f}% ｜ 全池 {df1['pool_intra'].mean()*100:+.3f}%")
    paired("model_intra", "random_intra", "买入日 模型 − 随机（应≈0）")
    print(f"  模型 持有期 {df1['model_hold'].mean()*100:+.3f}% ｜ 随机 {df1['random_hold'].mean()*100:+.3f}% ｜ "
          f"全池 {df1['pool_hold'].mean()*100:+.3f}%")
    paired("model_hold", "random_hold", "持有期 模型 − 随机300")
    paired("model_hold", "pool_hold", "持有期 模型 − 全池")

    # ── ② 全市场日历剖面（按"当月第几个交易日"）──
    allpx = px.copy()
    allpx["ym"] = allpx["date"].str[:7]
    allpx = allpx.sort_values(["symbol", "date"])
    allpx["intra"] = allpx["close"] / allpx["open"] - 1
    allpx["r"] = allpx.groupby(["symbol", "ym"]).cumcount() + 1
    last = allpx.groupby(["symbol", "ym"])["intra"].transform("size")
    allpx["is_last"] = allpx["r"] == last
    prof = allpx[allpx["r"] <= 10].groupby("r")["intra"].mean()
    print("\n【② 全市场日历剖面：当月第 N 个交易日的『开盘→收盘』】")
    print("  " + " ｜ ".join(f"第{int(i)}日 {v*100:+.3f}%" for i, v in prof.items()))
    d1 = allpx[allpx["r"] == 1].groupby("ym")["intra"].mean()
    dl = allpx[allpx["is_last"]].groupby("ym")["intra"].mean()
    do = allpx[(allpx["r"] > 1) & (~allpx["is_last"])].groupby("ym")["intra"].mean()
    for nm, s in [("月初第1日", d1), ("月末最后日", dl), ("其余交易日", do)]:
        t, p = stats.ttest_1samp(s, 0)
        print(f"  {nm}: 月均 {s.mean()*100:+.3f}%（n={len(s)} 个月） t={t:+.2f} p={p:.4f} "
              f"{'✅' if p < 0.05 else '❌不显著'}")

    # ── ③ 组合宽度 + β + 块自助 ──
    cache = json.load(open(CACHE))
    res, poolret = {N: [] for N in WIDTHS}, []
    for dt, de in month_end.items():
        m = dt[:7]
        if m not in cache or dt not in pc.columns or de not in pc.columns:
            continue
        sub = pd.DataFrame({"c0": pc[dt], "c1": pc[de]}).dropna()
        sub = sub[sub.index.isin(pool_idx)]
        sub["r"] = sub["c1"] / sub["c0"] - 1
        if len(sub) < 200:
            continue
        pred = pd.Series(dict(zip(cache[m]["symbols"], cache[m]["t2"])))
        sub["p"] = pred.reindex(sub.index)
        ok = sub.dropna(subset=["p"]).sort_values("p", ascending=False)
        poolret.append(ok["r"].mean())
        for N in WIDTHS:
            res[N].append(ok["r"].head(N).mean())
    df3 = pd.DataFrame({"pool": poolret})
    for N in WIDTHS:
        df3[f"top{N}"] = res[N]
    df3.to_csv(OUT / f"portfolio_width_alpha_{TAG}.csv", index=False, encoding="utf-8-sig")

    print(f"\n【③ 组合宽度（持有期，{len(df3)} 个月）】")
    print(f"  {'组合':>7s} {'月均超额':>9s} {'中位':>8s} {'>0月数':>8s} {'配对p':>7s} "
          f"{'β':>5s} | 块自助 95%CI（L=1/3/6/12）")
    for N in WIDTHS:
        e = (df3[f"top{N}"] - df3["pool"]).dropna()
        t, p = stats.ttest_rel(df3[f"top{N}"].dropna(), df3["pool"].dropna())
        x = df3["pool"].to_numpy()
        y = df3[f"top{N}"].to_numpy()
        beta = np.polyfit(x, y, 1)[0]
        cis = []
        for L in BLOCK_LS:
            lo, hi = block_bootstrap_mean(e.to_numpy(), L)
            cis.append(f"[{lo*100:+.3f},{hi*100:+.3f}]{'✗' if lo < 0 < hi else '✓'}")
        print(f"  TOP{N:<4d} {e.mean()*100:>+8.3f}% {np.median(e)*100:>+7.3f}% "
              f"{(e>0).sum():>4d}/{len(e):<3d} {p:>7.3f} {beta:>5.2f} | " + " ｜ ".join(cis))
    print("\n  注：CI 后 ✓=不跨 0（显著）、✗=跨 0。L=1 为 i.i.d. 自助（会低估方差、CI 偏窄）；")
    print("      L>1 为循环移动块自助（保留自相关）。**只认块自助（L≥3）的结论。**")
    print("\n【③b 剔除最好的 3 个月（刀切）】")
    for N in [10, 30]:
        e = (df3[f"top{N}"] - df3["pool"]).to_numpy()
        rest = np.delete(e, np.argsort(-e)[:3])
        tt = rest.mean() / (rest.std(ddof=1) / np.sqrt(len(rest)))
        print(f"  TOP{N}: 全样本 {e.mean()*100:+.3f}%/月 → 剔 3 月 {rest.mean()*100:+.3f}%/月（t={tt:+.2f}）")
    print(f"\n产物: {OUT}/buyday_effect_{TAG}.csv ｜ portfolio_width_alpha_{TAG}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
