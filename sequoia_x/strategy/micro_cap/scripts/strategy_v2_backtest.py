"""微盘策略 V2 —— 加入「非等权持仓」（最小方差组合优化）。

## 为什么这就是你说的"按前期研究结果"

项目 2026-10-03 做过 P0 方向一的「分钱方法」对比（六臂 70 个月，见 `docs/待办.md` #20）：
**最小方差最优**（+373.7% vs 等权 +277.8%），生产参数定为
`BUY_WEIGHT_PARAMS = {"lookback": 250, "cap": 0.10}`（一年窗口、单票上限 10%）。

⇒ 本脚本**照搬项目自己的优化器口径**（`model_selection_v2/backtest/monthly_engine.py`
的 `_estimate_cov` + `_solve_min_variance`），而不是另写一套，避免口径漂移：

  · 协方差用**不复权日收益**、clip 到 ±11%、日期**内连接对齐**
  · 收缩用 **sklearn Ledoit-Wolf**（不可用时退化为向对角收缩）
  · 求解用 **SLSQP**：min wᵀΣw s.t. Σw=1, 0 ≤ w ≤ cap
  · 对齐后样本 < 40 天 ⇒ **回退等权**（与生产一致）

⚠️ 单票上限与持仓数必须配对（项目 §一 明确写了）：
   N=10 + cap 0.10 ⇒ 10×0.10 = 100% ⇒ **等于强制等权**，优化器被废掉。
   所以 N=10 用 cap 0.15（老配置），N≥20 用 cap 0.10（生产配置）。

## 口径

- 初选池：非北交/非科创/非创业 + 非ST + 非退市 + 上市>250日 + 股价>2元
- 每月末在池内按**总市值最小**选 N 支
- 权重：等权 vs 最小方差（月内**买入并持有**，权重随涨跌自然漂移，不日内再平衡）
- 扣真实成本

用法：
    /home/zhulei/anaconda3/envs/zhulei_py312/bin/python strategy_v2_backtest.py [db|ths]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import microcap_data as md

PROJ = Path(__file__).resolve().parents[4]
BASE = PROJ / "sequoia_x" / "strategy" / "micro_cap"
INDEX_EXTRA = BASE / "data" / "index_extra.parquet"

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "ths"
NS = [10, 20, 30]
SLIP = 0.0020
LOOKBACK, MIN_SAMPLES = 250, 40          # 与生产 BUY_WEIGHT_PARAMS 一致
COMMISSION, TRANSFER = 0.00025, 0.00001
STAMP_BEFORE, STAMP_AFTER = 0.001, 0.0005
STAMP_CUTOFF = pd.Timestamp("2023-08-28")

pd.set_option("display.width", 200)


def log(msg: str = "") -> None:
    print(msg, flush=True)


def cap_for(n: int) -> float:
    """单票上限：N=10 必须放宽到 0.15，否则 10×0.10=100% 等于强制等权。"""
    return 0.15 if n <= 10 else 0.10


def min_variance_weights(panel: pd.DataFrame, cap: float) -> dict[str, float] | None:
    """panel: index=date, 列=symbol, 值=不复权收盘价（截至 ref_date）。
    返回 symbol→权重（和为 1）；样本不足/求解失败返回 None（调用方回退等权）。
    口径照搬项目 `monthly_engine._estimate_cov` + `_solve_min_variance`。
    """
    p = panel.tail(LOOKBACK + 1).dropna()
    if len(p) < MIN_SAMPLES + 1:
        return None
    rets = np.clip(p.pct_change().dropna().values, -0.11, 0.11)
    if rets.shape[0] < MIN_SAMPLES:
        return None
    try:
        from sklearn.covariance import LedoitWolf
        cov = LedoitWolf().fit(rets).covariance_
    except Exception:
        cov = np.cov(rets, rowvar=False) * 0.9 + np.diag(np.diag(np.cov(rets, rowvar=False))) * 0.1
    n = cov.shape[0]
    cap_eff = max(cap, 1.0 / n + 1e-9)
    res = minimize(
        lambda w: float(w @ cov @ w), np.ones(n) / n, method="SLSQP",
        bounds=[(0.0, cap_eff)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not res.success or not np.all(np.isfinite(res.x)):
        return None
    w = np.clip(res.x, 0, None)
    s = w.sum()
    if s <= 0:
        return None
    return dict(zip(p.columns, w / s))


log("=" * 100)
log(f"微盘策略 V2（等权 vs 最小方差）    数据源: {SOURCE}    滑点: {SLIP * 1e4:.0f}bp/边")
log(f"优化器口径: Ledoit-Wolf 收缩 + SLSQP 最小方差 | lookback {LOOKBACK} | 最少样本 {MIN_SAMPLES}")
log("=" * 100)

df_all = md.load(SOURCE)
df = md.apply_initial_pool(df_all)
log(f"初选池: {len(df):,} 行 / {df['symbol'].nunique():,} 只")

# 收盘价面板（按 symbol 分组索引，便于按月取窗口）
df = df.sort_values(["symbol", "date"])
close_by_sym = {s: g.set_index("date")["close"] for s, g in df.groupby("symbol")}
ret_by_sym = {s: g.set_index("date")["ret"] for s, g in df.groupby("symbol")}

ix = pd.read_parquet(INDEX_EXTRA)
bench_ret = ix[ix["symbol"] == "csi2000"].set_index("date")["close"].sort_index().pct_change()


def portfolio_from_weights(sel: pd.DataFrame, weighting: str) -> pd.Series:
    """sel: 列 ym / symbol / hold_ym。返回按 hold_ym 分月、池内指定权重的日度收益。"""
    out = []
    months = sorted(sel["hold_ym"].unique())
    for i, ym in enumerate(months):
        g = sel[sel["hold_ym"] == ym]
        syms = list(g["symbol"])
        # 权重在**上一个月末**（= 本月第一天之前）确定 ⇒ 用上月最后交易日为 ref
        prev_end = g["ref_date"].iloc[0]
        if weighting == "equal":
            w = pd.Series(1.0 / len(syms), index=syms)
        else:
            panel = pd.DataFrame({s: close_by_sym[s] for s in syms if s in close_by_sym})
            panel = panel[panel.index <= prev_end]
            wd = min_variance_weights(panel, cap_for(len(syms)))
            if wd is None:
                w = pd.Series(1.0 / len(syms), index=syms)
            else:
                w = pd.Series(wd)
        # 本月日收益
        r = pd.DataFrame({s: ret_by_sym[s] for s in syms if s in ret_by_sym})
        r = r[r.index.to_period("M") == ym]
        if r.empty:
            continue
        # ⚠️ 权重必须**只在当月有行情的股票上重新归一化**：
        #    否则某只股票当月无行情时，权重和 < 1 ⇒ 组合等于"部分空仓" ⇒ 收益被系统性压扁。
        w = w.reindex(r.columns).fillna(0.0)
        if w.sum() <= 0:
            continue
        w = w / w.sum()
        r = r.fillna(0.0)                      # 停牌日按 0 收益（真实：不动）
        nav = (1 + r).cumprod() * w            # 各股权重×净值
        tot = nav.sum(axis=1)
        port = tot.pct_change()
        port.iloc[0] = tot.iloc[0] - 1.0
        out.append(port)
    s = pd.concat(out)
    return s.sort_index()


def backtest(n: int, weighting: str, use_s3: bool) -> dict:
    members = md.pick_top_n(df, n=n, by="total_mv")
    # 记录每月末的 ref_date（= 选股那天），用于优化器取窗口
    is_me = df["date"] == df.groupby("ym")["date"].transform("max")
    me_dates = df[is_me].groupby("ym")["date"].max()
    members["ref_date"] = members["ym"].map(me_dates)
    port = portfolio_from_weights(members, weighting)

    share = md.pool_amount_share(df, members)
    sig = md.build_signals(port, share, bench_ret, ma=60, q_win=250, q_level=0.80)
    start = sig.dropna(how="any").index.min()
    sig, port = sig.loc[start:], port.reindex(sig.loc[start:].index)
    pos = (sig["S3_拥挤度"].shift(1).fillna(False).astype(bool) if use_s3
           else pd.Series(True, index=port.index))

    # 换手：月初换仓（N 支几乎全换）+ S3 开关
    w_by_ym = {ym: set(g["symbol"]) for ym, g in members.groupby("hold_ym")}
    yms = sorted(w_by_ym)
    repl = {ym: (1.0 if i == 0 else 1.0 - len(w_by_ym[ym] & w_by_ym[yms[i - 1]]) / len(w_by_ym[ym]))
            for i, ym in enumerate(yms)}
    first_day = {ym: x.min() for ym, x in pd.Series(port.index, index=port.index.to_period("M")).groupby(level=0)}
    rate = 2 * (COMMISSION + TRANSFER + SLIP) + pd.Series(
        np.where(port.index >= STAMP_CUTOFF, STAMP_AFTER, STAMP_BEFORE), index=port.index)
    t = pos.astype(float).diff().abs()
    t.iloc[0] = float(pos.iloc[0])
    prev = pos.shift(1).fillna(False)
    for ym, d in first_day.items():
        if pos.get(d, False) and prev.get(d, False):
            t.loc[d] = t.get(d, 0.0) + repl.get(ym, 0.0)
    t = t.clip(upper=1.0)

    net = port * pos - t * rate
    nav = (1 + net.fillna(0)).cumprod()
    ann = nav.iloc[-1] ** (244 / len(net)) - 1
    dd = (nav / nav.cummax() - 1).min()
    return {"ann": ann, "dd": dd, "calmar": ann / abs(dd) if dd < 0 else float("nan"),
            "inmkt": pos.mean(), "turn": t.sum() / (len(net) / 244)}


log("")
log("=" * 100)
log("结果（初选池 + 精选 N 支 + 真实成本）")
log("=" * 100)
rows = []
for n in NS:
    for weighting, tag in [("equal", "等权"), ("minvar", "最小方差")]:
        for use_s3, s3tag in [(False, "不加风控"), (True, "+S3 拥挤度")]:
            r = backtest(n, weighting, use_s3)
            rows.append({"持仓数": n, "权重": tag, "风控": s3tag,
                         "年化": f"{r['ann']:.1%}", "最大回撤": f"{r['dd']:.1%}",
                         "Calmar": f"{r['calmar']:.2f}", "在市": f"{r['inmkt']:.0%}",
                         "年换手": f"{r['turn']:.1f}"})
        log(f"  跑完 {n} 只 / {tag}")

log("")
log(pd.DataFrame(rows).to_string(index=False))
log("")
log("怎么读：")
log("  · 『权重』列 等权 vs 最小方差 —— 这是你说的『按前期研究结果非等权买入』的效果。")
log("  · 项目前期结论（70 个月、生产口径）：最小方差优于等权。本表用同一个优化器口径复核。")
