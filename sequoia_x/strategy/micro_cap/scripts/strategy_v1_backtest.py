"""微盘策略 V1 —— 按用户口径重做基线。

## 与之前那些分析的根本区别（重要）

之前做的是「**等权持有市值最小的 20%（约 800 只）**」—— 那等于一个**微盘指数**，
**不是**可落地的策略。用户的实际口径是：

    初选池：非北交所 + 非科创板 + 非创业板 + 非ST + 非退市 + 上市>250日 + 股价>2元
    每月在这个池子里**精选 N 支**（10 或 20 支）
    非等权买入（V1 先做等权基线，V2 再接最小方差）

⇒ 所以之前那些「回撤 −26%」的数字**套不到这里**：800 只等权天然分散，20 只集中持仓回撤会大得多。

## V1 做什么

1. 建正确的**初选池**
2. 每月按**总市值最小**选 N 支（N = 10 / 20 / 30）
3. **等权**持有（先立基线）
4. 扣真实交易成本（沿用 cost_model 的费率）
5. 对比：不加风控 vs 加 S3 拥挤度择时

用法：
    /home/zhulei/anaconda3/envs/zhulei_py312/bin/python strategy_v1_backtest.py [db|ths]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import microcap_data as md

PROJ = Path(__file__).resolve().parents[4]
BASE = PROJ / "sequoia_x" / "strategy" / "micro_cap"
INDEX_EXTRA = BASE / "data" / "index_extra.parquet"

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "ths"
SLIP = 0.0020
NS = [10, 20, 30]
COMMISSION, TRANSFER = 0.00025, 0.00001
STAMP_BEFORE, STAMP_AFTER = 0.001, 0.0005
STAMP_CUTOFF = pd.Timestamp("2023-08-28")

pd.set_option("display.width", 200)


def log(msg: str = "") -> None:
    print(msg, flush=True)


log("=" * 100)
log(f"微盘策略 V1（用户口径）    数据源: {SOURCE}    滑点: {SLIP * 1e4:.0f}bp/边")
log("初选池: 非北交/非科创/非创业 + 非ST + 非退市 + 上市>250日 + 股价>2元")
log("=" * 100)

df_all = md.load(SOURCE)
df = md.apply_initial_pool(df_all)
log("")
log(f"初选池: {len(df):,} 行 / {df['symbol'].nunique():,} 只    "
    f"（全市场 {df_all['symbol'].nunique():,} 只 ⇒ 剔掉 {df_all['symbol'].nunique() - df['symbol'].nunique():,} 只）")
pool_size = df.groupby("ym")["symbol"].nunique()
log(f"每月初选池规模: 中位 {pool_size.median():.0f} 只，最小 {pool_size.min()}，最大 {pool_size.max()}")

ix = pd.read_parquet(INDEX_EXTRA)
bench_ret = ix[ix["symbol"] == "csi2000"].set_index("date")["close"].sort_index().pct_change()
stamp = None


def backtest(n: int, use_s3: bool) -> dict:
    members = md.pick_top_n(df, n=n, by="total_mv")
    port = md.daily_portfolio(df, members)
    share = md.pool_amount_share(df, members)
    sig = md.build_signals(port, share, bench_ret, ma=60, q_win=250, q_level=0.80)
    start = sig.dropna(how="any").index.min()
    sig, port = sig.loc[start:], port.reindex(sig.loc[start:].index)
    pos = (sig["S3_拥挤度"].shift(1).fillna(False) if use_s3
           else pd.Series(True, index=port.index)).astype(bool)

    # 换手：月初换仓（N 支几乎全换）+ S3 开关
    pool_by_ym = {ym: set(x["symbol"]) for ym, x in members.groupby("hold_ym")}
    yms = sorted(pool_by_ym)
    repl = {ym: (1.0 if i == 0 else 1.0 - len(pool_by_ym[ym] & pool_by_ym[yms[i - 1]]) / len(pool_by_ym[ym]))
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

    gross = port * pos
    net = gross - t * rate
    nav = (1 + net.fillna(0)).cumprod()
    ann = nav.iloc[-1] ** (244 / len(net)) - 1
    dd = (nav / nav.cummax() - 1).min()
    return {"ann": ann, "dd": dd, "calmar": ann / abs(dd) if dd < 0 else float("nan"),
            "inmkt": pos.mean(), "turn": t.sum() / (len(net) / 244), "nav": nav,
            "n_pool": members.groupby("hold_ym").size().median(), "start": net.index.min()}


log("")
log("=" * 100)
log("结果（等权、扣真实成本）")
log("=" * 100)
rows = []
navs = {}
for n in NS:
    for use_s3, tag in [(False, "不加风控"), (True, "+S3 拥挤度")]:
        r = backtest(n, use_s3)
        navs[(n, tag)] = r["nav"]
        rows.append({"持仓数": n, "风控": tag,
                     "年化": f"{r['ann']:.1%}", "最大回撤": f"{r['dd']:.1%}",
                     "Calmar": f"{r['calmar']:.2f}", "在市": f"{r['inmkt']:.0%}",
                     "年换手": f"{r['turn']:.1f}"})
        log(f"  跑完 {n} 只 / {tag}: 实际池中位 {r['n_pool']:.0f} 只，区间 {r['start'].date()} 起")

log("")
log(pd.DataFrame(rows).to_string(index=False))
log("")
log("怎么读：")
log("  · 『最大回撤』是重点 —— 集中持仓 10~20 只，回撤会明显比 800 只等权（−59%）更深。")
log("  · 『年换手』= 一年换掉几个满仓；换成 N 支时月初几乎全换，换手自然高，成本要算足。")

# 与基准对照
bcn = (1 + bench_ret.reindex(navs[(20, "不加风控")].index).fillna(0)).cumprod()
log("")
log(f"对照组（中证2000）同期: 年化 {bcn.iloc[-1] ** (244 / len(bcn)) - 1:.1%}，"
    f"最大回撤 {(bcn / bcn.cummax() - 1).min():.1%}")
