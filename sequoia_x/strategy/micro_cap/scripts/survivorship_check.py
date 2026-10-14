"""量化「幸存者偏差」—— 把退市股放回去，收益掉多少？

## 问题是什么

数据里只有**现在还活着**的股票。那些**最终退市**的（2016-2026 窗口内 248 只）
被无声剔除了 ⇒ 回测**系统性高估收益**（买到的都是活下来的赢家）。

## 怎么量化

对同一套策略跑两遍：

  A. **不含退市股**（现状）
  B. **含退市股**（`fetch_delisted_stocks.py` 补回来的 248 只）

**两遍必须用同一个排序口径**（都按 **流通市值**），否则比的就不是"有没有退市股"，
而是"总市值 vs 流通市值"了。

为什么只能按流通市值：退市股**没有 xdxr 股本**，算不出总市值；
但 baostock 的换手率可以反推流通市值（两个来源口径已实测一致，差 0.0007%）。

用法：
    /home/zhulei/anaconda3/envs/zhulei_py312/bin/python survivorship_check.py [db|ths]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import microcap_data as md

PROJ = Path(__file__).resolve().parents[4]
INDEX_EXTRA = PROJ / "sequoia_x" / "strategy" / "micro_cap" / "data" / "index_extra.parquet"

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "ths"
SLIP = 0.0020
POOL_Q = 0.20
COMMISSION, TRANSFER = 0.00025, 0.00001
STAMP_BEFORE, STAMP_AFTER = 0.001, 0.0005
STAMP_CUTOFF = pd.Timestamp("2023-08-28")

pd.set_option("display.width", 200)


def log(msg: str = "") -> None:
    print(msg, flush=True)


ix = pd.read_parquet(INDEX_EXTRA)
bench_ret = ix[ix["symbol"] == "csi2000"].set_index("date")["close"].sort_index().pct_change()


def run(with_delisted: bool, by: str) -> dict:
    df = md.load(SOURCE, with_delisted=with_delisted)
    members = md.build_pool(df, q=POOL_Q, by=by)
    port = md.daily_portfolio(df, members)
    share = md.pool_amount_share(df, members)
    sig = md.build_signals(port, share, bench_ret, ma=60, q_win=250, q_level=0.80)
    start = sig.dropna(how="any").index.min()
    sig, port = sig.loc[start:], port.reindex(sig.loc[start:].index)

    pool_by_ym = {ym: set(x["symbol"]) for ym, x in members.groupby("hold_ym")}
    yms = sorted(pool_by_ym)
    repl = {ym: (1.0 if i == 0 else 1.0 - len(pool_by_ym[ym] & pool_by_ym[yms[i - 1]]) / len(pool_by_ym[ym]))
            for i, ym in enumerate(yms)}
    first_day = {ym: x.min() for ym, x in pd.Series(port.index, index=port.index.to_period("M")).groupby(level=0)}
    stamp = pd.Series(np.where(port.index >= STAMP_CUTOFF, STAMP_AFTER, STAMP_BEFORE), index=port.index)
    rate = 2 * (COMMISSION + TRANSFER + SLIP) + stamp

    def to_of(pos):
        pos = pos.reindex(port.index).fillna(False).astype(bool)
        t = pos.astype(float).diff().abs()
        t.iloc[0] = float(pos.iloc[0])
        prev = pos.shift(1).fillna(False)
        for ym, d in first_day.items():
            if pos.get(d, False) and prev.get(d, False):
                t.loc[d] = t.get(d, 0.0) + repl.get(ym, 0.0)
        return t.clip(upper=1.0)

    res = {"_rows": len(df), "_pool": members.groupby("hold_ym").size().median()}
    for label, pos in {"原版": pd.Series(True, index=port.index),
                       "+S3 拥挤度": sig["S3_拥挤度"].shift(1).fillna(False)}.items():
        net = port * pos.reindex(port.index).fillna(False).astype(bool) - to_of(pos) * rate
        nav = (1 + net.fillna(0)).cumprod()
        res[label] = (nav.iloc[-1] ** (244 / len(net)) - 1, (nav / nav.cummax() - 1).min())
    return res


log("=" * 96)
log(f"幸存者偏差量化    数据源: {SOURCE}    排序口径: 总市值(两边一致，微盘指数标准)    滑点: {SLIP * 1e4:.0f}bp/边")
log("⚠️ 不能用流通市值做对照 —— 那会把限售比例高的次新股拉进来，等于换了策略（实测 13.0%→4.2%）")
log("=" * 96)

rows = []
for flag, cn in [(False, "A 不含退市股（现状）"), (True, "B 含退市股（已补）")]:
    r = run(flag, "total_mv")
    log(f"  跑完 {cn}: 保留 {r['_rows']:,} 行，微盘池中位 {r['_pool']:.0f} 只")
    rows.append({"口径": cn,
                 "原版年化": f"{r['原版'][0]:.1%}", "原版回撤": f"{r['原版'][1]:.1%}",
                 "+S3 年化": f"{r['+S3 拥挤度'][0]:.1%}", "+S3 回撤": f"{r['+S3 拥挤度'][1]:.1%}"})

log("")
log("=" * 96)
log("结果（均按总市值排序、扣 20bp 滑点真实成本）")
log("=" * 96)
log(pd.DataFrame(rows).to_string(index=False))
log("")
log("怎么读：B 比 A 低多少，就是**幸存者偏差让收益虚高了多少**。")
log("        回撤同理 —— 如果 B 的回撤明显更深，说明我们之前也低估了风险。")
