"""风控信号稳健性检查 —— 换个参数它还成立吗？

## 这是「调优」还是「检查」？—— 说清楚

**不是调优。** 调优 = 挑出表现最好的参数当结论；本脚本做的是相反的事：
**换一组参数，看结论会不会翻车**。如果只有 80% 分位这一个点有效、79% 就没效，
那就是**过拟合**，不能信。

## 检查什么

主脚本（`risk_overlay_backtest.py`）跑出两个「通过」的信号：
- **S1 趋势**（净值 > N 日均线）
- **S3 拥挤度**（微盘成交额占比的 N 日均值 < 过去 M 日的 q 分位）

对这两个做参数扫描，看判据（回撤改善 / 年化保留）在参数空间里是不是**连续成立**。

用法：
    /home/zhulei/anaconda3/envs/zhulei_py312/bin/python robustness_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

import microcap_data as md       # 共用数据层

PROJ = Path(__file__).resolve().parents[4]
BASE = PROJ / "sequoia_x" / "strategy" / "micro_cap"

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "db"

pd.set_option("display.width", 200)


def log(msg: str = "") -> None:
    print(msg, flush=True)


# ── 数据（与主脚本同一套口径，统一在 microcap_data.py）──
df = md.load(SOURCE)
members = md.build_pool(df, q=0.20)
port = md.daily_portfolio(df, members)
share = md.pool_amount_share(df, members)
log(f"组合日收益 {len(port):,} 天，{port.index.min().date()} ~ {port.index.max().date()}  (source={SOURCE})")


def evaluate(pos: pd.Series) -> tuple[float, float, float]:
    """返回 (最大回撤, 年化, 在市比例)。pos 是已 shift 过的仓位。"""
    p = pos.reindex(port.index).fillna(False).astype(bool)
    r = (port * p).fillna(0.0)
    nav = (1 + r).cumprod()
    dd = (nav / nav.cummax() - 1).min()
    ann = nav.iloc[-1] ** (244 / len(r)) - 1
    return dd, ann, p.mean()


dd0, ann0, _ = evaluate(pd.Series(True, index=port.index))
log(f"\n原版（无风控）: 最大回撤 {dd0:.1%}，年化 {ann0:.1%}")
log("（判据：回撤改善 > 5pp 且 年化保留 > 67% 记为 ✅）")


def verdict(dd, ann) -> str:
    return "✅" if (dd - dd0 > 0.05 and ann > ann0 * 2 / 3) else "  "


# ── S1 趋势：换均线长度 ──
log("")
log("=" * 78)
log("【S1 趋势】净值 > N 日均线")
log("=" * 78)
log(f"{'均线N':>6} {'最大回撤':>10} {'年化':>9} {'年化保留':>9} {'在市':>7}  判定")
nav = (1 + port).cumprod()
for n in [10, 20, 40, 60, 90, 120, 180, 250]:
    pos = (nav > nav.rolling(n).mean()).shift(1)
    dd, ann, im = evaluate(pos)
    log(f"{n:>6} {dd:>10.1%} {ann:>9.1%} {ann / ann0:>9.0%} {im:>7.1%}  {verdict(dd, ann)}")

# ── S3 拥挤度：换 均线 / 分位窗口 / 分位阈值 ──
log("")
log("=" * 78)
log("【S3 拥挤度】微盘成交额占比 的 N 日均值 < 过去 M 日的 q 分位")
log("=" * 78)
log(f"{'均线N':>6} {'窗口M':>7} {'分位q':>7} {'最大回撤':>10} {'年化':>9} {'年化保留':>9} {'在市':>7}  判定")
for n in [20, 60, 120]:
    crowd = share.rolling(n).mean()
    for m in [250, 500]:
        for q in [0.6, 0.7, 0.8, 0.9]:
            th = crowd.rolling(m).quantile(q)
            pos = (crowd < th).shift(1)
            dd, ann, im = evaluate(pos)
            log(f"{n:>6} {m:>7} {q:>7.0%} {dd:>10.1%} {ann:>9.1%} {ann / ann0:>9.0%} {im:>7.1%}  {verdict(dd, ann)}")

# ── 崩盘月是否躲过（对表现最好的一类做抽查）──
log("")
log("=" * 78)
log("崩盘月抽查：S3 (均线60, 窗口250) 在不同分位阈值下，能否躲开原版最惨的 6 个月")
log("=" * 78)
mret_base = (1 + port).groupby(port.index.to_period("M")).prod() - 1
crowd = share.rolling(60).mean()
qs = [0.6, 0.7, 0.8, 0.9]
log("  月份      原版  " + "".join(f"{'q=' + format(q, '.0%'):>9}" for q in qs))
for ym, v in mret_base.nsmallest(6).items():
    sub = port.loc[ym.start_time:ym.end_time]
    cells = []
    for q in qs:
        pos = (crowd < crowd.rolling(250).quantile(q)).shift(1).reindex(sub.index).fillna(False).astype(bool)
        cells.append(f"{((1 + (sub * pos)).prod() - 1):>9.1%}")
    log(f"  {ym} {v:>7.1%}  " + "".join(cells))
log("（数字 = 该月在该阈值下的实际收益。0.0% = 整月空仓躲开了；与『原版』相同 = 完全没躲）")
