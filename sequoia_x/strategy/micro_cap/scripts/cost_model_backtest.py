"""微盘策略「真实交易成本」回测 —— 加成本后，风控层还成立吗？

## 为什么必须做这一步

前面所有回测都假设「空仓/进场是瞬时的、零成本」。但实测两个风控信号**在市时间只有 60% 左右**
⇒ **会频繁进出**；而微盘股流动性差 ⇒ **冲击成本可能吃掉相当一部分收益**。

**这是目前唯一可能把结论翻盘的因素**，所以必须堵上。

## 成本怎么算

策略的换手来自两处：
1. **风控开关**：仓位 0↔1 切换 ⇒ 全仓卖出或全仓买入（换手 100%）
2. **每月换池**：月末重选池子，成分股替换 ⇒ 换手 = 替换比例

对一次「完整换手」（卖出 100% + 买入 100%）的成本：

    卖出：佣金 + 印花税 + 过户费 + 滑点
    买入：佣金 + 过户费 + 滑点
    ⇒ 合计 = 2×(佣金 + 过户费 + 滑点) + 印花税

| 项目 | 取值 | 说明 |
|---|---|---|
| 佣金 | 0.025% 单边 | 万 2.5，行业常见 |
| 过户费 | 0.001% 单边 | 万 0.1 |
| 印花税 | **卖出单边 0.05%**（2023-08-28 起）/ **0.1%**（之前） | 2023-08-28 官方减半，按日期分段 |
| **滑点** | **情景扫描 0 / 10 / 20 / 40 bp 单边** | 最不确定的一项，所以给足区间 |

## 三个已知的保守之处（都让结果**偏乐观**）

1. **换手只算了成分股替换，没算权重漂移的再平衡** ⇒ 真实换手略高。
2. **假设全部按收盘价成交**，没建模日内滑点的非线性。
3. **没建模涨跌停无法成交**（最后一节做单独估计）。
   这一条在崩盘时特别关键 —— 风控让你「走」，但跌停板上**根本卖不掉**。

用法：
    /home/zhulei/anaconda3/envs/zhulei_py312/bin/python cost_model_backtest.py [db|ths]
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
POOL_Q = 0.20
Q_WIN, Q_LEVEL = 250, 0.80
MA_S3 = 60
SLIP_S3 = 0.0020

# ── 费率常量（改这里即可）──
COMMISSION = 0.00025      # 佣金 万2.5 / 单边
TRANSFER = 0.00001        # 过户费 万0.1 / 单边
STAMP_BEFORE = 0.001      # 印花税（2023-08-28 前）卖出 0.1%
STAMP_AFTER = 0.0005      # 印花税（2023-08-28 起）卖出 0.05%
STAMP_CUTOFF = pd.Timestamp("2023-08-28")
SLIPPAGE_SCENARIOS = [0.0, 0.0010, 0.0020, 0.0040]   # 单边 0 / 10 / 20 / 40 bp

pd.set_option("display.width", 200)


def log(msg: str = "") -> None:
    print(msg, flush=True)


log("=" * 100)
log(f"微盘策略真实交易成本回测    数据源: {SOURCE}")
log(f"费率: 佣金 {COMMISSION:.4%}/边 + 过户费 {TRANSFER:.4%}/边 + 印花税 "
    f"{STAMP_BEFORE:.2%}→{STAMP_AFTER:.2%}(2023-08-28 减半，卖出单边) + 滑点 情景扫描")
log("=" * 100)

# ── 1. 数据 / 组合 / 信号（与风控回测完全同一套定义）──
df = md.load(SOURCE)
members = md.build_pool(df, q=POOL_Q)
port = md.daily_portfolio(df, members)
share = md.pool_amount_share(df, members)

ix = pd.read_parquet(INDEX_EXTRA)
bench_ret = ix[ix["symbol"] == "csi2000"].set_index("date")["close"].sort_index().pct_change()

MAS = [20, 60, 120, 250]      # 扫多档均线：趋势信号越短越敏感 ⇒ 换手越高 ⇒ 越可能被成本吃掉
sigs = {ma: md.build_signals(port, share, bench_ret, ma=ma, q_win=Q_WIN, q_level=Q_LEVEL)
        for ma in MAS}
start = max(s.dropna(how="any").index.min() for s in sigs.values())   # 取最晚的起点，保证都有值
for ma in MAS:
    sigs[ma] = sigs[ma].loc[start:]
port = port.reindex(sigs[MA_S3].index)


def _pos(s: pd.Series) -> pd.Series:
    """把「当天收盘的信号」变成「当天的仓位」——必须 shift(1)，否则是穿越。"""
    return s.shift(1).fillna(False).astype(bool)


strategies = {
    "① 原版（无风控）": pd.Series(True, index=port.index),
    f"② +S3 拥挤度({MA_S3}/{Q_WIN}/80)": _pos(sigs[MA_S3]["S3_拥挤度"]),
}
for ma in MAS:
    strategies[f"③ +S1 趋势(MA{ma})"] = _pos(sigs[ma]["S1_趋势"])
strategies[f"④ S1(MA{max(MAS)}) 与 S3 都允许才在场"] = _pos(
    sigs[max(MAS)]["S1_趋势"] & sigs[MA_S3]["S3_拥挤度"]
)

# ── 2. 换手率 ──
pool_by_ym = {ym: set(g["symbol"]) for ym, g in members.groupby("hold_ym")}
yms = sorted(pool_by_ym)
replace = {}
for i, ym in enumerate(yms):
    cur = pool_by_ym[ym]
    replace[ym] = 1.0 if i == 0 else 1.0 - len(cur & pool_by_ym[yms[i - 1]]) / len(cur)

first_day = {ym: g.min() for ym, g in pd.Series(port.index, index=port.index.to_period("M")).groupby(level=0)}


def turnover(pos: pd.Series) -> pd.Series:
    """每日换手率（1.0 = 整个组合卖光或买满）。"""
    pos = pos.reindex(port.index).fillna(False).astype(bool)
    t = pos.astype(float).diff().abs()
    t.iloc[0] = float(pos.iloc[0])                     # 首日建仓
    prev = pos.shift(1).fillna(False)
    for ym, d in first_day.items():
        # 月初换池：只有「上一天也在场」时才需要单独再平衡；
        # 若正好当天刚进场，那笔开关换手已经包含买入，不重复计。
        if pos.get(d, False) and prev.get(d, False):
            t.loc[d] = t.get(d, 0.0) + replace.get(ym, 0.0)
    return t.clip(upper=1.0)


def cost_rate(slip: float) -> pd.Series:
    """每个交易日「完整换手一次」的综合费率（印花税按时段）。"""
    stamp = pd.Series(
        np.where(port.index >= STAMP_CUTOFF, STAMP_AFTER, STAMP_BEFORE), index=port.index
    )
    return 2 * (COMMISSION + TRANSFER + slip) + stamp


def perf(r: pd.Series, pos: pd.Series) -> tuple[float, float, float, float]:
    """返回 (年化, 最大回撤, Calmar, 年换手次数)"""
    r = r.fillna(0.0)
    nav = (1 + r).cumprod()
    dd = (nav / nav.cummax() - 1).min()
    ann = nav.iloc[-1] ** (244 / len(r)) - 1
    return ann, dd, (ann / abs(dd) if dd < 0 else float("nan")), turnover(pos).sum() / (len(r) / 244)


log("")
log("=" * 100)
log("换手与成本明细")
log("=" * 100)
log(f"{'方案':<30}{'年换手次数':>12}{'年化成本@20bp滑点':>20}")
for name, pos in strategies.items():
    to = turnover(pos).sum() / (len(port) / 244)
    drag = to * float(2 * (COMMISSION + TRANSFER + 0.0020) + STAMP_AFTER)
    log(f"{name:<30}{to:>12.1f}{drag:>20.2%}")
log("（『年化成本』= 一年的换手带来的成本，直接从该策略的年化收益里减掉这么多）")

# ── 3. 各滑点情景下的净表现 ──
log("")
log("=" * 100)
log("净表现（已扣佣金 / 印花税 / 过户费 + 各档滑点）")
log("=" * 100)
rows = []
for name, pos in strategies.items():
    pos = pos.reindex(port.index).fillna(False).astype(bool)
    gross = port * pos
    for slip in SLIPPAGE_SCENARIOS:
        net = gross - turnover(pos) * cost_rate(slip)
        ann, dd, calmar, to = perf(net, pos)
        rows.append({"方案": name, "滑点/边": f"{slip * 1e4:.0f}bp",
                     "年化": f"{ann:.1%}", "最大回撤": f"{dd:.1%}",
                     "Calmar": f"{calmar:.2f}", "年换手": f"{to:.1f}"})
res = pd.DataFrame(rows)
for name in strategies:
    log("")
    log(f"【{name}】")
    log(res[res["方案"] == name][["滑点/边", "年化", "最大回撤", "Calmar", "年换手"]].to_string(index=False))
log("")
log("怎么读：一列列往下看『年化』掉得快不快。如果加到 40bp 滑点后年化还是正的、")
log("        且回撤仍明显小于原版 ⇒ 风控层的价值是真的；反之就是纸面收益。")

# ── 4. 涨跌停：风控让你「走」，但走得掉吗？──
log("")
log("=" * 100)
log("⚠️ 涨跌停成交约束：风控日的『卖出』到底能不能执行")
log("=" * 100)
d = df.copy()
d["hold_ym"] = d["ym"]
held = d.merge(members[["hold_ym", "symbol"]], on=["hold_ym", "symbol"], how="inner")
limit_dn = (
    held.assign(is_limit_dn=held["ret"] <= -0.098)      # 近似：主板跌停 −10%
    .set_index(["date", "symbol"])["is_limit_dn"]
)

log(f"{'方案':<30}{'离场次数':>10}{'离场日跌停股占比(中位)':>24}{'最惨一次':>12}")
for name, pos in strategies.items():
    p = pos.reindex(port.index).fillna(False).astype(bool)
    exits = p.index[(~p.values) & p.shift(1).fillna(False).values]
    fracs = []
    for day in exits:
        try:
            fracs.append(float(limit_dn.xs(day, level="date").mean()))
        except KeyError:
            continue
    fx = np.median(fracs) if fracs else float("nan")
    mx = np.max(fracs) if fracs else float("nan")
    log(f"{name:<30}{len(exits):>10}{fx:>24.1%}{mx:>12.1%}")
log("")
log("怎么读：『离场日跌停股占比』= 风控发出离场信号那天，池子里有多少比例的股票处于跌停。")
log("        这部分当天**卖不掉**，只能等下一个非跌停日 ⇒ **真实回撤会比本模型更深**。")

# ── 5. 容量粗估 ──
log("")
log("=" * 100)
log("容量粗估：这套策略能装多少钱")
log("=" * 100)
h = held.groupby("date").agg(amt=("amount", "sum"), n=("symbol", "nunique"))
med_adv, med_n = h["amt"].median(), h["n"].median()
log(f"微盘池 单日成交额中位数: {med_adv / 1e8:,.2f} 亿元（池内约 {med_n:.0f} 只）")
log(f"单只股票 单日成交额中位数: {med_adv / med_n / 1e4:,.0f} 万元")
log("")
log(f"{'组合规模':>10}{'单只下单额':>14}{'参与率':>12}")
for cap in [1e7, 5e7, 1e8, 5e8, 1e9]:
    log(f"{cap / 1e8:>8.1f}亿{cap / med_n / 1e4:>12,.0f}万{cap / med_n / (med_adv / med_n):>12.2%}")
log("")
log("怎么读：参与率 = 单只下单额 ÷ 当日成交额。经验上超过 5~10% 冲击成本就明显上升。")
