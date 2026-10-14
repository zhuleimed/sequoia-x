"""V4 —— 空仓策略深度研究（用户重点要求）。

## 先复盘：为什么前面 5 次失败

V3b 试过的 5 类**全部失败**（半仓、加慢趋势、趋势单独、波动率单独、信号组合）。
失败原因是**它们都是"策略自己"的信号**（策略净值、策略波动）。

诊断已查明：剩下的 **−26.6% 最大回撤发生在 2019-04-19 ~ 2019-08-09，
那 77 个交易日 S3 一次都没触发** —— 因为那是 **跟随大盘** 的下跌，不是"微盘独立崩盘"。

⇒ **这次换真正不同类的东西：市场级信号。**

## 本脚本测三大类

### A. 市场级趋势（管"跟随大盘"的回撤）
  `CSI2000_MA60/120/250` 中证2000 跌破均线 ⇒ 空仓
  `HS300_MA60/120`       沪深300 跌破均线 ⇒ 空仓
  `BREADTH`              全市场**上涨家数占比**的均线过低 ⇒ 空仓（市场宽度）

### B. 拥挤度指标的改良（同一目标，换度量）
  `S3`      现状：微盘成交额占比的 60 日均值 < 过去 250 日的 80% 分位
  `S3_CHG`  改良：占比的 **20 日变化率**过高（拥挤在"加速"）⇒ 空仓
  `S3_TURN` 改良：微盘**换手率**（成交额/流通市值）过高 ⇒ 空仓

### C. 波动率目标（连续调仓，不是开关）
  `VOLTGT`  仓位 = min(1, 目标波动率 / 近 20 日已实现波动率)
            —— 波动大时自动减仓、波动小时满仓（与"开关"是两种不同机制）

用法：
    /home/zhulei/anaconda3/envs/zhulei_py312/bin/python strategy_v4_empty_study.py [db|ths]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import microcap_data as md
from strategy_v3_matrix import BASE, df, evaluate, empty_position, log, prepare

PROJ = Path(__file__).resolve().parents[4]
INDEX_EXTRA = BASE / "data" / "index_extra.parquet"

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "ths"
N, STOP = 30, -0.12

log("=" * 100)
log(f"V4 空仓策略深度研究    数据源: {SOURCE}    配置: {N} 只 / 等权 / 硬止损 −12%")
log("=" * 100)

# ── 基础组合 ──
sel, port, sig, hits = prepare(N, "equal", STOP)
idx = port.index

# ── 市场数据（中证2000 在 index_extra；沪深300 从 004 库取）──
ix = pd.read_parquet(INDEX_EXTRA)
mkt = pd.DataFrame(index=idx)
# ⚠️ 只用 index_extra 里的指数（2013-12 起，全覆盖）。
#    不要用 004 库的 index_daily —— 沪深300 在库里只有 2020 年以后的数据，
#    直接拿来做均线会让 2016-2019 全被判成"跌破均线"⇒ 那段被误当空仓 ⇒ 结果虚高。
for key, col in [("csi2000", "CSI2000"), ("gn2000", "GN2000"), ("csi1000", "CSI1000")]:
    s = ix[ix["symbol"] == key].set_index("date")["close"].sort_index()
    mkt[col] = s.reindex(idx)

# 全市场宽度：当日上涨家数占比
up = df.assign(_u=(df["ret"] > 0).astype(float)).groupby("date")["_u"].mean()
mkt["BREADTH"] = up.reindex(idx)

log(f"市场数据对齐: {mkt.notna().sum().to_dict()}")

# ── 拥挤度原料 ──
share = md.pool_amount_share(df, sel).reindex(idx)
share_long = share.rolling(60).mean()

# 微盘换手率：池内成交额 / 池内流通市值
d2 = df.copy()
d2["hold_ym"] = d2["ym"]
held = d2.merge(sel[["hold_ym", "symbol"]], on=["hold_ym", "symbol"], how="inner")
turn_rate = (held.groupby("date")["amount"].sum()
             / held.groupby("date")["float_mv"].sum().replace(0, np.nan)).reindex(idx)


def rule(cond: pd.Series) -> pd.Series:
    """把布尔条件转成 0/1 仓位（已 shift(1)，不穿越）。NaN 一律当"允许持有"。

    ⚠️ 必须先把 NaN 转成"允许"再 shift，否则数据缺失期会被当成"触发空仓"⇒ 结果虚高。
    """
    c = cond.astype("float").where(cond.notna(), 1.0)
    return c.shift(1).fillna(1.0)


def mkt_ma(col: str, n: int) -> pd.Series:
    px = mkt[col]
    return px > px.rolling(n).mean()


rows = []
k_all = pd.Series(1.0, index=idx)
rows.append(evaluate(port, sig, sel, k_all, hits, "不空仓（对照）", N, "等权", "−12%"))
k_s3 = empty_position(sig, "S3")
rows.append(evaluate(port, sig, sel, k_s3, hits, "S3 拥挤度（现状基准）", N, "等权", "−12%"))
log("  基准跑完")

# ── A. 市场级趋势 ──
for col in ["CSI2000", "CSI1000", "GN2000"]:
    for n in [60, 120, 250]:
        k = rule(mkt_ma(col, n))
        rows.append(evaluate(port, sig, sel, k, hits, f"A: {col} 跌破 MA{n}", N, "等权", "−12%"))
log("  A 组（市场级趋势）跑完")

# ── A2. 市场宽度 ──
b = mkt["BREADTH"]
for n in [20, 60]:
    k = rule(b.rolling(n).mean() > 0.45)
    rows.append(evaluate(port, sig, sel, k, hits, f"A: 宽度{n}日均 > 45%", N, "等权", "−12%"))

# ── B. 拥挤度改良 ──
for w in [20, 60]:
    chg = share - share.shift(w)
    k = rule(chg < chg.rolling(250).quantile(0.80))
    rows.append(evaluate(port, sig, sel, k, hits, f"B: 拥挤度{w}日变化率未过热", N, "等权", "−12%"))
tr60 = turn_rate.rolling(60).mean()
k = rule(tr60 < tr60.rolling(250).quantile(0.80))
rows.append(evaluate(port, sig, sel, k, hits, "B: 微盘换手率未过热", N, "等权", "−12%"))
log("  B 组（拥挤度改良）跑完")

# ── C. 波动率目标（连续调仓）──
rv = port.rolling(20).std() * np.sqrt(244)
for tgt in [0.25, 0.35]:
    k = (tgt / rv).clip(upper=1.0).shift(1).fillna(1.0)
    rows.append(evaluate(port, sig, sel, k, hits, f"C: 波动率目标 {tgt:.0%}", N, "等权", "−12%"))

# ── D. 组合：S3 或 市场趋势（互补）──
for col, n in [("CSI2000", 120), ("CSI2000", 250), ("CSI1000", 120)]:
    k = (k_s3 * rule(mkt_ma(col, n))).clip(0, 1)
    rows.append(evaluate(port, sig, sel, k, hits, f"D: S3 或 {col}MA{n} 任一即撤", N, "等权", "−12%"))
log("  C/D 组跑完")

log("")
res = pd.DataFrame(rows)
res["_c"] = pd.to_numeric(res["Calmar"], errors="coerce")
res = res.sort_values("_c", ascending=False).drop(columns="_c")
log("=" * 110)
log("结果（按 Calmar 降序）")
log("=" * 110)
log(res.to_string(index=False))
log("")
log("怎么读：")
log("  · 『在市』= 平均仓位（波动率目标那几行是连续值，不是 0/1 开关）。")
log("  · A 组管『跟随大盘』的回撤；B 组管『微盘独立崩盘』；C 组是连续调仓；D 组是互补组合。")

res.to_csv(BASE / "data" / "v4_empty_study.csv", index=False)
log("")
log(f"✅ 全表已存: {BASE / 'data' / 'v4_empty_study.csv'}")
