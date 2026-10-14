"""叠加研究 + 选股端降 beta（用户要求的两件事）。

## 任务 1：相关性表 —— 叠加到底有没有用

**前提**：其它资产要能在**微盘崩盘的时候不崩**才有分散价值。
所以除了全样本相关性，**重点看"微盘崩盘月"里它们跌多少**。

同场还测**简单叠加**（微盘 X% + 候选 1−X%）的效果 —— 用构造性判据看回撤/beta，不看收益高低
（按项目纪律：MDE ≈1.5%/月，收益类差异测不出）。

## 任务 2：选股端降 beta

诊断结论：剩下的大回撤是"普涨后的回调"= **高 beta 暴露** ⇒ 从选股源头压低。

做法（**保留微盘为主**）：先在初选池里取**市值最小的 M 只**，再在这 M 只里
按**过去 N 日波动率**取最低的 30 只。
对照：纯按市值最小取 30 只。

用法：
    /home/zhulei/anaconda3/envs/zhulei_py312/bin/python overlay_and_lowvol_study.py [db|ths]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import microcap_data as md
from strategy_v3_matrix import BASE, df, empty_position, evaluate, log, prepare

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "ths"
N, STOP = 30, -0.12
VOL_WIN, VOL_MIN = 120, 60          # 波动率估计窗口（交易日）
CAND_M = 100                        # 先在市值最小的 M 只里，再筛低波动

pd.set_option("display.width", 200)

# ── 基础组合（基准）──
sel, port, sig, hits = prepare(N, "equal", STOP)
k_s3 = empty_position(sig, "S3")
base = evaluate(port, sig, sel, k_s3, hits, "基准：纯市值最小 30 只", N, "等权", "−12%")
log("")
log("=" * 100)
log("基准（对照用）")
log("=" * 100)
log(pd.DataFrame([base]).to_string(index=False))

# ══════════════════════════════════════════════════════════
#  任务 1：相关性 + 叠加
# ══════════════════════════════════════════════════════════
log("")
log("=" * 100)
log("任务 1：候选资产 vs 微盘策略 —— 相关性 / 崩盘月表现 / 简单叠加")
log("=" * 100)

ast = pd.read_parquet(BASE / "data" / "assets_extra.parquet")
ast["date"] = pd.to_datetime(ast["date"])
ix = pd.read_parquet(BASE / "data" / "index_extra.parquet")
ix["date"] = pd.to_datetime(ix["date"])
for key, col in [("csi2000", "中证2000"), ("gn2000", "国证2000")]:
    s = ix[ix["symbol"] == key][["date", "close"]].copy()
    s["symbol"] = col
    ast = pd.concat([ast, s], ignore_index=True)

NAME = {"div_sh": "上证红利(高股息)", "sz50": "上证50", "hs300": "沪深300",
        "csi500": "中证500", "gem": "创业板指", "bond": "上证国债",
        "中证2000": "中证2000", "国证2000": "国证2000"}

# 月度收益（末日收盘比）
mret = {}
for key, g in ast.groupby("symbol"):
    s = g.set_index("date")["close"].sort_index()
    mret[NAME.get(key, key)] = s.resample("ME").last().pct_change()

# 微盘策略的月度收益（含空仓+止损）
strat_m = (1 + (port * k_s3).fillna(0)).groupby(port.index.to_period("M")).prod() - 1
strat_m.index = strat_m.index.to_timestamp("M")

mm = pd.DataFrame(mret)
mm["微盘策略"] = strat_m.reindex(mm.index)
mm = mm.dropna(how="all")

rows = []
for c in mm.columns:
    if c == "微盘策略":
        continue
    both = mm[["微盘策略", c]].dropna()
    crash = both[both["微盘策略"] <= -0.05]          # 微盘单月跌 ≥5% 视为崩盘月
    rows.append({
        "候选资产": c,
        "全样本相关性": f"{both['微盘策略'].corr(both[c]):+.2f}",
        "崩盘月数": len(crash),
        "崩盘月平均收益": f"{crash[c].mean():.1%}" if len(crash) else "—",
        "年化": f"{((1 + mm[c].dropna()).prod() ** (12 / mm[c].notna().sum()) - 1):.1%}",
        "最大回撤": f"{((1 + mm[c].dropna()).cumprod() / (1 + mm[c].dropna()).cumprod().cummax() - 1).min():.1%}",
    })
corr = pd.DataFrame(rows).sort_values("全样本相关性")
log("")
log(f"微盘策略月收益 ≤ −5% 的「崩盘月」共 {int((strat_m <= -0.05).sum())} 个")
log(corr.to_string(index=False))
log("")
log("怎么读：")
log("  · 『崩盘月平均收益』为正 = 微盘崩的时候它还涨 ⇒ **真分散**；也是负的 ⇒ 叠了没用。")
log("  · 『全样本相关性』越接近 0 越好；高股息/国债若接近 0 或为负，才有叠加价值。")

# 简单叠加：微盘 w + 候选 1−w
log("")
log("─" * 100)
log("简单叠加（微盘 + 单一候选，月度再平衡）：看回撤/beta 降多少（按纪律不看收益高低）")
log("─" * 100)
base_m = mm["微盘策略"].dropna()
mu = mm[["沪深300"]].reindex(base_m.index)["沪深300"]
log(f"{'候选':<16}{'微盘占比':>8}{'年化':>9}{'最大回撤':>10}{'对沪深300 beta':>14}")
for c in ["上证红利(高股息)", "上证国债", "中证2000"]:
    cs = mm[c].reindex(base_m.index).fillna(0)
    for w in [1.0, 0.8, 0.6]:
        r = w * base_m + (1 - w) * cs
        nav = (1 + r).cumprod()
        dd = (nav / nav.cummax() - 1).min()
        ann = nav.iloc[-1] ** (12 / len(r)) - 1
        b = np.cov(r, mu.fillna(0))[0, 1] / np.var(mu.fillna(0))
        log(f"{c:<16}{w:>8.0%}{ann:>9.1%}{dd:>10.1%}{b:>14.2f}")

# ══════════════════════════════════════════════════════════
#  任务 2：选股端降 beta（市值最小的 M 只 → 再取低波动 30 只）
# ══════════════════════════════════════════════════════════
log("")
log("=" * 100)
log("任务 2：选股端降 beta —— 先在市值最小的 100 只里，再按低波动取 30 只")
log("=" * 100)

d = df.sort_values(["symbol", "date"]).copy()
log("计算个股滚动波动率 ...")
d["vol"] = d.groupby("symbol")["ret"].transform(
    lambda s: s.rolling(VOL_WIN, min_periods=VOL_MIN).std()
)
log(f"  波动率可算出: {d['vol'].notna().mean():.1%}")

is_me = d["date"] == d.groupby("ym")["date"].transform("max")
me = d[is_me].dropna(subset=["total_mv", "vol"]).copy()
me = me.sort_values(["ym", "total_mv"])
me["_mv_rank"] = me.groupby("ym").cumcount()
cand = me[me["_mv_rank"] < CAND_M]                    # 市值最小的 100 只
cand = cand.sort_values(["ym", "vol"])
cand["_vol_rank"] = cand.groupby("ym").cumcount()
sel_lv = cand[cand["_vol_rank"] < N][["ym", "symbol"]].copy()
sel_lv["hold_ym"] = sel_lv["ym"] + 1
log(f"  低波动选股：每月 {sel_lv.groupby('hold_ym').size().median():.0f} 只")

from strategy_v3_matrix import build_port, me_dates  # noqa: E402

sel_lv["ref_date"] = sel_lv["ym"].map(me_dates)
port_lv, hits_lv = build_port(sel_lv, "equal", STOP)
share_lv = md.pool_amount_share(d, sel_lv)
sig_lv = md.build_signals(port_lv, share_lv,
                          ast[ast["symbol"] == "csi2000"].set_index("date")["close"]
                          .sort_index().pct_change(), ma=60, q_win=250, q_level=0.80)
start = sig_lv.dropna(how="any").index.min()
sig_lv, port_lv = sig_lv.loc[start:], port_lv.reindex(sig_lv.loc[start:].index)
k_lv = empty_position(sig_lv, "S3")

rows2 = [base,
         evaluate(port_lv, sig_lv, sel_lv, pd.Series(1.0, index=sig_lv.index), hits_lv,
                  "低波动选股（不空仓）", N, "等权", "−12%"),
         evaluate(port_lv, sig_lv, sel_lv, k_lv, hits_lv,
                  "低波动选股 + S3 空仓", N, "等权", "−12%")]
log("")
log(pd.DataFrame(rows2).to_string(index=False))
log("")
log("怎么读：与『基准』对比 —— 回撤降了多少？在市/换手有没有异常？")
