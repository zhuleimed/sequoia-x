"""量化「ST 过滤用今天的名字」这条未来函数，到底值多少。

## 问题是什么

早期脚本用 `stock_list.name`（**只有一个最新值**）剔除 ST 股。
这等于**提前知道了哪些股票"后来"会被 ST**，把最烂的那批在事发前就剔掉了 ⇒ **结果偏乐观**。

这是本项目目前**唯一确定的未来函数**。历史上项目因为这个吃过亏（见项目 CLAUDE.md 的教训），
所以不能只写一句"偏乐观"就算了，必须**量化**。

## 怎么量化

用三种口径各跑一遍同样的回测：
  A. `today` —— 用今天的名字剔除（现状，有未来函数，最乐观）
  B. `none`  —— 完全不过滤（无未来函数，但把 ST 也买了，最悲观/不现实）
  C. `proxy` —— **只用当时可见的价格信息**推断疑似 ST（无未来函数，接近现实）

A 与 C 的差距 ≈ 这条未来函数带来的虚高；C 落在 A 与 B 之间说明代理方向正确。

## 代理怎么做的

ST/*ST 股票的日涨跌幅被限制在 **±5%**（普通主板是 ±10%），所以：
「过去 20 日 |涨跌幅| **从未超过 5.4%**，且**至少有 1 天达到 4.6% 以上**」⇒ 判为疑似 ST。
只用 t 及之前的信息，**无未来函数**。

用法：
    /home/zhulei/anaconda3/envs/zhulei_py312/bin/python st_lookahead_check.py [db|ths]
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
SLIP = 0.0020        # 滑点情景固定用 20bp/边（中间档）
POOL_Q = 0.20

# 费率与 cost_model_backtest.py 保持一致（不改动那个脚本 —— 它是可执行脚本，import 会把它整段跑一遍）
COMMISSION, TRANSFER = 0.00025, 0.00001
STAMP_BEFORE, STAMP_AFTER = 0.001, 0.0005
STAMP_CUTOFF = pd.Timestamp("2023-08-28")

pd.set_option("display.width", 200)


def log(msg: str = "") -> None:
    print(msg, flush=True)


log("=" * 96)
log(f"ST 未来函数量化    数据源: {SOURCE}    滑点: {SLIP * 1e4:.0f}bp/边")
log("=" * 96)

# ── 第 1 步：代理准不准（拿今天的 ST 名单当答案对照）──
log("")
log("第 1 步：验证『价格反推 ST』代理的识别能力")
log("=" * 96)
raw = md.load(SOURCE, st_mode="none")          # 不剔除，才能看到全部股票
raw["_proxy"] = md.st_proxy_flag(raw)
raw["_today_st"] = raw["name"].astype(str).str.contains("ST", na=False)

recent = raw[raw["date"] >= raw["date"].max() - pd.Timedelta(days=365)]
by_sym = recent.groupby("symbol").agg(proxy=("_proxy", "mean"), st=("_today_st", "last"))
tp = by_sym[(by_sym["st"]) & (by_sym["proxy"] > 0.5)]
fn = by_sym[(by_sym["st"]) & (by_sym["proxy"] <= 0.5)]
fp = by_sym[(~by_sym["st"]) & (by_sym["proxy"] > 0.5)]
log(f"近一年被标为『今天仍是 ST』的股票: {int(by_sym['st'].sum())} 只")
log(f"  其中被代理**抓到**（>50% 的日子命中）: {len(tp)} 只  ⇒ 召回率 "
    f"{len(tp) / max(len(tp) + len(fn), 1):.1%}")
log(f"  代理**误报**（非 ST 却被标）: {len(fp)} 只  ⇒ 误报率 "
    f"{len(fp) / max(int((~by_sym['st']).sum()), 1):.1%}")
log("")
log("怎么读：召回率越高说明代理越能抓到真 ST；误报率越高说明它错杀的好股票越多。")
log("        这只是『代理准不准』的旁证，**真正的判据在第 2 步**（A 与 C 的回测差多少）。")

# ── 第 2 步：三种口径跑同一套回测 ──
log("")
log("第 2 步：三种 ST 口径下的回测对比")
log("=" * 96)

ix = pd.read_parquet(INDEX_EXTRA)
bench_ret = ix[ix["symbol"] == "csi2000"].set_index("date")["close"].sort_index().pct_change()


def run(mode: str) -> dict:
    df = md.load(SOURCE, st_mode=mode)
    members = md.build_pool(df, q=POOL_Q)
    port = md.daily_portfolio(df, members)
    share = md.pool_amount_share(df, members)
    sig = md.build_signals(port, share, bench_ret, ma=60, q_win=250, q_level=0.80)
    start = sig.dropna(how="any").index.min()
    sig, port = sig.loc[start:], port.reindex(sig.loc[start:].index)

    # 换手：月初换池 + S3 开关
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

    out = {}
    for label, pos in {
        "原版": pd.Series(True, index=port.index),
        "+S3 拥挤度": sig["S3_拥挤度"].shift(1).fillna(False),
    }.items():
        gross = port * pos.reindex(port.index).fillna(False).astype(bool)
        net = gross - to_of(pos) * rate
        nav = (1 + net.fillna(0)).cumprod()
        ann = nav.iloc[-1] ** (244 / len(net)) - 1
        dd = (nav / nav.cummax() - 1).min()
        out[label] = (ann, dd)
    out["_n"] = len(df)
    out["_pool"] = members.groupby("hold_ym").size().median()
    return out


rows = []
for mode, cn in [("today", "A 用今天的名字（有未来函数）"), ("none", "B 完全不剔除"), ("proxy", "C 价格反推（无未来函数）")]:
    r = run(mode)
    log(f"  跑完 {cn}: 保留 {r['_n']:,} 行，微盘池中位 {r['_pool']:.0f} 只")
    rows.append({
        "口径": cn,
        "原版年化": f"{r['原版'][0]:.1%}", "原版回撤": f"{r['原版'][1]:.1%}",
        "+S3 年化": f"{r['+S3 拥挤度'][0]:.1%}", "+S3 回撤": f"{r['+S3 拥挤度'][1]:.1%}",
    })

log("")
log("=" * 96)
log("结果（均已扣 20bp 滑点的真实成本）")
log("=" * 96)
log(pd.DataFrame(rows).to_string(index=False))
log("")
log("怎么读：对比 A 与 C —— **它们的差距就是这条未来函数带来的虚高**。")
log("        若 C 落在 A 和 B 之间，说明代理方向正确；若 A ≈ C，说明这条未来函数影响很小。")
