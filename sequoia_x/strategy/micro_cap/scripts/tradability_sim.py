"""涨跌停成交约束模拟 —— 崩盘时你到底跑不跑得掉？

## 为什么必须做

前面所有回测都假设"想卖就能卖"。但真实情况是：
**2024-01 那种崩盘里，大量微盘股连续一字跌停，挂单根本卖不出去。**

风控（S3 空仓 / 硬止损）发出的"卖出"指令，在跌停板上**执行不了**。
⇒ 纸面上"回撤 −21.9%"，实际可能深得多。

## 建模口径

**一字板判定**（用日线 high/low，不需要分钟数据）：
- **卖不掉**：`高 == 低` 且 `当日跌幅 ≤ −9.5%`（全天一个价 = 一字跌停，无买盘）
- **买不到**：`高 == 低` 且 `当日涨幅 ≥ +9.5%`（一字涨停，无卖盘）

（初选池已排除创业/科创/北交/ST ⇒ 涨跌停是 ±10%，判定阈值取 ±9.5% 留余量）

**卖不掉的处置**：继续持有，**每个后续交易日自动重试**，直到能卖掉为止（期间照常涨跌）。

**买不到的处置**：该笔资金留现金，下一日重试。

## 与前面版本的差异（本脚本是真正的逐股逐日模拟）

前面用 `port * k - 换手成本` 的向量化写法 —— **那假设成交 100% 成功**。
本脚本改成**逐个持仓追踪**，才能表达"卖不掉就继续拿着"。

用法：
    /home/zhulei/anaconda3/envs/zhulei_py312/bin/python tradability_sim.py [db|ths]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import microcap_data as md
from strategy_v3_matrix import empty_position, log, prepare

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "ths"
N, STOP = 30, -0.12
SLIP = 0.0020
COMMISSION, TRANSFER = 0.00025, 0.00001
STAMP_BEFORE, STAMP_AFTER = 0.001, 0.0005
STAMP_CUTOFF = pd.Timestamp("2023-08-28")
LIMIT_TOL = 0.095          # 涨跌停判定阈值（±10% 留余量）

BASE_PY = Path(__file__).resolve().parents[1]

# ── 组合与信号（与 V1-V4 同一套）──
sel, port_ref, sig, hits = prepare(N, "equal", STOP)
k = empty_position(sig, "S3")
days = port_ref.index

# 日期 → 当月的池子
pool_by_ym = {ym: set(g["symbol"]) for ym, g in sel.groupby("hold_ym")}
day_ym = pd.Series(days, index=days).dt.to_period("M")

# ── 逐股数据（ret / 一字标志）──
d = md.apply_initial_pool(md.load(SOURCE))
d = d[d["symbol"].isin(set(sel["symbol"]))].copy()
d["_flat"] = (d["high"] - d["low"]).abs() < 1e-9
# ⚠️ 两个口径给「上下界」（实测差距巨大，必须都跑）：
#   ① `yizi`  只有**一字跌停**（全天一个价、完全无买盘）才算卖不掉 —— **乐观下界**
#   ② `limit` 只要**收盘跌停**就算卖不掉 —— **保守上界**
#   实测 2024-02-05：池内 44.2% 收盘跌停，但只有 0.3% 是一字 ⇒ 两个口径差 100 倍以上！
#   真实值在两个之间（跌停价上通常仍有零星买盘，小资金能卖、大资金卖不掉）。
d["cant_sell_yizi"] = d["_flat"] & (d["ret"] <= -LIMIT_TOL)
d["cant_sell_limit"] = d["ret"] <= -LIMIT_TOL
d["cant_buy"] = d["_flat"] & (d["ret"] >= LIMIT_TOL)
log(f"逐股数据: {len(d):,} 行 / {d['symbol'].nunique()} 只")
log(f"跌停收盘（保守口径「卖不掉」）: {int(d['cant_sell_limit'].sum()):,} 行 "
    f"（占 {d['cant_sell_limit'].mean():.3%}）")
log(f"其中一字跌停（乐观口径）: {int(d['cant_sell_yizi'].sum()):,} 行 "
    f"（占 {d['cant_sell_yizi'].mean():.3%}）")
log(f"一字涨停日（买不到）: {int(d['cant_buy'].sum()):,} 行")

RET = {s: g.set_index("date")["ret"] for s, g in d.groupby("symbol")}
CANTS_YIZI = {s: g.set_index("date")["cant_sell_yizi"] for s, g in d.groupby("symbol")}
CANTS_LIM = {s: g.set_index("date")["cant_sell_limit"] for s, g in d.groupby("symbol")}
CANTB = {s: g.set_index("date")["cant_buy"] for s, g in d.groupby("symbol")}
ALLD = pd.DatetimeIndex(days)


def simulate(mode: str) -> dict:
    """逐股逐日模拟。

    mode: "none" 无约束（=前面的口径）｜"yizi" 只有一字跌停卖不掉（乐观下界）
          ｜"limit" 收盘跌停就卖不掉（保守上界）
    """
    apply_constraint = mode != "none"
    CANTS = CANTS_YIZI if mode == "yizi" else CANTS_LIM
    held: dict[str, float] = {}      # symbol -> 市值
    entry: dict[str, float] = {}     # symbol -> 入场时点归一净值
    cash = 1.0
    prev_target: set[str] = set()
    nav_list = []
    stats = {"sell_blocked": 0, "buy_blocked": 0, "frozen": 0}

    for i, day in enumerate(ALLD):
        # ① 持仓按当日收益更新
        for s in list(held):
            r = RET[s].get(day, np.nan) if s in RET else np.nan
            if pd.isna(r):
                stats["frozen"] += 1         # 无行情（停牌/退市）⇒ 冻结，按 0 收益
            else:
                held[s] *= (1 + r)

        # ⚠️ `hold_ym` 在 pick_top_n 里已经等于「持有月」（= 选股月 + 1），
        #    这里**不能再 +1**，否则会用到下个月的池子（既错位又偷看未来）。
        ym = day_ym.iloc[i]
        target = pool_by_ym.get(ym, set()) if k.get(day, 0.0) > 0.5 else set()

        # ② 目标变化 ⇒ 调仓（等权）
        if target != prev_target:
            # 卖出不在目标里的
            for s in list(held):
                if s in target:
                    continue
                blocked = (apply_constraint and s in CANTS
                           and bool(CANTS[s].get(day, False)))
                if blocked:
                    stats["sell_blocked"] += 1
                    continue
                stamp = STAMP_AFTER if day >= STAMP_CUTOFF else STAMP_BEFORE
                cash += held.pop(s) * (1 - (COMMISSION + TRANSFER + SLIP + stamp))
                entry.pop(s, None)
            # 买入（用当前总资产等权分配）
            if target:
                tot = cash + sum(held.values())
                budget = tot / len(target)
                for s in target:
                    if s in held:
                        continue
                    amt = min(budget, cash)
                    if amt <= 1e-9:
                        break
                    blocked = (apply_constraint and s in CANTB
                               and bool(CANTB[s].get(day, False)))
                    if blocked:
                        stats["buy_blocked"] += 1
                        continue
                    cost = amt * (COMMISSION + TRANSFER + SLIP)
                    held[s] = amt - cost
                    # ⚠️ entry 必须记**入场时的绝对市值**，不能记 1.0 ——
                    #    否则止损判定 `held/entry - 1` 会变成"0.033/1 - 1 = −97%"，一建仓就全被止损掉。
                    entry[s] = amt - cost
                    cash -= amt
            prev_target = target
            # 注意：即使 constr=False 也走同一路径，保证两边只有"约束"一个变量不同

        # ③ 硬止损（每只每日检查）
        for s in list(held):
            if s in entry and held[s] / entry[s] - 1 <= STOP:
                blocked = (apply_constraint and s in CANTS
                           and bool(CANTS[s].get(day, False)))
                if blocked:
                    stats["sell_blocked"] += 1
                    continue
                stamp = STAMP_AFTER if day >= STAMP_CUTOFF else STAMP_BEFORE
                cash += held.pop(s) * (1 - (COMMISSION + TRANSFER + SLIP + stamp))
                entry.pop(s, None)

        nav_list.append(cash + sum(held.values()))

    nav = pd.Series(nav_list, index=ALLD)
    r = nav.pct_change().fillna(0.0)
    ann = nav.iloc[-1] ** (244 / len(nav)) - 1
    dd = (nav / nav.cummax() - 1).min()
    return {"年化": f"{ann:.1%}", "最大回撤": f"{dd:.1%}",
            "Calmar": f"{ann / abs(dd):.2f}" if dd < 0 else "—",
            "卖不掉次数": stats["sell_blocked"], "买不到次数": stats["buy_blocked"],
            "无行情持仓日": stats["frozen"], "_nav": nav}


log("")
log("=" * 100)
log("模拟（逐股逐日）")
log("=" * 100)
res = {}
for mode, tag in [("none", "① 忽略涨跌停（=前面所有版本的口径）"),
                  ("yizi", "② 只有一字跌停卖不掉（乐观下界）"),
                  ("limit", "③ 收盘跌停就卖不掉（保守上界）")]:
    res[mode] = simulate(mode)
    log(f"  {tag} 跑完")

rows = [{"方案": tag, **{k2: v for k2, v in res[m].items() if k2 != "_nav"}}
        for m, tag in [("none", "① 忽略涨跌停（=前面所有版本的口径）"),
                       ("yizi", "② 只有一字跌停卖不掉（乐观下界）"),
                       ("limit", "③ 收盘跌停就卖不掉（保守上界）")]]
log("")
log(pd.DataFrame(rows).to_string(index=False))
log("")
d0 = float(res["none"]["最大回撤"].rstrip("%"))
for m in ["yizi", "limit"]:
    log(f"  {m}: 最大回撤 {res[m]['最大回撤']}（相对① {float(res[m]['最大回撤'].rstrip('%')) - d0:+.1f}pp）")
log("")
log("怎么读：②③ 是**上下界**，真实值在中间（跌停价通常仍有零星买盘：小资金能卖、大资金卖不掉）。")

# 崩盘月专项
log("")
log("─" * 100)
log("崩盘月专项（含约束）：回撤最深那段的逐月表现")
log("─" * 100)
nav_c = res["limit"]["_nav"]
dd = nav_c / nav_c.cummax() - 1
tr = dd.idxmin(); pk = nav_c[:tr].idxmax()
log(f"最大回撤 {dd.min():.1%}   从 {pk.date()} 到 {tr.date()}")
seg = d[(d["date"] >= pk) & (d["date"] <= tr)]
log(f"  该期间：收盘跌停 {int(seg['cant_sell_limit'].sum()):,} 行 / 一字跌停 "
    f"{int(seg['cant_sell_yizi'].sum()):,} 行（共 {len(seg):,} 行）")

# 最惨的那一天到底有多少跌停
worst_day = d[d["date"] == seg.loc[seg["cant_sell_limit"].idxmax(), "date"]] if len(seg) else None
if worst_day is not None:
    log(f"  最惨单日（{worst_day['date'].iloc[0].date()}）：池内 {len(worst_day)} 只中，"
        f"跌停收盘 {int(worst_day['cant_sell_limit'].sum())} 只、一字跌停 {int(worst_day['cant_sell_yizi'].sum())} 只")

for m in ["none", "yizi", "limit"]:
    res[m]["_nav"].to_frame("nav").to_csv(BASE_PY / "data" / f"nav_tradability_{m}.csv")
log("")
log("✅ 净值已存: data/nav_tradability_{none,yizi,limit}.csv")
