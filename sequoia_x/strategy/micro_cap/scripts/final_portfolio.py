"""最终方案 —— 微盘 80% + 国债 20%（用户 2026-10-13 拍板）。

## 方案全貌

```
初选池：  非北交/非科创/非创业 + 非ST + 非退市 + 上市>250日 + 股价>2元   （实测 3,209 只）
选股：    每月末按「总市值最小」取 30 只
持仓：    等权（**不做低波动筛选** —— 实测反而更差，见 叠加与降beta研究）
卖出：    月内硬止损 −12% + 月末清仓重建
空仓：    S3 拥挤度（微盘成交额占比过热 ⇒ 整段空仓）
组合层：  **微盘 80% + 上证国债 20%**，每月末再平衡回 80/20
```

## 本脚本要在 V1-V4 之外补上的三件事

1. **把国债仓位真正接进来**（前面只做了"月度粗略叠加"，现在是日频 + 真实再平衡成本）
2. **算准再平衡成本**（国债免印花税，佣金/过户费/滑点照算）
3. **对比两种"空仓时钱去哪"**

| 变体 | 微盘空仓时，那部分钱 | 说明 |
|---|---|---|
| `A` | **持现金（0 收益）** | 最保守的读法：80% 微盘 + 20% 国债各自独立 |
| `B` | **转入国债计息** | 更实际：空仓的资金不该白躺着 |

用法：
    /home/zhulei/anaconda3/envs/zhulei_py312/bin/python final_portfolio.py [db|ths]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from strategy_v3_matrix import BASE, empty_position, log, prepare

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "ths"
N, STOP = 30, -0.12
W_MICRO = 0.80                 # 微盘占比（用户拍板）
SLIP = 0.0020
COMMISSION, TRANSFER = 0.00025, 0.00001
STAMP_BEFORE, STAMP_AFTER = 0.001, 0.0005
STAMP_CUTOFF = pd.Timestamp("2023-08-28")

pd.set_option("display.width", 200)

# ── 微盘子策略（含空仓、止损、成本）──
sel, port, sig, hits = prepare(N, "equal", STOP)
k = empty_position(sig, "S3")
idx = port.index

# 微盘子策略的换手（月初换仓 + S3 开关）
w_by_ym = {ym: set(g["symbol"]) for ym, g in sel.groupby("hold_ym")}
yms = sorted(w_by_ym)
repl = {ym: (1.0 if i == 0 else 1.0 - len(w_by_ym[ym] & w_by_ym[yms[i - 1]]) / len(w_by_ym[ym]))
        for i, ym in enumerate(yms)}
first_day = {ym: x.min() for ym, x in pd.Series(idx, index=idx.to_period("M")).groupby(level=0)}
stamp = pd.Series(np.where(idx >= STAMP_CUTOFF, STAMP_AFTER, STAMP_BEFORE), index=idx)
rate_micro = 2 * (COMMISSION + TRANSFER + SLIP) + stamp
t_micro = k.astype(float).diff().abs()
t_micro.iloc[0] = float(k.iloc[0])
prev = k.shift(1).fillna(False)
for ym, d in first_day.items():
    if k.get(d, False) and prev.get(d, False):
        t_micro.loc[d] = t_micro.get(d, 0.0) + repl.get(ym, 0.0)
t_micro = t_micro.clip(upper=1.0)

micro_ret = (port * k - t_micro * rate_micro).fillna(0.0)

# ── 国债 ──
ast = pd.read_parquet(BASE / "data" / "assets_extra.parquet")
ast["date"] = pd.to_datetime(ast["date"])
bond_px = (ast[ast["symbol"] == "bond"].set_index("date")["close"].sort_index())
bond_ret = bond_px.pct_change().reindex(idx).fillna(0.0)

log("=" * 100)
log(f"最终方案：微盘 {W_MICRO:.0%} + 国债 {1 - W_MICRO:.0%}    数据源: {SOURCE}")
log(f"区间 {idx.min().date()} ~ {idx.max().date()}（{(len(idx) / 244):.1f} 年）")
log("=" * 100)

# ⚠️ 国债指数 2016-01 起才有；对齐后前段可能缺 ⇒ 缺失期视为 0 收益（不猜）
log(f"国债数据覆盖: {int((bond_ret != 0).sum()):,}/{len(bond_ret):,} 天")


def simulate(variant: str, w_micro: float) -> pd.Series:
    """逐日模拟：两条腿各自涨跌，每月末再平衡回目标权重（含成本）。"""
    vm, vb = w_micro, 1.0 - w_micro
    out = []
    is_first = idx.isin(list(first_day.values()))
    for i, d in enumerate(idx):
        r_m = micro_ret.iloc[i]
        r_b = bond_ret.iloc[i]
        # 微盘空仓时那部分钱的去向
        r_m_eff = r_m + (1.0 - k.iloc[i]) * r_b if variant == "B" else r_m
        vm *= (1 + r_m_eff)
        vb *= (1 + r_b)
        tot = vm + vb
        if is_first[i]:
            tm, tb = w_micro * tot, (1 - w_micro) * tot
            # 逐腿算成本：国债免印花税
            cost = 0.0
            if vm > tm:                                   # 卖微盘
                amt = vm - tm
                cost += amt * (COMMISSION + TRANSFER + SLIP + float(stamp.iloc[i]))
            elif tm > vm:                                 # 买微盘
                cost += (tm - vm) * (COMMISSION + TRANSFER + SLIP)
            if vb > tb:                                   # 卖国债（无印花税）
                cost += (vb - tb) * (COMMISSION + TRANSFER + SLIP)
            elif tb > vb:
                cost += (tb - vb) * (COMMISSION + TRANSFER + SLIP)
            tot -= cost
            vm, vb = w_micro * tot, (1 - w_micro) * tot
        out.append(vm + vb)
    nav = pd.Series(out, index=idx)
    return nav / nav.iloc[0]


def perf(nav: pd.Series) -> dict:
    r = nav.pct_change().fillna(0.0)
    ann = nav.iloc[-1] ** (244 / len(nav)) - 1
    dd = (nav / nav.cummax() - 1).min()
    return {"年化": f"{ann:.1%}", "最大回撤": f"{dd:.1%}",
            "Calmar": f"{ann / abs(dd):.2f}" if dd < 0 else "—"}


rows = []
navs = {}
pure = simulate("A", 1.0)
navs["纯微盘（100%）"] = pure
rows.append({"方案": "纯微盘（100%）", **perf(pure)})

for w in [0.9, 0.8, 0.7, 0.6]:
    for v, vtag in [("A", "A 空仓持现金"), ("B", "B 空仓转国债")]:
        nav = simulate(v, w)
        navs[f"微盘 {w:.0%} + 国债 {1 - w:.0%}（{vtag}）"] = nav
        rows.append({"方案": f"微盘 {w:.0%} + 国债 {1 - w:.0%}（{vtag}）", **perf(nav)})

log("")
log(pd.DataFrame(rows).to_string(index=False))
log("")
log("怎么读：")
log("  · 『A 空仓持现金』= 微盘空仓那段钱完全不计息（保守）；『B 空仓转国债』= 那部分也吃国债利息（实际）。")
log("  · 成本已含：每腿佣金+过户费+滑点(20bp)，微盘卖出另加印花税；国债免印花税。")

# ── 关键风险指标 ──
log("")
log("─" * 100)
log("关键风险指标（用户定位：低 beta、低回撤地持有）")
log("─" * 100)
hs = ast[ast["symbol"] == "hs300"].set_index("date")["close"].sort_index().pct_change().reindex(idx).fillna(0)
for name, nav in navs.items():
    r = nav.pct_change().fillna(0.0)
    beta = np.cov(r, hs)[0, 1] / np.var(hs) if np.var(hs) > 0 else float("nan")
    worst_m = (1 + r).groupby(idx.to_period("M")).prod().nsmallest(1).iloc[0] - 1
    log(f"  {name:<34} beta {beta:>5.2f}   最惨单月 {worst_m:>7.1%}")

navs["纯微盘（100%）"].to_frame("nav").to_csv(BASE / "data" / "final_nav_pure.csv")
navs[f"微盘 80% + 国债 20%（B 空仓转国债）"].to_frame("nav").to_csv(BASE / "data" / "final_nav_80_20.csv")
log("")
log(f"✅ 净值已存: data/final_nav_pure.csv / data/final_nav_80_20.csv")

# ══════════════════════════════════════════════════════════
#  正式实验产物（供 exp_contract.py 的 postflight 校验）
# ══════════════════════════════════════════════════════════
# postflight 会检查 <outdir>/bt_*_s*/monthly_returns.csv 的**首行月份**是否全落在声明窗口内，
# 以此发现"声明 6 个月、实际跑了 69 个月"这类事故 ⇒ 必须产出这个格式。
OUTDIR = Path(sys.argv[2]) if len(sys.argv) > 2 else (BASE / "output")
RUN = OUTDIR / "bt_microcap_s1"
RUN.mkdir(parents=True, exist_ok=True)

key = f"微盘 {W_MICRO:.0%} + 国债 {1 - W_MICRO:.0%}（B 空仓转国债）"
nav_final = navs[key]
monthly = (1 + nav_final.pct_change().fillna(0)).groupby(nav_final.index.to_period("M")).prod() - 1
months = [str(m) for m in monthly.index]
# 首行 = 月份表头（首格空），后续行 = 该方案的月度收益
pd.DataFrame({key: monthly.values}, index=months).T.to_csv(RUN / "monthly_returns.csv", header=months)
nav_final.to_frame("nav").to_csv(RUN / "daily_nav.csv")
pd.DataFrame(rows).to_csv(RUN / "summary.csv", index=False)
(RUN / "run_info.txt").write_text(
    f"name=final_portfolio_80_20\nsource={SOURCE}\nwindow={idx.min().date()}~{idx.max().date()}\n"
    f"days={len(idx)}\nn_micro={N}\nstop={STOP}\nw_micro={W_MICRO}\n",
    encoding="utf-8",
)
log(f"✅ 正式实验产物: {RUN}/  (monthly_returns.csv / daily_nav.csv / summary.csv)")
log(f"   产出月份 {months[0]} ~ {months[-1]}（共 {len(months)} 个）")
