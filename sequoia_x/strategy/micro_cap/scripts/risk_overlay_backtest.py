"""微盘策略「风控层」回测 —— 核心问题：能不能躲开那次崩盘？

## 为什么风控是这个策略的生死线

历史复盘（见 `../历史复盘_20261013.md`）结论：
- 微盘长期超额很大（7 年 6 年跑赢沪深300）；
- 但它会**独立崩盘**：2024-01 单月 −21%、2022-04 −16.5%，**而同期大盘几乎没跌**。

⇒ 一次崩盘就能抹掉多年超额。**所以本脚本只研究一件事：加一层择时能不能显著降低回撤。**

## 设计原则（防止自欺欺人）

1. **判据跑前定死**（写在这里，事后不改）：
   - 主判据：**最大回撤明显下降**（构造性判据，是算出来的、不是运气）
   - 次判据：年化收益不要大幅下降（掉超过 1/3 就不划算）
2. **参数不调优**：均线用 60 日、分位用 250 日的 80% —— 都是行业常用值，
   **不做网格搜索**。理由：样本里只有 2 次崩盘，任何「调出来的最优参数」都是过拟合。
3. **只用过去信息**：第 t 天的仓位由**第 t−1 天收盘时**的信号决定（`shift(1)`）。
4. **基准是「不加风控的原版」**，不是指数 —— 因为要回答的是「这层风控有没有用」。

## 4 个候选信号（都是「微盘自己」的指标，不是大盘的）

| 代号 | 名字 | 直觉 |
|---|---|---|
| S1 | 趋势 | 净值 > 60日均线 才持有（经典趋势跟踪） |
| S2 | **相对趋势：微盘 vs 中证2000** | 微盘**相对走弱**就撤（直击「独立崩盘」） |
| S3 | 拥挤度 | 微盘成交额占全市场比例过高（250日 80% 分位以上）就撤 |
| S4 | 波动率 | 微盘已实现波动过高就撤 |

用法：
    /home/zhulei/anaconda3/envs/zhulei_py312/bin/python risk_overlay_backtest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

import microcap_data as md       # 共用数据层：过滤 / 选池 / 日度收益

PROJ = Path(__file__).resolve().parents[4]
BASE = PROJ / "sequoia_x" / "strategy" / "micro_cap"
INDEX_EXTRA = BASE / "data" / "index_extra.parquet"

# 数据源：db = 004 库（2020-01 起，6.8 年，2 次崩盘）
#         ths = 同花顺十年 dump（2016-10 起，10 年，含 2017-18 小盘熊市）
SOURCE = sys.argv[1] if len(sys.argv) > 1 else "db"

pd.set_option("display.width", 220)

# ── 跑前定死的参数（不调优）──
POOL_Q = 0.20                 # 微盘池 = 总市值最小的 20%
MA_FAST = 60                  # 趋势均线长度
Q_WIN, Q_LEVEL = 250, 0.80    # 分位窗口 / 阈值

# ── 跑前定死的判据 ──
DD_REDUCE_MIN = 0.05          # 最大回撤至少改善 5 个百分点才算「有用」
RET_KEEP_MIN = 2 / 3          # 年化收益至少保留 2/3


def log(msg: str = "") -> None:
    print(msg, flush=True)


log("=" * 92)
log(f"微盘策略风控层回测    数据源: {SOURCE}")
log(f"参数（跑前定死）: 微盘池=总市值最小 {POOL_Q:.0%} | 趋势均线 {MA_FAST} 日 | 分位 {Q_WIN} 日 {Q_LEVEL:.0%}")
log(f"判据（跑前定死）: 最大回撤改善 ≥ {DD_REDUCE_MIN:.0%} 且 年化收益保留 ≥ {RET_KEEP_MIN:.0%}")
log("=" * 92)

# ── 1~3. 数据（过滤 / 选池 / 日度收益，口径统一在 microcap_data.py）──
df = md.load(SOURCE)
members = md.build_pool(df, q=POOL_Q)
port = md.daily_portfolio(df, members)
share = md.pool_amount_share(df, members)
log(f"每月微盘池规模中位数: {members.groupby('hold_ym').size().median():.0f} 只")
log(f"日度组合收益: {len(port):,} 天  {port.index.min().date()} ~ {port.index.max().date()}")

# ── 4. 基准：中证2000 ──
ix = pd.read_parquet(INDEX_EXTRA)
b = ix[ix["symbol"] == "csi2000"].set_index("date")["close"].sort_index()
bench_ret = b.pct_change().rename("bench_ret")

# ── 5. 构造信号（全部只用 t−1 及之前的信息来决定 t 的仓位）──
sig = md.build_signals(port, share, bench_ret, ma=MA_FAST, q_win=Q_WIN, q_level=Q_LEVEL)

# 有效起点 = 所有信号都能算出来的第一天
start = sig.dropna(how="any").index.min()
sig = sig.loc[start:]
port_o = port.reindex(sig.index)
bench_o = bench_ret.reindex(sig.index)
log(f"回测区间（所有信号可算）: {sig.index.min().date()} ~ {sig.index.max().date()}，{len(sig):,} 天")


def perf(r: pd.Series, pos: pd.Series | None, name: str) -> dict:
    r = r.fillna(0.0)
    n_years = len(r) / 244
    navv = (1 + r).cumprod()
    dd = (navv / navv.cummax() - 1).min()
    tot = navv.iloc[-1] - 1
    ann = (1 + tot) ** (1 / n_years) - 1 if n_years > 0 else float("nan")
    inmk = f"{pos.mean():.1%}" if pos is not None else "100.0%"
    return {
        "方案": name,
        "总收益": f"{tot:.1%}",
        "年化": f"{ann:.1%}",
        "最大回撤": f"{dd:.1%}",
        "Calmar": f"{ann / abs(dd):.2f}" if dd < 0 else "—",
        "在市时间": inmk,
    }


base_row = perf(port_o, None, "① 原版（无风控）")
bench_row = perf(bench_o, None, "② 对照：中证2000")
rows = [base_row, bench_row]
results: dict[str, pd.Series] = {}
for col in sig.columns:
    pos = sig[col].shift(1).fillna(False).astype(bool)     # ⚠️ t−1 的信号决定 t 的仓位
    r = port_o * pos
    results[col] = pos
    rows.append(perf(r, pos, f"③ +{col}"))

log("")
log("=" * 92)
log("结果")
log("=" * 92)
log(pd.DataFrame(rows).to_string(index=False))
log("")
log("怎么读：①看『最大回撤』比原版（−xx%）改善了多少 —— 这是主判据；")
log("        ②看『在市时间』—— 太低说明大部分时间空仓，那收益就不是靠策略赚的；")
log("        ③『Calmar』= 年化 ÷ 回撤，是「每承受 1 份回撤换回多少收益」，越高越划算。")

log("")
log("=" * 92)
log("崩盘月专项：原版跌得最惨的 6 个月，风控层躲掉了没有")
log("=" * 92)
log("（月份**自动取**『原版跌得最惨的 6 个月』，不写死 ⇒ 换数据区间时跟着变，不会漏掉新崩盘）")
mret_base = (1 + port_o).groupby(port_o.index.to_period("M")).prod() - 1
for ym, v in mret_base.nsmallest(6).items():
    sub = port_o.loc[ym.start_time:ym.end_time]
    parts = []
    for col in sig.columns:
        p = results[col].loc[sub.index]
        r = (sub * p).fillna(0)
        parts.append(f"{col.split('_')[0]}:{((1 + r).prod() - 1):+6.1%}")
    log(f"  {ym}  原版 {v:+6.1%}  ⇒  " + "  ".join(parts))

log("")
log("=" * 92)
log("结论（自动判定）")
log("=" * 92)
base_dd = float(base_row["最大回撤"].rstrip("%")) / 100
base_ann = float(base_row["年化"].rstrip("%")) / 100
for col in sig.columns:
    r = port_o * results[col]
    navv = (1 + r.fillna(0)).cumprod()
    dd = (navv / navv.cummax() - 1).min()
    ann = navv.iloc[-1] ** (244 / len(r)) - 1
    ok_dd = (dd - base_dd) > DD_REDUCE_MIN
    ok_ret = ann > base_ann * RET_KEEP_MIN
    verdict = "✅ 通过" if (ok_dd and ok_ret) else ("⚠️ 只改善回撤" if ok_dd else "❌ 不通过")
    log(f"  {col:<20} 回撤改善 {dd - base_dd:+.1%}  年化保留 {ann / base_ann:.0%}  {verdict}")

log("")
log("=" * 92)
log("⚠️ 必须说清楚的局限（不要把上面的数字当成已验证的策略）")
log("=" * 92)
log(f"  1. 回测区间 {sig.index.min().date()} ~ {sig.index.max().date()}（约 {len(sig) / 244:.1f} 年，"
    f"数据源 {SOURCE}）。样本有限，任何「有效」的风控都可能只是恰好拟合了区间内的几次崩盘。")
log("  2. 本回测假设『空仓』是瞬时的、无成本的；真实调仓有冲击成本，而且可能反复被打脸（假信号）。")
log("  3. 微盘池的 ST 过滤仍用了「今天的名字」，偏乐观。")
log("  4. 正确用法：把结果当『探索性线索』，方向上做交叉验证后，再走 exp_contract 正式立项。")
