"""微盘策略 V3 —— 「卖出规则」×「空仓策略」实验矩阵。

## 用户的两条要求

1. **选股以微盘为主**（不做高股息那条岔路）
2. **空仓策略保留，并进一步优化、改进**

所以本脚本在 V2（初选池 + 精选 N 支 + 非等权）基础上，把两个维度做成矩阵：

### 维度 A：卖出规则（月内硬止损，项目 `#18` 的 E1 规则）
| 取值 | 含义 |
|---|---|
| `none` | 不止损（V2 的现状） |
| `-12%` | 单票浮亏 ≤ −12% 即卖出，转现金（**项目生产规则**） |

### 维度 B：空仓策略（在 V2 的 S3 拥挤度上做优化改进）
| 取值 | 含义 | 改进点 |
|---|---|---|
| `none` | 不空仓 | 对照 |
| `S3` | 拥挤度超 80% 分位 ⇒ **全空仓** | V2 现状 |
| `S3_half` | 拥挤度超 80% 分位 ⇒ **半仓**（0.5） | **改进①：分级减仓**，避免"全撤后踏空" |
| `S3_slow` | 拥挤度 **或** 慢速趋势（MA250）任一触发 ⇒ 全空仓 | **改进②：加慢速趋势作互补**（MA20/60 换手太高被成本打死，250 日才活下来） |
| `S3_half_slow` | 上面两个改进叠加 | **改进③** |

用法：
    /home/zhulei/anaconda3/envs/zhulei_py312/bin/python strategy_v3_matrix.py [db|ths]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import microcap_data as md

PROJ = Path(__file__).resolve().parents[4]
BASE = PROJ / "sequoia_x" / "strategy" / "micro_cap"
INDEX_EXTRA = BASE / "data" / "index_extra.parquet"

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "ths"
NS = [20, 30]
STOPS = [(None, "不止损"), (-0.12, "止损 −12%")]
EMPTY_MODES = [("none", "不空仓"), ("S3", "S3 全空仓"), ("S3_half", "S3 半仓"),
               ("S3_slow", "S3+慢趋势 全空仓"), ("S3_half_slow", "S3+慢趋势 半仓")]
SLIP = 0.0020
LOOKBACK, MIN_SAMPLES = 250, 40
COMMISSION, TRANSFER = 0.00025, 0.00001
STAMP_BEFORE, STAMP_AFTER = 0.001, 0.0005
STAMP_CUTOFF = pd.Timestamp("2023-08-28")

pd.set_option("display.width", 220)


def log(msg: str = "") -> None:
    print(msg, flush=True)


def cap_for(n: int) -> float:
    return 0.15 if n <= 10 else 0.10


def min_variance_weights(panel: pd.DataFrame, cap: float) -> dict[str, float] | None:
    """口径照搬项目 `monthly_engine._estimate_cov` + `_solve_min_variance`。"""
    p = panel.tail(LOOKBACK + 1).dropna()
    if len(p) < MIN_SAMPLES + 1:
        return None
    rets = np.clip(p.pct_change().dropna().values, -0.11, 0.11)
    if rets.shape[0] < MIN_SAMPLES:
        return None
    try:
        from sklearn.covariance import LedoitWolf
        cov = LedoitWolf().fit(rets).covariance_
    except Exception:
        c = np.cov(rets, rowvar=False)
        cov = c * 0.9 + np.diag(np.diag(c)) * 0.1
    n = cov.shape[0]
    cap_eff = max(cap, 1.0 / n + 1e-9)
    res = minimize(lambda w: float(w @ cov @ w), np.ones(n) / n, method="SLSQP",
                   bounds=[(0.0, cap_eff)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                   options={"maxiter": 500, "ftol": 1e-12})
    if not res.success or not np.all(np.isfinite(res.x)):
        return None
    w = np.clip(res.x, 0, None)
    s = w.sum()
    return dict(zip(p.columns, w / s)) if s > 0 else None


log("=" * 110)
log(f"微盘策略 V3 矩阵    数据源: {SOURCE}    滑点: {SLIP * 1e4:.0f}bp/边")
log("=" * 110)

df = md.apply_initial_pool(md.load(SOURCE))
log(f"初选池: {len(df):,} 行 / {df['symbol'].nunique():,} 只")
df = df.sort_values(["symbol", "date"])
close_by_sym = {s: g.set_index("date")["close"] for s, g in df.groupby("symbol")}
ret_by_sym = {s: g.set_index("date")["ret"] for s, g in df.groupby("symbol")}

ix = pd.read_parquet(INDEX_EXTRA)
bench_ret = ix[ix["symbol"] == "csi2000"].set_index("date")["close"].sort_index().pct_change()

is_me = df["date"] == df.groupby("ym")["date"].transform("max")
me_dates = df[is_me].groupby("ym")["date"].max()


def build_port(sel: pd.DataFrame, weighting: str, stop: float | None) -> tuple[pd.Series, int]:
    """按月拼出日度收益；返回 (收益序列, 止损触发次数)。"""
    out, hits_total = [], 0
    for ym in sorted(sel["hold_ym"].unique()):
        g = sel[sel["hold_ym"] == ym]
        syms = list(g["symbol"])
        if weighting == "equal":
            w = pd.Series(1.0 / len(syms), index=syms)
        else:
            panel = pd.DataFrame({s: close_by_sym[s] for s in syms if s in close_by_sym})
            panel = panel[panel.index <= g["ref_date"].iloc[0]]
            wd = min_variance_weights(panel, cap_for(len(syms)))
            w = pd.Series(wd) if wd else pd.Series(1.0 / len(syms), index=syms)

        r = pd.DataFrame({s: ret_by_sym[s] for s in syms if s in ret_by_sym})
        r = r[r.index.to_period("M") == ym]
        if r.empty:
            continue
        w = w.reindex(r.columns).fillna(0.0)
        if w.sum() <= 0:
            continue
        w = w / w.sum()
        r = r.fillna(0.0)                       # 停牌日 = 0 收益（价格冻结）
        nav = (1 + r).cumprod()                 # 各股相对月初的净值（起点 1）
        if stop is not None:
            nav, h = md.apply_stop_loss(nav, stop)
            hits_total += h
        tot = (nav * w).sum(axis=1)
        p = tot.pct_change()
        p.iloc[0] = tot.iloc[0] - 1.0
        out.append(p)
    return pd.concat(out).sort_index(), hits_total


def empty_position(sig: pd.DataFrame, mode: str) -> pd.Series:
    """空仓/减仓的仓位系数（1=满仓, 0.5=半仓, 0=空仓）。已 shift(1)，不穿越。"""
    crowd = sig["S3_拥挤度"].shift(1).fillna(True).astype(bool)   # True=允许持有
    if mode == "none":
        return pd.Series(1.0, index=sig.index)
    if mode == "S3":
        return crowd.astype(float)
    if mode == "S3_half":
        return crowd.map({True: 1.0, False: 0.5}).astype(float)
    slow = sig["S1_趋势"].shift(1).fillna(True).astype(bool)
    both = crowd & slow
    if mode == "S3_slow":
        return both.astype(float)
    if mode == "S3_half_slow":
        return both.map({True: 1.0, False: 0.5}).astype(float)
    raise ValueError(mode)


def evaluate(port: pd.Series, sig: pd.DataFrame, sel: pd.DataFrame, k_full: pd.Series,
             hits: int, mode_tag: str, n: int, wtag: str, stop_tag: str) -> dict:
    """在**已建好的组合**上套用某个空仓策略（组合构建在调用方，只做一次）。"""
    k = k_full
    w_by_ym = {ym: set(g["symbol"]) for ym, g in sel.groupby("hold_ym")}
    yms = sorted(w_by_ym)
    repl = {ym: (1.0 if i == 0 else 1.0 - len(w_by_ym[ym] & w_by_ym[yms[i - 1]]) / len(w_by_ym[ym]))
            for i, ym in enumerate(yms)}
    first_day = {ym: x.min() for ym, x in pd.Series(port.index, index=port.index.to_period("M")).groupby(level=0)}
    rate = 2 * (COMMISSION + TRANSFER + SLIP) + pd.Series(
        np.where(port.index >= STAMP_CUTOFF, STAMP_AFTER, STAMP_BEFORE), index=port.index)
    t = k.diff().abs()
    t.iloc[0] = float(k.iloc[0])
    prev_k = k.shift(1).fillna(0.0)
    for ym, d in first_day.items():
        if k.get(d, 0.0) > 0 and prev_k.get(d, 0.0) > 0:
            t.loc[d] = t.get(d, 0.0) + repl.get(ym, 0.0) * k.get(d, 0.0)
    t = t.clip(upper=1.0)

    net = port * k - t * rate
    nav = (1 + net.fillna(0)).cumprod()
    ann = nav.iloc[-1] ** (244 / len(net)) - 1
    dd = (nav / nav.cummax() - 1).min()
    return {"持仓": n, "权重": wtag, "止损": stop_tag, "空仓策略": mode_tag,
            "年化": f"{ann:.1%}", "最大回撤": f"{dd:.1%}",
            "Calmar": f"{ann / abs(dd) if dd < 0 else float('nan'):.2f}",
            "在市": f"{k.mean():.0%}", "年换手": f"{t.sum() / (len(net) / 244):.1f}",
            "止损次数": hits}


def prepare(n: int, weighting: str, stop, sig_ma: int = 250):
    """建好一套组合 + 信号（供 main 与其它实验脚本复用，避免重复计算）。"""
    sel = md.pick_top_n(df, n=n, by="total_mv")
    sel["ref_date"] = sel["ym"].map(me_dates)
    port_raw, hits = build_port(sel, weighting, stop)
    share = md.pool_amount_share(df, sel)
    sig = md.build_signals(port_raw, share, bench_ret, ma=60, q_win=250, q_level=0.80)
    # S1_趋势 改用慢速均线（MA20/60 实测换手太高、会被成本打死）
    sig_slow = md.build_signals(port_raw, share, bench_ret, ma=sig_ma, q_win=250, q_level=0.80)
    sig["S1_趋势"] = sig_slow["S1_趋势"]
    start = sig.dropna(how="any").index.min()
    sig = sig.loc[start:]
    return sel, port_raw.reindex(sig.index), sig, hits


def main() -> None:
    log("")
    rows = []
    for n in NS:
        for weighting, wtag in [("equal", "等权"), ("minvar", "最小方差")]:
            for stop, stop_tag in STOPS:
                sel, port, sig, hits = prepare(n, weighting, stop)
                for mode, mode_tag in EMPTY_MODES:
                    k = empty_position(sig, mode)
                    rows.append(evaluate(port, sig, sel, k, hits, mode_tag, n, wtag, stop_tag))
                log(f"  跑完 {n} 只 / {wtag} / {stop_tag}")

    res = pd.DataFrame(rows)
    res["_c"] = pd.to_numeric(res["Calmar"], errors="coerce")
    log("")
    log("=" * 110)
    log("全部结果（按 Calmar 降序前 20）")
    log("=" * 110)
    log(res.sort_values("_c", ascending=False).head(20).drop(columns="_c").to_string(index=False))
    log("")
    log("怎么读：")
    log("  · 『止损』列 = 月内硬止损（项目 #18 的 E1 规则：单票浮亏 ≤ −12% 卖出）。")
    log("  · 『空仓策略』列 = 在拥挤度基础上做的三个改进（分级减仓 / 加慢速趋势 / 两者叠加）。")
    log("  · 重点看『最大回撤』降了多少、『年换手』涨了多少（换手高 = 成本高 = 实盘更难）。")

    res.drop(columns="_c").to_csv(BASE / "data" / "v3_matrix.csv", index=False)
    log("")
    log(f"✅ 全表已存: {BASE / 'data' / 'v3_matrix.csv'}")


if __name__ == "__main__":
    main()
