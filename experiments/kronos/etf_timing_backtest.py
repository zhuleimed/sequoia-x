#!/usr/bin/env python3
"""中证1000 ETF 择时策略回测（2026-08-09）

策略: Kronos 零样本 base 预测中证1000 未来 5 日收益,
信号 = pred_ret5 > -k%（k=0.25 为验证阈值, 校正看跌偏置）→ 持有 512100 / 空仓。

执行模型（真实可交易口径）:
  - 信号在 ref 日收盘后可得（输入 ≤ref 最近 120 日）→ **T+1 开盘价执行**
  - 持有期收益: ETF 从 ref_i 后首个交易日 open → ref_{i+1} 后首个交易日 open
  - 成本: 状态变化（空↔多）付单边 0.05%（佣金 0.03% + 滑点 0.02%; ETF 无印花税）

数据: 019 项目 etf_daily.db（512100 南方中证1000ETF, 2019-03 起）
信号: experiments/kronos/output/index_timing_check_4y.json
输出: 年化/波动/最大回撤/夏普/胜率/暴露/调仓次数 + output/etf_timing_backtest.json

用法: py312 python experiments/kronos/etf_timing_backtest.py [--k 0.25] [--pred 4y]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ETF_DB = "/public/home/hpc/zhulei/superman/quant/code/019_etf_daily_sync_and_backtest/data/etf_daily.db"
ETF_CODE = "512100"   # 南方中证1000ETF
INDEX_CODE = "000852" # 中证1000 指数（019 库, 2019-03 起, 无数据跳变）
OUT_DIR = PROJECT_ROOT / "experiments/kronos/output"
COST = 0.0005         # 单边: 佣金 0.03% + 滑点 0.02%
RF = 0.02             # 无风险利率（年化, 夏普用）


def load_index() -> pd.DataFrame:
    """段收益基准 = 中证1000 指数 open（019 库）。

    2026-08-09 数据审计结论: 019 库 512100 ETF 历史价格不可靠（2022-09-05 拆分点
    前后缩放不一致, 拆前价 0.98 与真实净值 ~6.8 差 7 倍; 腾讯 qfq 接口仅 640 行;
    baostock 仅 144 行）→ 段收益改用指数 000852 open（1803 行全量, 唯一跳变
    2024-10-08 为真实暴涨开盘）。ETF 跟踪误差 <0.5%/年, 5 日段尺度可忽略;
    ETF 可交易性（流动性）单独报告。
    """
    conn = sqlite3.connect(ETF_DB)
    df = pd.read_sql(
        "SELECT date, open FROM index_daily WHERE symbol=? ORDER BY date",
        conn, params=[INDEX_CODE])
    conn.close()
    return df


def report_etf_liquidity() -> None:
    """512100 流动性报告（日均成交量, 万手→元估算）。"""
    conn = sqlite3.connect(ETF_DB)
    df = pd.read_sql(
        "SELECT date, volume FROM etf_daily WHERE symbol=? ORDER BY date",
        conn, params=[ETF_CODE])
    conn.close()
    med = df["volume"].median() / 1e4   # 手（×100 股/手）
    med_1y = df.tail(240)["volume"].median() / 1e4
    print(f"流动性 {ETF_CODE}: 日均成交 {med:.0f} 万手, 近 1 年 {med_1y:.0f} 万手 "
          f"（≈{med_1y*3*1e4/1e8:.1f} 亿元/日, 充分）")


def backtest(pred_file: str, k: float, idx: pd.DataFrame) -> dict:
    res = json.loads(Path(pred_file).read_text())
    sigs = [(r["date"], 1 if r["pred_ret5"] > -k / 100.0 else 0) for r in res["rows"]]
    dates = idx["date"].tolist()
    opens = idx["open"].values

    def exec_open(ref_date: str) -> float | None:
        """ref 后首个交易日的指数 open（T+1 开盘执行, ETF 与指数同步交易）。"""
        for i, d in enumerate(dates):
            if d > ref_date:
                return float(opens[i])
        return None

    # ── 逐信号段执行 ──
    rows = []
    for i, (ref, sig) in enumerate(sigs):
        buy_px = exec_open(ref)
        sell_ref = sigs[i + 1][0] if i + 1 < len(sigs) else None
        sell_px = exec_open(sell_ref) if sell_ref else None
        if buy_px is None or sell_px is None:
            continue
        ret = sell_px / buy_px - 1.0
        rows.append({"ref": ref, "sig": sig, "ret": ret,
                     "buy_date": next(d for d in dates if d > ref),
                     "sell_date": sell_ref})

    # ── 净值与成本 ──
    df = pd.DataFrame(rows)
    state = df["sig"].values
    navs, costs = [], []
    nav = 1.0
    for i in range(len(df)):
        if i > 0 and state[i] != state[i - 1]:
            nav *= (1 - COST)          # 状态变化 → 付单边成本
            costs.append(COST)
        if state[i] == 1:
            nav *= (1 + df["ret"].iloc[i])
        navs.append(nav)
    df["nav"] = navs
    n_flips = int((state[1:] != state[:-1]).sum())

    # ── 指标 ──
    start, end = df["buy_date"].iloc[0], df["sell_date"].iloc[-1]
    years = (pd.to_datetime(end) - pd.to_datetime(start)).days / 365.25
    final = navs[-1]
    ann_ret = final ** (1 / years) - 1
    seg_ret = np.where(state == 1, df["ret"].values, 0.0)
    ann_vol = np.std(seg_ret) * np.sqrt(52)
    sharpe = (ann_ret - RF) / ann_vol if ann_vol > 0 else np.nan
    # 最大回撤（段级净值）
    peak = np.maximum.accumulate(navs)
    mdd = float(((np.array(navs) - peak) / peak).min())
    # 买入持有（同期, 无成本）
    bh_nav = float(np.prod(1 + df["ret"].values))
    bh_ann = bh_nav ** (1 / years) - 1
    # 胜率（持仓段）
    held = df[df["sig"] == 1]
    win = float((held["ret"] > 0).mean()) if len(held) else np.nan

    return {
        "etf": ETF_CODE, "k": k, "n_seg": len(df), "years": round(years, 2),
        "start": start, "end": end,
        "strat_nav": float(final), "strat_ann_ret": float(ann_ret),
        "ann_vol": float(ann_vol), "sharpe": float(sharpe),
        "max_drawdown": mdd, "win_rate": win,
        "exposure": float((df["sig"] == 1).mean()),
        "n_flips": n_flips, "n_costs": len(costs),
        "bh_nav": bh_nav, "bh_ann_ret": float(bh_ann),
        "excess_nav": float(final / bh_nav - 1),
        "segments": df.to_dict(orient="records"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=float, default=0.25, help="看多阈值 k%%（默认 0.25）")
    ap.add_argument("--pred", default="4y", choices=["2y", "4y"])
    args = ap.parse_args()

    pred_file = OUT_DIR / f"index_timing_check_{args.pred}.json"
    idx = load_index()
    report_etf_liquidity()
    print(f"段收益基准: 中证1000 指数 {INDEX_CODE}（{len(idx)} 行, "
          f"{idx['date'].iloc[0]} ~ {idx['date'].iloc[-1]}）")
    print(f"信号: {pred_file.name}, 阈值 k={args.k}%（信号=pred_ret5 > -{args.k}%）")

    r = backtest(str(pred_file), args.k, idx)
    print("\n" + "═" * 62)
    print(f"📊 中证1000 ETF 择时回测（512100, {r['start']} ~ {r['end']}, {r['n_seg']} 段 / {r['years']} 年）")
    print("═" * 62)
    print(f"策略净值      {r['strat_nav']:>10.2%}  (年化 {r['strat_ann_ret']:+.1%})")
    print(f"买入持有      {r['bh_nav']:>10.2%}  (年化 {r['bh_ann_ret']:+.1%})")
    print(f"超额净值      {r['excess_nav']:>10.2%}")
    print(f"年化波动      {r['ann_vol']*100:>9.1f}%   夏普 {r['sharpe']:+.2f}（无风险 {RF:.0%}）")
    print(f"最大回撤      {r['max_drawdown']*100:>9.1f}%")
    print(f"持仓胜率      {r['win_rate']*100:>9.1f}%   暴露 {r['exposure']*100:.0f}%")
    print(f"调仓次数      {r['n_flips']} 次（双边成本 {COST:.2%}/次）")

    out = OUT_DIR / "etf_timing_backtest.json"
    out.write_text(json.dumps(r, ensure_ascii=False, indent=2, default=str))
    print(f"\n结果已写 {out}")


if __name__ == "__main__":
    main()
