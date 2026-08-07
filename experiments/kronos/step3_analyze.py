#!/usr/bin/env python3
"""3a 单月 Rank IC 分析（2026-08-07, 与 analyze_monthly_ic 同口径）

输入: experiments/kronos/output/month_<YM>.jsonl  [{code, exp_ret}]
y2 = 未来 20 交易日超额收益（个股收益 - 沪深300 同期, 与 V2 口径一致）
Rank IC = Spearman(exp_ret, y2)（全截面）

用法: py312 python step3_analyze.py --month 2026-06
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DB = str(PROJECT_ROOT / "data/sequoia_v2.db")
HORIZON = 20  # 与 V2 T2 一致


def load_predictions(month: str) -> pd.DataFrame:
    fp = PROJECT_ROOT / "experiments/kronos/output" / f"month_{month}.jsonl"
    rows = []
    for line in fp.read_text(encoding="utf-8").strip().splitlines():
        if line:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def future_ret(code: str, ref_date: str, conn: sqlite3.Connection) -> float | None:
    """ref_date 后 20 交易日收益（用第 20 个交易日收盘价）。"""
    rows = conn.execute(
        "SELECT close FROM stock_daily WHERE symbol=? AND date>? ORDER BY date LIMIT ?",
        (code, ref_date, HORIZON)).fetchall()
    if len(rows) < HORIZON:
        return None
    close_now = conn.execute(
        "SELECT close FROM stock_daily WHERE symbol=? AND date=?",
        (code, ref_date)).fetchone()
    if not close_now or not close_now[0]:
        return None
    return rows[-1][0] / close_now[0] - 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default="2026-06")
    args = ap.parse_args()

    preds = load_predictions(args.month)
    if preds.empty:
        print(f"❌ 无预测文件: month_{args.month}.jsonl")
        return
    print(f"预测: {len(preds)} 只")

    # 指数基准（沪深300 同期收益）
    conn = sqlite3.connect(DB)
    idx_rows = conn.execute(
        "SELECT close FROM index_daily WHERE symbol='sh.000300' AND date>? ORDER BY date LIMIT ?",
        (args.month + "-30", HORIZON)).fetchall()
    ref_row = conn.execute(
        "SELECT close FROM index_daily WHERE symbol='sh.000300' AND date<=? "
        "ORDER BY date DESC LIMIT 1", (args.month + "-30",)).fetchone()
    if len(idx_rows) < HORIZON or not ref_row:
        print("❌ 沪深300 数据不足（index_daily）")
        return
    idx_ret = idx_rows[-1][0] / ref_row[0] - 1.0
    print(f"沪深300 {HORIZON} 日基准收益: {idx_ret:+.2%}")

    # 逐股 y2（超额收益）
    codes, exps, y2s = [], [], []
    for r in preds.itertuples():
        fr = future_ret(r.code, r.date, conn)
        if fr is None:
            continue
        codes.append(r.code)
        exps.append(r.exp_ret)
        y2s.append(fr - idx_ret)
    conn.close()

    exps = np.array(exps)
    y2s = np.array(y2s)
    ic, p = spearmanr(exps, y2s)
    print(f"\n有效样本: {len(codes)} 只")
    print(f"Rank IC = {ic:+.4f} (p={p:.4f})")
    print(f"exp_ret 分布: mean={exps.mean():+.2%} std={exps.std():.2%} "
          f"min={exps.min():+.2%} max={exps.max():+.2%}")
    # 分组检验: 预测前 10% vs 后 10% 的实际超额
    n = len(codes)
    k = max(n // 10, 1)
    top_idx = np.argsort(-exps)[:k]
    bot_idx = np.argsort(exps)[:k]
    print(f"预测 TOP10% 实际 y2 均值: {y2s[top_idx].mean():+.2%}")
    print(f"预测 BOT10% 实际 y2 均值: {y2s[bot_idx].mean():+.2%}")
    print(f"TOP - BOT 价差: {y2s[top_idx].mean() - y2s[bot_idx].mean():+.2%}")


if __name__ == "__main__":
    main()
