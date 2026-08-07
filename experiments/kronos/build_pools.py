#!/usr/bin/env python3
"""按月动态股票池生成器（2026-08-08, 用户要求: 回测每月池子动态变化）

背景: 70 个月回测必须用【该月时点】的股票池, 而非当前 .stock_pool.json——
  用今天 2978 只套 2020 年 = 幸存者偏差（当年未上市股票混入）+ 次新/低价过滤失效。

规则（与 build_prediction_cache._filter_stock_pool 一致, 按月评估）:
  1. 板块剔除: 688/689(科创) 300/301(创业) 4xx/8xx(北交所)
  2. ST/退市: stock_list.name（当前快照近似; 历史 ST 状态无数据源, 记录局限）
  3. 次新: listed_date ≤ ref_date - 365 天（上市满 1 年）
  4. 低价: ref_date 时点(该日或最近) close ≥ 2 元
  5. 历史充足: ref_date 前 ≥ 140 交易日数据（Kronos 需要 120+20）

输出: output/backtest_v2/kronos_pools/{YM}.json（每月一个池子, 供 step2_launch 使用）

用法: py312 python build_pools.py [--start 2020-09] [--end 2026-06]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DB = str(PROJECT_ROOT / "data/sequoia_v2.db")
POOLS_DIR = PROJECT_ROOT / "output/backtest_v2/kronos_pools"
MIN_HISTORY = 140  # Kronos: 120 输入 + 20 未来（历史 ≥ 140 交易日）


def month_last_trade_date(conn: sqlite3.Connection, ym: str) -> str | None:
    """该月最后交易日。"""
    y, m = int(ym[:4]), int(ym[5:7])
    nxt = f"{y + (1 if m == 12 else 0)}-{1 if m == 12 else m + 1:02d}-01"
    row = conn.execute(
        "SELECT MAX(date) FROM stock_daily WHERE date>=? AND date<?", (ym + "-01", nxt)).fetchone()
    return row[0] if row else None


def build_month_pool(conn: sqlite3.Connection, ref_date: str) -> list[str]:
    """按 ref_date 时点构建股票池。"""
    # 1. 全部股票 + 名称 + 上市日
    stocks = conn.execute(
        "SELECT symbol, name, listed_date FROM stock_list").fetchall()
    exclude_prefixes = ('688', '689', '300', '301', '4', '8')

    # 2. ref_date 时点每只股票的最近收盘价（一次查询全市场）
    close_map = dict(conn.execute(
        "SELECT symbol, close FROM stock_daily "
        "WHERE (symbol, date) IN (SELECT symbol, MAX(date) FROM stock_daily "
        "                        WHERE date<=? GROUP BY symbol)", (ref_date,)).fetchall())

    # 3. 每只股票 ref_date 前的数据天数（历史充足检查）
    hist_days = dict(conn.execute(
        "SELECT symbol, COUNT(*) FROM stock_daily WHERE date<=? GROUP BY symbol",
        (ref_date,)).fetchall())

    cutoff = (date.fromisoformat(ref_date) - timedelta(days=365)).isoformat()
    pool = []
    for symbol, name, listed in stocks:
        name = name or ""
        if symbol.startswith(exclude_prefixes):
            continue
        if "ST" in name or "退" in name:
            continue
        if listed and listed > cutoff:  # 上市未满 1 年
            continue
        if close_map.get(symbol, 0) < 2.0:
            continue
        if hist_days.get(symbol, 0) < MIN_HISTORY:
            continue
        pool.append(symbol)
    return pool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-09")
    ap.add_argument("--end", default="2026-06")
    args = ap.parse_args()

    # 月份序列
    months = []
    y, m = int(args.start[:4]), int(args.start[5:7])
    ey, em = int(args.end[:4]), int(args.end[5:7])
    while (y, m) <= (ey, em):
        months.append(f"{y}-{m:02d}")
        m += 1
        if m > 12:
            m, y = 1, y + 1

    conn = sqlite3.connect(DB)
    POOLS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"生成 {len(months)} 个月动态池 ({args.start} ~ {args.end})...")
    for i, ym in enumerate(months):
        ref = month_last_trade_date(conn, ym)
        if ref is None:
            print(f"  {ym}: 无数据, 跳过")
            continue
        pool = build_month_pool(conn, ref)
        (POOLS_DIR / f"{ym}.json").write_text(json.dumps(pool))
        print(f"  [{i+1}/{len(months)}] {ym} (ref={ref}): {len(pool)} 只")
    conn.close()
    print(f"\n✅ 完成, 输出 {POOLS_DIR}/")


if __name__ == "__main__":
    main()
