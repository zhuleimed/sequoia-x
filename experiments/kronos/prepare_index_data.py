#!/usr/bin/env python3
"""方案 B 指数微调数据准备（2026-08-09）: DB index_daily → 单序列 CSV

与 prepare_finetune_data.py 同格式（symbol/timestamps/open/high/low/close/volume/amount）,
但为单指数序列（中证1000 sh.000852）。amount 缺失 → volume×均价 估算（同规则）。
数据截止 --end（默认 2026-02-28, 防评估月泄漏, 与 3b 同口径）。

用法: py312 python experiments/kronos/prepare_index_data.py [--end 2026-02-28]
"""
import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB = str(PROJECT_ROOT / "data/sequoia_v2.db")
OUT_DIR = PROJECT_ROOT / "experiments/kronos/finetune_csv/data"
INDEX_CODE = "sh.000852"   # 中证1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default="2026-02-28")
    args = ap.parse_args()

    conn = sqlite3.connect(DB)
    df = pd.read_sql(
        "SELECT date, open, high, low, close, volume FROM index_daily "
        "WHERE symbol=? AND date<=? ORDER BY date",
        conn, params=[INDEX_CODE, args.end])
    conn.close()
    if len(df) < 300:
        raise RuntimeError(f"{INDEX_CODE} 数据不足: {len(df)} 行")

    df = df.rename(columns={"date": "timestamps"})
    df.insert(0, "symbol", INDEX_CODE)
    # amount 估算: volume×均价（与推理管线同规则, 指数源无 amount 列）
    df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)
    df["timestamps"] = pd.to_datetime(df["timestamps"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fp = OUT_DIR / f"index_{INDEX_CODE}_{args.end}.csv"
    df.to_csv(fp, index=False)
    print(f"✅ 已写 {fp}: {len(df)} 条, {df['timestamps'].min().date()} ~ "
          f"{df['timestamps'].max().date()}")
    print(f"   字段: {list(df.columns)}; 前 2 行:\n{df.head(2).to_string()}")
    # 自检: 无 NaN
    print(f"   NaN 数: {int(df.isna().sum().sum())}")


if __name__ == "__main__":
    main()
