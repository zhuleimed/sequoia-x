#!/usr/bin/env python3
"""3b 微调数据准备（2026-08-08）: DB → 多股票 CSV

从 sequoia_v2.db 导出 A 股日线为官方 finetune_csv 格式（多股票单文件, 带 symbol 列,
供改造后的 CustomKlineDataset 按 symbol 分组切窗）。

设计:
  - 股票: 评估池（kronos_pools/2026-03.json）随机抽 N 只（固定 seed, 可复现）
  - 截止: --end 日期之前全历史（微调数据截止 < 最早评估月输入窗口, 与 T2/T4 滚动窗口口径一致）
  - 字段: symbol/timestamps/open/high/low/close/volume/amount
  - amount 清洗: DB 腾讯源未入库 → volume×均价 估算（与推理管线 predict_one 同规则）
  - 输出: experiments/kronos/finetune_csv/data/a_share_{end}_{n}.csv

用法: py312 python prepare_finetune_data.py --n 50 --end 2026-02-28 --seed 42
"""
import argparse
import json
import random
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB = str(PROJECT_ROOT / "data/sequoia_v2.db")
POOL = PROJECT_ROOT / "output/backtest_v2/kronos_pools/2026-03.json"
OUT_DIR = PROJECT_ROOT / "experiments/kronos/finetune_csv/data"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="抽 N 只股票")
    ap.add_argument("--end", default="2026-02-28", help="数据截止日")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pool = json.loads(POOL.read_text())
    rng = random.Random(args.seed)
    codes = rng.sample(pool, min(args.n, len(pool)))
    print(f"抽样 {len(codes)} 只（seed={args.seed}, 池子 {len(pool)} 只）")

    conn = sqlite3.connect(DB)
    frames = []
    for i, c in enumerate(codes):
        df = pd.read_sql(
            "SELECT date, open, high, low, close, volume, amount FROM stock_daily "
            "WHERE symbol=? AND date<=? ORDER BY date",
            conn, params=[c, args.end])
        if len(df) < 200:  # 至少 200 天才有切窗价值
            print(f"  跳过 {c}: 仅 {len(df)} 天")
            continue
        df = df.rename(columns={"date": "timestamps"})
        df.insert(0, "symbol", c)
        # amount 清洗: volume×均价（与 predict_one 同规则）
        df["amount"] = df["amount"].fillna(
            df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1))
        frames.append(df)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(codes)} 只已读")
    conn.close()

    if not frames:
        print("❌ 无有效股票")
        return
    out = pd.concat(frames, ignore_index=True)
    out["timestamps"] = pd.to_datetime(out["timestamps"])
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"a_share_{args.end}_{len(codes)}.csv"
    out.to_csv(fp, index=False)
    print(f"✅ 已写 {fp}: {len(out)} 条, {out['symbol'].nunique()} 只股票, "
          f"时间 {out['timestamps'].min().date()} ~ {out['timestamps'].max().date()}")
    print(f"   字段: {list(out.columns)}")
    # 自检: 每只股票行数分布
    print(out.groupby("symbol").size().describe().round(1).to_string())


if __name__ == "__main__":
    main()
