#!/usr/bin/env python3
"""3a 第二步: 单月全量 Kronos 零样本预测（2026-08-07）

多进程并行（12 workers, 每 worker 独立加载模型）+ 断点续跑（jsonl）+ 进度日志。
输出: experiments/kronos/output/month_<YM>.jsonl  [{code, date, exp_ret, pred_close, close}]

用法:
  py312 python step2_monthly.py --month 2026-06 [--workers 12] [--limit N]

设计（对齐 V2 口径）:
  - ref_date = 该月最后交易日（DB 查）
  - 输入: 每股 ≤ref_date 最近 120 天 OHLCV（无 look-ahead）
  - 预测: Kronos-base 零样本, 未来 20 交易日 close（T=0.2, top_p=0.9, 30 采样）
  - 输出: exp_ret = median(pred_close)/close - 1（截面排序分）
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

# ── 线程控制（铁律一）: KMP_AFFINITY 清除 + OMP 限制（12 workers × 2 线程 ≤ 36 核）──
os.environ.pop("KMP_AFFINITY", None)
os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "3")
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "experiments/kronos"))
sys.path.insert(0, str(PROJECT_ROOT))

DB = str(PROJECT_ROOT / "data/sequoia_v2.db")
MODELS_DIR = Path("/public/home/hpc/zhulei/superman/quant/code/models")
POOL_PATH = PROJECT_ROOT / "output/backtest_v2/.stock_pool.json"
OUT_DIR = PROJECT_ROOT / "experiments/kronos/output"
LOOKBACK = 120
PRED_LEN = 20


def get_month_last_trade_date(ym: str) -> str:
    """该月最后交易日（DB 中该月的 MAX(date)）。"""
    conn = sqlite3.connect(DB)
    row = conn.execute(
        "SELECT MAX(date) FROM stock_daily WHERE date >= ? AND date < ?",
        (ym + "-01", f"{int(ym[:4]) + (1 if ym[5:] == '12' else 0)}-{1 if ym[5:] == '12' else int(ym[5:]) + 1:02d}-01"),
    ).fetchone()
    conn.close()
    return row[0]


def load_pool(limit: int = 0) -> list[str]:
    symbols = json.loads(POOL_PATH.read_text())
    return symbols[:limit] if limit > 0 else symbols


def worker_init():
    """每 worker 加载一次模型（进程常驻, 不重复加载）。"""
    global _PREDICTOR
    from model import Kronos, KronosTokenizer, KronosPredictor
    tokenizer = KronosTokenizer.from_pretrained(
        str(MODELS_DIR / "Kronos-Tokenizer-base"), local_files_only=True)
    model = Kronos.from_pretrained(str(MODELS_DIR / "Kronos-base"), local_files_only=True)
    _PREDICTOR = KronosPredictor(model, tokenizer, device="cpu", max_context=512)


def predict_one(code: str, ref_date: str) -> dict | None:
    """单股推理: 输入 ≤ref_date 最近 120+20 天 → exp_ret。"""
    global _PREDICTOR
    conn = sqlite3.connect(DB)
    df = pd.read_sql(
        "SELECT date, open, high, low, close, volume, amount FROM stock_daily "
        "WHERE symbol=? AND date<=? ORDER BY date DESC LIMIT ?",
        conn, params=[code, ref_date, LOOKBACK + PRED_LEN + 20])
    conn.close()
    if df is None or len(df) < LOOKBACK + 10:
        return None
    df = df.iloc[::-1].reset_index(drop=True)

    x_df = df.loc[:LOOKBACK - 1, ["open", "high", "low", "close", "volume", "amount"]].copy()
    # amount 清洗: DB 腾讯源基本未入库 → volume×均价估算（Kronos 要求无 NaN）
    x_df["amount"] = x_df["amount"].fillna(
        x_df["volume"] * x_df[["open", "high", "low", "close"]].mean(axis=1))
    x_df = x_df.ffill().bfill()
    x_ts = pd.to_datetime(df["date"].iloc[:LOOKBACK])
    y_ts = pd.to_datetime(df["date"].iloc[LOOKBACK:LOOKBACK + PRED_LEN])

    pred_df = _PREDICTOR.predict(
        df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
        pred_len=PRED_LEN, T=0.2, top_p=0.9, sample_count=30, verbose=False)
    pred_close = float(np.median(pred_df["close"].values))
    close = float(df["close"].iloc[-1])
    if close <= 0:
        return None
    return {
        "code": code,
        "date": ref_date,
        "close": close,
        "pred_close": pred_close,
        "exp_ret": pred_close / close - 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", default="2026-06")
    parser.add_argument("--workers", type=int, default=36)
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 只（调试）")
    args = parser.parse_args()

    ref_date = get_month_last_trade_date(args.month)
    if ref_date is None:
        print(f"❌ 无 {args.month} 数据")
        return
    symbols = load_pool(args.limit)
    print(f"单月全量预测: month={args.month} ref_date={ref_date} pool={len(symbols)} 只 "
          f"workers={args.workers}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / f"month_{args.month}.jsonl"
    # ── 断点续跑: 已完成的 code 跳过 ──
    done = set()
    if out_file.exists():
        for line in out_file.read_text(encoding="utf-8").strip().splitlines():
            if line:
                try:
                    done.add(json.loads(line)["code"])
                except Exception:
                    pass
        print(f"断点续跑: 已完成 {len(done)}/{len(symbols)}")

    pending = [s for s in symbols if s not in done]
    if not pending:
        print("全部完成")
        return

    # ── 多进程并行（每 worker 独立模型 + SQLite 连接）──
    # 线程配置: 推理耗时≈反比于 OMP 线程数（36线程=15.9s/股, 2线程=101s/股 实测）
    # → 默认 workers=12 × OMP=3 = 36 核满配（铁律一: 总线程≤核数×1.5 安全）
    # fork 模式（spawn 启动开销大; fork+低线程稳定）
    from multiprocessing import Pool
    _ctx = __import__("multiprocessing").get_context(os.environ.get("MP_CTX", "fork"))
    with _ctx.Pool(args.workers, initializer=worker_init) as pool:
        for i, result in enumerate(
                pool.imap_unordered(_run_one, [(c, ref_date) for c in pending], chunksize=5)):
            if result is not None:
                with open(out_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result) + "\n")
                n_ok += 1
            done_cnt = len(done) + i + 1
            # 进度日志（铁律一: 跑到哪/还要多久/速率）
            if done_cnt % 100 == 0 or done_cnt == len(symbols):
                el = time.time() - t0
                rate = el / done_cnt
                eta = rate * (len(symbols) - done_cnt)
                print(f"  [{datetime.now():%H:%M:%S}] {done_cnt}/{len(symbols)} "
                      f"成功={n_ok} 耗时={el:.0f}s 速率={rate:.1f}s/股 ETA={eta/60:.0f}min",
                      flush=True)

    el = time.time() - t0
    print(f"\n✅ 完成: {n_ok}/{len(symbols)} 只, 总耗时 {el/60:.1f}min, "
          f"输出 {out_file}")


def _run_one(args: tuple) -> dict | None:
    code, ref_date = args
    try:
        return predict_one(code, ref_date)
    except Exception:
        return None


if __name__ == "__main__":
    main()
