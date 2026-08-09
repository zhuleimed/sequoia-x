#!/usr/bin/env python3
"""V3 修订二: 合成训练标签生成（2026-08-09, Kronos 合成 K 线 → T2/T4 数据增强）

对池子股票在多个 ref 日用 Kronos（aggregate=False 全路径, 2026-08-09 修复）生成
未来 20 日合成路径 → 合成标签 y_synth = 校准后的路径收益（20 日, 与缓存 y2 同口径）。

校准（分布匹配真实）:
  y_synth = (y_raw - μ_gen) / σ_gen × σ_real + μ_real
  其中 μ_real/σ_real = 该股票历史 20 日滚动收益分布（demean 用全市场均值 0）
  —— 不注入 Kronos 的均值偏置（均值回归启发式）

输出: experiments/kronos/output/synth_labels_<n>_<m>.json
  {"symbol": {"YYYY-MM-DD": y_synth, ...}, ...}  (ref 日期 → 校准后合成 20 日收益)

用法: env -u KMP_AFFINITY -u OMP_NUM_THREADS nohup py312 python -u \
  experiments/kronos/synth_train_data.py --n 50 --refs 12 --workers 12 \
  > logs/synth_train_data.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.pop("KMP_AFFINITY", None)
os.environ["OMP_NUM_THREADS"] = "3"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "experiments/kronos"))
sys.path.insert(0, str(PROJECT_ROOT))

import step2_monthly as S2       # 个股数据加载/predict_one 基础设施
from model.kronos import calc_time_stamps
OUT_DIR = PROJECT_ROOT / "experiments/kronos/output"
LOOKBACK = 120
PRED_LEN = 20                    # 与缓存 y2 同口径（20 日收益）
PREDICTOR = None                 # worker 内加载


def worker_init():
    """每 worker 独立加载模型（fork, 铁律一）。"""
    global PREDICTOR
    from model import Kronos, KronosTokenizer, KronosPredictor
    tokenizer = KronosTokenizer.from_pretrained(str(S2.TOKENIZER_DIR), local_files_only=True)
    model = Kronos.from_pretrained(str(S2.PREDICTOR_DIR), local_files_only=True)
    PREDICTOR = KronosPredictor(model, tokenizer, device="cpu", max_context=512)


def gen_synth_label(symbol: str, ref_date: str, T: float, hist_real: tuple,
                    df: pd.DataFrame) -> float | None:
    """Kronos 生成 (symbol, ref) 合成 20 日收益, 校准后返回。

    hist_real: (μ, σ) 该股历史 20 日滚动收益分布（校准用）。
    df: 调用方预加载的 ≤ref 最近 LOOKBACK 天（升序）——避免每 ref 查 DB（性能修复）。
    """
    global PREDICTOR
    if len(df) < LOOKBACK:
        return None
    x_df = df[["open", "high", "low", "close", "volume", "amount"]].copy()
    x_df["amount"] = x_df["amount"].fillna(
        x_df["volume"] * x_df[["open", "high", "low", "close"]].mean(axis=1))
    x_df = x_df.ffill().bfill()
    x_ts = pd.to_datetime(df["date"])

    from index_timing_check import next_n_trade_dates
    future = next_n_trade_dates(ref_date, PRED_LEN)
    y_ts = pd.to_datetime(pd.Series(future))

    x = x_df[["open", "high", "low", "close", "volume", "amount"]].values.astype(np.float32)
    x_mean, x_std = x.mean(axis=0), x.std(axis=0)
    x = np.clip((x - x_mean) / (x_std + 1e-5), -PREDICTOR.clip, PREDICTOR.clip)[np.newaxis]
    x_stamp = calc_time_stamps(x_ts).values.astype(np.float32)[np.newaxis]
    y_stamp = calc_time_stamps(y_ts).values.astype(np.float32)[np.newaxis]

    import torch
    preds = PREDICTOR.generate(
        torch.from_numpy(x), torch.from_numpy(x_stamp), torch.from_numpy(y_stamp),
        PRED_LEN, T, 0, 0.9, 10, False, aggregate=False)   # 10 路径（原 30, 2026-08-09 提速 3 倍）
    paths = preds[0, :, :, 3] * (x_std[3] + 1e-5) + x_mean[3]   # close 列, 反标准化
    if paths.shape[0] < 10:
        return None
    y_raw = paths[:, -1] / paths[:, 0] - 1.0                     # 每条路径 20 日收益
    mu_g, sg_g = float(np.mean(y_raw)), float(np.std(y_raw))
    mu_r, sg_r = hist_real
    if sg_g < 1e-6 or sg_r < 1e-6:
        return None
    # 校准: 去生成偏置 + σ 匹配真实, 均值对齐真实（全市场/个股历史 20 日收益）
    y_cal = (y_raw - mu_g) / sg_g * sg_r + mu_r
    # ⚠️ 2026-08-09: 返回路径级标签数组（不取均值）——模型条件均值对 ref 不敏感
    # （均值回归启发式趋常数）, 取均值会丢失采样多样性; 每条路径 = 一个训练样本
    return [float(v) for v in y_cal]


def _worker(args: tuple) -> dict:
    """每只股票: 一次 DB 加载全历史（≤最晚 ref）→ 按 ref 切片 + 一次 hist。

    ⚠️ 2026-08-09 性能修复: 原实现每个 ref 各查一次 DB（12 refs = 12 次 1300 天查询）
    + hist 各算一次 → 12 worker 并发 SQLite 锁竞争, ETA 158min。改为单次加载。
    """
    symbol, refs, T = args
    conn = sqlite3.connect(S2.DB)
    df_all = pd.read_sql(
        "SELECT date, open, high, low, close, volume, amount FROM stock_daily "
        "WHERE symbol=? AND date<=? ORDER BY date",
        conn, params=[symbol, refs[0]])
    conn.close()
    if len(df_all) < LOOKBACK + 30:
        return {symbol: {}}
    # 该股历史 20 日滚动收益分布（校准目标, 一次计算）
    closes = df_all["close"].reset_index(drop=True)
    rets = (closes.shift(-PRED_LEN) / closes - 1).dropna()
    if len(rets) < 20:
        return {symbol: {}}
    hist = (float(rets.mean()), float(rets.std()))

    out = {}
    for ref in refs:
        df = df_all[df_all["date"] <= ref].tail(LOOKBACK)
        if len(df) < LOOKBACK:
            continue
        ys = gen_synth_label(symbol, ref, T, hist, df)
        if ys is not None:
            out[ref] = ys       # 路径级标签数组 [y1, y2, ..., y10]
    return {symbol: out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="股票数（池子随机抽样）")
    ap.add_argument("--refs", type=int, default=12, help="每只股票 ref 日数（近 N 个月每月末）")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--T", type=float, default=0.5, help="采样温度")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pool = S2.load_pool()
    rng = random.Random(args.seed)
    codes = rng.sample(pool, min(args.n, len(pool)))
    print(f"[{datetime.now():%H:%M:%S}] 合成标签生成: {len(codes)} 只 × {args.refs} refs "
          f"× T={args.T}, workers={args.workers}", flush=True)

    # ref 日 = 近 refs 个月的月末最后交易日（DB 直查, 防未来月份）
    conn = sqlite3.connect(S2.DB)
    yms = [r[0] for r in conn.execute(
        "SELECT DISTINCT substr(date,1,7) FROM stock_daily ORDER BY date DESC LIMIT ?",
        (args.refs,)).fetchall()]
    conn.close()
    months = [S2.get_month_last_trade_date(ym) for ym in yms]
    months = [m for m in months if m]
    print(f"  ref 日: {months}", flush=True)
    if len(months) < 2:
        print("❌ ref 日不足")
        return

    # 多进程并行（fork + 每进程独立模型）
    from multiprocessing import Pool
    ctx = __import__("multiprocessing").get_context("fork")
    tasks = [(c, months, args.T) for c in codes]
    merged: dict = {}
    t0 = datetime.now()
    with ctx.Pool(args.workers, initializer=worker_init) as pool:
        for i, res in enumerate(pool.imap_unordered(_worker, tasks)):
            merged.update(res)
            if (i + 1) % 10 == 0 or i + 1 == len(tasks):
                n_syn = sum(len(v) for v in merged.values())
                el = (datetime.now() - t0).total_seconds()
                rate = el / (i + 1)
                eta = rate * (len(tasks) - i - 1)
                print(f"  [{datetime.now():%H:%M:%S}] {i+1}/{len(tasks)} 只, "
                      f"合成标签 {n_syn} 个, 速率 {rate:.1f}s/只, ETA {eta/60:.0f}min",
                      flush=True)

    n_total = sum(len(v) for v in merged.values())
    print(f"✅ 完成: {len(merged)} 只 / {n_total} 个合成标签", flush=True)
    out = OUT_DIR / f"synth_labels_{args.n}_{args.refs}.json"
    out.write_text(json.dumps(merged, ensure_ascii=False))
    print(f"已写 {out}", flush=True)


if __name__ == "__main__":
    main()
