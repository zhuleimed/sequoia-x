#!/usr/bin/env python3
"""V3 修订二: 合成完整序列生成（真·数据增强, 2026-08-09）

滚动生成"完整合成股票"序列（特征+标签全自洽, 与标签替换增强的本质区别）:
  种子: 真实股票最近 120 天 OHLCV
  循环: Kronos 条件生成未来 20 天（10 路径取中位路径, 稳定延续）→ 拼接 →
        再以新序列尾部 120 天为条件生成…… ×9 轮 → 300 天完整序列
  校准: 每轮生成路径 20 日收益做 σ 匹配（该股票历史 20 日收益 σ）+ demean 0
输出: experiments/kronos/output/synth_series/ 每只一个 CSV
      (timestamps/open/high/low/close/volume/amount, 合成序列)

用法: env -u KMP_AFFINITY -u OMP_NUM_THREADS nohup py312 python -u \
  experiments/kronos/synth_full_series.py --n 24 --workers 12 \
  > logs/synth_full_series.log 2>&1 &
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.pop("KMP_AFFINITY", None)
os.environ["OMP_NUM_THREADS"] = "3"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "experiments/kronos"))
sys.path.insert(0, str(PROJECT_ROOT))

import step2_monthly as S2
from index_timing_check import next_n_trade_dates
from model.kronos import calc_time_stamps

SEED_LEN = 120          # 种子长度（真实历史）
GEN_LEN = 20            # 每轮生成长度
ROUNDS = 9              # 滚动轮数 → 总长 120 + 9×20 = 300 天
OUT_DIR = PROJECT_ROOT / "experiments/kronos/output/synth_series"
PREDICTOR = None


def worker_init():
    """每 worker 独立加载模型（fork, 铁律一）。"""
    global PREDICTOR
    from model import Kronos, KronosTokenizer, KronosPredictor
    tokenizer = KronosTokenizer.from_pretrained(str(S2.TOKENIZER_DIR), local_files_only=True)
    model = Kronos.from_pretrained(str(S2.PREDICTOR_DIR), local_files_only=True)
    PREDICTOR = KronosPredictor(model, tokenizer, device="cpu", max_context=512)


def _gen_one_step(hist: pd.DataFrame, ref_date: str, T: float,
                  sigma_real: float) -> pd.DataFrame | None:
    """以 hist(≤ref 最近 120 天) 为条件, 生成未来 20 天（10 路径中位 + σ 校准）。

    返回生成段 DataFrame（open/high/low/close/volume/amount 6 列, 含日期）。
    """
    import torch
    if len(hist) < SEED_LEN:
        return None
    df = hist.tail(SEED_LEN).reset_index(drop=True)
    x_df = df[["open", "high", "low", "close", "volume", "amount"]].copy()
    x_df["amount"] = x_df["amount"].fillna(
        x_df["volume"] * x_df[["open", "high", "low", "close"]].mean(axis=1))
    x_df = x_df.ffill().bfill()
    x_ts = pd.to_datetime(df["date"])
    future = next_n_trade_dates(ref_date, GEN_LEN)
    y_ts = pd.to_datetime(pd.Series(future))

    x = x_df[["open", "high", "low", "close", "volume", "amount"]].values.astype(np.float32)
    x_mean, x_std = x.mean(axis=0), x.std(axis=0)
    x = np.clip((x - x_mean) / (x_std + 1e-5), -PREDICTOR.clip, PREDICTOR.clip)[np.newaxis]
    x_stamp = calc_time_stamps(x_ts).values.astype(np.float32)[np.newaxis]
    y_stamp = calc_time_stamps(y_ts).values.astype(np.float32)[np.newaxis]

    preds = PREDICTOR.generate(
        torch.from_numpy(x), torch.from_numpy(x_stamp), torch.from_numpy(y_stamp),
        GEN_LEN, T, 0, 0.9, 10, False, aggregate=False)
    paths = preds[0] * (x_std + 1e-5) + x_mean      # (10, 20, 6) 反标准化
    if paths.shape[0] < 5:
        return None
    med = np.median(paths, axis=0)                  # 中位路径（稳定延续）
    # σ 校准（2026-08-09 修复）: 温和 clip 0.5-1.5——原 0.3-3.0 在滚动拼接时
    # 每轮放大累积导致价格崩溃（20 日 σ 269%）
    scale = sigma_real / (np.std(paths[:, -1, 3] / paths[:, 0, 3] - 1) + 1e-6)
    scale = float(np.clip(scale, 0.5, 1.5))
    base = med[0, 3]
    med[:, 3] = base * (1 + (med[:, 3] / base - 1) * scale)
    med[:, [0, 1, 2]] = base + (med[:, [0, 1, 2]] - base) * scale
    med = np.maximum(med, base * 0.5)            # 价格下限保护（防负/崩）
    med = np.nan_to_num(med, nan=base)           # 防 NaN 传播（滚动拼接前兜底）
    out = pd.DataFrame(med, columns=["open", "high", "low", "close", "volume", "amount"])
    out.insert(0, "date", future)
    return out


def _gen_series(symbol: str, seed: pd.DataFrame, T: float) -> pd.DataFrame | None:
    """滚动生成 300 天完整合成序列（120 种子 + 9×20 生成）。"""
    hist = seed.copy()
    segs = [hist]
    last_date = hist["date"].iloc[-1]
    # 该股历史 20 日收益 σ（校准目标）
    closes = seed["close"].reset_index(drop=True)
    rets = (closes.shift(-GEN_LEN) / closes - 1).dropna()
    sigma_real = float(rets.std()) if len(rets) > 20 else 0.05
    for r in range(ROUNDS):
        seg = _gen_one_step(hist, last_date, T, sigma_real)
        if seg is None or len(seg) < GEN_LEN:
            return None
        segs.append(seg)
        hist = pd.concat([hist.tail(SEED_LEN - GEN_LEN), seg], ignore_index=True)
        last_date = seg["date"].iloc[-1]
    return pd.concat(segs, ignore_index=True)


def _worker(args: tuple):
    symbol, T, out_dir = args
    out_dir = Path(out_dir)      # ⚠️ 2026-08-10 修复: main 传 str, mkdir 需 Path
    conn = sqlite3.connect(S2.DB)
    seed = pd.read_sql(
        "SELECT date, open, high, low, close, volume, amount FROM stock_daily "
        "WHERE symbol=? ORDER BY date DESC LIMIT ?",
        conn, params=[symbol, SEED_LEN])
    conn.close()
    if len(seed) < SEED_LEN:
        return symbol, None
    seed = seed.iloc[::-1].reset_index(drop=True)
    # amount 清洗（2026-08-09: 种子真实数据 amount 大量 NaN——腾讯源未入库,
    # 与 predict_one 同规则: volume×均价 估算 + ffill/bfill）
    seed["amount"] = seed["amount"].fillna(
        seed["volume"] * seed[["open", "high", "low", "close"]].mean(axis=1))
    seed = seed.ffill().bfill()
    series = _gen_series(symbol, seed, T)
    if series is None:
        return symbol, None
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"syn_{symbol}.csv"
    series.to_csv(fp, index=False)
    return symbol, len(series)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24, help="合成股票数（池子随机抽样）")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--T", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import random
    pool = S2.load_pool()
    rng = random.Random(args.seed)
    codes = rng.sample(pool, min(args.n, len(pool)))
    print(f"[{datetime.now():%H:%M:%S}] 合成完整序列生成: {len(codes)} 只 × "
          f"{SEED_LEN + ROUNDS * GEN_LEN} 天, workers={args.workers}, T={args.T}",
          flush=True)

    from multiprocessing import Pool
    ctx = __import__("multiprocessing").get_context("fork")
    t0 = datetime.now()
    n_ok = 0
    with ctx.Pool(args.workers, initializer=worker_init) as pool:
        for i, (sym, n) in enumerate(pool.imap_unordered(
                _worker, [(c, args.T, str(OUT_DIR)) for c in codes])):
            if n:
                n_ok += 1
            if (i + 1) % 6 == 0 or i + 1 == len(codes):
                el = (datetime.now() - t0).total_seconds()
                print(f"  [{datetime.now():%H:%M:%S}] {i+1}/{len(codes)} 只 "
                      f"成功 {n_ok}, 速率 {el/(i+1):.1f}s/只, "
                      f"ETA {(el/(i+1))*(len(codes)-i-1)/60:.0f}min", flush=True)
    print(f"✅ 完成: {n_ok}/{len(codes)} 只合成序列 → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
