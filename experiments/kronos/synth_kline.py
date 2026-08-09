#!/usr/bin/env python3
"""Kronos 合成 K 线生成验证（2026-08-09, 论文第三能力: 保真度 +22%, 首次探索）

验证 Kronos 条件采样生成能力: 给定中证1000 最近 120 天历史, 采样生成未来 N 天路径。
评估: 生成的 N 日收益分布 vs 真实历史滚动 N 日收益分布（均值/标准差/偏度/分位数）,
以及不同 T 参数（温度, 控制生成多样性）的影响。

用途: ①数据增强（合成 K 线扩训练集）②压力测试（极端场景生成测模拟盘稳健性）
输出: output/synth_kline_<ref>.json + 生成路径 CSV

用法: py312 python experiments/kronos/synth_kline.py [--ref 2026-08-07] [--pred-len 20]
      [--samples 300] [--T 0.2,0.5,1.0]
"""
from __future__ import annotations

import argparse
import json
import os
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

import index_timing_check as ITC
OUT_DIR = PROJECT_ROOT / "experiments/kronos/output"
HIST_YEARS = 5   # 真实分布对比窗口（滚动 N 日收益）


def real_distribution(ref: str, pred_len: int) -> dict:
    """真实历史: 过去 HIST_YEARS 年的滚动 pred_len 日收益分布（指数 000852）。"""
    dates = ITC.ensure_index_data()
    i = dates.index(ref)
    hist = dates[max(0, i - 260 * HIST_YEARS):i + 1]
    conn = __import__("sqlite3").connect(ITC.DB)
    df = pd.read_sql(
        "SELECT date, close FROM index_daily WHERE symbol=? AND date<=? ORDER BY date",
        conn, params=[ITC.INDEX_CODE, ref])
    conn.close()
    closes = df.set_index("date")["close"]
    rets = (closes.shift(-pred_len) / closes - 1).dropna()
    return {"n": len(rets), "mean": float(rets.mean()), "std": float(rets.std()),
            "skew": float(rets.skew()), "p5": float(rets.quantile(0.05)),
            "p50": float(rets.quantile(0.50)), "p95": float(rets.quantile(0.95)),
            "min": float(rets.min()), "max": float(rets.max())}


def generate_paths(ref: str, pred_len: int, samples: int, T: float) -> np.ndarray:
    """条件采样生成: 返回 (samples, pred_len) 的 close 路径。

    ⚠️ 2026-08-09 修复: 官方 predict() 末尾 np.mean(sample 维) 只返回平均路径
    → 直接调 generate(aggregate=False) 拿全采样路径（(1, samples, pred_len, feat)）。
    """
    import torch
    from model.kronos import calc_time_stamps
    conn = __import__("sqlite3").connect(ITC.DB)
    df = pd.read_sql(
        "SELECT date, open, high, low, close, volume FROM index_daily "
        "WHERE symbol=? AND date<=? ORDER BY date DESC LIMIT ?",
        conn, params=[ITC.INDEX_CODE, ref, ITC.LOOKBACK])
    future = ITC.next_n_trade_dates(ref, pred_len)
    conn.close()
    df = df.iloc[::-1].reset_index(drop=True)
    x_df = df[["open", "high", "low", "close", "volume"]].copy()
    x_df["amount"] = x_df["volume"] * x_df[["open", "high", "low", "close"]].mean(axis=1)
    x_ts = pd.to_datetime(df["date"])
    y_ts = pd.to_datetime(pd.Series(future))

    # 构造输入（与 predict 前段一致: 标准化 + clip）
    x = x_df[["open", "high", "low", "close", "volume", "amount"]].values.astype(np.float32)
    x_mean, x_std = x.mean(axis=0), x.std(axis=0)
    x = np.clip((x - x_mean) / (x_std + 1e-5), -ITC._PREDICTOR.clip, ITC._PREDICTOR.clip)[np.newaxis]
    x_stamp = calc_time_stamps(x_ts).values.astype(np.float32)[np.newaxis]
    y_stamp = calc_time_stamps(y_ts).values.astype(np.float32)[np.newaxis]

    preds = ITC._PREDICTOR.generate(
        torch.from_numpy(x), torch.from_numpy(x_stamp), torch.from_numpy(y_stamp),
        pred_len, T, 0, 0.9, samples, False, aggregate=False)
    # (1, samples, pred_len, feat) → close 列 idx=3 → (samples, pred_len)
    paths = preds[0, :, :, 3] * (x_std[3] + 1e-5) + x_mean[3]   # 反标准化
    return paths


def _gen_worker(args: tuple) -> tuple:
    """worker: 生成 (T, n) 条路径（fork 后每进程独立加载模型, 铁律一）。"""
    T, n, ref, pred_len = args
    paths = generate_paths(ref, pred_len, n, T)
    return T, paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="2026-08-07")
    ap.add_argument("--pred-len", type=int, default=20)
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--T", default="0.2,0.5,1.0")
    ap.add_argument("--workers", type=int, default=6,
                    help="并行进程数（每 T 分片; 铁律一: 多进程并行, 推荐 3-6）")
    args = ap.parse_args()

    ITC.ensure_index_data()
    real = real_distribution(args.ref, args.pred_len)
    print(f"📊 合成 K 线生成验证（ref={args.ref}, 未来 {args.pred_len} 日, "
          f"采样 {args.samples} 条/T × {len(args.T.split(','))} 个 T, workers={args.workers}）",
          flush=True)
    print(f"真实历史分布（{HIST_YEARS} 年滚动 {args.pred_len} 日收益, n={real['n']}）: "
          f"均值 {real['mean']*100:+.2f}% | σ {real['std']*100:.2f}% | "
          f"偏度 {real['skew']:+.2f} | P5/P50/P95 "
          f"{real['p5']*100:+.1f}/{real['p50']*100:+.1f}/{real['p95']*100:+.1f}%", flush=True)

    # ── 多进程并行（铁律一）: 采样分片给 workers, fork 后每进程独立加载模型 ──
    Ts = [float(x) for x in args.T.split(",")]
    n_chunks = max(1, args.workers // len(Ts))          # 每 T 分几片
    tasks = [(T, args.samples // n_chunks, args.ref, args.pred_len)
             for T in Ts for _ in range(n_chunks)]

    from multiprocessing import Pool
    ctx = __import__("multiprocessing").get_context("fork")
    results: dict[float, list[np.ndarray]] = {T: [] for T in Ts}
    with ctx.Pool(len(tasks), initializer=ITC._load_predictor) as pool:
        for i, (T, paths) in enumerate(pool.imap_unordered(_gen_worker, tasks)):
            results[T].append(paths)
            done = sum(len(p) for p in results[T])
            print(f"  [{datetime.now():%H:%M:%S}] T={T:.1f} 完成 {done}/{args.samples} "
                  f"条（{i+1}/{len(tasks)} 分片）", flush=True)

    print("", flush=True)
    gen_out: dict[str, dict] = {}
    for T in Ts:
        paths = np.concatenate(results[T], axis=0)
        rets = pd.Series(paths[:, -1] / paths[:, 0] - 1)   # 末日/首日-1, 同真实口径
        gen = {"n": len(rets), "mean": float(rets.mean()), "std": float(rets.std()),
               "skew": float(rets.skew()), "p5": float(np.quantile(rets, 0.05)),
               "p50": float(np.quantile(rets, 0.50)), "p95": float(np.quantile(rets, 0.95)),
               "min": float(rets.min()), "max": float(rets.max())}
        gen_out[str(T)] = gen
        print(f"  T={T:.1f}: 均值 {gen['mean']*100:+.2f}% | σ {gen['std']*100:.2f}% | "
              f"偏度 {gen['skew']:+.2f} | P5/P50/P95 "
              f"{gen['p5']*100:+.1f}/{gen['p50']*100:+.1f}/{gen['p95']*100:+.1f}%"
              f" | min/max {gen['min']*100:+.1f}/{gen['max']*100:+.1f}%", flush=True)
        if T == 0.5:    # 保存中档 T 的路径样本
            pd.DataFrame(paths).to_csv(
                OUT_DIR / f"synth_paths_{args.ref}_{args.pred_len}d.csv", index=False,
                header=[f"day{i+1}" for i in range(args.pred_len)])

    out = {"ref": args.ref, "pred_len": args.pred_len, "samples": args.samples,
           "workers": args.workers, "real": real, "gen_T": gen_out}
    (OUT_DIR / f"synth_kline_{args.ref}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n✅ 结果已写 output/synth_kline_{args.ref}.json + 路径 CSV", flush=True)


if __name__ == "__main__":
    main()
