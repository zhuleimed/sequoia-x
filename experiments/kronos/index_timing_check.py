#!/usr/bin/env python3
"""中金路线可行性验证：零样本 Kronos 对中证1000 指数未来 5 日择时（2026-08-09）

背景：方向四 3b 证伪（个股 20 日 RankIC 增量均值 -0.0507, 输出=均值回归启发式）。
中金研报《大语言时序模型KRONOS的A股择时应用》(2025-10-11, 郑文才等) 显示：
  - 标准版（零样本）Kronos 预测中证1000 未来 5 日收盘价 vs 真实值 Spearman = 0.732
  - 滚动微调 + 滚动调整推理参数后 → 0.856, 2025 年策略收益 33.9%, 年化超额 9%
本脚本先验证零样本版在该场景（指数 5 日价格预测, 非个股 20 日排序）的可行性。

口径（对齐中金）:
  - 标的: 中证1000 (sh.000852; 若 index_daily 缺失则腾讯源补齐, 幂等)
  - 输入: ≤ref 最近 120 日 OHLCV（amount=volume×均价 估算, 与个股管线 predict_one 一致）
  - 预测: Kronos-base 零样本, pred_len=5, T=0.2, top_p=0.9, 30 采样, 取 median
  - 滚动: 每 5 交易日一个 ref 点（中金为逐日滚动, 5 日步长省算力, 重叠不影响口径）
  - 指标: ①预测 vs 真实 5 日收益 Spearman（中金口径近似）②方向准确率
         ③简单择时（信号=预测涨→持指数/跌→空仓）累计收益 vs 买入持有
  - 输出: output/index_timing_check.json + 逐点明细

用法: py312 python experiments/kronos/index_timing_check.py [--days 365] [--workers 12]
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

# ── 线程控制（铁律一）──
os.environ.pop("KMP_AFFINITY", None)
os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "3")
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "experiments/kronos"))
sys.path.insert(0, str(PROJECT_ROOT))

DB = str(PROJECT_ROOT / "data/sequoia_v2.db")
MODELS_DIR = Path("/public/home/hpc/zhulei/superman/quant/code/models")
TOKENIZER_DIR = Path(os.environ.get("KRONOS_TOKENIZER_DIR", str(MODELS_DIR / "Kronos-Tokenizer-base")))
PREDICTOR_DIR = Path(os.environ.get("KRONOS_PREDICTOR_DIR", str(MODELS_DIR / "Kronos-base")))
OUT_DIR = PROJECT_ROOT / "experiments/kronos/output"
INDEX_CODE = "sh.000852"      # 中证1000
LOOKBACK = 120
PRED_LEN = 5                  # 中金口径: 未来 5 日收盘价
SAMPLES = 30
T = 0.2
TOP_P = 0.9


def ensure_index_data() -> list[str]:
    """确保 index_daily 有中证1000; 缺失则腾讯源补齐（幂等）。返回全部交易日。"""
    conn = sqlite3.connect(DB)
    n = conn.execute("SELECT COUNT(*) FROM index_daily WHERE symbol=?", (INDEX_CODE,)).fetchone()[0]
    if n < 300:
        print(f"⏳ {INDEX_CODE} 缺失/不完整 ({n} 行), 腾讯源补齐...", flush=True)
        from sequoia_x.data.tencent_source import TencentSource
        df = TencentSource().get_daily("sh000852", days=1600)
        if df is None or len(df) < 300:
            conn.close()
            raise RuntimeError("中证1000 拉取失败")
        df["symbol"] = INDEX_CODE
        df = df[["symbol", "date", "open", "close", "high", "low", "volume"]]
        conn.executemany(
            "INSERT OR REPLACE INTO index_daily (symbol, date, open, close, high, low, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [tuple(x) for x in df.itertuples(index=False)])
        conn.commit()
        n = len(df)
    dates = [r[0] for r in conn.execute(
        "SELECT date FROM index_daily WHERE symbol=? ORDER BY date", (INDEX_CODE,)).fetchall()]
    conn.close()
    print(f"✅ {INDEX_CODE}: {n} 行, {dates[0]} ~ {dates[-1]}", flush=True)
    return dates


def _load_predictor():
    """每 worker 加载一次模型（进程常驻）。"""
    global _PREDICTOR
    from model import Kronos, KronosTokenizer, KronosPredictor
    tokenizer = KronosTokenizer.from_pretrained(str(TOKENIZER_DIR), local_files_only=True)
    model = Kronos.from_pretrained(str(PREDICTOR_DIR), local_files_only=True)
    _PREDICTOR = KronosPredictor(model, tokenizer, device="cpu", max_context=512)


def predict_index(ref_date: str) -> dict | None:
    """单点: ≤ref 最近 120 日指数 OHLCV → 预测未来 5 日收盘价。

    与 predict_one 同规则: y_timestamp 用指数自身未来交易日（时间特征, 无价格泄漏）,
    amount 用 volume×均价 估算（指数源无 amount 列）。
    """
    global _PREDICTOR
    conn = sqlite3.connect(DB)
    df = pd.read_sql(
        "SELECT date, open, high, low, close, volume FROM index_daily "
        "WHERE symbol=? AND date<=? ORDER BY date DESC LIMIT ?",
        conn, params=[INDEX_CODE, ref_date, LOOKBACK])
    future_dates = [r[0] for r in conn.execute(
        "SELECT date FROM index_daily WHERE symbol=? AND date>? ORDER BY date LIMIT ?",
        (INDEX_CODE, ref_date, PRED_LEN)).fetchall()]
    # 实际 5 日后收盘价（评估用, 非模型输入）
    actual = conn.execute(
        "SELECT close FROM index_daily WHERE symbol=? AND date>? ORDER BY date LIMIT ?",
        (INDEX_CODE, ref_date, PRED_LEN)).fetchall()
    conn.close()
    if df is None or len(df) < LOOKBACK or len(future_dates) < PRED_LEN:
        return None
    df = df.iloc[::-1].reset_index(drop=True)

    x_df = df[["open", "high", "low", "close", "volume"]].copy()
    x_df["amount"] = x_df["volume"] * x_df[["open", "high", "low", "close"]].mean(axis=1)
    x_df = x_df.ffill().bfill()
    x_ts = pd.to_datetime(df["date"])
    y_ts = pd.to_datetime(pd.Series(future_dates))

    pred_df = _PREDICTOR.predict(
        df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
        pred_len=PRED_LEN, T=T, top_p=TOP_P, sample_count=SAMPLES, verbose=False)
    pred_close5 = float(np.median(pred_df["close"].values))  # 未来 5 日收盘价中位数
    close = float(df["close"].iloc[-1])
    actual_close5 = float(actual[PRED_LEN - 1][0])
    if close <= 0 or actual_close5 <= 0:
        return None
    return {
        "date": ref_date,
        "close": close,
        "pred_close5": pred_close5,
        "pred_ret5": pred_close5 / close - 1.0,
        "actual_ret5": actual_close5 / close - 1.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365, help="回溯窗口（交易日历天数, 近 1 年默认）")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="index_timing_check.json", help="输出文件名")
    args = ap.parse_args()

    dates = ensure_index_data()
    t0 = time.time()
    # 每 5 交易日一个 ref 点（ref 需有足够历史 + 未来 5 日）, 近 N 个交易日窗口
    start_date = dates[max(0, len(dates) - args.days)]
    refs = [dates[i] for i in range(LOOKBACK, len(dates) - PRED_LEN)
            if dates[i] >= start_date and i % 5 == 0]
    print(f"ref 点数: {len(refs)}（{refs[0]} ~ {refs[-1]}, 每 5 交易日）workers={args.workers}", flush=True)

    from multiprocessing import Pool
    _ctx = __import__("multiprocessing").get_context("fork")
    results = []
    with _ctx.Pool(args.workers, initializer=_load_predictor) as pool:
        for r in pool.imap_unordered(predict_index, refs):
            if r is not None:
                results.append(r)
    results.sort(key=lambda r: r["date"])
    el = time.time() - t0
    print(f"推理完成: {len(results)}/{len(refs)} 点, 耗时 {el/60:.1f}min", flush=True)
    if len(results) < 30:
        print("❌ 有效点太少, 无法评估")
        return

    # ── 评估（中金口径）──
    import scipy.stats as st
    pred_ret = np.array([r["pred_ret5"] for r in results])
    act_ret = np.array([r["actual_ret5"] for r in results])
    spear, p = st.spearmanr(pred_ret, act_ret)          # 预测 vs 真实 5 日收益
    spear_c, _ = st.spearmanr(
        [r["pred_close5"] for r in results], [r["close"] * (1 + r["actual_ret5"]) for r in results])
    acc = float(np.mean(np.sign(pred_ret) == np.sign(act_ret)))  # 方向准确率
    # 简单择时: 信号=预测涨→持指数 5 日, 预测跌→空仓（0 收益）
    sig_ret = np.where(pred_ret > 0, act_ret, 0.0)
    strat_nav = float(np.prod(1 + sig_ret))
    bh_nav = float(np.prod(1 + act_ret))
    # 周期内总收益（5 日段连乘 = 近似全期）
    n_period = len(act_ret)
    excess = strat_nav / bh_nav - 1

    print("\n" + "═" * 60)
    print(f"📊 中证1000 零样本 Kronos 未来5日择时验证（{results[0]['date']} ~ {results[-1]['date']}, {len(results)} 点）")
    print("═" * 60)
    print(f"① Spearman(预测收益, 真实收益) = {spear:+.4f} (p={p:.4f})  [中金标准版参考 0.732]")
    print(f"② Spearman(预测收盘, 真实收盘) = {spear_c:+.4f}")
    print(f"③ 方向准确率 = {acc*100:.1f}%  [基准 50%]")
    print(f"④ 择时累计 = {strat_nav:+.1%} | 买入持有 = {bh_nav:+.1%} | 超额 = {excess:+.1%}")
    print(f"⑤ 预测收益分布: 均值 {pred_ret.mean():+.2%}, 正信号占比 {(pred_ret>0).mean():.1%}")

    result = {
        "index": INDEX_CODE, "pred_len": PRED_LEN, "model": str(PREDICTOR_DIR),
        "n_points": len(results), "range": [results[0]["date"], results[-1]["date"]],
        "spearman_ret": float(spear), "spearman_p": float(p),
        "spearman_close": float(spear_c), "accuracy": acc,
        "strat_nav": strat_nav, "buyhold_nav": bh_nav, "excess": excess,
        "rows": results,
    }
    out = OUT_DIR / args.out
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n结果已写 {out}", flush=True)


if __name__ == "__main__":
    main()
