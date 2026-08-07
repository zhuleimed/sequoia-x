#!/usr/bin/env python3
"""3a 第一步: 单股 Kronos 零样本推理测时（2026-08-07）

验证: 权重加载 + OHLCV 输入构造 + 推理输出 + 耗时（决定 70 个月全量成本）。

用法:
  py312 python step1_timing.py [code] [ref_date]
  例:   python step1_timing.py 600519 2026-06-30

依赖: experiments/kronos/model/（ZNDX 拷贝的 Kronos 官方本地副本）+
      quant/code/models/Kronos-base（官方权重, local_files_only 离线加载）
"""
import sys
import time
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "experiments/kronos"))
sys.path.insert(0, str(PROJECT_ROOT))

from model import Kronos, KronosTokenizer, KronosPredictor

DB = PROJECT_ROOT / "data/sequoia_v2.db"
MODELS_DIR = Path("/public/home/hpc/zhulei/superman/quant/code/models")
LOOKBACK = 120      # 与 V2 window 一致
PRED_LEN = 20       # 与 V2 T2 预测窗口一致（未来 20 交易日）


def load_ohlcv(code: str, ref_date: str | None = None, n: int = 260) -> pd.DataFrame:
    """读 DB 日线（≤ref_date 或最近 n 行）。"""
    conn = sqlite3.connect(str(DB))
    if ref_date:
        df = pd.read_sql(
            "SELECT * FROM stock_daily WHERE symbol=? AND date<=? ORDER BY date",
            conn, params=[code, ref_date])
    else:
        df = pd.read_sql(
            "SELECT * FROM stock_daily WHERE symbol=? ORDER BY date DESC LIMIT ?",
            conn, params=[code, n])
        df = df.iloc[::-1].reset_index(drop=True)
    conn.close()
    return df


def main() -> None:
    code = sys.argv[1] if len(sys.argv) > 1 else "600519"
    ref_date = sys.argv[2] if len(sys.argv) > 2 else None

    df = load_ohlcv(code, ref_date)
    if df is None or len(df) < LOOKBACK + PRED_LEN:
        print(f"数据不足: {len(df) if df is not None else 0} 行（需 ≥{LOOKBACK + PRED_LEN}）")
        return
    print(f"数据: {code} {len(df)} 行 ({df['date'].iloc[0]} ~ {df['date'].iloc[-1]})")

    # ── 模型加载（离线 local_files_only, 不计入推理成本）──
    t0 = time.time()
    tokenizer = KronosTokenizer.from_pretrained(
        str(MODELS_DIR / "Kronos-Tokenizer-base"), local_files_only=True)
    model = Kronos.from_pretrained(str(MODELS_DIR / "Kronos-base"), local_files_only=True)
    predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)
    print(f"模型加载: {time.time()-t0:.1f}s")

    # ── 输入构造（无 look-ahead: 只用 ref_date 前数据; y_timestamp 是待预测区间）──
    df = df.copy()
    df["timestamps"] = pd.to_datetime(df["date"])
    x_df = df.loc[:LOOKBACK - 1, ["open", "high", "low", "close", "volume", "amount"]].copy()
    # amount 清洗: DB 腾讯源 amount 基本未入库(NaN) → 按 Kronos 估算方式补 volume×均价
    # （Kronos 校验要求 price/volume/amount 全部无 NaN）
    x_df["amount"] = x_df["amount"].fillna(
        x_df["volume"] * x_df[["open", "high", "low", "close"]].mean(axis=1))
    x_df = x_df.ffill().bfill()  # 兜底: 极端缺失行用前/后值
    x_ts = df.loc[:LOOKBACK - 1, "timestamps"]
    y_ts = df.loc[LOOKBACK:LOOKBACK + PRED_LEN - 1, "timestamps"]

    # ── 推理（复用 ZNDX 参数: T=0.2 近贪婪, top_p=0.9, 30 次采样）──
    t0 = time.time()
    pred_df = predictor.predict(
        df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
        pred_len=PRED_LEN, T=0.2, top_p=0.9, sample_count=30, verbose=False,
    )
    t_pred = time.time() - t0

    # ── 输出适配: 预测 close 中位数 → 预期收益（截面排序用）──
    pred_close = pred_df["close"].values
    actual_close = df["close"].values[-1]
    exp_ret = float(np.median(pred_close)) / actual_close - 1.0
    print(f"\n预测(median) close={np.median(pred_close):.2f} | 当前 close={actual_close:.2f} "
          f"| 预期 20 日收益={exp_ret:+.2%}")
    print(f"pred_close 分布: p25={np.percentile(pred_close,25):.2f} "
          f"median={np.median(pred_close):.2f} p75={np.percentile(pred_close,75):.2f} "
          f"std={pred_close.std():.2f}")

    # ── 成本评估（关键输出）──
    print(f"\n耗时: 推理 {t_pred:.1f}s/股")
    print(f"  → 单月 2900 只: {t_pred*2900/3600:.1f}h（单进程）")
    print(f"  → 70 个月: {t_pred*2900*70/3600:.0f}h 单进程 / 12 进程 ≈ {t_pred*2900*70/3600/12:.0f}h")


if __name__ == "__main__":
    main()
