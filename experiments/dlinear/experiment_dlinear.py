"""V3 方向二实验：DLinear vs T4 LSTM 70 个月逐月 IC 对照（2026-08-06）

体检 T4：用 10 秒级训练的线性模型（DLinear）对照 15min 训练的 LSTM，
回答"LSTM 的时序记忆在 80 维日线特征上是否真的有用"。

对照组（零重训成本）: production prediction_cache.json 的 t4 预测（72 个月全有值）
实验组: DLinear（纯 numpy，无新依赖）
  - 输入 (n, 120, 80) → 移动平均分解(kernel=25) → 趋势 + 残差
  - 各自线性层 (9600→1) → y2 = 趋势预测 + 残差预测
  - SGD 20 epochs, batch 256, lr 0.01, weight decay 1e-4（5000 条 × 9600 维过参数化，正则必需）

口径与生产 T4 完全一致（公平对照，方向一教训"除被比较对象外一切相同"）:
  - 训练数据: 80 维缓存（62cf234c5440），12 月滚动窗口，5000 尾部抽样（生产 build 同款）
  - 预测特征: ref_date 前 120 天 × 80 维（include_market_state=False，与 t4_monthly_worker 同款）
  - 预测对象: 全股票池（.stock_pool.json）

工程（沿用方向一已验证配置）:
  - 8 workers × 特征 8 进程（ProcessPoolExecutor 嵌套，KMP_AFFINITY 清除）
  - nohup 解绑 + 断点续跑（.tmp/dlinear/{month}.json）
  - 输出: output/backtest_v2/experiments/dlinear_predictions.json + ic_report.csv

用法:
  python experiments/dlinear/experiment_dlinear.py --month 2026-06        # 单月验证
  python experiments/dlinear/experiment_dlinear.py                         # 全量 70 个月
  python experiments/dlinear/experiment_dlinear.py --analyze               # IC 对照分析
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── 铁律一：线程控制，必须在 import numpy 之前 ──
os.environ.pop("KMP_AFFINITY", None)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# ── 常量 ──
CACHE_DIR_80 = PROJECT_ROOT / "data/cache/v2_dataset/62cf234c5440"   # 80 维缓存（T4/DLinear 用）
OUT_DIR = PROJECT_ROOT / "output/backtest_v2/experiments/dlinear"
TMP_DIR = PROJECT_ROOT / ".tmp/dlinear"
PREDICTIONS_PATH = PROJECT_ROOT / "output/backtest_v2/experiments/dlinear_predictions.json"
IC_REPORT_PATH = PROJECT_ROOT / "output/backtest_v2/experiments/dlinear_ic_report.csv"
DB_PATH = str(PROJECT_ROOT / "data/sequoia_v2.db")
STOCK_POOL_PATH = PROJECT_ROOT / "output/backtest_v2/.stock_pool.json"

TRAIN_MONTHS = 12
MAX_TRAIN_SAMPLES = 5000      # 与生产 build 的 T4 口径一致
FEAT_WORKERS = 8
N_POOL_WORKERS = 8

# DLinear 超参
MA_KERNEL = 25                # 移动平均分解核
L2 = 1.0                      # Ridge 正则（5000 条 × 9600 维过参数化；SGD 已弃用——数值发散）
SEED = 42


# ══════════════════ DLinear 纯 numpy 实现 ══════════════════

def moving_average(x: np.ndarray, kernel: int = MA_KERNEL) -> np.ndarray:
    """沿时间轴（axis=1）移动平均分解趋势分量（nearest padding 保持长度）。"""
    from scipy.ndimage import convolve1d
    k = np.ones(kernel, dtype=np.float32) / kernel
    # origin 居中：convolve1d 输出长度不变；nearest 边界填充
    return convolve1d(x, k, axis=1, mode="nearest", origin=0).astype(np.float32)


def train_dlinear(X: np.ndarray, y: np.ndarray, l2: float = 1.0) -> tuple:
    """训练 DLinear（Ridge 闭式解——2026-08-06 修正：SGD 在 9600 维展平特征上数值发散
    （梯度爆炸 NaN），线性模型用正规方程一步到位，数学最优且不会发散）。

    Args:
        X: (n, 120, 80) 训练特征（float32）
        y: (n,) y2 标签
        l2: Ridge 正则强度（5000 样本 × 9600 维过参数化，正则必需；λ=1.0 标准起步）

    Returns:
        (Wt, Wr, bias): 趋势/残差权重与共享偏置（y 中心化后无截距项）
    """
    n = len(X)
    trend = moving_average(X)
    resid = X - trend
    # float64：9600×9600 矩阵求逆需要高精度（float32 条件数大时精度不足）
    Xt = trend.reshape(n, -1).astype(np.float64)   # (n, 9600)
    Xr = resid.reshape(n, -1).astype(np.float64)
    y_c = y.astype(np.float64) - float(y.mean())   # 中心化 → 无截距

    def ridge(Xm: np.ndarray) -> np.ndarray:
        XtX = Xm.T @ Xm + l2 * np.eye(Xm.shape[1], dtype=np.float64)
        return np.linalg.solve(XtX, Xm.T @ y_c)

    Wt = ridge(Xt)
    Wr = ridge(Xr)
    return Wt, Wr, float(y.mean())


def predict_dlinear(X: np.ndarray, model: tuple) -> np.ndarray:
    """预测 y2（截面排序用）。"""
    Wt, Wr, bias = model
    n = len(X)
    trend = moving_average(X)
    resid = X - trend
    Xt = trend.reshape(n, -1).astype(np.float64)
    Xr = resid.reshape(n, -1).astype(np.float64)
    return (Xt @ Wt + Xr @ Wr + bias).astype(np.float64)


def _build_one_features_80(args: tuple):
    """单只股票 80 维特征构建（复制自 build_prediction_cache._build_one_features，
    仅 include_market_state=False——T4/DLinear 同口径，隔离规则 §3.1）。"""
    from sequoia_x.model_selection_v2.features import _extract_per_day_features
    sym, df, idx_df, cfg = args
    if df is None or len(df) < cfg.window + 10:
        return None
    per_day = _extract_per_day_features(df, idx_df, cfg, include_market_state=False)
    if len(per_day) < cfg.window:
        return None
    return sym, per_day[-cfg.window:]


# ══════════════════ 单月 worker ══════════════════

def _dlinear_month_worker(args: tuple) -> tuple:
    """单月 DLinear 训练+预测（模块级，供 ProcessPoolExecutor 调用）。

    Returns:
        (month, n_valid, pred_std, train_seconds) 或 (month, None, 0, 0)
    """
    month, cfg_dict, max_pool_size = args

    import sqlite3
    import pandas as pd

    _os = __import__("os")
    _os.environ.pop("KMP_AFFINITY", None)
    _os.environ["OMP_NUM_THREADS"] = "1"
    _os.environ["OPENBLAS_NUM_THREADS"] = "1"
    _os.environ["MKL_NUM_THREADS"] = "1"
    _os.environ["NUMEXPR_NUM_THREADS"] = "1"
    print(f"[Worker {month}] 启动诊断: CPU核={_os.cpu_count()} OMP=1 KMP清除", flush=True)

    marker = TMP_DIR / f"{month}.json"
    if marker.exists():
        print(f"[Worker {month}] 跳过：已完成 ({marker})", flush=True)
        return month, None, 0, 0

    # ── mmap 加载 80 维缓存 ──
    X = np.load(str(CACHE_DIR_80 / "X.npy"), mmap_mode="r")
    y2 = np.load(str(CACHE_DIR_80 / "y2.npy"), mmap_mode="r")
    with open(CACHE_DIR_80 / "dates.json") as f:
        dates = json.load(f)
    dates_arr = np.array(dates)
    print(f"[Worker {month}] mmap加载: X={X.shape} (80维), {len(set(dates))}采样日期", flush=True)

    # ── 训练截止日（上月最后交易日）──
    ym_year, ym_month = int(month[:4]), int(month[5:7])
    prev_m = ym_month - 1
    prev_y = ym_year
    if prev_m <= 0:
        prev_m += 12
        prev_y -= 1
    conn = sqlite3.connect(DB_PATH)
    last_date_row = conn.execute(
        "SELECT MAX(date) FROM stock_daily WHERE date >= ? AND date < ?",
        (f"{prev_y}-{prev_m:02d}-01", month + "-01"),
    ).fetchone()
    train_end_date = last_date_row[0] if last_date_row and last_date_row[0] else (month + "-01")

    # ── 股票池 ──
    from build_prediction_cache import _filter_stock_pool
    if STOCK_POOL_PATH.exists():
        stock_pool = json.loads(STOCK_POOL_PATH.read_text())
    else:
        all_symbols = [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM stock_daily").fetchall()]
        stock_pool = _filter_stock_pool(all_symbols, DB_PATH)
    if max_pool_size > 0 and len(stock_pool) > max_pool_size:
        import random
        random.seed(42)
        stock_pool = random.sample(stock_pool, max_pool_size)
    conn.close()

    # ── 训练数据（12 月窗口 + 5000 尾部抽样，与生产 T4 同口径）──
    end_ym = train_end_date[:7]
    ey, em = int(end_ym[:4]), int(end_ym[5:7])
    sm = em - TRAIN_MONTHS
    sy = ey
    if sm <= 0:
        sm += 12
        sy -= 1
    train_start = f"{sy}-{sm:02d}-01"
    mask = (dates_arr >= train_start) & (dates_arr <= train_end_date)
    n_train = mask.sum()
    if n_train < 100:
        print(f"[Worker {month}] ⚠ 训练样本不足({n_train})，跳过", flush=True)
        return month, None, 0, 0
    X_tr = X[mask]
    y_tr = y2[mask]
    if n_train > MAX_TRAIN_SAMPLES:
        X_tr = X_tr[-MAX_TRAIN_SAMPLES:]
        y_tr = y_tr[-MAX_TRAIN_SAMPLES:]
    print(f"[Worker {month}] 训练数据: {len(y_tr)} 条 (窗口 {train_start} ~ {train_end_date}, "
          f"5000尾部抽样与T4同口径)", flush=True)

    # ── 重建 cfg ──
    from sequoia_x.model_selection_v2.config import V2Config
    cfg = V2Config()
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)

    # ── 训练 DLinear ──
    t0 = time.time()
    print(f"[Worker {month}] Step1: DLinear训练(samples={len(y_tr)}, Ridge λ={L2})...", flush=True)
    model = train_dlinear(X_tr, y_tr)
    print(f"[Worker {month}] Step1完成 ({time.time()-t0:.0f}s)", flush=True)
    del X_tr, y_tr
    import gc
    gc.collect()

    # ── OHLCV 预加载 → 80 维特征构建 ──
    print(f"[Worker {month}] Step2: OHLCV预加载({len(stock_pool)}只)...", flush=True)
    conn = sqlite3.connect(DB_PATH)
    ph = ",".join("?" * len(stock_pool))
    ohlcv_df = pd.read_sql(
        f"SELECT * FROM stock_daily WHERE symbol IN ({ph}) AND date <= ? ORDER BY symbol, date",
        conn, params=stock_pool + [train_end_date])
    idx_df = pd.read_sql(
        "SELECT * FROM index_daily WHERE symbol='sh.000300' AND date <= ? ORDER BY date",
        conn, params=(train_end_date,))
    conn.close()
    ohlcv_cache = {sym: g.reset_index(drop=True) for sym, g in ohlcv_df.groupby("symbol")}
    print(f"[Worker {month}] OHLCV={len(ohlcv_df)}行 {len(ohlcv_cache)}只, "
          f"80维特征构建({FEAT_WORKERS}进程并行)...", flush=True)

    from concurrent.futures import ProcessPoolExecutor
    tasks = [(sym, ohlcv_cache.get(sym), idx_df, cfg) for sym in stock_pool]
    with ProcessPoolExecutor(max_workers=FEAT_WORKERS) as ex:
        results = list(ex.map(_build_one_features_80, tasks, chunksize=100))
    X_list = [r[1] for r in results if r is not None]
    sym_list = [r[0] for r in results if r is not None]
    if not X_list:
        print(f"[Worker {month}] ❌ 特征构建全部失败", flush=True)
        return month, None, 0, 0
    X_pred = np.stack(X_list, axis=0)
    n_valid = len(X_pred)
    print(f"[Worker {month}] 特征完成: {n_valid}/{len(stock_pool)} 有效", flush=True)

    # ── 预测 + 自检（铁律一）──
    pred = predict_dlinear(X_pred, model)
    pred_std = float(np.std(pred))
    if pred_std < 1e-7:
        print(f"[Worker {month}] ❌ 严重: DLinear预测无方差! std={pred_std:.2e}", flush=True)
    else:
        print(f"[Worker {month}] ✅ DLinear pred: std={pred_std:.4f} "
              f"range=[{pred.min():.3f},{pred.max():.3f}]", flush=True)

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({
        "month": month, "train_end_date": train_end_date, "n": n_valid,
        "symbols": sym_list, "dlinear": [float(v) for v in pred],
    }))
    print(f"[Worker {month}] ✅ 完成并保存 ({marker})", flush=True)
    return month, n_valid, pred_std, time.time() - t0


# ══════════════════ 主流程 ══════════════════

def get_experiment_months(start_month: str = "2020-09", end_month: str = "2026-06") -> list[str]:
    y0, m0 = int(start_month[:4]), int(start_month[5:7])
    y1, m1 = int(end_month[:4]), int(end_month[5:7])
    months = []
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        months.append(f"{y}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def run_experiment(months: list[str], max_pool_size: int = 0, n_workers: int = N_POOL_WORKERS):
    from concurrent.futures import ProcessPoolExecutor

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    from sequoia_x.model_selection_v2.config import get_config
    cfg = get_config()
    cfg_dict = {k: v for k, v in cfg.__dict__.items() if not k.startswith("_")}

    print(f"═══ DLinear 对照实验（{len(months)} 个月）═══", flush=True)
    print(f"  并行: {n_workers} workers × 特征{FEAT_WORKERS}进程 | 窗口: {TRAIN_MONTHS}月 | "
          f"抽样: {MAX_TRAIN_SAMPLES}(与T4同口径) | MA核: {MA_KERNEL} | Ridge λ={L2}", flush=True)
    print(f"  KMP_AFFINITY: 已清除 | 环境: {sys.executable}", flush=True)

    t_start = time.time()
    args_list = [(m, cfg_dict, max_pool_size) for m in months]
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(_dlinear_month_worker, args_list))

    done = [r for r in results if r and r[1]]
    print(f"\n═══ 完成: {len(done)}/{len(months)} 个月 总耗时 {time.time()-t_start:.0f}s ═══", flush=True)
    for m, n, std, sec in done:
        print(f"  {m}: n={n} std={std:.4f} 训练+预测={sec:.0f}s", flush=True)

    merge_predictions(months)


def merge_predictions(months: list[str]):
    merged = {}
    for month in months:
        marker = TMP_DIR / f"{month}.json"
        if marker.exists():
            d = json.loads(marker.read_text())
            merged[month] = {"symbols": d["symbols"], "dlinear": d["dlinear"]}
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_PATH.write_text(json.dumps(merged))
    print(f"合并完成: {len(merged)} 个月 → {PREDICTIONS_PATH}")


# ══════════════════ IC 对照分析（--analyze）══════════════════

def _load_y2_for_month(month: str, symbols: list[str]):
    import sqlite3
    import pandas as pd

    conn = sqlite3.connect(DB_PATH)
    last_date = conn.execute(
        "SELECT MAX(date) FROM stock_daily WHERE date LIKE ?", (month + "%",)
    ).fetchone()[0]
    if last_date is None:
        conn.close()
        return None, None
    ph = ",".join("?" * len(symbols))
    prices = pd.read_sql(
        f"SELECT symbol, date, close FROM stock_daily "
        f"WHERE symbol IN ({ph}) AND date >= ? ORDER BY symbol, date",
        conn, params=symbols + [last_date])
    idx_df = pd.read_sql(
        "SELECT date, close FROM index_daily WHERE symbol='sh.000300' AND date >= ? ORDER BY date",
        conn, params=(last_date,))
    conn.close()

    t_close, t20_close = {}, {}
    for sym, g in prices.groupby("symbol"):
        g = g.sort_values("date")
        row_t = g[g["date"] == last_date]
        if row_t.empty:
            continue
        t_close[sym] = float(row_t["close"].iloc[0])
        future = g[g["date"] > last_date]
        if len(future) >= 20:
            t20_close[sym] = float(future["close"].iloc[19])
    idx_t = idx_df[idx_df["date"] == last_date]
    if idx_t.empty:
        return None, None
    idx_future = idx_df[idx_df["date"] > last_date]
    if len(idx_future) < 20:
        return None, None
    idx_ret = float(idx_future["close"].iloc[19]) / float(idx_t["close"].iloc[0]) - 1

    valid, y2s = [], []
    for sym in symbols:
        if sym in t_close and sym in t20_close:
            stock_ret = t20_close[sym] / t_close[sym] - 1
            valid.append(sym)
            y2s.append(np.clip(stock_ret - idx_ret, -0.5, 0.5))
    return valid, np.array(y2s)


def analyze_ic_report():
    """逐月 IC 对照：DLinear vs T4（production t4）vs T2（production，参考），同股票交集。"""
    from scipy.stats import spearmanr
    from build_prediction_cache import OUTPUT_PATH

    cache = json.loads(OUTPUT_PATH.read_text())
    exp = {}
    for mf in sorted(TMP_DIR.glob("*.json")):
        d = json.loads(mf.read_text())
        if "dlinear" in d:
            exp[d["month"]] = {"symbols": d["symbols"], "dlinear": d["dlinear"]}

    months = sorted(set(cache.keys()) & set(exp.keys()))
    print(f"═══ 逐月 IC 对照：DLinear vs T4 vs T2（{len(months)} 个月，同股票交集）═══")

    rows = []
    for month in months:
        t4_syms = cache[month]["symbols"]
        dl_syms = exp[month]["symbols"]
        common = sorted(set(t4_syms) & set(dl_syms))
        if len(common) < 50:
            continue
        valid, y2 = _load_y2_for_month(month, common)
        if valid is None or len(valid) < 50:
            continue
        y2_aligned = np.array([y2[valid.index(s)] for s in valid])
        t4_pred = np.array([cache[month]["t4"][t4_syms.index(s)] for s in valid])
        dl_pred = np.array([exp[month]["dlinear"][dl_syms.index(s)] for s in valid])
        t2_pred = np.array([cache[month]["t2"][t4_syms.index(s)] for s in valid])
        ic_t4, _ = spearmanr(t4_pred, y2_aligned)
        ic_dl, _ = spearmanr(dl_pred, y2_aligned)
        ic_t2, _ = spearmanr(t2_pred, y2_aligned)
        corr_dl_t4, _ = spearmanr(dl_pred, t4_pred)
        rows.append({
            "month": month, "n": len(valid),
            "t4_ic": round(ic_t4, 4), "dlinear_ic": round(ic_dl, 4),
            "t2_ic": round(ic_t2, 4),
            "delta_dl_vs_t4": round(ic_dl - ic_t4, 4),
            "corr_dl_t4": round(corr_dl_t4, 4),
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    print(df[["month", "n", "t4_ic", "dlinear_ic", "t2_ic", "delta_dl_vs_t4", "corr_dl_t4"]]
          .to_string(index=False))

    print("\n═══ 总体统计 ═══")
    print(f"  月份数: {len(df)}")
    print(f"  T4 IC 均值: {df['t4_ic'].mean():+.4f}（基线，生产 LSTM）")
    print(f"  DLinear IC 均值: {df['dlinear_ic'].mean():+.4f}")
    print(f"  T2 IC 均值(参考): {df['t2_ic'].mean():+.4f}")
    print(f"  DLinear vs T4 差值均值: {df['delta_dl_vs_t4'].mean():+.4f}")
    print(f"  DLinear 赢过 T4 的月份占比: {(df['delta_dl_vs_t4'] > 0).mean()*100:.0f}%")
    print(f"  DLinear 正 IC 月份占比: {(df['dlinear_ic'] > 0).mean()*100:.0f}% (T4: {(df['t4_ic'] > 0).mean()*100:.0f}%)")
    print(f"  corr(DLinear, T4) 均值: {df['corr_dl_t4'].mean():+.3f}")

    print("\n═══ 按年度汇总 ═══")
    df["year"] = df["month"].str[:4]
    for y, g in df.groupby("year"):
        print(f"  {y}: T4={g['t4_ic'].mean():+.3f} DLinear={g['dlinear_ic'].mean():+.3f} "
              f"T2={g['t2_ic'].mean():+.3f} dl胜率={(g['delta_dl_vs_t4']>0).mean()*100:.0f}%")

    ic_t4_m = df["t4_ic"].mean()
    ic_dl_m = df["dlinear_ic"].mean()
    pos_t4 = (df["t4_ic"] > 0).mean()
    pos_dl = (df["dlinear_ic"] > 0).mean()
    print("\n═══ 门槛判定（V3 §7.4）═══")
    print(f"  ① DLinear IC 均值 > T4?  {ic_dl_m:+.4f} vs {ic_t4_m:+.4f} → {'✅ 达标' if ic_dl_m > ic_t4_m else '❌ 未达标'}")
    print(f"  ② 正 IC 月占比 ≥ T4?  {pos_dl*100:.0f}% vs {pos_t4*100:.0f}% → {'✅ 达标' if pos_dl >= pos_t4 else '❌ 未达标'}")
    print(f"  ③ corr(DLinear, T4) < 0.5（融合前提）?  {df['corr_dl_t4'].mean():+.3f} → {'✅ 达标' if df['corr_dl_t4'].mean() < 0.5 else '❌ 未达标'}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(IC_REPORT_PATH, index=False)
    print(f"\n已保存: {IC_REPORT_PATH}")


def main():
    ap = argparse.ArgumentParser(description="V3 方向二：DLinear vs T4 LSTM 逐月 IC 对照")
    ap.add_argument("--month", help="仅跑单月（测试）")
    ap.add_argument("--start-month", default="2020-09")
    ap.add_argument("--end-month", default="2026-06")
    ap.add_argument("--max-stocks", type=int, default=0, help="限制股票池（测试）")
    ap.add_argument("--workers", type=int, default=N_POOL_WORKERS)
    ap.add_argument("--analyze", action="store_true", help="只做 IC 对照分析")
    args = ap.parse_args()

    if args.analyze:
        analyze_ic_report()
        return

    months = [args.month] if args.month else get_experiment_months(args.start_month, args.end_month)
    print(f"实验月份: {months[0]} ~ {months[-1]}（{len(months)} 个月）")
    run_experiment(months, max_pool_size=args.max_stocks, n_workers=args.workers)


if __name__ == "__main__":
    main()
