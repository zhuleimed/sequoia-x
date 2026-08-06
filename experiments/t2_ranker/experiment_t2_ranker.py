"""V3 方向一实验：LGBMRanker vs LGBMRegressor 71 个月逐月 IC 对照（2026-08-06）

对比对象：
  - T2 回归（现有生产）：LGBMRegressor, huber objective, 预测 y2 数值
  - T2-Ranker（实验）  ：LGBMRanker, lambdarank objective, 直接优化排序（NDCG@10）

口径与生产 build_prediction_cache.py 完全一致（保证对照公平）：
  - 训练数据：88 维缓存（13132147f8e8），12 月滚动窗口，尾部 MAX_TRAIN_SAMPLES=5000 抽样
  - 训练截止日：目标月上一月的最后交易日（train_end_date）
  - 预测特征：ref_date 前 120 天窗口 × 88 维（实时构建，与 build 相同）
  - 预测对象：全股票池（.stock_pool.json）

Ranker 特有：
  - group = 每个采样日的样本数（同截面为一组，lightgbm 要求组内样本连续）
  - label = y2 按全训练样本 6 分位分档（0~5），label_gain=[0,1,3,7,15,31]
  - 验证集 = 最后 ~10% 采样日（时序划分，防泄漏），early stopping
  - 预测输出 raw_score（未 sigmoid，用于排序）

用法：
  python scripts/experiment_t2_ranker.py --month 2026-07          # 单月测试
  python scripts/experiment_t2_ranker.py                           # 全量 71 个月
  python scripts/experiment_t2_ranker.py --analyze                 # 逐月 IC 对照分析
  python scripts/experiment_t2_ranker.py --max-stocks 300 --month 2026-07  # 小池测试

输出：
  output/backtest_v2/experiments/t2_ranker_predictions.json   # 71 个月 ranker 预测
  output/backtest_v2/experiments/t2_ranker_ic_report.csv      # 逐月 IC 对照（--analyze）
  .tmp/t2_ranker/{month}.json                                 # 单月完成标记（断点续跑）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── 铁律一：线程控制，必须在 import numpy 之前 ──
os.environ.pop("KMP_AFFINITY", None)          # 必须 pop（.bashrc 的绑核，24 worker 抢 1 核）
os.environ["OMP_NUM_THREADS"] = "1"           # 硬赋值（.bashrc 的 36 必须覆盖）
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

# 项目根目录（experiments/t2_ranker/../..）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))  # 复用 build_prediction_cache 的模块级函数（只读，不改动）

from sequoia_x.model_selection_v2.config import V2Config, get_config

logger_path = PROJECT_ROOT / "logs"

# ── 常量（与 build_prediction_cache.py 完全一致）──
CACHE_DIR_88 = PROJECT_ROOT / "data/cache/v2_dataset/13132147f8e8"  # 88 维缓存（T2/T1/T3）
OUT_DIR = PROJECT_ROOT / "output/backtest_v2/experiments/t2_ranker"
TMP_DIR = PROJECT_ROOT / ".tmp/t2_ranker"
PREDICTIONS_PATH = PROJECT_ROOT / "output/backtest_v2/experiments/t2_ranker_predictions.json"
IC_REPORT_PATH = PROJECT_ROOT / "output/backtest_v2/experiments/t2_ranker_ic_report.csv"
DB_PATH = str(PROJECT_ROOT / "data/sequoia_v2.db")
STOCK_POOL_PATH = PROJECT_ROOT / "output/backtest_v2/.stock_pool.json"

TRAIN_MONTHS = 12          # 滚动窗口（V2 定稿）
# ⚠️ 2026-08-06 实验设计修正（V3 文档 §6 记录）：
#   生产 build 用 MAX_TRAIN_SAMPLES=5000 尾部抽样 ≈ 仅最后 1.7 个采样日（24 采样日中）——
#   实测对 ranker 灾难性（2 组训练 → 泛化 IC -0.41）。正式对照实验必须"除目标函数外
#   一切相同"：回归与 ranker 都用 12 月窗口全量样本（24 采样日 ~7 万条）训练。
MAX_TRAIN_SAMPLES = 0      # 0 = 全量（不用尾部抽样）；生产口径参考值 5000（已在诊断中记录）
FEAT_WORKERS = 8           # 特征构建并行数（每 worker 内嵌套，build 同款）
# ⚠️ 2026-08-06 内存事故教训（V3 文档 §6.7）：LightGBM 10560 维特征训练实测峰值
#   ~17GB/worker → 并发预算铁律：workers × 实测峰值 ≤ 内存总量 × 0.8。
#   187GB × 0.8 / 17GB ≈ 8 workers 为安全上限（10+ 会超内存换页，慢 5 倍）
N_POOL_WORKERS = 8         # 月份并行数（内存安全上限）

# Ranker 超参（沿用 T2 回归的默认超参，仅 objective/metric 不同；不重新 Optuna）
RANKER_PARAMS = {
    "num_leaves": 31, "learning_rate": 0.1, "subsample": 0.8,
    "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0,
    "min_child_samples": 20,
}
LABEL_GAIN = [0, 1, 3, 7, 15, 31]   # 6 档（label 0~5），指数增益
N_LABEL_QUANTILES = 6               # y2 分位数档数


def get_experiment_months(start_month: str = "2020-09", end_month: str = "2026-06") -> list[str]:
    """实验月份列表（与 70 个月回测一致：2020-09 ~ 2026-06）。
    注意：end_month 不能晚于"y2 答案已揭晓"的月份——该月最后交易日 + 20 交易日需 ≤ 数据最新日期。"""
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


def train_ranker(
    X_tr_2d: np.ndarray, y_tr: np.ndarray, dates_tr: np.ndarray, cfg: V2Config,
) -> tuple:
    """训练 LGBMRanker（lambdarank）。

    Args:
        X_tr_2d: (n, 10560) 训练特征（12 月滚动窗口抽样后）
        y_tr:    (n,) y2 标签
        dates_tr: (n,) 每样本的采样日（构建 group 用，与 X_tr 同序）
        cfg: V2 配置

    Returns:
        model: lgb.Booster（rank 模型）
    """
    import lightgbm as lgb

    n = len(y_tr)
    if n < 200:
        raise ValueError(f"训练样本不足: {n}")

    # ── label 分档：y2 全训练样本 6 分位 → 0~5（越大越好）──
    qs = np.percentile(y_tr, np.linspace(0, 100, N_LABEL_QUANTILES + 1)[1:-1])
    label = np.digitize(y_tr, qs)  # 0~5
    label = np.clip(label, 0, N_LABEL_QUANTILES - 1)

    # ── group：按采样日计数（缓存按日期排序，同组连续）──
    _, counts = np.unique(dates_tr, return_counts=True)
    group = counts.tolist()

    # ── ⚠️ 2026-08-06 第二版修正（V3 文档 §6.7）：固定轮数训练，不做 early stopping ──
    #    实测 23 个月中 ranker 树数<10 占 48%、回归轮数<10 占 61%——时序尾部验证在
    #    A 股高噪声下 ndcg/RMSE 波动极大，早停第 2-9 轮就触发 → 模型几乎未学习。
    #    修复：两模型均固定 300 轮（lr=0.1 常规配置），绝对公平、确定性。

    params = {
        **RANKER_PARAMS,
        "objective": "lambdarank",
        "label_gain": LABEL_GAIN,
        "metric": "ndcg",
        "ndcg_eval_at": [10],
        "verbosity": -1,
        "n_jobs": cfg.n_jobs,
        "random_state": cfg.random_seed,
        "boost_from_average": False,
    }
    # group 必须与行顺序对应（缓存按日期排序，同组连续）
    _, tr_counts = np.unique(dates_tr, return_counts=True)
    train_data = lgb.Dataset(X_tr_2d, label=label, group=tr_counts.tolist())

    # 固定 300 轮（不早停，见上方注释）
    model = lgb.train(
        params, train_data,
        num_boost_round=300,
        callbacks=[lgb.log_evaluation(0)],
    )
    return model


def train_reg_fixed(X_tr_2d: np.ndarray, y_tr: np.ndarray, cfg: V2Config):
    """固定轮数版 LGBMRegressor（对照组，保证与 ranker 完全公平：同为 300 轮、无早停）。

    复制自生产 sequoia_x/model_selection_v2/models/tree_reg.py::train_reg 的参数与 objective
    （huber + alpha 0.1），仅去掉 TimeSeriesSplit 早停，改为固定 num_boost_round=300。
    """
    import lightgbm as lgb

    params = {
        "num_leaves": 31, "learning_rate": 0.1, "subsample": 0.8,
        "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0,
        "min_child_samples": 20,
        "objective": "huber",
        "alpha": 0.1,  # Huber delta
        "metric": "rmse",
        "verbosity": -1,
        "n_jobs": cfg.n_jobs,
        "random_state": cfg.random_seed,
    }
    train_data = lgb.Dataset(X_tr_2d, label=y_tr)
    model = lgb.train(
        params, train_data,
        num_boost_round=300,
        callbacks=[lgb.log_evaluation(0)],
    )
    return model


def _ranker_month_worker(args: tuple) -> tuple:
    """单月 ranker 训练+预测（模块级，供 multiprocessing.Pool 调用）。

    Args:
        args: (month, cfg_dict, max_pool_size)

    Returns:
        (month, n_valid, pred_std, train_seconds) 或 (month, None, 0, 0) 若失败/已跳过
    """
    month, cfg_dict, max_pool_size = args

    import sqlite3
    import pandas as pd

    # ── 铁律一：worker 内启动诊断 + 线程控制（与 build 完全一致）──
    _os = __import__("os")
    _os.environ.pop("KMP_AFFINITY", None)
    _os.environ["OMP_NUM_THREADS"] = "1"
    _os.environ["OPENBLAS_NUM_THREADS"] = "1"
    _os.environ["MKL_NUM_THREADS"] = "1"
    _os.environ["NUMEXPR_NUM_THREADS"] = "1"
    print(f"[Worker {month}] 启动诊断: CPU核={_os.cpu_count()} OMP=1 KMP清除", flush=True)

    # ── 断点续跑（铁律四）──
    marker = TMP_DIR / f"{month}.json"
    if marker.exists():
        print(f"[Worker {month}] 跳过：已完成 ({marker})", flush=True)
        return month, None, 0, 0

    # ── mmap 加载缓存 ──
    X = np.load(str(CACHE_DIR_88 / "X.npy"), mmap_mode="r")
    y2 = np.load(str(CACHE_DIR_88 / "y2.npy"), mmap_mode="r")
    with open(CACHE_DIR_88 / "dates.json") as f:
        dates = json.load(f)
    dates_arr = np.array(dates)
    print(f"[Worker {month}] mmap加载: X={X.shape}, {len(set(dates))}采样日期", flush=True)

    # ── 确定训练截止日（上月最后交易日，与 build 一致）──
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

    # ── 提取训练数据（12 月滚动窗口 + 尾部 5000 抽样，与 build 一致）──
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
    dates_tr = dates_arr[mask]
    if MAX_TRAIN_SAMPLES > 0 and n_train > MAX_TRAIN_SAMPLES:
        # 生产口径尾部抽样（默认关闭：对照实验用全量；5000 的失败诊断见 V3 文档 §6）
        X_tr = X_tr[-MAX_TRAIN_SAMPLES:]
        y_tr = y_tr[-MAX_TRAIN_SAMPLES:]
        dates_tr = dates_tr[-MAX_TRAIN_SAMPLES:]
    X_tr_2d = X_tr.reshape(len(X_tr), -1).astype(np.float32)  # float32: 7万×10560 ≈ 3GB
    print(f"[Worker {month}] 训练数据: {len(y_tr)} 条, {len(set(dates_tr))} 采样日"
          f" (窗口 {train_start} ~ {train_end_date}, 全量)", flush=True)

    # ── 重建 cfg ──
    cfg = V2Config()
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)

    # ── 训练 LGBMRanker（实验组）──
    t0 = time.time()
    print(f"[Worker {month}] Step1: LGBMRanker训练(samples={len(y_tr)})...", flush=True)
    model_ranker = train_ranker(X_tr_2d, y_tr, dates_tr, cfg)
    print(f"[Worker {month}] Step1完成: ranker树数={model_ranker.num_trees()} ({time.time()-t0:.0f}s)", flush=True)

    # ── 训练 LGBMRegressor（对照组，全量数据；目标函数不同，其余完全一致；固定 300 轮）──
    t1 = time.time()
    print(f"[Worker {month}] Step1b: LGBMRegressor训练(全量, 固定300轮, 对照)...", flush=True)
    model_reg = train_reg_fixed(X_tr_2d, y_tr, cfg)
    print(f"[Worker {month}] Step1b完成: 树数={model_reg.num_trees()} ({time.time()-t1:.0f}s)", flush=True)

    # ── ⚠️ 内存释放（2026-08-06 事故教训）：24 worker × 20GB 超售 187GB 内存 → 换页慢 5 倍 + OOM 风险
    #    X_tr/X_tr_2d 合计 ~6GB/worker，训练完成后立即释放，再进入特征构建阶段
    import gc
    del X_tr, X_tr_2d, y_tr, dates_tr, mask
    gc.collect()
    print(f"[Worker {month}] 训练数据已释放 (RSS→{__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss/1048576:.1f}GB)", flush=True)

    # ── OHLCV 预加载 → 特征构建（与 build 相同流程）──
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
    # ⚠️ 特征构建 8 进程并行：主进程已改用 ProcessPoolExecutor（非 daemon，worker 可嵌套，
    #    合法——daemon Pool 禁止嵌套见 BACKTEST_PLAN §20.4）
    print(f"[Worker {month}] OHLCV={len(ohlcv_df)}行 {len(ohlcv_cache)}只, "
          f"特征构建({FEAT_WORKERS}进程并行)...", flush=True)

    from build_prediction_cache import _build_one_features
    from concurrent.futures import ProcessPoolExecutor
    tasks = [(sym, ohlcv_cache.get(sym), idx_df, cfg) for sym in stock_pool]
    with ProcessPoolExecutor(max_workers=FEAT_WORKERS) as ex:
        results = list(ex.map(_build_one_features, tasks, chunksize=100))
    X_list = [r[1] for r in results if r is not None]
    sym_list = [r[0] for r in results if r is not None]
    if not X_list:
        print(f"[Worker {month}] ❌ 特征构建全部失败", flush=True)
        return month, None, 0, 0
    X_pred = np.stack(X_list, axis=0)
    n_valid = len(X_pred)
    X_pred_2d = X_pred.reshape(n_valid, -1)
    print(f"[Worker {month}] 特征完成: {n_valid}/{len(stock_pool)} 有效", flush=True)

    # ── 预测（ranker: raw_score 用于排序；回归: y2 预测值）──
    pred_rk = model_ranker.predict(X_pred_2d, raw_score=True).flatten()
    pred_reg = model_reg.predict(X_pred_2d).flatten()
    rk_std = float(np.std(pred_rk))
    reg_std = float(np.std(pred_reg))
    # ── 铁律一：运行时自检 ──
    if rk_std < 1e-7:
        print(f"[Worker {month}] ❌ 严重: ranker预测无方差! std={rk_std:.2e}", flush=True)
    if reg_std < 1e-7:
        print(f"[Worker {month}] ❌ 严重: 回归预测无方差! std={reg_std:.2e}", flush=True)
    if rk_std >= 1e-7:
        print(f"[Worker {month}] ✅ ranker pred: std={rk_std:.4f} "
              f"range=[{pred_rk.min():.3f},{pred_rk.max():.3f}]", flush=True)
    if reg_std >= 1e-7:
        print(f"[Worker {month}] ✅ 回归 pred: std={reg_std:.4f} "
              f"range=[{pred_reg.min():.3f},{pred_reg.max():.3f}]", flush=True)

    # ── 保存完成标记（断点续跑）──
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({
        "month": month, "train_end_date": train_end_date, "n": n_valid,
        "symbols": sym_list,
        "t2_ranker": [float(v) for v in pred_rk],
        "t2_reg_full": [float(v) for v in pred_reg],
        "ranker_trees": model_ranker.num_trees(),
        "reg_trees": model_reg.num_trees(),
    }))
    print(f"[Worker {month}] ✅ 完成并保存 ({marker})", flush=True)
    return month, n_valid, rk_std, time.time() - t0


def run_experiment(months: list[str], max_pool_size: int = 0, n_workers: int = N_POOL_WORKERS):
    """主流程：并行训练所有月份（断点续跑）。

    ⚠️ 2026-08-06 第二版：multiprocessing.Pool → ProcessPoolExecutor——
    daemon Pool worker 禁止嵌套子进程（§20.4），ProcessPoolExecutor 的 worker 非 daemon，
    worker 内可再开 8 进程特征并行（build 单月模式的合法嵌套）。
    """
    from concurrent.futures import ProcessPoolExecutor

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    cfg = get_config()
    cfg_dict = {k: v for k, v in cfg.__dict__.items() if not k.startswith("_")}

    pending = months
    print(f"═══ LGBMRanker 对照实验（{len(months)} 个月）═══", flush=True)
    print(f"  并行: {n_workers} workers × 特征{FEAT_WORKERS}进程 | 窗口: {TRAIN_MONTHS}月 | "
          f"抽样: {MAX_TRAIN_SAMPLES}", flush=True)
    print(f"  KMP_AFFINITY: 已清除 | 环境: {sys.executable}", flush=True)

    t_start = time.time()
    args_list = [(m, cfg_dict, max_pool_size) for m in pending]
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(_ranker_month_worker, args_list))

    done = [r for r in results if r and r[1]]
    failed = [r for r in results if r and r[1] is None and r[0] not in months]
    print(f"\n═══ 完成: {len(done)}/{len(months)} 个月 总耗时 {time.time()-t_start:.0f}s ═══", flush=True)
    for m, n, std, sec in done:
        print(f"  {m}: n={n} std={std:.4f} 训练+预测={sec:.0f}s", flush=True)

    merge_predictions(months)


def merge_predictions(months: list[str]):
    """合并单月完成文件 → 总预测文件。"""
    merged = {}
    for month in months:
        marker = TMP_DIR / f"{month}.json"
        if marker.exists():
            d = json.loads(marker.read_text())
            merged[month] = {
                "symbols": d["symbols"],
                "t2_ranker": d["t2_ranker"],
                "t2_reg_full": d.get("t2_reg_full", []),
            }
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_PATH.write_text(json.dumps(merged))
    print(f"合并完成: {len(merged)} 个月 → {PREDICTIONS_PATH}")


# ══════════════════════════ IC 对照分析（--analyze）══════════════════════════

def _load_y2_for_month(month: str, symbols: list[str]):
    """计算某月每只股票的实际 y2（未来 20 交易日超额收益，与 analyze_monthly_ic 同口径）。

    Returns:
        (valid_symbols, y2_arr) 或 (None, None) 若数据不足
    """
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
    """逐月 IC 三向对照（同股票交集口径）：
       - t2_ranker   : LGBMRanker（全量训练，实验组）
       - t2_reg_full : LGBMRegressor（全量训练，公平对照组）
       - t2 (生产)    : prediction_cache 的回归（5000 尾部抽样，生产参考）
    """
    from scipy.stats import spearmanr

    from build_prediction_cache import OUTPUT_PATH
    cache = json.loads(OUTPUT_PATH.read_text())
    # 直接从 TMP_DIR 读单月结果（不依赖合并文件，运行中可做中间分析）
    exp = {}
    for mf in sorted(TMP_DIR.glob("*.json")):
        d = json.loads(mf.read_text())
        if "t2_ranker" in d:
            exp[d["month"]] = {"symbols": d["symbols"],
                               "t2_ranker": d["t2_ranker"],
                               "t2_reg_full": d.get("t2_reg_full", [])}

    months = sorted(set(cache.keys()) & set(exp.keys()))
    print(f"═══ 逐月 IC 三向对照（{len(months)} 个月，同股票交集）═══")

    rows = []
    for month in months:
        reg_syms = cache[month]["symbols"]
        rk_syms = exp[month]["symbols"]
        common = sorted(set(reg_syms) & set(rk_syms))
        if len(common) < 50:
            continue
        valid, y2 = _load_y2_for_month(month, common)
        if valid is None or len(valid) < 50:
            continue
        y2_aligned = np.array([y2[valid.index(s)] for s in valid])
        # 同一股票集合上的三套预测
        reg_prod = np.array([cache[month]["t2"][reg_syms.index(s)] for s in valid])
        reg_full = np.array([exp[month]["t2_reg_full"][rk_syms.index(s)] for s in valid])
        rk_pred = np.array([exp[month]["t2_ranker"][rk_syms.index(s)] for s in valid])
        ic_reg_prod, _ = spearmanr(reg_prod, y2_aligned)
        ic_reg_full, _ = spearmanr(reg_full, y2_aligned)
        ic_rk, _ = spearmanr(rk_pred, y2_aligned)
        corr_rk_reg, _ = spearmanr(rk_pred, reg_full)
        rows.append({
            "month": month, "n": len(valid),
            "t2_prod_ic": round(ic_reg_prod, 4),     # 生产（5000 抽样）
            "t2_full_ic": round(ic_reg_full, 4),     # 回归全量（对照组）
            "t2_ranker_ic": round(ic_rk, 4),         # ranker 全量（实验组）
            "delta_rk_vs_full": round(ic_rk - ic_reg_full, 4),
            "delta_full_vs_prod": round(ic_reg_full - ic_reg_prod, 4),
            "corr_rk_reg": round(corr_rk_reg, 4),
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    print(df[["month", "n", "t2_prod_ic", "t2_full_ic", "t2_ranker_ic",
              "delta_rk_vs_full", "delta_full_vs_prod", "corr_rk_reg"]].to_string(index=False))

    print("\n═══ 总体统计 ═══")
    print(f"  月份数: {len(df)}")
    print(f"  生产回归 IC 均值: {df['t2_prod_ic'].mean():+.4f}（5000 尾部抽样）")
    print(f"  全量回归 IC 均值: {df['t2_full_ic'].mean():+.4f}（公平对照组）")
    print(f"  Ranker IC 均值:   {df['t2_ranker_ic'].mean():+.4f}（全量训练）")
    print(f"  全量 vs 生产(抽样损失): {df['delta_full_vs_prod'].mean():+.4f}")
    print(f"  Ranker vs 全量回归: {df['delta_rk_vs_full'].mean():+.4f}")
    print(f"  Ranker 赢过全量回归的月份占比: {(df['delta_rk_vs_full'] > 0).mean()*100:.0f}%")
    print(f"  Ranker 正 IC 月份占比: {(df['t2_ranker_ic'] > 0).mean()*100:.0f}% (全量回归: {(df['t2_full_ic'] > 0).mean()*100:.0f}%)")
    print(f"  corr(ranker, 全量回归) 均值: {df['corr_rk_reg'].mean():+.3f}")

    print("\n═══ 按年度汇总 ═══")
    df["year"] = df["month"].str[:4]
    for y, g in df.groupby("year"):
        print(f"  {y}: 生产={g['t2_prod_ic'].mean():+.3f} 全量回归={g['t2_full_ic'].mean():+.3f} "
              f"ranker={g['t2_ranker_ic'].mean():+.3f} rk胜率={(g['delta_rk_vs_full']>0).mean()*100:.0f}%")

    # 门槛判定（V3 文档 §6.5，对照基线更新为全量回归）
    ic_mean_rk = df["t2_ranker_ic"].mean()
    ic_mean_full = df["t2_full_ic"].mean()
    pos_rk = (df["t2_ranker_ic"] > 0).mean()
    pos_full = (df["t2_full_ic"] > 0).mean()
    print("\n═══ 门槛判定（V3 §6.5，基线=全量回归）═══")
    print(f"  ① Ranker IC 均值 > 全量回归?  {ic_mean_rk:+.4f} vs {ic_mean_full:+.4f} → {'✅ 达标' if ic_mean_rk > ic_mean_full else '❌ 未达标'}")
    print(f"  ② 正 IC 月占比 ≥ 全量回归?  {pos_rk*100:.0f}% vs {pos_full*100:.0f}% → {'✅ 达标' if pos_rk >= pos_full else '❌ 未达标'}")
    print(f"  ③ corr(ranker, 回归) < 0.5（融合前提）?  {df['corr_rk_reg'].mean():+.3f} → {'✅ 达标' if df['corr_rk_reg'].mean() < 0.5 else '❌ 未达标'}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(IC_REPORT_PATH, index=False)
    print(f"\n已保存: {IC_REPORT_PATH}")


def main():
    ap = argparse.ArgumentParser(description="V3 方向一：LGBMRanker vs LGBMRegressor 逐月 IC 对照")
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
