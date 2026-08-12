#!/usr/bin/env python
"""T4 LSTM 单月训练 Worker。

每个进程独立完成一个月份的：训练数据提取 → LSTM 训练 → 特征计算 → 预测 → 保存。

用法:
    python scripts/t4_monthly_worker.py --month 2025-08

由启动器并行调用 11 个实例，或手动单独运行。
"""
from __future__ import annotations

import sys, os, json, time, sqlite3, argparse, logging
from pathlib import Path

# ── 路径设置：项目根目录 ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── 环境变量：必须在 import tensorflow 之前设置 ──
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.pop("KMP_AFFINITY", None)         # 清除 .bashrc 锁核（CLAUDE.md 铁律）
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"    # 并行时每进程 2 线程
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"

import numpy as np
import pandas as pd

from sequoia_x.model_selection_v2.config import get_config
from sequoia_x.model_selection_v2.models.deep_lstm import train_lstm, predict_lstm
from sequoia_x.model_selection_v2.features import _extract_per_day_features

# ── 常量 ──
# T4 LSTM 使用 80 维特征（不含市场状态），树模型使用 88 维（含市场状态）。
# §3.6 已验证：LSTM 能从 120 步时序中隐式推断市场状态，显式特征反而引入噪声。
CACHE_DIR = PROJECT_ROOT / "data/cache/v2_dataset/62cf234c5440"  # 80维缓存 (2026-08-01 重建, 400728样本)
CACHE_PATH = PROJECT_ROOT / "output/backtest_v2/prediction_cache.json"
TMP_DIR = PROJECT_ROOT / "output/backtest_v2/.t4_tmp"
LOG_DIR = PROJECT_ROOT / "output/backtest_v2/.t4_logs"
STOCK_POOL_PATH = PROJECT_ROOT / "output/backtest_v2/.stock_pool.json"
DB_PATH = PROJECT_ROOT / "data/sequoia_v2.db"

# ── 月份列表（动态生成 2020-08 ~ 2026-06，共 71 个月）──
def _gen_months(start: str, end: str) -> list[str]:
    months = []
    sy, sm = int(start[:4]), int(start[5:7])
    ey, em = int(end[:4]), int(end[5:7])
    y, m = sy, sm
    while (y, m) <= (ey, em):
        months.append(f"{y}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


ALL_MONTHS = _gen_months("2020-08", "2026-12")


def setup_logger(month: str) -> logging.Logger:
    """为单个月份配置独立的文件日志（铁律一：详尽日志）。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"t4_{month}.log"

    logger = logging.getLogger(f"t4_{month}")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fh = logging.FileHandler(str(log_file), mode="a")
    # 强制无缓冲：每行日志立即写入磁盘，确保进度日志实时可见
    fh.stream.reconfigure(line_buffering=True) if hasattr(fh.stream, "reconfigure") else None
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)
    return logger


def load_shared_data():
    """加载所有 Worker 共享的只读数据（mmap + 股票池 + 日期）。"""
    X = np.load(str(CACHE_DIR / "X.npy"), mmap_mode="r")
    y2 = np.load(str(CACHE_DIR / "y2.npy"), mmap_mode="r")
    with open(CACHE_DIR / "dates.json") as f:
        dates = json.load(f)
    with open(STOCK_POOL_PATH) as f:
        stock_pool = json.load(f)
    return X, y2, np.array(dates), stock_pool


def compute_train_end(month: str) -> str:
    """计算该月份训练数据的截止日期（上月最后交易日）。"""
    ym_y, ym_m = int(month[:4]), int(month[5:7])
    pm = ym_m - 1
    py = ym_y
    if pm <= 0:
        pm += 12
        py -= 1

    conn = sqlite3.connect(str(DB_PATH))
    row = conn.execute(
        "SELECT MAX(date) FROM stock_daily WHERE date>=? AND date<?",
        (f"{py}-{pm:02d}-01", month + "-01"),
    ).fetchone()
    conn.close()
    return row[0] if row and row[0] else month + "-01"


def extract_train_data(
    X: np.ndarray,
    y2: np.ndarray,
    dates_arr: np.ndarray,
    train_end: str,
    max_samples: int = 5000,
) -> tuple[np.ndarray, np.ndarray]:
    """提取该月份的训练数据：train_end 前 12 个月，最多 max_samples 条。"""
    ey, em = int(train_end[:4]), int(train_end[5:7])
    sm = em - 12
    sy = ey
    if sm <= 0:
        sm += 12
        sy -= 1

    mask = (dates_arr >= f"{sy}-{sm:02d}-01") & (dates_arr <= train_end)
    n = mask.sum()
    if n < 100:
        raise ValueError(f"训练样本不足：仅 {n} 条")

    idx = np.where(mask)[0]
    if len(idx) > max_samples:
        idx = idx[-max_samples:]

    return X[idx], y2[idx]


def load_ohlcv_data(stock_pool: list[str], train_end: str):
    """加载股票日线 + 指数数据。"""
    conn = sqlite3.connect(str(DB_PATH))
    ph = ",".join("?" * len(stock_pool))
    ohlcv = pd.read_sql(
        f"SELECT * FROM stock_daily WHERE symbol IN ({ph}) AND date<=? "
        f"ORDER BY symbol, date",
        conn,
        params=stock_pool + [train_end],
    )
    idx = pd.read_sql(
        "SELECT * FROM index_daily WHERE symbol='sh.000300' AND date<=? "
        "ORDER BY date",
        conn,
        params=(train_end,),
    )
    conn.close()
    return ohlcv, idx


def build_features_and_predict(
    model,
    ohlcv: pd.DataFrame,
    idx: pd.DataFrame,
    stock_pool: list[str],
    cfg,
    logger: logging.Logger,  # 铁律一：传入 logger 用于进度汇报
) -> tuple[list[str], np.ndarray]:
    """对股票池逐只提取特征并预测。"""
    oc = {s: g.reset_index(drop=True) for s, g in ohlcv.groupby("symbol")}
    idx_empty = idx.empty

    Xl, sl = [], []
    n_pool = len(stock_pool)
    t_feat_loop = time.time()
    for i, s in enumerate(stock_pool):
        df = oc.get(s)
        if df is None or len(df) < cfg.window + 10:
            continue
        try:
            per_day = _extract_per_day_features(
                df, idx if not idx_empty else None, cfg,
                include_market_state=False, symbol=s,  # T4 用 80 维
            )
        except Exception:
            continue
        if len(per_day) < cfg.window:
            continue
        Xl.append(per_day[-cfg.window:])
        sl.append(s)
        # 进度日志（铁律一：特征提取每 500 只汇报一次进度）
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t_feat_loop
            rate = elapsed / max(len(sl), 1)
            remaining = rate * (n_pool - i - 1)
            pct = 100 * (i + 1) // n_pool
            logger.info(
                f"  特征提取进度: {i+1}/{n_pool} ({pct}%), "
                f"有效={len(sl)}, 耗时={elapsed:.0f}s, "
                f"速率={rate:.2f}s/stock, ETA={remaining:.0f}s"
            )

    if not Xl:
        raise RuntimeError("特征提取失败：无有效股票")

    Xp = np.stack(Xl)
    pred = predict_lstm(model, Xp).flatten()
    return sl, pred


def update_cache(month: str, symbols: list[str], pred: np.ndarray):
    """原子更新 prediction_cache.json 中该月份的 T4 预测。

    先写临时文件再 rename，保证原子性。同时写 .t4_tmp/ 标记已完成。
    """
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # 读当前缓存
    cache = json.loads(CACHE_PATH.read_text())

    # 标记该月份 T4 已完成（t4_tmp 快照：优先存完整条目，否则存 symbols+pred 供 merge 对齐）
    out_file = TMP_DIR / f"t4_{month}.json"
    if month in cache:
        # 主缓存已有该月（T2/T1/T3 就绪）→ 直接更新 t4 字段
        t4_map = dict(zip(symbols, [float(v) for v in pred]))
        cache[month]["t4"] = [t4_map.get(s, 0.0) for s in cache[month]["symbols"]]

        # 原子写入（写临时 + rename）
        tmp_path = CACHE_PATH.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
        tmp_path.rename(CACHE_PATH)

        with open(out_file, "w") as f:
            json.dump(cache[month], f, ensure_ascii=False)
    else:
        # 主缓存尚无该月（build 未完成，T4 与 build 并行的正常时序）→
        # 不抛错，只存 (symbols, t4) 快照，等 build 完成后由 --merge 对齐合并
        snapshot = {
            "symbols": symbols,
            "t4": [float(v) for v in pred],
            "_pending_merge": True,
        }
        with open(out_file, "w") as f:
            json.dump(snapshot, f, ensure_ascii=False)
        logger.warning(
            f"主缓存暂无 {month}（build 并行中），T4 结果已存 t4_tmp，"
            f"待 --merge 合并"
        )


def run_one_month(month: str) -> dict:
    """执行单个月份的完整 T4 训练+预测流程。

    Returns:
        {"month": str, "status": "ok"|"skip"|"error",
         "pred_mean": float, "pred_std": float, "elapsed": float}
    """
    logger = setup_logger(month)
    t_start = time.time()

    # ── 铁律一：启动诊断日志 ──
    import multiprocessing
    logger.info("=" * 60)
    logger.info(
        f"T4 Worker 启动 | month={month} | PID={os.getpid()} | "
        f"CPU={os.cpu_count()}核 | "
        f"TF_INTRA={os.environ.get('TF_NUM_INTRAOP_THREADS','?')} "
        f"TF_INTER={os.environ.get('TF_NUM_INTEROP_THREADS','?')} "
        f"OMP={os.environ.get('OMP_NUM_THREADS','?')}"
    )

    # ── 断点续跑：检查是否已完成 ──
    out_file = TMP_DIR / f"t4_{month}.json"
    if out_file.exists():
        logger.info(f"跳过：已完成（{out_file}）")
        return {"month": month, "status": "skip", "elapsed": time.time() - t_start}

    try:
        # 1. 加载共享数据
        logger.info("加载共享数据...")
        X, y2, dates_arr, stock_pool = load_shared_data()
        logger.info(
            f"数据加载完成 | X={X.shape} y2={y2.shape} "
            f"pool={len(stock_pool)} stocks dates={len(dates_arr)}"
        )

        # 2. 提取训练数据
        train_end = compute_train_end(month)
        logger.info(f"训练截止日: {train_end}")
        X_tr, y_tr = extract_train_data(X, y2, dates_arr, train_end)
        logger.info(
            f"训练数据 | samples={len(X_tr)} "
            f"y_mean={float(y_tr.mean()):.4f} y_std={float(y_tr.std()):.4f}"
        )

        # 3. 训练 T4 LSTM
        cfg = get_config()
        logger.info(
            f"T4 配置 | units={cfg.lstm_units} "
            f"num_transformers={cfg.lstm_num_transformers} "
            f"dropout={cfg.lstm_dropout_rate} lr={cfg.lstm_learning_rate} "
            f"l2={cfg.lstm_l2_reg} batch={cfg.lstm_batch_size}"
        )
        model = train_lstm(X_tr, y_tr, cfg, search_optuna=False, model_id=f"t4_{month}")

        # 4. 加载 OHLCV + 特征提取 + 预测
        logger.info("加载 OHLCV 数据...")
        ohlcv, idx = load_ohlcv_data(stock_pool, train_end)
        logger.info(f"OHLCV: {len(ohlcv)} rows, index: {len(idx)} rows")

        logger.info(f"特征提取+预测（{len(stock_pool)} 只股票）...")
        t_feat = time.time()
        symbols, pred = build_features_and_predict(
            model, ohlcv, idx, stock_pool, cfg, logger
        )
        feat_elapsed = time.time() - t_feat

        pred_mean = float(pred.mean())
        pred_std = float(pred.std())
        logger.info(
            f"预测完成 | stocks={len(symbols)} "
            f"mean={pred_mean:.4f} std={pred_std:.4f} "
            f"耗时={feat_elapsed:.0f}s "
            f"({feat_elapsed/len(symbols):.1f}s/stock)"
        )

        # ── 铁律一：运行时自检（预测方差验证）──
        if pred_std < 1e-7:
            logger.error(
                f"自检失败：预测标准差={pred_std:.2e} < 1e-7，常数预测！中止。"
            )
            raise RuntimeError(f"预测方差为零：pred_std={pred_std:.2e}")

        # 5. 更新缓存
        logger.info("更新 prediction_cache.json...")
        update_cache(month, symbols, pred)

        elapsed = time.time() - t_start
        logger.info(
            f"==== T4 {month} 完成 | "
            f"总耗时={elapsed:.0f}s ({elapsed/60:.1f}min) | "
            f"pred_mean={pred_mean:.4f} pred_std={pred_std:.4f} "
            f"stocks={len(symbols)} ===="
        )

        return {
            "month": month,
            "status": "ok",
            "pred_mean": pred_mean,
            "pred_std": pred_std,
            "n_stocks": len(symbols),
            "elapsed": elapsed,
        }

    except Exception as e:
        elapsed = time.time() - t_start
        logger.error(f"T4 {month} 失败 | {type(e).__name__}: {e} | 耗时={elapsed:.0f}s")
        import traceback
        logger.error(traceback.format_exc())
        return {"month": month, "status": "error", "error": str(e), "elapsed": elapsed}


# ════════════════════════════════════════════════════════════════
#  主入口
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T4 LSTM 单月训练 Worker")
    parser.add_argument(
        "--month", required=True,
        help=f"月份，格式 YYYY-MM，可选: {', '.join(ALL_MONTHS)}",
    )
    args = parser.parse_args()

    if args.month not in ALL_MONTHS:
        print(f"错误：无效月份 '{args.month}'，可选: {ALL_MONTHS}", file=sys.stderr)
        sys.exit(1)

    result = run_one_month(args.month)

    # 最后一行输出 JSON 结果，方便启动器收集
    print(json.dumps(result, ensure_ascii=False))
    sys.exit(0 if result["status"] in ("ok", "skip") else 1)
