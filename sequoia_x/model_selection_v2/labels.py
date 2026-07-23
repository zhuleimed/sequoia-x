"""model_selection_v2 - 多任务标签构建模块。

为每个采样日期的每只股票构建 3 个标签：
  y1: 5日涨跌方向（二分类，0=跌 1=涨）
  y2: 20日超额收益率（相对沪深300，回归）
  y3: 20日日收益率年化波动率（回归，用于风险度量）
"""
from __future__ import annotations
import numpy as np
import sqlite3
from datetime import datetime
from sequoia_x.data.engine import DataEngine
from sequoia_x.core.logger import get_logger
from sequoia_x.model_selection_v2.config import V2Config, get_config
from sequoia_x.model_selection_v2.features import build_stock_features

logger = get_logger(__name__)


def _get_sample_dates(engine: DataEngine, cfg: V2Config) -> list[str]:
    """获取采样日期列表：每月 2 天（月初 5 日 + 月中 15 日）。

    自动跳过数据不足的早期日期（需 window+30=150 个历史交易日）。
    """
    import pandas as pd
    min_history = cfg.window + 30  # 特征构建需要的最少历史天数
    conn = sqlite3.connect(engine.db_path)
    all_dates = pd.read_sql(
        "SELECT DISTINCT date FROM stock_daily WHERE date >= ? AND date <= ? ORDER BY date",
        conn, params=(cfg.sample_start, cfg.sample_end)
    )["date"].tolist()
    conn.close()

    if len(all_dates) <= min_history:
        logger.warning(f"总交易天数({len(all_dates)})不足最小历史需求({min_history})")
        return []

    # 跳过前 min_history 个交易日（数据不足以构建特征）
    valid_start_idx = min_history
    logger.info(
        f"跳过前 {valid_start_idx} 个交易日（需>{min_history}天历史），"
        f"首个有效采样日: {all_dates[valid_start_idx]}"
    )

    # 每月取 2 天：第 5 个交易日 和 第 15 个交易日（或最接近的）
    valid_dates = all_dates[valid_start_idx:]
    monthly = {}
    for d in valid_dates:
        ym = d[:7]
        if ym not in monthly:
            monthly[ym] = []
        monthly[ym].append(d)

    sample_dates = []
    for ym, dates in sorted(monthly.items()):
        if len(dates) >= 15:
            sample_dates.append(dates[4])
            sample_dates.append(dates[14])
        elif len(dates) >= 5:
            sample_dates.append(dates[4])
            sample_dates.append(dates[-1])

    return sample_dates


def _compute_label_t1(
    symbol: str, ref_date: str, engine: DataEngine, cfg: V2Config
) -> float | None:
    """计算 T1 标签：5 日涨跌方向。

    在 stock_daily 中查找 ref_date 之后的第 predict_horizon_t1 个交易日，
    比较收盘价。数据不足返回 None。
    """
    conn = sqlite3.connect(engine.db_path)
    rows = conn.execute(
        "SELECT date, close FROM stock_daily WHERE symbol=? AND date > ? ORDER BY date LIMIT ?",
        (symbol, ref_date, cfg.predict_horizon_t1 + 2)
    ).fetchall()
    conn.close()

    if len(rows) < cfg.predict_horizon_t1:
        return None

    ref_close = rows[0][1]
    target_close = rows[cfg.predict_horizon_t1 - 1][1]
    if ref_close is None or target_close is None or ref_close <= 0:
        return None

    ret = (target_close - ref_close) / ref_close
    return 1.0 if ret > 0 else 0.0


def _compute_label_t2(
    symbol: str, ref_date: str, engine: DataEngine, cfg: V2Config
) -> float | None:
    """计算 T2 标签：20 日超额收益率。

    个股 20 日收益率 - 沪深 300 的 20 日收益率。
    """
    conn = sqlite3.connect(engine.db_path)
    # 个股
    stock_rows = conn.execute(
        "SELECT close FROM stock_daily WHERE symbol=? AND date > ? ORDER BY date LIMIT ?",
        (symbol, ref_date, cfg.predict_horizon_t2 + 2)
    ).fetchall()
    # 指数（优先从 index_daily 表查询 sh.000300，fallback 到 stock_daily 的 000300）
    idx_rows = conn.execute(
        "SELECT close FROM index_daily WHERE symbol='sh.000300' AND date > ? ORDER BY date LIMIT ?",
        (ref_date, cfg.predict_horizon_t2 + 2)
    ).fetchall()
    if len(idx_rows) < cfg.predict_horizon_t2:
        idx_rows = conn.execute(
            "SELECT close FROM stock_daily WHERE symbol='000300' AND date > ? ORDER BY date LIMIT ?",
            (ref_date, cfg.predict_horizon_t2 + 2)
        ).fetchall()
    conn.close()

    if len(stock_rows) < cfg.predict_horizon_t2 or len(idx_rows) < cfg.predict_horizon_t2:
        return None

    stock_ret = (stock_rows[cfg.predict_horizon_t2 - 1][0] / stock_rows[0][0]) - 1.0
    idx_ret = (idx_rows[cfg.predict_horizon_t2 - 1][0] / idx_rows[0][0]) - 1.0
    ret = stock_ret - idx_ret

    # 异常值裁剪
    return float(np.clip(ret, -0.5, 0.5))


def _compute_label_t3(
    symbol: str, ref_date: str, engine: DataEngine, cfg: V2Config
) -> float | None:
    """计算 T3 标签：20 日日收益率年化波动率。"""
    conn = sqlite3.connect(engine.db_path)
    rows = conn.execute(
        "SELECT close FROM stock_daily WHERE symbol=? AND date > ? ORDER BY date LIMIT ?",
        (symbol, ref_date, cfg.predict_horizon_t3 + 2)
    ).fetchall()
    conn.close()

    if len(rows) < cfg.predict_horizon_t3 + 1:
        return None

    closes = np.array([r[0] for r in rows[:cfg.predict_horizon_t3 + 1] if r[0] is not None])
    if len(closes) < 10:
        return None

    daily_rets = np.diff(closes) / closes[:-1]
    vol = float(np.std(daily_rets) * np.sqrt(252))
    return min(vol, 2.0)


def _process_date_stocks(args: tuple) -> tuple[str, str, int]:
    """Worker：处理一个采样日期，结果写入临时 .npz 文件，返回 (ref_date, tmp_path, n_samples)。

    不通过 multiprocessing pipe 传大数据，避免序列化 OOM/断管。
    """
    import tempfile
    ref_date, symbols, cfg = args
    # 极简 engine：绕过 __init__（避免12进程争SQLite DDL锁），仅 db_path 即可
    from sequoia_x.core.config import Settings as _Settings
    engine = DataEngine.__new__(DataEngine)
    engine.db_path = _Settings().db_path

    X_list, y1_list, y2_list, y3_list = [], [], [], []
    for symbol in symbols:
        try:
            X_i, _ = build_stock_features(symbol, ref_date, engine, cfg)
            if X_i is None:
                continue
            y1 = _compute_label_t1(symbol, ref_date, engine, cfg)
            y2 = _compute_label_t2(symbol, ref_date, engine, cfg)
            y3 = _compute_label_t3(symbol, ref_date, engine, cfg)
            if y1 is None or y2 is None or y3 is None:
                continue
            X_list.append(X_i)
            y1_list.append(y1)
            y2_list.append(y2)
            y3_list.append(y3)
        except Exception:
            continue

    # 写入临时文件（避免 pipe 传输大数据）
    tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False, prefix="v2data_")
    np.savez_compressed(tmp, X=np.array(X_list, dtype=np.float32) if X_list else np.array([]),
                        y1=np.array(y1_list, dtype=np.int32), y2=np.array(y2_list, dtype=np.float32),
                        y3=np.array(y3_list, dtype=np.float32))
    tmp.close()
    return ref_date, tmp.name, len(X_list)


def build_training_dataset(
    engine: DataEngine, cfg: V2Config | None = None,
    symbols: list[str] | None = None,
    n_workers: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """构建完整训练数据集（多进程并行，每进程独立 DataEngine）。

    Args:
        engine: DataEngine 实例（仅用于获取采样日期和股票池）。
        cfg: V2Config 配置。
        symbols: 股票列表，默认从 engine.get_base_stock_pool() 获取。
        n_workers: 并行进程数（默认 12，不超过 CPU 限制）。

    Returns:
        (X, y1, y2, y3, date_labels)
    """
    import time
    from multiprocessing import Pool

    if cfg is None:
        cfg = get_config()
    if symbols is None:
        symbols = engine.get_base_stock_pool()

    dates = _get_sample_dates(engine, cfg)
    logger.info(f"采样日期: {len(dates)} 天 ({dates[0]} ~ {dates[-1]})")
    logger.info(f"股票池: {len(symbols)} 只, 并行: {n_workers} workers")

    t0 = time.time()

    # 多进程并行：每个 worker 处理一个日期，结果写入临时 .npz 文件
    tasks = [(d, symbols, cfg) for d in dates]
    all_X, all_y1, all_y2, all_y3, all_dates = [], [], [], [], []
    completed = 0
    tmp_files: list[str] = []

    with Pool(processes=n_workers) as pool:
        for ref_date, tmp_path, n_samples in pool.imap_unordered(
            _process_date_stocks, tasks
        ):
            if n_samples > 0:
                data = np.load(tmp_path)
                all_X.append(data["X"])
                all_y1.append(data["y1"])
                all_y2.append(data["y2"])
                all_y3.append(data["y3"])
                all_dates.extend([ref_date] * n_samples)
                data.close()
            tmp_files.append(tmp_path)
            completed += 1
            if completed % 10 == 0 or completed == 1:
                elapsed = time.time() - t0
                logger.info(
                    f"  已完成 {completed}/{len(dates)} 日期, "
                    f"累计 {sum(len(x) for x in all_X)} 样本, {elapsed:.0f}s"
                )

    # 清理临时文件
    import os as _os
    for f in tmp_files:
        try:
            _os.unlink(f)
        except OSError:
            pass

    elapsed = time.time() - t0
    total_samples = sum(len(x) for x in all_X)
    logger.info(f"数据集构建完成: {total_samples} 样本, {elapsed:.0f}s")

    if total_samples == 0:
        return (np.array([]).reshape(0, cfg.window, 0),
                np.array([]), np.array([]), np.array([]),
                [])

    X = np.concatenate(all_X, axis=0)
    y1 = np.concatenate(all_y1, axis=0).astype(np.int32)
    y2 = np.concatenate(all_y2, axis=0).astype(np.float32)
    y3 = np.concatenate(all_y3, axis=0).astype(np.float32)

    logger.info(
        f"X.shape={X.shape}, "
        f"y1 涨比例={y1.mean():.2%}, "
        f"y2 均值={y2.mean():.4f}, "
        f"y3 均值={y3.mean():.4f}"
    )
    return X, y1, y2, y3, date_list


# ════════════════════════════════════════════════════════════
#  CLI（独立测试用）
# ════════════════════════════════════════════════════════════

def main() -> None:
    """CLI: 测试数据集构建（限制股票数量快速验证）。"""
    from sequoia_x.core.config import Settings
    cfg = get_config()
    engine = DataEngine(Settings())
    pool = engine.get_base_stock_pool()[:50]  # 只取50只测试
    X, y1, y2, y3, dates = build_training_dataset(engine, cfg, symbols=pool)
    print(f"X={X.shape}, y1={y1.shape}, y2={y2.shape}, y3={y3.shape}")
    print(f"样本日期范围: {dates[0]} ~ {dates[-1]}")


if __name__ == "__main__":
    main()
