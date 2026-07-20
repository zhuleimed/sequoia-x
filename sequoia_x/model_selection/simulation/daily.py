"""LSTM 每日预测 → 买入信号 → 模拟盘更新。

流程：
1. 同步行情数据到 LSTM 模拟盘 DB（SimEngine 需在同一 DB 中访问 stock_daily）
2. 加载最新 LSTM 模型，预测全量股票收益率
3. 按 min_pred_return(0.01) 过滤，取 top 2
4. 通过 submit_buy_signals() 写入买入信号
5. 执行 SimEngine.run_daily() 驱动 LSTM 账户模拟盘

CLI:
    python -m sequoia_x.model_selection.simulation.daily
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from sequoia_x.core.config import Settings
from sequoia_x.core.logger import get_logger
from sequoia_x.data.engine import DataEngine
from sequoia_x.model_selection.config import LSTMConfig, get_config
from sequoia_x.model_selection.predict import predict_all
from sequoia_x.simulation.signals import submit_buy_signals
from sequoia_x.simulation.engine import SimEngine

logger = get_logger(__name__)


# ════════════════════════════════════════════════════════════
#  行情数据同步（SimEngine 需要 stock_daily + index_daily）
# ════════════════════════════════════════════════════════════


def _sync_stock_tables(cfg: LSTMConfig) -> None:
    """将 stock_daily 和 index_daily 从主 DB 同步到 LSTM 模拟盘 DB。

    SQLite ATTACH 方式跨库拷贝，增量式（INSERT OR IGNORE），首次完整拷贝后
    每日仅追加新增数据行。SimEngine._get_today_data() 直接查询
    self.db_path 下的 stock_daily，因此模拟盘 DB 中必须有这张表。
    """
    main_path = Path(cfg.db_path)
    lstm_path = Path(cfg.sim_db_path)
    if not main_path.exists():
        logger.warning(f"主行情数据库不存在: {main_path}，跳过同步")
        return

    lstm_path.parent.mkdir(parents=True, exist_ok=True)
    abs_main = str(main_path.resolve())

    t0 = time.time()
    with sqlite3.connect(str(lstm_path)) as dst:
        dst.execute(f"ATTACH DATABASE '{abs_main}' AS src")

        # stock_daily: 先创建空表（如果不存在），再增量插入
        dst.execute(
            "CREATE TABLE IF NOT EXISTS stock_daily ("
            "  id       INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  symbol   TEXT    NOT NULL,"
            "  date     TEXT    NOT NULL,"
            "  open     REAL,"
            "  high     REAL,"
            "  low      REAL,"
            "  close    REAL,"
            "  volume   REAL,"
            "  turnover REAL,"
            "  amount      REAL,"
            "  pctChg      REAL,"
            "  peTTM       REAL,"
            "  pbMRQ       REAL,"
            "  psTTM       REAL,"
            "  pcfNcfTTM   REAL,"
            "  UNIQUE (symbol, date)"
            ")"
        )
        dst.execute("INSERT OR IGNORE INTO stock_daily SELECT * FROM src.stock_daily")

        # index_daily
        dst.execute(
            "CREATE TABLE IF NOT EXISTS index_daily ("
            "  id       INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  symbol   TEXT    NOT NULL,"
            "  date     TEXT    NOT NULL,"
            "  open     REAL,"
            "  high     REAL,"
            "  low      REAL,"
            "  close    REAL,"
            "  volume   REAL,"
            "  UNIQUE (symbol, date)"
            ")"
        )
        dst.execute("INSERT OR IGNORE INTO index_daily SELECT * FROM src.index_daily")

        # 索引
        for idx in [
            "CREATE INDEX IF NOT EXISTS idx_sd_sym_date ON stock_daily(symbol, date)",
            "CREATE INDEX IF NOT EXISTS idx_id_date ON index_daily(date)",
        ]:
            try:
                dst.execute(idx)
            except sqlite3.OperationalError:
                pass

        dst.execute("DETACH DATABASE src")

    elapsed = time.time() - t0
    logger.debug(f"行情数据同步完成（{elapsed:.1f}s）: {lstm_path}")


# ════════════════════════════════════════════════════════════
#  主流程
# ════════════════════════════════════════════════════════════


def run_lstm_daily(
    cfg: LSTMConfig | None = None,
) -> dict:
    """执行 LSTM 每日模拟盘流程：预测 → 信号 → 模拟。

    Args:
        cfg: 配置对象，默认从 get_config() 获取。

    Returns:
        {"status": "ok"|"skipped"|"error", "symbols": [...], "sim_results": {...}}
    """
    if cfg is None:
        cfg = get_config()

    today_str = time.strftime("%Y-%m-%d")
    result: dict = {"date": today_str, "status": "ok", "symbols": [], "sim_results": {}}

    logger.info(f"═══ LSTM 每日模拟盘 [{today_str}] ═══")
    t0 = time.time()

    # ── 1. 同步行情数据 ──
    _sync_stock_tables(cfg)

    # ── 2. 预测全量股票 ──
    logger.info("LSTM 预测全量股票...")
    settings = Settings()  # 主 DB（行情数据）
    engine = DataEngine(settings)
    predictions = predict_all(engine, cfg=cfg)

    if not predictions:
        logger.warning("LSTM 预测无结果，跳过")
        result["status"] = "skipped"
        return result

    # ── 3. 过滤 + 取 top N ──
    top_symbols = [
        s for s, r in predictions
        if r >= cfg.min_pred_return
    ][:cfg.top_n_buy_per_day]

    if not top_symbols:
        logger.info(
            f"LSTM 预测: 无股票超过 min_pred_return={cfg.min_pred_return:.0%}，跳过"
        )
        result["status"] = "skipped"
        return result

    result["symbols"] = top_symbols
    logger.info(
        f"LSTM 选股: {top_symbols} "
        f"(阈值={cfg.min_pred_return:.0%}, top={cfg.top_n_buy_per_day})"
    )

    # ── 4. 写入买入信号 → LSTM 模拟盘 DB ──
    submit_buy_signals(
        db_path=cfg.sim_db_path,
        symbols=top_symbols,
        strategy_name=cfg.strategy_name,
        top_n=cfg.top_n_buy_per_day,
    )

    # ── 5. 运行 SimEngine（LSTM 账户） ──
    lstm_settings = Settings(db_path=cfg.sim_db_path)
    sim_engine = SimEngine(lstm_settings)
    sim_results = sim_engine.run_daily()
    result["sim_results"] = sim_results

    elapsed = time.time() - t0
    logger.info(
        f"═══ LSTM 每日模拟盘完成 [{today_str}] "
        f"选股={top_symbols} 耗时={elapsed:.0f}s ═══"
    )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="LSTM 每日预测+模拟盘更新")
    parser.add_argument("--top", type=int, help="覆盖 top_n_buy_per_day")
    parser.add_argument("--threshold", type=float, help="覆盖 min_pred_return")
    args = parser.parse_args()

    cfg = get_config()
    if args.top is not None:
        cfg.top_n_buy_per_day = args.top
    if args.threshold is not None:
        cfg.min_pred_return = args.threshold

    run_lstm_daily(cfg)


if __name__ == "__main__":
    main()
