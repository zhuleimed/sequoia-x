#!/usr/bin/env python
"""历史数据回填：Tencent/Sina API 直接拉取 2020 至今全部日线。

原理:
    sync_daily(force=True) 调用 get_daily(code, days=5)，只拉 5 条。
    本脚本直接调 TencentSource.get_daily(code, days=1600)，拉全部历史。

用法:
    nohup python -u scripts/backfill_history.py > logs/backfill_direct_$(date +%Y%m%d_%H%M).log 2>&1 &

数据源:
    OHLCV: Tencent → Sina（双源智能切换）
    估值 + 指数: 复用 sync 模块的增量同步
"""

import sys, os, time, sqlite3, logging
from pathlib import Path
from datetime import date

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / f"backfill_direct_{date.today().strftime('%Y%m%d')}_{os.getpid()}.log"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(str(log_file)),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("backfill")

HISTORY_DAYS = 1600  # ~6 年交易日，覆盖 2020-01-01 至今


def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info(f"历史数据回填 | days={HISTORY_DAYS} | {date.today()}")
    logger.info(f"PID={os.getpid()} | 日志={log_file}")
    logger.info(f"数据源: Tencent(主力) → Sina(后备)")
    logger.info("=" * 60)

    from sequoia_x.core.config import Settings
    from sequoia_x.data.engine import DataEngine
    from sequoia_x.data.tencent_source import TencentSource
    import pandas as pd

    settings = Settings()
    engine = DataEngine(settings)
    tdx = TencentSource()

    # ── 获取股票列表 ──
    symbols = engine.get_local_symbols()
    logger.info(f"本地股票: {len(symbols)} 只")

    # ── 逐只拉取全量历史日线 ──
    total_written = 0
    stock_count = 0
    t_batch = time.time()

    for i, sym in enumerate(symbols):
        # 腾讯代码格式
        if sym.startswith("6") or sym.startswith("5"):
            tc_code = f"sh{sym}"
        else:
            tc_code = f"sz{sym}"

        try:
            df = tdx.get_daily(tc_code, days=HISTORY_DAYS)
        except Exception:
            df = None

        if df is None or df.empty:
            continue

        # 过滤：只保留 2020-01-01 之后的数据
        df = df[df["date"] >= "2020-01-01"].copy()
        if df.empty:
            continue

        # 写入：INSERT OR REPLACE（覆盖已有日期，不产生重复）
        df["symbol"] = sym
        df = df[["symbol", "date", "open", "close", "high", "low", "volume"]]

        try:
            conn = sqlite3.connect(settings.db_path)
            # 用原始 SQL 实现 INSERT OR REPLACE，避免 pandas to_sql 的重复键报错
            rows = [tuple(x) for x in df.itertuples(index=False)]
            conn.executemany(
                "INSERT OR REPLACE INTO stock_daily (symbol, date, open, close, high, low, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()
            conn.close()
            total_written += len(df)
            stock_count += 1
        except Exception as e:
            logger.debug(f"  写入失败 {sym}: {e}")

        # 进度日志（铁律一：每 500 只汇报）
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t_batch
            rate = elapsed / (i + 1)
            remaining = rate * (len(symbols) - i - 1)
            logger.info(
                f"进度: {i+1}/{len(symbols)} "
                f"({100*(i+1)//len(symbols)}%), "
                f"写入={stock_count}只/{total_written}条, "
                f"速率={rate:.2f}s/只, ETA={remaining:.0f}s"
            )

    # ── 指数日线回填 ──
    logger.info("指数日线回填 (Tencent API)...")
    INDEX_CODES = {
        "sh.000001": "sh000001",   # 上证指数
        "sh.000016": "sh000016",   # 上证50
        "sh.000300": "sh000300",   # 沪深300
        "sh.000905": "sh000905",   # 中证500
        "sz.399001": "sz399001",   # 深证成指
        "sz.399106": "sz399106",   # 深证综指
    }
    idx_written = 0
    for db_code, api_code in INDEX_CODES.items():
        try:
            df = tdx.get_daily(api_code, days=HISTORY_DAYS)
            if df is not None and not df.empty:
                df = df[df["date"] >= "2020-01-01"].copy()
                df["symbol"] = db_code
                df = df[["symbol", "date", "open", "close", "high", "low", "volume"]]
                conn = sqlite3.connect(settings.db_path)
                conn.executemany(
                    "INSERT OR REPLACE INTO index_daily (symbol, date, open, close, high, low, volume) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [tuple(x) for x in df.itertuples(index=False)],
                )
                conn.commit()
                conn.close()
                idx_written += len(df)
                logger.info(f"  {db_code}: {df['date'].min()} ~ {df['date'].max()}, {len(df)} 条")
        except Exception as e:
            logger.warning(f"  {db_code} 失败: {e}")
    logger.info(f"指数回填完成: {idx_written} 条")

    # ── 估值数据回填（TDX→baostock，覆盖全部历史缺口）──
    logger.info("估值数据回填 (peTTM/pbMRQ, 全量)...")
    from sequoia_x.data.sync import DataSync
    sync_mgr = DataSync(settings)
    r_val = sync_mgr._fill_valuation_gaps(days=2500)  # 覆盖 ~6 年历史缺口
    logger.info(f"估值回填: filled={r_val.get('total_filled', 0)} 条")

    # ── 运行时自检 ──
    logger.info("=" * 60)
    logger.info("运行时自检：")
    conn = sqlite3.connect(settings.db_path)
    r = conn.execute(
        "SELECT MIN(date), MAX(date), COUNT(DISTINCT date), COUNT(*), "
        "COUNT(DISTINCT symbol) FROM stock_daily"
    ).fetchone()
    logger.info(f"  stock_daily: {r[0]} ~ {r[1]}, {r[2]} 交易日, {r[3]} 条, {r[4]} 只")

    r = conn.execute(
        "SELECT COUNT(DISTINCT symbol), COUNT(*) FROM stock_daily WHERE date < '2024-01-01'"
    ).fetchone()
    logger.info(f"  2024年前: {r[0]} 只股票, {r[1]} 条记录")

    r = conn.execute("SELECT COUNT(*) FROM stock_daily WHERE peTTM > 0").fetchone()
    logger.info(f"  peTTM 有值: {r[0]} 条")

    for row in conn.execute(
        "SELECT symbol, MIN(date), MAX(date), COUNT(*) FROM index_daily GROUP BY symbol"
    ).fetchall():
        logger.info(f"  index {row[0]}: {row[1]} ~ {row[2]}, {row[3]} 条")

    conn.close()

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(f"回填完成 | 总耗时={elapsed:.0f}s ({elapsed/60:.1f}min) | 日志={log_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
