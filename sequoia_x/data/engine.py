"""数据引擎模块：负责 SQLite 行情数据存储与行情数据查询。"""

import sqlite3
from pathlib import Path

import pandas as pd

from sequoia_x.core.config import Settings
from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


def _migrate_columns(conn: sqlite3.Connection, table: str,
                     columns: list[tuple[str, str]]) -> None:
    """安全地给已有表新增列（列已存在则跳过）。"""
    for col_name, col_def in columns:
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_def}")
        except sqlite3.OperationalError:
            pass


# ── 建表 SQL ──

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS stock_daily (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol   TEXT    NOT NULL,
    date     TEXT    NOT NULL,
    open     REAL,
    high     REAL,
    low      REAL,
    close    REAL,
    volume   REAL,
    turnover REAL,
    amount      REAL,
    pctChg      REAL,
    peTTM       REAL,
    pbMRQ       REAL,
    psTTM       REAL,
    pcfNcfTTM   REAL,
    UNIQUE (symbol, date)
);
"""

_CREATE_INDEX_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS index_daily (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol   TEXT    NOT NULL,
    date     TEXT    NOT NULL,
    open     REAL,
    high     REAL,
    low      REAL,
    close    REAL,
    volume   REAL,
    amount   REAL,
    pctChg   REAL,
    UNIQUE (symbol, date)
);
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_symbol_date ON stock_daily (symbol, date);
"""

_CREATE_SYNC_LOG_SQL = """
CREATE TABLE IF NOT EXISTS sync_log (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    date             TEXT    NOT NULL UNIQUE,
    status           TEXT    NOT NULL,
    stock_count      INTEGER DEFAULT 0,
    delisted_count   INTEGER DEFAULT 0,
    new_listed_count INTEGER DEFAULT 0,
    backfilled_days  INTEGER DEFAULT 0,
    is_trade_day      INTEGER DEFAULT 1,
    api_status        TEXT    DEFAULT '',
    coverage          REAL    DEFAULT 0.0,
    duration_seconds  REAL    DEFAULT 0.0,
    error_msg        TEXT,
    created_at       TEXT    DEFAULT (datetime('now','localtime'))
);
"""


class DataEngine:
    """行情数据引擎，负责 SQLite 存储和行情数据查询。"""

    def __init__(self, settings: Settings) -> None:
        self.db_path: str = settings.db_path
        self.start_date: str = settings.start_date
        self._init_db()

    def _init_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE_TABLE_SQL)
            conn.execute(_CREATE_INDEX_SQL)
            conn.execute(_CREATE_INDEX_TABLE_SQL)
            conn.execute(_CREATE_SYNC_LOG_SQL)
            # 为旧版 sync_log 表补充新字段（兼容已有数据库）
            _migrate_columns(conn, "sync_log", [
                ("is_trade_day", "INTEGER DEFAULT 1"),
                ("api_status", "TEXT DEFAULT ''"),
                ("coverage", "REAL DEFAULT 0.0"),
                ("duration_seconds", "REAL DEFAULT 0.0"),
            ])
            # 为旧版 stock_daily 表补充新字段（peTTM等估值指标 + amount）
            _migrate_columns(conn, "stock_daily", [
                ("pctChg", "REAL"),
                ("peTTM", "REAL"),
                ("pbMRQ", "REAL"),
                ("psTTM", "REAL"),
                ("pcfNcfTTM", "REAL"),
                ("amount", "REAL"),
            ])
            conn.commit()
        logger.info(f"数据库初始化完成：{self.db_path}")

    def _get_last_date(self, symbol: str) -> str | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT MAX(date) FROM stock_daily WHERE symbol = ?",
                (symbol,),
            ).fetchone()
        return row[0] if row and row[0] else None

    def _get_last_date_range(self) -> str | None:
        """获取数据库中最新的日期（全局）。"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT MAX(date) FROM stock_daily").fetchone()
        return row[0] if row and row[0] else None

    def get_ohlcv(self, symbol: str) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql(
                "SELECT * FROM stock_daily WHERE symbol = ? ORDER BY date",
                conn,
                params=(symbol,),
            )
        return df

    @staticmethod
    def _to_baostock_code(symbol: str) -> str:
        """将纯数字代码转为 baostock 格式：6/9开头 -> sh，其余 -> sz。"""
        prefix = "sh" if symbol.startswith(("6", "9")) else "sz"
        return f"{prefix}.{symbol}"

    # ── 基础股票池 ──

    def get_base_stock_pool(self) -> list[str]:
        """获取基础股票池（三步过滤）。

        1. 板块剔除：科创板(688/689)、创业板(300/301)、北交所(4xx/8xx)
        2. 质量剔除：ST/*ST/退市股、上市不满1年的次新股
        3. 价格剔除：最新收盘价 < 2元的低价股

        Returns:
            符合条件的股票代码列表。
        """
        import baostock as bs
        from datetime import date, timedelta

        lg = bs.login()
        if lg.error_code != "0":
            logger.error(f"baostock 登录失败: {lg.error_msg}")
            return []

        try:
            rs = bs.query_stock_basic(code_name="", code="")
            candidates: list[str] = []
            today = date.today()
            one_year_ago = today - timedelta(days=365)

            while rs.next():
                row = rs.get_row_data()
                code_num = row[0].split(".")[1]  # "600000"
                name = row[1]  # 股票名称
                ipo_str = row[2]  # 上市日期
                s_type = row[4]  # "1" = 股票
                status = row[5]  # "1" = 上市

                if status != "1" or s_type != "1":
                    continue
                if code_num.startswith(("688", "689", "300", "301")):
                    continue
                if code_num.startswith(("4", "8")):
                    continue
                if "ST" in name or "退" in name:
                    continue
                try:
                    ipo_date = date.fromisoformat(ipo_str)
                    if ipo_date > one_year_ago:
                        continue
                except (ValueError, TypeError):
                    continue

                candidates.append(code_num)

            logger.info(f"基础池（板块+ST+次新过滤）: {len(candidates)} 只")
        except Exception as e:
            logger.error(f"获取基础股票池失败: {e}")
            return []
        finally:
            bs.logout()

        # 价格过滤：最新收盘价 < 2 剔除
        if candidates:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    ph = ",".join("?" for _ in candidates)
                    rows = conn.execute(
                        f"SELECT symbol, close FROM stock_daily "
                        f"WHERE symbol IN ({ph}) "
                        f"AND (symbol, date) IN "
                        f"(SELECT symbol, MAX(date) FROM stock_daily GROUP BY symbol)",
                        candidates,
                    ).fetchall()
                    price_ok = {r[0]: r[1] for r in rows}
                    before = len(candidates)
                    candidates = [
                        s for s in candidates
                        if s not in price_ok or price_ok[s] >= 2.0
                    ]
                    dropped = before - len(candidates)
                    if dropped:
                        logger.info(f"价格过滤（<2元）: 剔除 {dropped} 只")
            except Exception as e:
                logger.warning(f"价格过滤失败（数据未回填?）: {e}")

        logger.info(f"基础股票池最终: {len(candidates)} 只")
        return candidates

    # ── 股票列表 ──

    def get_all_symbols(self) -> list[str]:
        """通过 baostock 获取全市场 A 股代码列表。"""
        import baostock as bs

        lg = bs.login()
        if lg.error_code != "0":
            logger.error(f"baostock 登录失败: {lg.error_msg}")
            return []

        try:
            rs = bs.query_stock_basic(code_name="", code="")
            symbols: list[str] = []
            while rs.next():
                row = rs.get_row_data()
                code = row[0]  # "sh.600000" or "sz.000001"
                status = row[4]  # "1" = 上市
                stock_type = row[5]  # "1" = 股票
                if status == "1" and stock_type == "1":
                    symbols.append(code.split(".")[1])
            logger.info(f"获取股票列表完成，共 {len(symbols)} 只")
            return symbols
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return []
        finally:
            bs.logout()

    def get_local_symbols(self) -> list[str]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT DISTINCT symbol FROM stock_daily"
            ).fetchall()
        return [row[0] for row in rows]
