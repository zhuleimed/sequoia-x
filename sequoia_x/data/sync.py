"""数据同步模块：负责 baostock → SQLite 的全量/增量数据同步。

DataSync 专注于数据同步管线，持有 DataEngine 实例以复用查询能力。
"""

from __future__ import annotations

import sqlite3
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from sequoia_x.core.config import Settings
from sequoia_x.data.tencent_source import TencentSource
from sequoia_x.core.logger import get_logger
from sequoia_x.data.engine import DataEngine

logger = get_logger(__name__)


class DataSync:
    """数据同步模块，负责 baostock → SQLite 的全量/增量数据同步。

    持有 DataEngine 引用以复用查询能力（get_all_symbols, get_local_symbols 等），
    管理 baostock 连接生命周期（单会话复用）。
    """

    @staticmethod
    def _bs_get_data(rs) -> pd.DataFrame:
        """安全获取 baostock 查询结果（兼容 pandas 3.0 移除 df.append）。

        因 baostock 的 ResultData.get_data() 内部使用 df.append()，
        在 pandas >= 2.0 中已移除，故此方法手动逐行拼接。
        """
        rows: list[list[str]] = []
        while rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows, columns=rs.fields)

    def __init__(self, settings: Settings) -> None:
        """初始化 DataSync。

        Args:
            settings: 系统配置对象（需含 db_path, start_date）。
        """
        self.settings: Settings = settings
        self.db_path: str = settings.db_path
        self.start_date: str = settings.start_date
        self.engine = DataEngine(settings)
        self._bs_logged_in: bool = False
        self._init_db()

    def _init_db(self) -> None:
        """初始化数据库表结构。

        调用 engine._init_db() 确保 stock_daily 和 sync_log 表存在，
        同时补充 sync_log 增强字段（兼容已有数据库的 ALTER TABLE 迁移）。
        """
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.engine._init_db()
        # 为旧版 sync_log 表补充新字段（DataEngine._init_db 已处理，此处做二次保障）
        try:
            with sqlite3.connect(self.db_path) as conn:
                for col_sql in [
                    "ALTER TABLE sync_log ADD COLUMN is_trade_day INTEGER DEFAULT 1",
                    "ALTER TABLE sync_log ADD COLUMN api_status TEXT DEFAULT ''",
                    "ALTER TABLE sync_log ADD COLUMN coverage REAL DEFAULT 0.0",
                    "ALTER TABLE sync_log ADD COLUMN duration_seconds REAL DEFAULT 0.0",
                ]:
                    try:
                        conn.execute(col_sql)
                    except sqlite3.OperationalError:
                        pass
                conn.commit()
        except Exception:
            pass

    # ── baostock 连接管理 ──

    def _bs_login(self) -> bool:
        """登录 baostock，设置 _bs_logged_in 标志。

        Returns:
            True 表示登录成功。
        """
        if self._bs_logged_in:
            return True
        import baostock as bs
        lg = bs.login()
        if lg.error_code != "0":
            logger.error(f"baostock 登录失败: {lg.error_msg}")
            return False
        self._bs_logged_in = True
        logger.info("baostock 登录成功")
        return True

    def _bs_logout(self) -> None:
        """登出 baostock（若已登录）。"""
        if not self._bs_logged_in:
            return
        import baostock as bs
        bs.logout()
        self._bs_logged_in = False
        logger.info("baostock 已登出")

    # ── 批量写入（性能优化） ──

    def _open_db(self) -> None:
        """打开持久化 SQLite 连接（批量写入优化）。"""
        self._db_conn = sqlite3.connect(self.db_path)
        self._db_conn.execute("PRAGMA journal_mode=WAL")
        self._db_conn.execute("PRAGMA synchronous=NORMAL")
        self._batch_buffer: list[dict] = []

    def _close_db(self) -> None:
        """关闭持久化连接。"""
        if hasattr(self, '_batch_buffer') and self._batch_buffer:
            self._flush_batch()
        if hasattr(self, '_db_conn') and self._db_conn:
            self._db_conn.close()
            self._db_conn = None

    def _flush_batch(self) -> int:
        """将缓冲区中的数据批量写入 stock_daily。

        使用持久化连接 + 事务包裹，避免逐只写入时的连接/提交开销。
        """
        if not self._batch_buffer:
            return 0

        all_stock_cols: list[str] = [
            "symbol", "date", "open", "high", "low", "close", "volume", "turnover",
            "amount", "pctChg", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM",
        ]
        # 取第一条记录的列作为基准
        cols_present: list[str] = [c for c in all_stock_cols if c in self._batch_buffer[0]]
        placeholders: str = ", ".join(f":{c}" for c in cols_present)
        cols_str: str = ", ".join(cols_present)

        count: int = len(self._batch_buffer)
        self._db_conn.executemany(
            f"INSERT OR REPLACE INTO stock_daily ({cols_str}) VALUES ({placeholders})",
            self._batch_buffer,
        )
        self._db_conn.commit()
        self._batch_buffer.clear()

        logger.debug(f"_flush_batch: 批量写入 {count} 条")
        return count

    def _buffer_row(self, record: dict) -> None:
        """将一行数据加入批量缓冲区，达到阈值自动刷盘。"""
        self._batch_buffer.append(record)
        if len(self._batch_buffer) >= 500:
            self._flush_batch()

    def _write_to_db(self, df: pd.DataFrame) -> int:
        """清洗 DataFrame 并加入批量缓冲区（不再立即写盘）。

        对于小 DataFrame（1行），直接放入缓冲区由 _flush_batch 批量写入。
        对于大 DataFrame（回填场景），直接写入以控制内存。

        Args:
            df: 包含行情字段的 DataFrame。

        Returns:
            处理的行数。
        """
        if df.empty:
            return 0

        numeric_cols: list[str] = [
            "open", "high", "low", "close", "volume", "turnover", "amount",
            "pctChg", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["close"])

        # 停牌处理：成交量为空填0，其他字段向前填充
        if "volume" in df.columns:
            df["volume"] = df["volume"].fillna(0.0)

        # 确保按日期排序，使前向填充顺序正确
        if "date" in df.columns:
            df = df.sort_values("date")

        # 其他数值字段：用上一交易日数据向前填充，仍为空则填0
        ffill_cols = [c for c in numeric_cols if c != "volume" and c in df.columns]
        if ffill_cols:
            df[ffill_cols] = df[ffill_cols].ffill().fillna(0.0)

        if df.empty:
            return 0

        count: int = len(df)

        if count <= 5:
            # 小批量：放入缓冲区
            stock_cols: list[str] = [
                "symbol", "date", "open", "high", "low", "close", "volume", "turnover",
                "amount", "pctChg", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM",
            ]
            cols_present: list[str] = [c for c in stock_cols if c in df.columns]
            for rec in df[cols_present].to_dict("records"):
                self._buffer_row(rec)
        else:
            # 大批量：直接写盘（回填场景，数据量大）
            all_stock_cols: list[str] = [
                "symbol", "date", "open", "high", "low", "close", "volume", "turnover",
                "amount", "pctChg", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM",
            ]
            cols_present: list[str] = [c for c in all_stock_cols if c in df.columns]
            records = df[cols_present].to_dict("records")
            placeholders: str = ", ".join(f":{c}" for c in cols_present)
            cols_str: str = ", ".join(cols_present)
            self._db_conn.executemany(
                f"INSERT OR REPLACE INTO stock_daily ({cols_str}) VALUES ({placeholders})",
                records,
            )
            self._db_conn.commit()

        logger.info(f"_write_to_db: 写入 {count} 条记录（{df['date'].nunique()} 个日期）")
        return count

    def _log_sync(self, result: dict, duration: float) -> None:
        """将同步结果写入增强版 sync_log 表。

        Args:
            result: run_full() 返回的结果字典。
            duration: 同步耗时（秒）。
        """
        today_str = date.today().strftime("%Y-%m-%d")
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO sync_log "
                    "(date, status, stock_count, delisted_count, new_listed_count, "
                    "backfilled_days, is_trade_day, api_status, coverage, duration_seconds, error_msg) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        today_str,
                        result.get("status", "unknown"),
                        result.get("stock_count", 0),
                        result.get("delisted", 0),
                        result.get("new_listed", 0),
                        result.get("backfilled", 0),
                        0 if not result.get("is_trade_day", True) else 1,
                        result.get("api_status", ""),
                        result.get("coverage", 0.0),
                        duration,
                        result.get("error", ""),
                    ),
                )
                conn.commit()
            logger.info(f"sync_log 写入完成: status={result.get('status')}")
        except Exception as e:
            logger.warning(f"sync_log 写入失败: {e}")

    # ── 交易日历 ──

    def is_trade_day(self, check_date: date | None = None) -> bool:
        """判断指定日期是否为 A 股交易日。

        判断策略（三层）：
        1. 周末过滤 — 周六/周日一定不开盘（最快路径）
        2. baostock query_trade_dates — 在线时精确判断（含节假日）
        3. chinese_calendar 离线库 — baostock 不可用时回退
        4. fail-open — 以上均失败时假定为交易日，避免漏数据

        Args:
            check_date: 待检查日期，默认当天。

        Returns:
            True 表示是交易日或无法确定（fail-open）。
        """
        if check_date is None:
            check_date = date.today()
        date_str: str = check_date.strftime("%Y-%m-%d")

        # 第1层：周末一定不开盘
        if check_date.weekday() >= 5:  # 5=周六, 6=周日
            logger.info(f"is_trade_day: {date_str} 是周末，非交易日")
            return False

        # 第2层：baostock 在线精确判断
        if self._bs_login():
            import baostock as bs
            try:
                rs = bs.query_trade_dates(start_date=date_str, end_date=date_str)
                if rs.error_code == "0":
                    data = self._bs_get_data(rs)
                    if data is not None and not data.empty:
                        is_trading = data["is_trading_day"].iloc[0]
                        return is_trading == "1"
            except Exception as e:
                logger.debug(f"is_trade_day: baostock 查询异常 {e}，尝试离线判断")

        # 第3层：chinese_calendar 离线节假日判断
        try:
            from chinese_calendar import is_workday
            result = is_workday(check_date)
            if result:
                logger.info(f"is_trade_day: {date_str} chinese_calendar 判断为工作日")
            else:
                logger.info(f"is_trade_day: {date_str} chinese_calendar 判断为节假日")
            return result
        except ImportError:
            logger.debug("is_trade_day: chinese_calendar 未安装")
        except Exception as e:
            logger.debug(f"is_trade_day: chinese_calendar 异常 {e}")

        # 第4层：fail-open，假定为交易日
        logger.warning(f"is_trade_day: {date_str} 所有数据源均不可用，假定为交易日")
        return True

    # ── 股票列表同步 ──

    def get_active_stocks(self) -> dict:
        """通过 baostock query_stock_basic 获取全量 A 股并对比本地变化。

        仅保留沪深 A 股（sh.6 / sz.0 / sz.3 开头），与本地 stock_daily
        已有 symbol 做差集计算，输出新增和退市列表。

        Returns:
            dict:
                - symbols (list[str]): 远程全量 symbol 列表（纯数字代码）
                - new_listed (list[str]): 远程有但本地无的新上市股票
                - delisted (list[str]): 本地有但远程无的退市股票
                - count (int): 远程股票总数
        """
        if not self._bs_login():
            logger.error("get_active_stocks: baostock 登录失败")
            return {"symbols": [], "new_listed": [], "delisted": [], "count": 0}

        import baostock as bs
        try:
            rs = bs.query_stock_basic(code_name="", code="")
            if rs.error_code != "0":
                logger.error(
                    f"get_active_stocks: query_stock_basic 失败 "
                    f"code={rs.error_code}"
                )
                return {"symbols": [], "new_listed": [], "delisted": [], "count": 0}

            raw = self._bs_get_data(rs)
            if raw.empty:
                logger.warning("get_active_stocks: 查询结果为空")
                return {"symbols": [], "new_listed": [], "delisted": [], "count": 0}

            # 仅保留沪深 A 股：sh.6 / sz.0 / sz.3 开头，且 type="1"(股票) status="1"(上市)
            mask = (
                raw["code"].str.startswith(("sh.6", "sz.0", "sz.3"))
                & (raw["type"] == "1")
                & (raw["status"] == "1")
            )
            filtered = raw.loc[mask, "code"]

            remote_symbols: list[str] = filtered.str.split(".").str[1].tolist()
            local_symbols: list[str] = self.engine.get_local_symbols()

            remote_set: set[str] = set(remote_symbols)
            local_set: set[str] = set(local_symbols)

            new_listed: list[str] = sorted(remote_set - local_set)
            delisted: list[str] = sorted(local_set - remote_set)

            logger.info(
                f"get_active_stocks: 远程 {len(remote_symbols)} 只，"
                f"本地 {len(local_symbols)} 只，"
                f"新增 {len(new_listed)} 只，退市 {len(delisted)} 只"
            )
            return {
                "symbols": remote_symbols,
                "new_listed": new_listed,
                "delisted": delisted,
                "count": len(remote_symbols),
            }
        except Exception as e:
            logger.error(f"get_active_stocks 异常: {e}")
            return {"symbols": [], "new_listed": [], "delisted": [], "count": 0}

    def sync_stock_list(self) -> dict:
        """将活跃股票列表持久化到 SQLite stock_list 表。

        逻辑分支：
        - 若本地 stock_list 为空 → 全量 INSERT OR IGNORE
        - 否则 → 新股 INSERT，退市股 UPDATE delisted_date

        Returns:
            dict:
                - status (str): "ok" 或 "error"
                - new_listed (list[str]): 新上市股票
                - delisted (list[str]): 退市股票
                - total (int): 总股票数
        """
        active = self.get_active_stocks()
        if not active.get("symbols"):
            logger.warning("sync_stock_list: get_active_stocks 返回空")
            return {"status": "error", "new_listed": [], "delisted": [], "total": 0}

        try:
            with sqlite3.connect(self.db_path) as conn:
                # 确保 stock_list 表存在
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS stock_list (
                        symbol       TEXT PRIMARY KEY,
                        listed_date  TEXT,
                        delisted_date TEXT,
                        updated_at   TEXT DEFAULT (datetime('now','localtime'))
                    )"""
                )

                count_row = conn.execute(
                    "SELECT COUNT(*) FROM stock_list"
                ).fetchone()
                is_empty = count_row[0] == 0 if count_row else True

                if is_empty:
                    # 全量写入
                    for sym in active["symbols"]:
                        conn.execute(
                            "INSERT OR IGNORE INTO stock_list (symbol) VALUES (?)",
                            (sym,),
                        )
                    logger.info(
                        f"sync_stock_list: 全量写入 {active['count']} 只股票"
                    )
                else:
                    # 增量：新股 INSERT
                    for sym in active["new_listed"]:
                        conn.execute(
                            "INSERT OR IGNORE INTO stock_list (symbol) VALUES (?)",
                            (sym,),
                        )
                    # 退市股更新 delisted_date
                    today_str: str = date.today().strftime("%Y-%m-%d")
                    for sym in active["delisted"]:
                        conn.execute(
                            "UPDATE stock_list SET delisted_date = ? WHERE symbol = ?",
                            (today_str, sym),
                        )
                    if active["new_listed"]:
                        logger.info(
                            f"sync_stock_list: 新增 {len(active['new_listed'])} 只"
                        )
                    if active["delisted"]:
                        logger.info(
                            f"sync_stock_list: 退市 {len(active['delisted'])} 只"
                        )

                conn.commit()

            return {
                "status": "ok",
                "new_listed": active["new_listed"],
                "delisted": active["delisted"],
                "total": active["count"],
            }
        except Exception as e:
            logger.error(f"sync_stock_list 异常: {e}")
            return {"status": "error", "new_listed": [], "delisted": [], "total": 0}

    # ── 增量日线同步 ──

    def _get_local_last_dates(self) -> dict[str, str]:
        """查询本地 stock_daily 表中每个 symbol 的最大日期。

        Returns:
            dict: {symbol: "YYYY-MM-DD", ...}，异常时返回空字典。
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT symbol, MAX(date) FROM stock_daily GROUP BY symbol"
                ).fetchall()
            result: dict[str, str] = {}
            for row in rows:
                if row[0] is not None and row[1] is not None:
                    result[row[0]] = row[1]
            return result
        except Exception as e:
            logger.warning(f"_get_local_last_dates 异常: {e}")
            return {}

    def sync_daily(self, force: bool = False) -> dict:
        """增量日线同步：拉取每个 symbol 的最新日线数据写入 stock_daily。

        非 force 模式：
        1. 检查今日是否为交易日（非交易日跳过）
        2. 时间门控：当前时间 < 配置的 sync_after_hour:30 时跳过
        3. 通过 _get_local_last_dates 获取各 symbol 最新日期
        4. 仅拉取 last_date+1 至今天的数据

        force 模式：从 start_date 全量拉取所有 symbol。

        Bug 修复（对比原始 sync_today_bulk）：
        - last_local_date 作用域修复：增量模式的 start_date 通过
          _get_local_last_dates 统一获取，避免局部变量未定义问题。
        - 不再重复 login/logout：由 _bs_login / _bs_logout 统一管理
          会话生命周期。

        Args:
            force: 是否强制全量拉取（忽略交易日判断和时间门控）。

        Returns:
            dict:
                - status (str): "ok" | "skipped" | "error"
                - stock_count (int): 成功写入的股票数量
                - is_trade_day (bool): 是否为交易日
                - error (str): 错误信息（仅 status="error" 时有内容）
        """
        today_str: str = date.today().strftime("%Y-%m-%d")

        # ── 非 force 模式下的前置检查 ──
        if not force:
            # 交易日检查
            if not self.is_trade_day():
                logger.info("sync_daily: 非交易日，跳过同步")
                return {
                    "status": "skipped",
                    "stock_count": 0,
                    "is_trade_day": False,
                    "error": "",
                }

            # 时间门控
            now: datetime = datetime.now()
            sync_hour: int = int(getattr(self.settings, "sync_after_hour", 17))
            if now.hour < sync_hour or (now.hour == sync_hour and now.minute < 30):
                logger.info(
                    f"sync_daily: 时间门控跳过（{now.strftime('%H:%M')} < "
                    f"{sync_hour:02d}:30）"
                )
                return {
                    "status": "skipped",
                    "stock_count": 0,
                    "is_trade_day": True,
                    "error": "",
                }

            # 增量模式：从本地已有 symbol 出发
            symbols: list[str] = self.engine.get_local_symbols()
            if not symbols:
                symbols = self.engine.get_all_symbols()

            # 补充 stock_list 中有但尚未拉取过数据的股票（新股）
            try:
                with sqlite3.connect(self.db_path) as conn:
                    listed = conn.execute(
                        "SELECT symbol FROM stock_list WHERE delisted_date IS NULL"
                    ).fetchall()
                listed_syms = set(s[0] for s in listed)
                existing = set(symbols)
                new_syms = sorted(listed_syms - existing)
                if new_syms:
                    symbols.extend(new_syms)
                    logger.info(f"sync_daily: 补充 {len(new_syms)} 只新股: {' '.join(new_syms[:10])}{'...' if len(new_syms) > 10 else ''}")
            except Exception as e:
                logger.debug(f"sync_daily: 查询 stock_list 补充新股失败: {e}")

            # 获取每个 symbol 的最新日期（Bug 修复 #1：作用域统一在块开头）
            last_dates: dict[str, str] = self._get_local_last_dates()

            # 计算每个 symbol 的拉取起始日期
            symbol_starts: dict[str, str] = {}
            for sym in symbols:
                last_date: str | None = last_dates.get(sym)
                if last_date is not None:
                    dt: datetime = datetime.strptime(last_date, "%Y-%m-%d")
                    start: str = (dt + timedelta(days=1)).strftime("%Y-%m-%d")
                else:
                    start = self.start_date
                # 仅当起始日期不晚于今天时才拉取
                if start <= today_str:
                    symbol_starts[sym] = start

            logger.info(
                f"sync_daily 增量模式: {len(symbol_starts)}/{len(symbols)} 只待拉取"
            )
        else:
            # force 模式：跳过交易日判断和时间门控，保持增量逻辑
            # 起始日期使用 last_date（而非 last_date+1），通过 INSERT OR REPLACE 去重
            # 这样可以覆盖已有记录的新增字段（如 peTTM 等估值指标）
            symbols = self.engine.get_local_symbols()
            if not symbols:
                symbols = self.engine.get_all_symbols()

            # 补充 stock_list 中有但尚未拉取过数据的股票（新股）
            try:
                with sqlite3.connect(self.db_path) as conn:
                    listed = conn.execute(
                        "SELECT symbol FROM stock_list WHERE delisted_date IS NULL"
                    ).fetchall()
                listed_syms = set(s[0] for s in listed)
                existing = set(symbols)
                new_syms = sorted(listed_syms - existing)
                if new_syms:
                    symbols.extend(new_syms)
                    logger.info(f"sync_daily force: 补充 {len(new_syms)} 只新股: {' '.join(new_syms[:10])}{'...' if len(new_syms) > 10 else ''}")
            except Exception as e:
                logger.debug(f"sync_daily force: 查询 stock_list 补充新股失败: {e}")
            last_dates = self._get_local_last_dates()
            symbol_starts = {}
            for sym in symbols:
                last_date = last_dates.get(sym)
                if last_date is not None:
                    start = last_date  # 从最后日期开始（覆盖模式）
                else:
                    start = self.start_date
                if start <= today_str:
                    symbol_starts[sym] = start
            logger.info(
                f"sync_daily force 模式（覆盖）: {len(symbol_starts)}/{len(symbols)} 只待拉取"
            )

        if not symbol_starts:
            logger.info("sync_daily: 无待拉取股票，跳过")
            return {
                "status": "skipped",
                "stock_count": 0,
                "is_trade_day": True,
                "error": "",
            }

        # Bug 修复 #2：由 _bs_login 统一管理会话，不再重复 login
        if not self._bs_login():
            return {
                "status": "error",
                "stock_count": 0,
                "is_trade_day": True,
                "error": "baostock 登录失败",
            }

        # 批量写入优化：打开持久化 SQLite 连接
        self._open_db()

        import baostock as bs

        stock_count: int = 0
        symbols_list: list[str] = list(symbol_starts.keys())
        consecutive_errors: int = 0
        max_consecutive_errors: int = 10

        # 主动重连与速度监控参数
        requests_since_login: int = 0
        max_requests_per_conn: int = 1400
        batch_start_time: float = time.time()
        batch_start_count: int = 0
        log_batch_size: int = 100
        # baostock 是否可用的标记（首次失败后跳过，直接走 Tencent）
        baostock_available: bool = True
        skip_baostock_count: int = 0
        skip_baostock_retry_threshold: int = 500  # 跳过500次后再试一次 baostock（baostock不稳定时避免频繁重试拖慢Tencent）

        def _log_batch_progress(current_idx: int) -> None:
            nonlocal batch_start_time, batch_start_count
            now = time.time()
            elapsed = now - batch_start_time
            batch_count = current_idx - batch_start_count
            rate = f"{elapsed / batch_count:.2f}s" if batch_count > 0 else "N/A"
            conn_age = requests_since_login
            logger.info(
                f"sync_daily 进度: {current_idx}/{len(symbols_list)} "
                f"(有效写入 {stock_count}) "
                f"[本批次{batch_count}只 {elapsed:.0f}s {rate}/只 "
                f"连续错误{consecutive_errors} "
                f"连接已用{conn_age}次请求]"
            )
            batch_start_time = now
            batch_start_count = current_idx

        def _proactive_reconnect() -> bool:
            nonlocal requests_since_login, consecutive_errors
            logger.info(
                f"sync_daily: 主动重连"
                f"(已执行 {requests_since_login} 次请求)"
            )
            self._bs_logout()
            time.sleep(0.3)
            ok = self._bs_login()
            if ok:
                requests_since_login = 0
                consecutive_errors = 0
                logger.info("sync_daily: 主动重连成功")
            else:
                logger.error("sync_daily: 主动重连失败")
            return ok

        try:
            for i, sym in enumerate(symbols_list):
                bs_code = self.engine._to_baostock_code(sym)
                start = symbol_starts[sym]

                # ① 主动重连: 每连接1400次请求后重新登录
                if requests_since_login >= max_requests_per_conn and i > 0:
                    if not _proactive_reconnect():
                        logger.error("sync_daily: 主动重连失败，终止同步")
                        break

                # ② 速度监控: 最近50只平均>4s/只，提前重连
                if (i - batch_start_count) >= 50 and i > 0:
                    batch_elapsed = time.time() - batch_start_time
                    avg_per_stock = batch_elapsed / (i - batch_start_count)
                    if avg_per_stock > 4.0 and requests_since_login > 500:
                        logger.warning(
                            f"sync_daily: 检测到连接减速 "
                            f"(最近{i - batch_start_count}只平均{avg_per_stock:.2f}s/只)"
                        )
                        _proactive_reconnect()

                try:
                    if consecutive_errors >= max_consecutive_errors:
                        logger.warning(
                            f"sync_daily: 连续{consecutive_errors}次错误，尝试重连baostock..."
                        )
                        self._bs_logout()
                        reconnect_ok = False
                        for retry in range(5):
                            wait = 2 ** retry
                            logger.info(f"sync_daily: 等待{wait}s后第{retry+1}次重连...")
                            time.sleep(wait)
                            if self._bs_login():
                                consecutive_errors = 0
                                requests_since_login = 0
                                reconnect_ok = True
                                logger.info(f"sync_daily: baostock重连成功(第{retry+1}次)")
                                break
                            logger.warning(f"sync_daily: 第{retry+1}次重连失败")
                        if not reconnect_ok:
                            logger.error("sync_daily: baostock 5次重连均失败，终止")
                            break

                    data = None
                    # ===== 尝试 baostock（如上次失败则跳过）=====
                    if baostock_available:
                        try:
                            bs_rs = bs.query_history_k_data_plus(
                                bs_code,
                                "date,open,high,low,close,volume,amount,turn,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM",
                                start_date=start,
                                end_date=today_str,
                                frequency="d",
                                adjustflag="2",
                            )
                            requests_since_login += 1
                            if bs_rs.error_code == "0":
                                bs_data = self._bs_get_data(bs_rs)
                                if bs_data is not None and not bs_data.empty:
                                    bs_data["symbol"] = sym
                                    bs_data.rename(columns={"turn": "turnover"}, inplace=True)
                                    data = bs_data
                                    if not baostock_available:
                                        baostock_available = True
                                        logger.info("sync_daily: baostock 恢复可用")
                            else:
                                if baostock_available:
                                    baostock_available = False
                                    logger.info(f"sync_daily: baostock 返回错误 {bs_rs.error_code}，切换 Tencent")
                        except Exception as bs_e:
                            logger.debug(f"sync_daily {sym}: baostock exception {bs_e}")
                            if baostock_available:
                                baostock_available = False
                                skip_baostock_count = 0
                                logger.info("sync_daily: baostock 异常，切换到 Tencent 直连模式")

                    # ===== baostock 跳过或失败，尝试 Tencent =====
                    if data is None:
                        skip_baostock_count += 1
                        # 每跳过 50 次尝试重连一次 baostock
                        if (not baostock_available and 
                            skip_baostock_count >= skip_baostock_retry_threshold):
                            baostock_available = True
                            skip_baostock_count = 0
                            logger.info("sync_daily: 达到阈值，尝试恢复 baostock")
                        if baostock_available:
                            continue  # 让下一轮循环再试 baostock
                        logger.debug(f"sync_daily {sym}: baostock 不可用，使用 Tencent")
                        try:
                            tc_code = TencentSource.to_baostock_code(bs_code)
                            tc = TencentSource()
                            tc_df = tc.get_daily(tc_code, 5)
                            if tc_df is not None and not tc_df.empty:
                                # 取最新一行（对应今天）
                                latest = tc_df.iloc[-1:].copy()
                                if not latest.empty:
                                    latest["symbol"] = sym
                                    # 从实时查询补充 amount
                                    rt = tc.get_realtime(tc_code)
                                    latest["amount"] = rt["amount"] if rt else 0.0
                                    # 计算 pctChg（始终设置，默认 0.0）
                                    latest["pctChg"] = 0.0
                                    if len(tc_df) >= 2:
                                        prev_close = tc_df.iloc[-2]["close"]
                                        if prev_close and prev_close > 0:
                                            latest["pctChg"] = round(
                                                (latest.iloc[0]["close"] - prev_close) / prev_close * 100, 2
                                            )
                                    latest["turnover"] = 0.0
                                    latest["peTTM"] = None
                                    latest["pbMRQ"] = None
                                    latest["psTTM"] = None
                                    latest["pcfNcfTTM"] = None
                                    data = latest
                                    consecutive_errors = 0
                                    logger.debug(f"sync_daily {sym}: Tencent 替代成功")
                        except Exception as tc_e:
                            logger.debug(f"sync_daily {sym}: Tencent also failed {tc_e}")

                    if data is None:
                        consecutive_errors += 1
                        if consecutive_errors <= 3:
                            logger.debug(f"sync_daily {sym}: 所有数据源均失败 连续第{consecutive_errors}次")
                        continue

                    written = self._write_to_db(data)
                    if written > 0:
                        stock_count += 1
                        consecutive_errors = 0
                    else:
                        consecutive_errors += 1
                        if consecutive_errors <= 3:
                            logger.debug(f"sync_daily {sym}: 写入0条 连续第{consecutive_errors}次")

                except Exception as e:
                    consecutive_errors += 1
                    if consecutive_errors <= 3:
                        logger.warning(
                            f"sync_daily {sym}: {e} 连续第{consecutive_errors}次"
                        )
                    continue

                time.sleep(0.15)

                if (i + 1) % log_batch_size == 0:
                    _log_batch_progress(i + 1)

            if (i + 1) % log_batch_size != 0 and (i + 1) > 0:
                _log_batch_progress(i + 1)

            logger.info(
                f"sync_daily 完成: 共{len(symbols_list)}只, "
                f"有效写入{stock_count}只"
            )
            self._close_db()
            return {
                "status": "ok",
                "stock_count": stock_count,
                "is_trade_day": True,
                "error": "",
            }
        except Exception as e:
            logger.error(f"sync_daily 异常: {e}")
            self._close_db()
            return {
                "status": "error",
                "stock_count": stock_count,
                "is_trade_day": True,
                "error": str(e),
            }

    # ── 缺失诊断与修复 ──

    def check_missing(self, days: int = 5) -> dict:
        """检查最近 N 个交易日的数据完整性。

        诊断逻辑：
        1. 通过 _get_local_last_dates 获取各 symbol 最新日期
        2. 找到全局 latest_date（所有 symbol 中最新的日期）
        3. 从 latest_date 往前推 days*2 个日历日作为检查区间（确保覆盖 N 个交易日）
        4. 用 baostock query_trade_dates 获取区间内的理论交易日列表
        5. 对每个 symbol，查询其在交易日列表中的实际覆盖，标记缺失

        Args:
            days: 检查最近多少个交易日（默认 5）。

        Returns:
            dict:
                - status (str): "ok" 或 "error"
                - latest_date (str): 全体 symbol 中最新的数据日期 "YYYY-MM-DD"
                - trade_days_expected (int): 理论交易日数
                - total_missing (int): 缺失总数（各 symbol 缺失交易日之和）
                - missing_by_symbol (dict[str, list[str]]): {symbol: [缺失日期列表], ...}
        """
        last_dates: dict[str, str] = self._get_local_last_dates()
        if not last_dates:
            logger.warning("check_missing: 本地无任何行情数据")
            return {
                "status": "ok",
                "latest_date": "",
                "trade_days_expected": 0,
                "total_missing": 0,
                "missing_by_symbol": {},
            }

        # 找到全局最新的数据日期
        all_dates: list[str] = list(last_dates.values())
        latest_date: str = max(all_dates)
        dt_latest: datetime = datetime.strptime(latest_date, "%Y-%m-%d")
        # 用 days*2 日历日确保覆盖 N 个交易日（周末/节假日多退少补）
        dt_start: datetime = dt_latest - timedelta(days=days * 2)
        start_str: str = dt_start.strftime("%Y-%m-%d")

        # 获取检查区间内的理论交易日列表
        if not self._bs_login():
            logger.error("check_missing: baostock 登录失败")
            return {
                "status": "error",
                "latest_date": latest_date,
                "trade_days_expected": 0,
                "total_missing": 0,
                "missing_by_symbol": {},
            }

        import baostock as bs
        trade_days: list[str] = []
        try:
            rs = bs.query_trade_dates(start_date=start_str, end_date=latest_date)
            if rs.error_code == "0":
                data = self._bs_get_data(rs)
                if not data.empty:
                    trade_days = data.loc[
                        data["is_trading_day"] == "1", "calendar_date"
                    ].tolist()
        except Exception as e:
            logger.warning(f"check_missing: 交易日查询异常: {e}，假定全为交易日")
        # 回退：query_trade_dates 异常或返回错误码时，生成区间内所有日历日
        if not trade_days:
            cursor: datetime = dt_start
            while cursor <= dt_latest:
                trade_days.append(cursor.strftime("%Y-%m-%d"))
                cursor += timedelta(days=1)

        if not trade_days:
            logger.warning("check_missing: 检查区间内无交易日")
            return {
                "status": "ok",
                "latest_date": latest_date,
                "trade_days_expected": 0,
                "total_missing": 0,
                "missing_by_symbol": {},
            }

        trade_days_set: set[str] = set(trade_days)

        # 批量查询区间内所有 symbol 的实际覆盖日期
        actual_dates_by_symbol: dict[str, set[str]] = {}
        try:
            with sqlite3.connect(self.db_path) as conn:
                placeholders: str = ",".join(["?"] * len(trade_days))
                rows = conn.execute(
                    f"SELECT symbol, date FROM stock_daily "
                    f"WHERE date IN ({placeholders})",
                    trade_days,
                ).fetchall()
            for sym, d in rows:
                if sym not in actual_dates_by_symbol:
                    actual_dates_by_symbol[sym] = set()
                actual_dates_by_symbol[sym].add(d)
        except Exception as e:
            logger.warning(f"check_missing: 批量查询 stock_daily 异常: {e}")
            return {
                "status": "error",
                "latest_date": latest_date,
                "trade_days_expected": len(trade_days),
                "total_missing": 0,
                "missing_by_symbol": {},
            }

        # 对比每个 symbol 的缺失
        missing_by_symbol: dict[str, list[str]] = {}
        total_missing: int = 0

        for sym, sym_last in last_dates.items():
            # 仅检查最新日期落在区间内的 symbol（或至少靠近）
            if sym_last < start_str:
                # 该 symbol 的数据远落后于全局最新，暂不归入此类短期缺失
                continue
            actual: set[str] = actual_dates_by_symbol.get(sym, set())
            missing: list[str] = sorted(trade_days_set - actual)
            if missing:
                missing_by_symbol[sym] = missing
                total_missing += len(missing)

        logger.info(
            f"check_missing: 区间 {start_str}~{latest_date}，"
            f"理论 {len(trade_days)} 个交易日，"
            f"{len(missing_by_symbol)} 只股票缺失 {total_missing} 条数据"
        )
        return {
            "status": "ok",
            "latest_date": latest_date,
            "trade_days_expected": len(trade_days),
            "total_missing": total_missing,
            "missing_by_symbol": missing_by_symbol,
        }

    def repair_missing(self, days: int = 5, max_stocks: int | None = None) -> dict:
        """修复最近 N 个交易日内的缺失数据。

        修复策略：
        1. 调用 check_missing(days) 获取缺失报告
        2. 按缺失日期数量降序排列（max_stocks=None 则全部修复）
        3. 对每只股票，用 bs.query_history_k_data_plus 拉取缺失范围数据
        4. _write_to_db 写入，失败股票记录到重试队列，最多重试 2 轮

        Args:
            days: 检查范围（天），默认 5。
            max_stocks: 最多修复多少只股票，None 表示不限制（全部修复）。

        Returns:
            dict:
                - status (str): "ok" | "skipped" | "error"
                - checked (int): 检查区间的理论股票数
                - affected_stocks (int): 实际修复的股票数
                - total_filled (int): 写入的总记录数
        """
        t0: float = time.time()
        report: dict = self.check_missing(days)

        if report.get("status") == "error":
            return {
                "status": "error",
                "checked": 0,
                "affected_stocks": 0,
                "total_filled": 0,
            }

        missing: dict[str, list[str]] = report.get("missing_by_symbol", {})
        if not missing:
            logger.info("repair_missing: 无需修复，数据完整")
            return {
                "status": "skipped",
                "checked": len(report.get("missing_by_symbol", {})),
                "affected_stocks": 0,
                "total_filled": 0,
            }

        # 按缺失数量降序排列
        ranked_all: list[tuple[str, list[str]]] = sorted(
            missing.items(), key=lambda kv: len(kv[1]), reverse=True
        )
        ranked: list[tuple[str, list[str]]] = (
            ranked_all if max_stocks is None else ranked_all[:max_stocks]
        )

        logger.info(
            f"repair_missing: 共 {len(ranked_all)} 只股票缺失，"
            f"本次修复 {'全部' if max_stocks is None else f'前{len(ranked)}'} 只"
        )

        self._bs_login()
        import baostock as bs

        self._open_db()  # 批量写入优化

        logger.info(f"repair_missing: 开始修复 {len(ranked)} 只股票")
        affected: int = 0
        total_filled: int = 0
        failed_retry: list[tuple[str, str, str]] = []  # (sym, earliest, latest)
        _last_progress: float = time.time()  # 进度日志时间戳

        try:
            for sym, missing_dates in ranked:
                if not missing_dates:
                    continue
                bs_code: str = self.engine._to_baostock_code(sym)
                # 缺失日期区间：最早到最晚，一次查询覆盖
                earliest: str = min(missing_dates)
                latest_str: str = max(missing_dates)

                try:
                    rs = bs.query_history_k_data_plus(
                        bs_code,
                        "date,open,high,low,close,volume,amount,turn,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM",
                        start_date=earliest,
                        end_date=latest_str,
                        frequency="d",
                        adjustflag="2",
                    )
                    data = None
                    if rs.error_code == "0":
                        bs_data: pd.DataFrame = self._bs_get_data(rs)
                        if bs_data is not None and not bs_data.empty:
                            data = bs_data
                    if data is None:
                        tc_code = TencentSource.to_baostock_code(bs_code)
                        tc = TencentSource(request_interval=0.15)
                        tc_df = tc.get_daily(tc_code, 30)
                        if tc_df is not None and not tc_df.empty:
                            tc_df["date"] = tc_df["date"].astype(str)
                            mask = tc_df["date"].between(earliest, latest_str)
                            subset = tc_df[mask].copy()
                            if not subset.empty:
                                subset["symbol"] = sym
                                subset["turnover"] = 0.0
                                subset["peTTM"] = None
                                subset["pbMRQ"] = None
                                subset["psTTM"] = None
                                subset["pcfNcfTTM"] = None
                                data = subset
                                logger.debug(f"repair_missing {sym}: baostock 失败，Tencent 补 {len(data)} 条")
                    if data is None:
                        failed_retry.append((sym, earliest, latest_str))
                        continue

                    data["symbol"] = sym
                    data.rename(columns={"turn": "turnover"}, inplace=True)
                    n: int = self._write_to_db(data)
                    if n > 0:
                        affected += 1
                        total_filled += n
                    else:
                        failed_retry.append((sym, earliest, latest_str))

                except Exception as e:
                    logger.warning(f"repair_missing {sym} 拉取异常: {e}")
                    continue

                # 每处理 50 只打印一次进度日志
                if (affected + total_filled) > 0 and time.time() - _last_progress > 30:
                    _last_progress = time.time()
                    logger.info(
                        f"repair_missing 进度: {len(ranked)} 只中已处理 "
                        f"(affected={affected}, filled={total_filled}, "
                        f"retry_queue={len(failed_retry)})"
                    )

            # ── 重试失败股票（最多 2 轮，指数退避） ──
            for retry_round in range(1, 3):
                if not failed_retry:
                    break
                retry_sleep: float = 0.5 * retry_round  # 0.5s → 1.0s
                logger.info(
                    f"repair_missing: 第 {retry_round} 轮重试 "
                    f"{len(failed_retry)} 只失败股票（等待 {retry_sleep}s）"
                )
                time.sleep(retry_sleep)

                still_failed: list[tuple[str, str, str]] = []
                for sym, earliest, latest_str in failed_retry:
                    try:
                        rs = bs.query_history_k_data_plus(
                            self.engine._to_baostock_code(sym),
                            "date,open,high,low,close,volume,amount,turn,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM",
                            start_date=earliest,
                            end_date=latest_str,
                            frequency="d",
                            adjustflag="2",
                        )
                        data = None
                        if rs.error_code == "0":
                            bs_data = self._bs_get_data(rs)
                            if bs_data is not None and not bs_data.empty:
                                data = bs_data
                        if data is None:
                            tc_code = TencentSource.to_baostock_code(sym)
                            tc = TencentSource(request_interval=0.15)
                            tc_df = tc.get_daily(tc_code, 30)
                            if tc_df is not None and not tc_df.empty:
                                tc_df["date"] = tc_df["date"].astype(str)
                                mask = tc_df["date"].between(earliest, latest_str)
                                subset = tc_df[mask].copy()
                                if not subset.empty:
                                    subset["symbol"] = sym
                                    subset["turnover"] = 0.0
                                    subset["peTTM"] = None
                                    subset["pbMRQ"] = None
                                    subset["psTTM"] = None
                                    subset["pcfNcfTTM"] = None
                                    data = subset
                                    logger.debug(f"repair_missing {sym}: retry Tencent 补 {len(data)} 条")
                        if data is None:
                            still_failed.append((sym, earliest, latest_str))
                            continue
                        data["symbol"] = sym
                        data.rename(columns={"turn": "turnover"}, inplace=True)
                        n = self._write_to_db(data)
                        if n > 0:
                            affected += 1
                            total_filled += n
                        else:
                            still_failed.append((sym, earliest, latest_str))
                        time.sleep(0.15)
                    except Exception:
                        still_failed.append((sym, earliest, latest_str))
                failed_retry = still_failed

            if failed_retry:
                logger.warning(
                    f"repair_missing: {len(failed_retry)} 只股票 2 轮重试后仍失败"
                )

            # 构建结果并记录到 sync_log
            result: dict = {
                "status": "ok",
                "stock_count": affected,
                "is_trade_day": True,
                "api_status": "ok",
                "coverage": 1.0,
                "error": "",
            }
            self._log_sync(result, time.time() - t0)

            logger.info(
                f"repair_missing 完成: 修复 {affected} 只，"
                f"写入 {total_filled} 条"
            )
            self._close_db()
            return {
                "status": "ok",
                "checked": len(report.get("missing_by_symbol", {})),
                "affected_stocks": affected,
                "total_filled": total_filled,
            }
        except Exception as e:
            logger.error(f"repair_missing 异常: {e}")
            self._close_db()
            return {
                "status": "error",
                "checked": len(report.get("missing_by_symbol", {})),
                "affected_stocks": affected,
                "total_filled": total_filled,
            }

    # ── 历史回填 ──

    def backfill(self) -> dict:
        """历史日线数据全量回填（从 settings.start_date 到今天的所有交易日）。

        回填策略：
        1. 获取活跃股票列表
        2. 生成 start_date → today 的全部交易日
        3. 对每只股票按约 800 条一批分段拉取（适配 baostock 单次查询上限）
        4. 每批写入 stock_daily

        Returns:
            dict:
                - status (str): "ok" | "error"
                - stock_count (int): 成功回填的股票数
                - total_records (int): 写入的总记录数
                - duration_seconds (float): 耗时（秒）
                - error (str): 错误信息
        """
        t0: float = time.time()

        active: dict = self.get_active_stocks()
        symbols: list[str] = active.get("symbols", [])
        if not symbols:
            logger.error("backfill: 无法获取活跃股票列表")
            return {
                "status": "error",
                "stock_count": 0,
                "total_records": 0,
                "duration_seconds": 0.0,
                "error": "无法获取活跃股票列表",
            }

        today_str: str = date.today().strftime("%Y-%m-%d")

        # 获取全部交易日列表
        if not self._bs_login():
            return {
                "status": "error",
                "stock_count": 0,
                "total_records": 0,
                "duration_seconds": time.time() - t0,
                "error": "baostock 登录失败",
            }

        import baostock as bs

        trade_days: list[str] = []
        try:
            rs = bs.query_trade_dates(
                start_date=self.start_date, end_date=today_str
            )
            if rs.error_code == "0":
                data = self._bs_get_data(rs)
                if not data.empty:
                    trade_days = data.loc[
                        data["is_trading_day"] == "1", "calendar_date"
                    ].tolist()
        except Exception as e:
            logger.error(f"backfill: 交易日查询异常: {e}")
            return {
                "status": "error",
                "stock_count": 0,
                "total_records": 0,
                "duration_seconds": time.time() - t0,
                "error": str(e),
            }

        if not trade_days:
            logger.warning("backfill: 交易日列表为空")
            return {
                "status": "error",
                "stock_count": 0,
                "total_records": 0,
                "duration_seconds": time.time() - t0,
                "error": "交易日列表为空",
            }

        # 按约 800 条一批切分交易日列表
        batch_size: int = 800
        trade_batches: list[list[str]] = [
            trade_days[i : i + batch_size]
            for i in range(0, len(trade_days), batch_size)
        ]

        logger.info(
            f"backfill: {len(symbols)} 只股票，"
            f"{len(trade_days)} 个交易日，{len(trade_batches)} 批"
        )

        stock_count: int = 0
        total_records: int = 0

        try:
            for s_idx, sym in enumerate(symbols):
                bs_code: str = self.engine._to_baostock_code(sym)
                sym_records: int = 0

                for batch in trade_batches:
                    batch_start: str = batch[0]
                    batch_end: str = batch[-1]

                    try:
                        rs = bs.query_history_k_data_plus(
                            bs_code,
                            "date,open,high,low,close,volume,amount,turn,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM",
                            start_date=batch_start,
                            end_date=batch_end,
                            frequency="d",
                            adjustflag="2",
                        )
                        if rs.error_code != "0":
                            continue
                        chunk: pd.DataFrame = self._bs_get_data(rs)
                        if chunk.empty:
                            continue

                        chunk["symbol"] = sym
                        chunk.rename(columns={"turn": "turnover"}, inplace=True)
                        n: int = self._write_to_db(chunk)
                        sym_records += n
                    except Exception:
                        continue

                if sym_records > 0:
                    stock_count += 1
                    total_records += sym_records

                # 每 10 只股票打一次进度日志（历史回填较慢）
                if (s_idx + 1) % 10 == 0:
                    elapsed: float = time.time() - t0
                    logger.info(
                        f"backfill 进度: {s_idx + 1}/{len(symbols)} 只，"
                        f"已写入 {total_records} 条（{elapsed:.0f}s）"
                    )

                # 期间保持连接活跃
                time.sleep(0.1)

            duration: float = time.time() - t0
            result: dict = {
                "status": "ok",
                "stock_count": stock_count,
                "is_trade_day": True,
                "api_status": "ok",
                "coverage": 1.0,
                "backfilled": len(trade_days),
                "error": "",
            }
            self._log_sync(result, duration)

            logger.info(
                f"backfill 完成: {stock_count}/{len(symbols)} 只，"
                f"共 {total_records} 条，耗时 {duration:.0f}s"
            )
            return {
                "status": "ok",
                "stock_count": stock_count,
                "total_records": total_records,
                "duration_seconds": duration,
                "error": "",
            }
        except Exception as e:
            logger.error(f"backfill 异常: {e}")
            return {
                "status": "error",
                "stock_count": stock_count,
                "total_records": total_records,
                "duration_seconds": time.time() - t0,
                "error": str(e),
            }

    # ── 指数日线同步 ──

    # 6 大指数 baostock 代码映射
    INDEX_CODES: dict[str, str] = {
        "sh.000001": "上证指数",
        "sh.000016": "上证50",
        "sh.000300": "沪深300",
        "sh.000905": "中证500",
        "sz.399001": "深证成指",
        "sz.399106": "深证综指",
    }

    def _write_index_to_db(self, df: pd.DataFrame) -> int:
        """将指数 DataFrame 写入 index_daily 表。"""
        if df.empty:
            return 0

        for col in ["open", "high", "low", "close", "volume", "amount", "pctChg"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close"])

        if df.empty:
            return 0

        count: int = len(df)
        records = df.to_dict("records")
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO index_daily
                   (symbol, date, open, high, low, close, volume, amount, pctChg)
                   VALUES (:symbol, :date, :open, :high, :low, :close, :volume, :amount, :pctChg)""",
                records,
            )
            conn.commit()
        logger.info(f"_write_index_to_db: 写入 {count} 条指数记录")
        return count

    def _fill_valuation_gaps(self, days: int = 5) -> dict:
        """回填 peTTM/pbMRQ/psTTM/pcfNcfTTM/pctChg 为空的记录（用 baostock 更新）。

        当数据由 TencentSource 写入时不含估值字段和涨跌幅，此方法在 baostock 可用时回填。
        """
        today_str: str = date.today().strftime("%Y-%m-%d")
        start_date: str = (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")

        try:
            import baostock as bs

            if not self._bs_login():
                logger.warning("_fill_valuation_gaps: baostock 登录失败，跳过")
                return {"status": "skipped", "filled": 0}

            self._open_db()
            # 快速探测 baostock 是否真的可用
            test_rs = bs.query_trade_dates(start_date=today_str, end_date=today_str)
            if test_rs.error_code != "0":
                logger.warning("_fill_valuation_gaps: baostock 不可用，跳过（不影响管线）")
                self._bs_logout()
                self._close_db()
                return {"status": "skipped", "filled": 0, "reason": "baostock 不可用"}

            logger.info("_fill_valuation_gaps: baostock 可用，开始回填估值字段")
            null_rows = self._db_conn.execute(
                "SELECT symbol, date FROM stock_daily "
                "WHERE date >= ? AND date <= ? "
                "AND (peTTM IS NULL OR pbMRQ IS NULL OR psTTM IS NULL "
                "     OR pcfNcfTTM IS NULL OR pctChg IS NULL) "
                "ORDER BY symbol, date",
                (start_date, today_str)
            ).fetchall()

            if not null_rows:
                logger.info("_fill_valuation_gaps: 无缺失字段")
                self._bs_logout()
                return {"status": "ok", "filled": 0}

            t0 = time.time()
            logger.info(f"_fill_valuation_gaps: 发现 {len(null_rows)} 条缺失字段的记录")
            filled = 0
            failed = 0

            for sym, dt in null_rows:
                try:
                    bs_code = self.engine._to_baostock_code(sym)
                    rs = bs.query_history_k_data_plus(
                        bs_code,
                        "peTTM,pbMRQ,psTTM,pcfNcfTTM,pctChg",
                        start_date=dt, end_date=dt,
                        frequency="d", adjustflag="2",
                    )
                    if rs.error_code == "0":
                        data = self._bs_get_data(rs)
                        if data is not None and not data.empty:
                            row = data.iloc[0]
                            pe = row.get("peTTM")
                            pb = row.get("pbMRQ")
                            ps = row.get("psTTM")
                            pcf = row.get("pcfNcfTTM")
                            pct = row.get("pctChg")
                            self._db_conn.execute(
                                "UPDATE stock_daily SET peTTM=?, pbMRQ=?, psTTM=?, pcfNcfTTM=?, pctChg=? "
                                "WHERE symbol=? AND date=?",
                                (pe, pb, ps, pcf, pct, sym, dt)
                            )
                            filled += 1
                except Exception as e:
                    failed += 1
                    if failed <= 3:
                        logger.debug(f"_fill_valuation_gaps {sym} {dt}: {e}")

                if (filled + failed) % 100 == 0:
                    self._db_conn.commit()
                    elapsed = time.time() - t0
                    logger.info(
                        f"_fill_valuation_gaps 进度: {filled+failed}/{len(null_rows)} "
                        f"条 (filled={filled}, failed={failed}, {elapsed:.0f}s)"
                    )

            self._db_conn.commit()
            self._bs_logout()
            self._close_db()
            logger.info(f"_fill_valuation_gaps: 回填 {filled} 条，失败 {failed} 条")
            return {"status": "ok", "filled": filled, "failed": failed}

        except Exception as e:
            logger.error(f"_fill_valuation_gaps 异常: {e}")
            if hasattr(self, '_db_conn') and self._db_conn:
                try: self._db_conn.close(); self._db_conn = None
                except: pass
            return {"status": "error", "filled": 0, "error": str(e)}

    def sync_index_daily(self, force: bool = False) -> dict:
        """同步 6 大指数日线数据到 index_daily 表。

        指数包括：上证指数、上证50、沪深300、中证500、深证成指、深证综指。
        数据与 stock_daily 表完全隔离，存储在 index_daily 表中。

        Args:
            force: 是否强制拉取（忽略交易日判断）。

        Returns:
            {"status": "ok"/"skipped"/"error", "index_count": int, "error": ""}
        """
        if not force and not self.is_trade_day():
            logger.info("sync_index_daily: 非交易日，跳过")
            return {"status": "skipped", "index_count": 0, "error": ""}

        if not self._bs_login():
            return {"status": "error", "index_count": 0, "error": "baostock 登录失败"}

        import baostock as bs

        today_str: str = date.today().strftime("%Y-%m-%d")
        index_count: int = 0

        try:
            for bs_code, name in self.INDEX_CODES.items():
                # 从本地获取最新日期
                last_date: str | None = None
                with sqlite3.connect(self.db_path) as conn:
                    row = conn.execute(
                        "SELECT MAX(date) FROM index_daily WHERE symbol = ?",
                        (bs_code,),
                    ).fetchone()
                    if row and row[0]:
                        last_date = row[0]

                start_date: str = last_date or self.start_date
                if last_date:
                    dt = datetime.strptime(last_date, "%Y-%m-%d")
                    start_date = (dt + timedelta(days=1)).strftime("%Y-%m-%d")

                if start_date > today_str:
                    continue

                data = None
                try:
                    rs = bs.query_history_k_data_plus(
                        bs_code,
                        "date,open,high,low,close,volume,amount,pctChg",
                        start_date=start_date,
                        end_date=today_str,
                        frequency="d",
                        adjustflag="2",
                    )
                    if rs.error_code == "0":
                        bs_data = self._bs_get_data(rs)
                        if bs_data is not None and not bs_data.empty:
                            data = bs_data
                except Exception:
                    pass
                if data is None:
                    tc_code = TencentSource.to_baostock_code(bs_code)
                    tc = TencentSource(request_interval=0.15)
                    tc_df = tc.get_daily(tc_code, 30)
                    if tc_df is not None and not tc_df.empty:
                        tc_df["date"] = tc_df["date"].astype(str)
                        mask = tc_df["date"].between(start_date, today_str)
                        idx_val = tc_df[mask].copy()
                        if not idx_val.empty:
                            for col in ["open", "high", "low", "close", "volume", "amount", "pctChg"]:
                                if col not in idx_val.columns:
                                    idx_val[col] = 0.0
                            idx_val["symbol"] = bs_code
                            data = idx_val
                            logger.info(f"sync_index_daily {name}: baostock 失败，Tencent 补 {len(data)} 条")
                if data is None:
                    logger.warning(f"sync_index_daily {name}: 双数据源均失败，跳过")
                    continue

                data["symbol"] = bs_code
                n: int = self._write_index_to_db(data)
                if n > 0:
                    index_count += 1
                    logger.info(f"sync_index_daily: {name} 写入 {n} 条")

                time.sleep(0.15)

            logger.info(f"sync_index_daily 完成: {index_count}/6 个指数")
            return {"status": "ok", "index_count": index_count, "error": ""}
        except Exception as e:
            import traceback
            logger.error(f"sync_index_daily 异常: {e}\n{traceback.format_exc()}")
            return {"status": "error", "index_count": index_count, "error": str(e)}

    # ── 完整同步管线 ──

    def run_full(self) -> dict:
        """完整同步管线：stock_list 同步 + 增量日线同步 + 缺失补填。

        管线阶段：
        Phase 1: sync_stock_list() — 同步股票列表
        Phase 2: sync_daily(force=False) — 增量日线同步（含交易日/时间门控）
        Phase 3: repair_missing(days=5) — 修复最近 5 天的缺失数据
        Phase 4: _fill_valuation_gaps(days=5) — 回填估值字段(None→实际值)
        Phase 5: sync_index_daily() — 同步 6 大指数日线

        任一阶段返回 status="error" 则终止后续阶段。
        Phase 2 返回 status="skipped"（非交易日/时间未到）不终止，继续 Phase 3。
        Phase 4 error 不终止管线（指数同步失败不影响股票选股）。

        Returns:
            dict:
                - status (str): "ok" | "error"
                - phases (dict): 各阶段结果
                - error (str): 错误信息
        """
        t0: float = time.time()
        phases: dict[str, dict] = {}

        # Phase 1: 股票列表同步
        logger.info("run_full Phase 1: 股票列表同步")
        r1: dict = self.sync_stock_list()
        phases["stock_list"] = r1
        if r1.get("status") == "error":
            logger.error("run_full Phase 1 失败，终止管线")
            self._log_sync(
                {
                    "status": "error",
                    "error": "Phase 1 stock_list 失败",
                    "is_trade_day": True,
                    "stock_count": 0,
                    "api_status": r1.get("status", ""),
                    "coverage": 0.0,
                },
                time.time() - t0,
            )
            return {
                "status": "error",
                "phases": phases,
                "error": "Phase 1 stock_list 失败",
            }

        # Phase 2: 增量日线同步
        logger.info("run_full Phase 2: 增量日线同步")
        r2: dict = self.sync_daily(force=False)
        phases["daily_sync"] = r2
        if r2.get("status") == "error":
            logger.error("run_full Phase 2 失败，终止管线")
            self._log_sync(
                {
                    "status": "error",
                    "error": "Phase 2 sync_daily 失败",
                    "is_trade_day": r2.get("is_trade_day", True),
                    "stock_count": r2.get("stock_count", 0),
                    "api_status": "error",
                    "coverage": 0.0,
                },
                time.time() - t0,
            )
            return {
                "status": "error",
                "phases": phases,
                "error": "Phase 2 sync_daily 失败",
            }

        # Phase 3: 缺失补填（Phase 2 skipped 也继续）
        logger.info("run_full Phase 3: 缺失补填")
        r3: dict = self.repair_missing(days=5)
        phases["repair"] = r3

        # Phase 4: 估值字段回填（仅当 baostock 可用时）
        logger.info("run_full Phase 4: 估值字段回填")
        r4v: dict = self._fill_valuation_gaps(days=5)

        # Phase 5: 指数日线同步（error 不终止管线）
        logger.info("run_full Phase 5: 指数日线同步")
        r4: dict = self.sync_index_daily()
        phases["index_sync"] = r4

        # 汇总结果
        overall_status: str = "ok"
        error_msg: str = ""
        if r3.get("status") == "error":
            overall_status = "error"
            error_msg = "Phase 3 repair_missing 失败"
        elif r4.get("status") == "error":
            # 指数同步失败不视为整体失败
            logger.warning(f"run_full: 指数同步失败（不影响选股）: {r4.get('error')}")

        duration: float = time.time() - t0
        self._log_sync(
            {
                "status": overall_status,
                "stock_count": r2.get("stock_count", 0),
                "is_trade_day": r2.get("is_trade_day", True),
                "api_status": "ok",
                "coverage": 1.0,
                "error": error_msg,
            },
            duration,
        )

        logger.info(
            f"run_full 完成: status={overall_status}, "
            f"phases={{stock_list: {r1.get('status')}, "
            f"daily: {r2.get('status')}, repair: {r3.get('status')}, "
            f"index: {r4.get('status')}}}"
        )
        return {
            "status": overall_status,
            "phases": phases,
            "error": error_msg,
        }
