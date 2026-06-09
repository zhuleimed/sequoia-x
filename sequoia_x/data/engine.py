"""数据引擎模块：负责 SQLite 行情数据存储与 baostock 增量同步。"""

import sqlite3
from pathlib import Path

import pandas as pd

from sequoia_x.core.config import Settings
from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


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
    error_msg        TEXT,
    created_at       TEXT    DEFAULT (datetime('now','localtime'))
);
"""


class DataEngine:
    """行情数据引擎，负责 SQLite 存储和 baostock 数据同步。"""

    def __init__(self, settings: Settings) -> None:
        self.db_path: str = settings.db_path
        self.start_date: str = settings.start_date
        self._init_db()

    def _init_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE_TABLE_SQL)
            conn.execute(_CREATE_INDEX_SQL)
            conn.execute(_CREATE_SYNC_LOG_SQL)
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

    # ══════════════════════════════════════════════════
    #  交易日历
    # ══════════════════════════════════════════════════

    def get_trade_calendar(self, start_date: str, end_date: str) -> list[str]:
        """从 baostock 获取指定日期范围内的交易日列表。"""
        import baostock as bs

        lg = bs.login()
        if lg.error_code != "0":
            logger.error(f"baostock 登录失败: {lg.error_msg}")
            return []

        try:
            rs = bs.query_trade_dates(start_date=start_date, end_date=end_date)
            dates = []
            while rs.next():
                row = rs.get_row_data()
                if row[1] == "1":  # is_trading_day
                    dates.append(row[0])
            return dates
        except Exception as e:
            logger.error(f"获取交易日历失败: {e}")
            return []
        finally:
            bs.logout()

    # ══════════════════════════════════════════════════
    #  数据同步 — 19:10 独立运行
    # ══════════════════════════════════════════════════

    def sync_and_clean(self) -> dict:
        """完整数据同步 + 清洗。

        1. 对比 baostock 上市列表：清理退市股、发现新股
        2. 获取交易日历，检查缺失交易日并回填
        3. 拉取当日增量数据
        4. 记录 sync_log

        Returns:
            {"status":"success"/"failed", "stock_count":int, "delisted":int,
             "new_listed":int, "backfilled":int, "latest_date":str, "error":str}
        """
        import time
        import baostock as bs
        from datetime import date, timedelta

        today_str = date.today().strftime("%Y-%m-%d")
        result = {
            "status": "failed",
            "stock_count": 0,
            "delisted": 0,
            "new_listed": 0,
            "backfilled": 0,
            "latest_date": "",
            "error": "",
        }

        lg = bs.login()
        if lg.error_code != "0":
            result["error"] = f"baostock登录失败: {lg.error_msg}"
            logger.error(result["error"])
            return result

        try:
            # ── 1. 获取当前上市股票列表 ──
            rs = bs.query_stock_basic(code_name="", code="")
            active_stocks: dict[str, tuple[str, str]] = {}
            while rs.next():
                row = rs.get_row_data()
                code_num = row[0].split(".")[1]
                name = row[1]
                ipo = row[2]
                s_type = row[4]
                status = row[5]
                if status == "1" and s_type == "1":
                    active_stocks[code_num] = (name, ipo)

            local_symbols = self.get_local_symbols()

            # ── 2. 清理退市股 ──
            to_delete = [s for s in local_symbols if s not in active_stocks]
            if to_delete:
                with sqlite3.connect(self.db_path) as conn:
                    for sym in to_delete:
                        conn.execute("DELETE FROM stock_daily WHERE symbol = ?", (sym,))
                    conn.commit()
                logger.info(
                    f"清理退市股: 移除 {len(to_delete)} 只: {' '.join(to_delete[:10])}"
                )
            result["delisted"] = len(to_delete)

            # ── 3. 发现新股 ──
            new_stocks = [s for s in active_stocks if s not in local_symbols]
            result["new_listed"] = len(new_stocks)
            if new_stocks:
                logger.info(f"发现新股: {len(new_stocks)} 只")

            local_after = self.get_local_symbols()

            # ── 4. 获取交易日历 & 检查缺失 ──
            start_check = (date.today() - timedelta(days=15)).strftime("%Y-%m-%d")
            bs.logout()  # 换个连接取 trade_dates
            trade_dates = self.get_trade_calendar(start_check, today_str)

            missing_dates: list[str] = []
            if local_after and trade_dates:
                with sqlite3.connect(self.db_path) as conn:
                    for td in trade_dates:
                        # 跳过当日（baostock 17:30 后才入库）
                        if td == today_str:
                            continue
                        cnt = conn.execute(
                            "SELECT COUNT(DISTINCT symbol) FROM stock_daily WHERE date = ?",
                            (td,),
                        ).fetchone()[0]
                        # 覆盖不足 50% 视为缺失
                        if cnt < len(local_after) * 0.5:
                            missing_dates.append(td)

            if missing_dates:
                logger.info(
                    f"发现 {len(missing_dates)} 个待补充交易日: {missing_dates}"
                )
                total_synced = 0
                all_backfill_rows: list[list] = []
                active_local = [s for s in local_after if s in active_stocks]

                for md in missing_dates:
                    lg2 = bs.login()
                    if lg2.error_code != "0":
                        continue
                    day_rows: list[list] = []
                    try:
                        for sym in active_local:
                            code = self._to_baostock_code(sym)
                            rs = bs.query_history_k_data_plus(
                                code,
                                "date,open,high,low,close,volume,amount",
                                start_date=md,
                                end_date=md,
                                frequency="d",
                                adjustflag="1",
                            )
                            if rs.error_code == "0":
                                while rs.next():
                                    day_rows.append([sym] + rs.get_row_data())
                            time.sleep(0.15)
                    finally:
                        bs.logout()

                    if day_rows:
                        all_backfill_rows.extend(day_rows)
                        total_synced += 1

                if all_backfill_rows:
                    df = pd.DataFrame(
                        all_backfill_rows,
                        columns=[
                            "symbol", "date", "open", "high",
                            "low", "close", "volume", "turnover",
                        ],
                    )
                    for col in ["open", "high", "low", "close", "volume", "turnover"]:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    df = df.dropna(subset=["close"])
                    with sqlite3.connect(self.db_path) as conn:
                        for d in df["date"].unique():
                            conn.execute(
                                "DELETE FROM stock_daily WHERE date = ?", (d,)
                            )
                        df.to_sql(
                            "stock_daily",
                            conn,
                            if_exists="append",
                            index=False,
                            method="multi",
                            chunksize=500,
                        )
                        conn.commit()

                result["backfilled"] = total_synced
                logger.info(f"缺失数据回填完成: {total_synced} 个交易日")

            # ── 5. 拉取今日增量数据 ──
            today_count = self.sync_today_bulk()
            logger.info(f"今日增量数据: {today_count} 条")

            # ── 6. 统计结果 ──
            local_final = self.get_local_symbols()
            result["stock_count"] = len(local_final)
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute("SELECT MAX(date) FROM stock_daily").fetchone()
                result["latest_date"] = row[0] or ""

            result["status"] = "success"

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"sync_and_clean 异常: {e}")
            import traceback

            traceback.print_exc()
        finally:
            bs.logout()

        # ── 记录 sync_log ──
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO sync_log "
                    "(date, status, stock_count, delisted_count, new_listed_count, backfilled_days) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        today_str,
                        result["status"],
                        result["stock_count"],
                        result["delisted"],
                        result["new_listed"],
                        result["backfilled"],
                    ),
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"sync_log 写入失败: {e}")

        logger.info(f"sync_and_clean 完成: 状态={result['status']} "
                     f"股票{result['stock_count']} 退市{result['delisted']} "
                     f"新股{result['new_listed']} 补填{result['backfilled']}天")
        return result

    # ══════════════════════════════════════════════════
    #  数据完整性检查 — 20:55 选股前执行
    # ══════════════════════════════════════════════════

    def check_data_completeness(self) -> dict:
        """检查数据完整性：确认最近交易日数据是否已完整拉取。

        Returns:
            {"is_complete": bool,
             "latest_trade_day": str,
             "coverage": float,
             "total_stocks": int,
             "stocks_with_data": int,
             "last_sync_status": str}
        """
        import baostock as bs
        from datetime import date, timedelta

        today_str = date.today().strftime("%Y-%m-%d")
        result = {
            "is_complete": False,
            "latest_trade_day": "",
            "coverage": 0.0,
            "total_stocks": 0,
            "stocks_with_data": 0,
            "last_sync_status": "unknown",
        }

        # 1. 获取最近交易日（排除今日 — 盘前/盘中查不到今日数据）
        check_start = (date.today() - timedelta(days=10)).strftime("%Y-%m-%d")
        all_trade_dates = self.get_trade_calendar(check_start, today_str)
        trade_dates = [d for d in all_trade_dates if d != today_str]
        if not trade_dates:
            result["error"] = "无法获取交易日历"
            return result

        latest_td = trade_dates[-1]
        result["latest_trade_day"] = latest_td

        # 2. 本地股票总数
        local_symbols = self.get_local_symbols()
        result["total_stocks"] = len(local_symbols)
        if not local_symbols:
            result["error"] = "本地数据库为空"
            return result

        # 3. 统计最新交易日有数据的股票数
        with sqlite3.connect(self.db_path) as conn:
            cnt = conn.execute(
                "SELECT COUNT(DISTINCT symbol) FROM stock_daily WHERE date = ?",
                (latest_td,),
            ).fetchone()[0]

        result["stocks_with_data"] = cnt
        result["coverage"] = round(cnt / len(local_symbols), 4)
        # 覆盖率 > 85% 视为完整（允许少数停牌股）
        result["is_complete"] = cnt >= len(local_symbols) * 0.85

        # 4. 检查最近 sync_log
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT status FROM sync_log WHERE date = ? ORDER BY id DESC LIMIT 1",
                    (latest_td,),
                ).fetchone()
                if row:
                    result["last_sync_status"] = row[0]
        except Exception:
            pass

        logger.info(
            f"数据完整性检查: 最新日={latest_td} "
            f"覆盖率={result['coverage']:.1%} "
            f"({cnt}/{len(local_symbols)}) "
            f"结果={'完整' if result['is_complete'] else '不完整'}"
        )
        return result

    # ══════════════════════════════════════════════════
    #  数据同步 — 增量拉取
    # ══════════════════════════════════════════════════

    def sync_today_bulk(self) -> int:
        """单进程顺序通过 baostock 拉取增量数据（后复权），写入 SQLite。

        注意：baostock API 的 SocketUtil 为单例模式，且服务器端对同一用户
        有并发连接限制。多进程并行会导致连接冲突（login 死锁、logout failed、
        数据丢失），故使用单进程顺序拉取 + 请求间隔 200ms。

        自动判断：若当前时间 < 17:30（baostock 日线入库时间），跳过今日拉取。
        """
        from datetime import date, timedelta, datetime, time as dtime
        import time
        import baostock as bs

        today_str = date.today().strftime("%Y-%m-%d")

        # ── 盘前/盘中跳过：baostock 17:30 后才入库日线数据 ──
        now = datetime.now()
        baostock_update_time = dtime(17, 30)
        if now.time() < baostock_update_time:
            # 但还是要检查是否有非今日的缺失（比如昨天因为异常没拉到）
            last_local_date = self._get_last_date_range()
            if last_local_date and last_local_date >= (date.today() - timedelta(days=3)).strftime("%Y-%m-%d"):
                logger.info(
                    f"当前 {now.strftime('%H:%M')}，baostock 日线 17:30 后入库，"
                    f"且本地数据最新到 {last_local_date}，跳过增量拉取"
                )
                return 0
            else:
                logger.info(
                    f"本地数据较旧（最新 {last_local_date}），"
                    f"虽未到 17:30 仍尝试拉取"
                )

        tasks = []
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT symbol, MAX(date) FROM stock_daily GROUP BY symbol"
            ).fetchall()

        if not rows:
            logger.warning("本地无股票数据，请先执行 --backfill")
            return 0

        for symbol, last_date in rows:
            if last_date and last_date >= today_str:
                continue
            start = today_str
            if last_date:
                start = (date.fromisoformat(last_date) + timedelta(days=1)).strftime(
                    "%Y-%m-%d"
                )
            tasks.append((symbol, self._to_baostock_code(symbol), start, today_str))

        if not tasks:
            logger.info("所有股票已是最新，无需更新")
            return 0

        _start_time = time.time()
        logger.info(
            f"需要更新 {len(tasks)} 只股票，单进程顺序拉取（每只间隔200ms）..."
        )

        all_rows: list[list] = []
        lg = bs.login()
        if lg.error_code != "0":
            logger.error(f"baostock 登录失败: {lg.error_msg}")
            return 0

        try:
            for i, (symbol, bs_code, start, end) in enumerate(tasks):
                rs = bs.query_history_k_data_plus(
                    bs_code,
                    "date,open,high,low,close,volume,amount",
                    start_date=start,
                    end_date=end,
                    frequency="d",
                    adjustflag="1",
                )
                if rs.error_code == "0":
                    while rs.next():
                        all_rows.append([symbol] + rs.get_row_data())
                time.sleep(0.2)

                if (i + 1) % 500 == 0:
                    logger.info(
                        f"增量同步进度: {i + 1}/{len(tasks)}, "
                        f"已获取 {len(all_rows)} 条"
                    )
        finally:
            bs.logout()

        if not all_rows:
            logger.info("无新数据（可能非交易日或所有股票无成交）")
            return 0

        df = pd.DataFrame(
            all_rows,
            columns=[
                "symbol", "date", "open", "high",
                "low", "close", "volume", "turnover",
            ],
        )
        for col in ["open", "high", "low", "close", "volume", "turnover"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close"])
        df = df[df["volume"] > 0]

        count = len(df)
        with sqlite3.connect(self.db_path) as conn:
            for d in df["date"].unique().tolist():
                conn.execute("DELETE FROM stock_daily WHERE date = ?", (d,))
            df.to_sql(
                "stock_daily",
                conn,
                if_exists="append",
                index=False,
                method="multi",
                chunksize=500,
            )
            conn.commit()

        logger.info(
            f"增量同步完成: 写入 {count} 条数据（遍历 {len(tasks)} 只）"
            f"耗时 {time.time() - _start_time:.0f}秒"
        )
        return count

    def backfill(self, symbols: list[str]) -> None:
        """单进程顺序回填历史K线数据。

        注意：baostock API 不支持多进程并发。单进程每只间隔 200ms 防封。
        每 200 只重新 login 一次（防止长连接超时）。
        已入库的自动 skip，中断后可重跑续传。
        """
        import time
        from datetime import date, timedelta
        import baostock as bs

        today_str = date.today().strftime("%Y-%m-%d")
        start_time = time.time()
        reconnect_interval = 200

        success = 0
        failed = 0
        skipped = 0

        # 构建任务列表（跳过已完成的）
        tasks = []
        for symbol in symbols:
            last_date = self._get_last_date(symbol)
            if last_date and last_date >= today_str:
                skipped += 1
                continue
            start = last_date or self.start_date
            if last_date:
                start = (
                    date.fromisoformat(last_date) + timedelta(days=1)
                ).strftime("%Y-%m-%d")
            tasks.append((symbol, self._to_baostock_code(symbol), start, today_str))

        if not tasks:
            logger.info("所有股票已是最新，无需回填")
            return

        total = len(tasks)
        logger.info(
            f"回填启动：{total} 只股票需处理（跳过已有 {skipped} 只），"
            f"单进程顺序拉取，每 {reconnect_interval} 只重连一次"
        )

        for batch_start in range(0, total, reconnect_interval):
            batch = tasks[batch_start : batch_start + reconnect_interval]

            lg = bs.login()
            if lg.error_code != "0":
                logger.error(f"baostock 登录失败: {lg.error_msg}")
                return

            all_rows: list[list] = []
            try:
                for symbol, bs_code, start, end in batch:
                    rs = bs.query_history_k_data_plus(
                        bs_code,
                        "date,open,high,low,close,volume,amount",
                        start_date=start,
                        end_date=end,
                        frequency="d",
                        adjustflag="1",
                    )
                    if rs.error_code == "0":
                        while rs.next():
                            all_rows.append([symbol] + rs.get_row_data())
                    time.sleep(0.2)
            finally:
                bs.logout()

            batch_success = len(set(r[0] for r in all_rows)) if all_rows else 0
            success += batch_success
            failed += len(batch) - batch_success

            # 写入数据库
            if all_rows:
                df = pd.DataFrame(
                    all_rows,
                    columns=[
                        "symbol", "date", "open", "high",
                        "low", "close", "volume", "amount",
                    ],
                )
                df = df.rename(columns={"amount": "turnover"})
                for col in ["open", "high", "low", "close", "volume", "turnover"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.dropna(subset=["close"])
                df = df[df["volume"] > 0]

                with sqlite3.connect(self.db_path) as conn:
                    for d in df["date"].unique():
                        conn.execute("DELETE FROM stock_daily WHERE date = ?", (d,))
                    df.to_sql(
                        "stock_daily",
                        conn,
                        if_exists="append",
                        index=False,
                        method="multi",
                        chunksize=500,
                    )

            progress = min(batch_start + reconnect_interval, total)
            elapsed = time.time() - start_time
            rate = progress / elapsed if elapsed > 0 else 0
            logger.info(
                f"回填进度: {progress}/{total} | "
                f"成功 {success} 失败 {failed} 跳过 {skipped} | "
                f"速率 {rate:.1f} 只/秒 | "
                f"已用 {elapsed/60:.0f}分钟"
            )

        # 最终报告
        elapsed = time.time() - start_time
        logger.info(
            f"回填完成 — 成功: {success} | "
            f"失败: {failed} | "
            f"跳过: {skipped} | "
            f"耗时: {elapsed/60:.1f}分钟 | "
            f"平均: {elapsed/total:.2f}秒/只"
        )

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
