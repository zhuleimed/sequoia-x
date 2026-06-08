"""数据引擎模块：负责 SQLite 行情数据存储与 baostock 增量同步。"""

import sqlite3
from pathlib import Path

import pandas as pd

from sequoia_x.core.config import Settings
from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


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


def _bs_fetch_batch(tasks: list) -> list:
    """多进程 worker：独立 login，批量拉取 baostock 数据。
    
    每个 worker 内每只股票处理完后休眠 200ms，防止请求过于密集被封。
    """
    import time
    import baostock as bs
    bs.login()
    results = []
    for symbol, bs_code, start, end in tasks:
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,open,high,low,close,volume,amount",
            start_date=start,
            end_date=end,
            frequency="d",
            adjustflag="1",  # 后复权
        )
        if rs.error_code != "0":
            time.sleep(0.2)
            continue
        while rs.next():
            results.append([symbol] + rs.get_row_data())
        time.sleep(0.2)  # 200ms 间隔，防止封 IP
    bs.logout()
    return results


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
            conn.commit()
        logger.info(f"数据库初始化完成：{self.db_path}")

    def _get_last_date(self, symbol: str) -> str | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT MAX(date) FROM stock_daily WHERE symbol = ?",
                (symbol,),
            ).fetchone()
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

    # ── 数据同步 ──

    def sync_today_bulk(self) -> int:
        """多进程并行通过 baostock 拉取增量数据（后复权），写入 SQLite。"""
        from datetime import date, timedelta
        from multiprocessing import Pool

        today_str = date.today().strftime("%Y-%m-%d")

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
                start = (date.fromisoformat(last_date) + timedelta(days=1)).strftime("%Y-%m-%d")
            tasks.append((symbol, self._to_baostock_code(symbol), start, today_str))

        if not tasks:
            logger.info("所有股票已是最新，无需更新")
            return 0

        logger.info(f"需要更新 {len(tasks)} 只股票，启动多进程并行拉取...")

        n_workers = min(8, len(tasks))
        chunks = [tasks[i::n_workers] for i in range(n_workers)]

        with Pool(n_workers) as pool:
            batch_results = pool.map(_bs_fetch_batch, chunks)

        all_rows = []
        for batch in batch_results:
            all_rows.extend(batch)

        if not all_rows:
            logger.info("无新数据（可能非交易日）")
            return 0

        df = pd.DataFrame(all_rows, columns=["symbol", "date", "open", "high", "low", "close", "volume", "turnover"])
        for col in ["open", "high", "low", "close", "volume", "turnover"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close"])
        df = df[df["volume"] > 0]

        count = len(df)
        with sqlite3.connect(self.db_path) as conn:
            for d in df["date"].unique().tolist():
                conn.execute("DELETE FROM stock_daily WHERE date = ?", (d,))
            df.to_sql("stock_daily", conn, if_exists="append", index=False, method="multi", chunksize=500)
            conn.commit()

        logger.info(f"sync_today_bulk: 写入 {count} 条数据")
        return count

    def backfill(self, symbols: list[str]) -> None:
        """多进程并行回填历史K线数据。

        分批处理 + 请求间隔控制，避免 baostock API 被封。
        - 每批 200 只股票，9 进程并行拉取
        - 每个 worker 内每只股票间隔 200ms
        - 已入库的自动 skip，中断后可重跑续传
        - SQLite 只在主进程写入，避免并发冲突
        """
        import time
        from datetime import date, timedelta
        from multiprocessing import Pool

        today_str = date.today().strftime("%Y-%m-%d")
        start_time = time.time()
        n_workers = 9
        batch_size = 200

        # 构建任务列表（跳过已完成的）
        tasks = []
        for symbol in symbols:
            last_date = self._get_last_date(symbol)
            if last_date and last_date >= today_str:
                continue
            start = last_date or self.start_date
            if last_date:
                start = (date.fromisoformat(last_date) + timedelta(days=1)).strftime("%Y-%m-%d")
            tasks.append((symbol, self._to_baostock_code(symbol), start, today_str))

        if not tasks:
            logger.info("所有股票已是最新，无需回填")
            return

        total = len(tasks)
        logger.info(f"回填启动：{total} 只股票需处理，{n_workers} 进程并行")

        total_success = 0
        total_failed = 0

        # 分批处理
        for batch_start in range(0, total, batch_size):
            batch = tasks[batch_start:batch_start + batch_size]
            chunks = [batch[i::n_workers] for i in range(n_workers)]
            chunks = [c for c in chunks if c]

            all_rows = []

            with Pool(n_workers) as pool:
                batch_results = pool.map(_bs_fetch_batch, chunks)

            for rows in batch_results:
                if rows:
                    all_rows.extend(rows)

            batch_success = len(set(r[0] for r in all_rows)) if all_rows else 0
            total_success += batch_success
            total_failed += (len(batch) - batch_success)

            # 写入数据库
            if all_rows:
                df = pd.DataFrame(all_rows, columns=["symbol", "date", "open", "high", "low", "close", "volume", "amount"])
                df = df.rename(columns={"amount": "turnover"})
                for col in ["open", "high", "low", "close", "volume", "turnover"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.dropna(subset=["close"])
                df = df[df["volume"] > 0]

                with sqlite3.connect(self.db_path) as conn:
                    for d in df["date"].unique():
                        conn.execute("DELETE FROM stock_daily WHERE date = ?", (d,))
                    df.to_sql("stock_daily", conn, if_exists="append", index=False, method="multi", chunksize=500)

            progress = min(batch_start + batch_size, total)
            elapsed = time.time() - start_time
            rate = progress / elapsed if elapsed > 0 else 0
            logger.info(
                f"回填进度: {progress}/{total} | "
                f"成功 {total_success} 失败 {total_failed} | "
                f"速率 {rate:.1f} 只/秒 | "
                f"已用 {elapsed/60:.0f}分钟"
            )

        # 最终报告
        elapsed = time.time() - start_time
        logger.info(
            f"回填完成 — 成功: {total_success} | "
            f"失败: {total_failed} | "
            f"跳过: {total - total_success - total_failed} | "
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
            candidates = []
            today = date.today()
            one_year_ago = today - timedelta(days=365)

            while rs.next():
                row = rs.get_row_data()
                code_num = row[0].split(".")[1]      # "600000"
                name = row[1]                          # 股票名称
                ipo_str = row[2]                       # 上市日期
                s_type = row[4]                        # "1" = 股票
                status = row[5]                        # "1" = 上市

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
                        f"AND date = (SELECT MAX(date) FROM stock_daily WHERE symbol = stock_daily.symbol)",
                        candidates,
                    ).fetchall()
                    price_ok = {r[0]: r[1] for r in rows}
                    before = len(candidates)
                    candidates = [s for s in candidates if s in price_ok and price_ok[s] >= 2.0]
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
            symbols = []
            while rs.next():
                row = rs.get_row_data()
                code = row[0]           # "sh.600000" or "sz.000001"
                status = row[4]         # "1" = 上市
                stock_type = row[5]     # "1" = 股票
                if status == "1" and stock_type == "1":
                    symbols.append(code.split(".")[1])  # 提取纯数字代码
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
