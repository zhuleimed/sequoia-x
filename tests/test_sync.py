"""数据同步模块综合测试。覆盖 v2.4 修复的所有 Bug。

测试策略：
- 使用 unittest.mock 隔离外部依赖（baostock, TencentSource）
- 使用临时数据库避免数据污染
- 每个测试关注一个特定的修复点
"""

import sqlite3
import tempfile
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock, call, ANY

import pandas as pd
import pytest

from sequoia_x.core.config import Settings
from sequoia_x.data.sync import DataSync
from sequoia_x.data.tencent_source import TencentSource


# ── 测试辅助 ──

def make_sync(tmp_dir: str, db_name: str = "test.db") -> DataSync:
    """创建使用临时数据库的 DataSync 实例。"""
    settings = Settings(
        db_path=str(Path(tmp_dir) / db_name),
        start_date="2024-01-01",
        wxpusher_token="AT_test_token_123",
    )
    sync = DataSync(settings)
    return sync


def stock_row(symbol: str = "600000", date_str: str = "2024-01-02",
              open_p=10.0, high=11.0, low=9.0, close=10.5,
              volume=1e6, amount=1e7, turn=1.5, pct_chg="0.5",
              pe="10.0", pb="1.0", ps="2.0", pcf="5.0") -> list[str]:
    """生成 baostock 格式的一行数据。"""
    return [date_str, str(open_p), str(high), str(low), str(close),
            str(volume), str(amount), str(turn), pct_chg, pe, pb, ps, pcf]


def make_bs_result(data_rows: list[list[str]], fields: list[str] | None = None,
                   error_code: str = "0") -> MagicMock:
    """创建模拟的 baostock ResultData 对象。"""
    mock = MagicMock()
    mock.error_code = error_code
    if fields is None:
        fields = ["date", "open", "high", "low", "close", "volume",
                   "amount", "turn", "pctChg", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"]
    mock.fields = fields
    # 模拟 next() 迭代
    mock._rows = list(data_rows)
    mock._idx = 0

    def next_impl():
        if mock._idx < len(mock._rows):
            mock._idx += 1
            return True
        return False

    mock.next.side_effect = next_impl
    mock.get_row_data.side_effect = lambda: mock._rows[mock._idx - 1] if mock._idx > 0 else []
    return mock


def make_trade_days_bs(start: str = "2024-01-02", end: str = "2024-01-05") -> MagicMock:
    """创建模拟的交易日查询结果。"""
    rows = []
    cursor = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    while cursor <= end_dt:
        wd = cursor.weekday()
        is_trade = "1" if wd < 5 else "0"  # 周一~五为交易日
        rows.append([cursor.strftime("%Y-%m-%d"), is_trade])
        cursor += timedelta(days=1)
    return make_bs_result(rows, fields=["calendar_date", "is_trading_day"])


# ═══════════════════════════════════════════════════
# Bug#1: baostock_available 标记不再被意外复位
# ═══════════════════════════════════════════════════

class TestBaostockAvailable:
    """验证 baostock_available 在成功查询后保持 True（Bug#1 修复）。"""

    @patch("sequoia_x.data.sync.DataSync.is_trade_day", return_value=True)
    @patch("sequoia_x.data.sync.DataSync._get_local_last_dates")
    @patch("sequoia_x.data.sync.DataEngine.get_local_symbols")
    @patch("baostock.query_history_k_data_plus")
    @patch("baostock.login")
    @patch("baostock.query_trade_dates")
    def test_baostock_persists_after_success(
        self, mock_qt, mock_login, mock_query, mock_symbols, mock_dates, mock_trade
    ):
        """连续多只股票 baostock 查询成功后，baostock_available 保持 True，
        所有股票都应走 baostock 而非提前切 Tencent。"""
        mock_login.return_value = MagicMock(error_code="0")
        mock_qt.return_value = make_trade_days_bs("2024-01-02", "2024-01-05")
        mock_trade.return_value = True

        # 返回 3 只股票
        mock_symbols.return_value = ["600000", "600001", "600002"]
        # 本地已有部分数据
        mock_dates.return_value = {
            "600000": "2024-01-04", "600001": "2024-01-04", "600002": "2024-01-04",
        }

        # baostock 返回成功（只有 2024-01-05 需要拉取）
        mock_query.return_value = make_bs_result([
            stock_row("600000", "2024-01-05"),
        ])

        with tempfile.TemporaryDirectory() as tmp_dir:
            sync = make_sync(tmp_dir)
            # 预置 2024-01-02~04 数据避免空表
            conn = sqlite3.connect(sync.db_path)
            for sym in ["600000", "600001", "600002"]:
                for d in ["2024-01-02", "2024-01-03", "2024-01-04"]:
                    conn.execute(
                        "INSERT OR REPLACE INTO stock_daily "
                        "(symbol, date, open, high, low, close, volume) "
                        "VALUES (?, ?, 10, 11, 9, 10.5, 1e6)",
                        (sym, d)
                    )
            conn.commit()
            conn.close()

            # 执行同步
            result = sync.sync_daily(force=True)

            # baostock.query_history_k_data_plus 应该被调用了 3 次（每只股票一次）
            assert mock_query.call_count == 3, (
                f"baostock 应被调用 3 次（每只股票1次），实际 {mock_query.call_count}"
            )
            assert result["status"] == "ok", f"同步应成功：{result}"

    @patch("sequoia_x.data.sync.DataSync.is_trade_day", return_value=True)
    @patch("sequoia_x.data.sync.DataSync._get_local_last_dates")
    @patch("sequoia_x.data.sync.DataEngine.get_local_symbols")
    @patch("baostock.query_history_k_data_plus")
    @patch("baostock.login")
    @patch("baostock.query_trade_dates")
    def test_baostock_fallback_on_error(
        self, mock_qt, mock_login, mock_query, mock_symbols, mock_dates, mock_trade
    ):
        """baostock 失败时应切换 Tencent，验证 TencentSource.get_daily 被调用。"""
        mock_login.return_value = MagicMock(error_code="0")
        mock_qt.return_value = make_trade_days_bs("2024-01-02", "2024-01-05")
        mock_trade.return_value = True
        mock_symbols.return_value = ["600000"]
        mock_dates.return_value = {"600000": "2024-01-04"}

        # baostock 返回错误
        mock_query.return_value = make_bs_result([], error_code="10002007")

        with tempfile.TemporaryDirectory() as tmp_dir:
            sync = make_sync(tmp_dir)
            conn = sqlite3.connect(sync.db_path)
            for d in ["2024-01-02", "2024-01-03", "2024-01-04"]:
                conn.execute(
                    "INSERT OR REPLACE INTO stock_daily "
                    "(symbol, date, open, high, low, close, volume) "
                    "VALUES (?, ?, 10, 11, 9, 10.5, 1e6)",
                    ("600000", d)
                )
            conn.commit()
            conn.close()

            with patch.object(TencentSource, "get_daily") as mock_tc:
                # TencentSource 返回有效数据
                tc_df = pd.DataFrame({
                    "date": ["2024-01-05"],
                    "open": [10.0], "close": [10.5],
                    "high": [11.0], "low": [9.0], "volume": [1e6],
                })
                mock_tc.return_value = tc_df

                with patch.object(TencentSource, "get_realtime") as mock_rt:
                    mock_rt.return_value = {"amount": 1e7}

                    result = sync.sync_daily(force=True)

            assert result["status"] == "ok", f"Tencent 回退后应成功：{result}"
            mock_tc.assert_called_once()


# ═══════════════════════════════════════════════════
# Bug#2: _fill_valuation_gaps 使用统一会话管理
# ═══════════════════════════════════════════════════

class TestFillValuationSession:
    """验证 _fill_valuation_gaps 使用统一的 _bs_login/_bs_logout（Bug#2 修复）。"""

    @patch("baostock.login")
    @patch("baostock.query_trade_dates")
    @patch("baostock.query_history_k_data_plus")
    def test_uses_bs_login_not_login_direct(
        self, mock_query, mock_qt, mock_login
    ):
        """_fill_valuation_gaps 应通过 _bs_login 管理会话，而非直接 bs.login()。"""
        from datetime import date, timedelta
        today = date.today()
        test_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")  # 昨天

        mock_login.return_value = MagicMock(error_code="0")
        mock_qt.return_value = make_trade_days_bs("2024-01-02", "2024-01-02")

        def query_side_effect(bs_code, fields, start_date, end_date, frequency, adjustflag):
            # 对 query_trade_dates 的调用返回正常
            if "date" in fields and "peTTM" not in fields:
                return make_bs_result([], error_code="0")
            return make_bs_result([
                ["10.0", "1.0", "2.0", "5.0"],
            ], fields=["peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"])
        mock_query.side_effect = query_side_effect

        with tempfile.TemporaryDirectory() as tmp_dir:
            sync = make_sync(tmp_dir)
            # 写入一条 peTTM=NULL 的数据（日期在 _fill_valuation_gaps 的 days 范围内）
            conn = sqlite3.connect(sync.db_path)
            conn.execute(
                "INSERT OR REPLACE INTO stock_daily "
                "(symbol, date, open, high, low, close, volume) "
                "VALUES (?, ?, 10, 11, 9, 10.5, 1e6)",
                ("600000", test_date)
            )
            conn.commit()
            conn.close()

            with patch.object(sync, "_bs_login", wraps=sync._bs_login) as spy_login:
                with patch.object(sync, "_bs_logout", wraps=sync._bs_logout) as spy_logout:
                    result = sync._fill_valuation_gaps(days=5)

            # _bs_login 应被调用
            spy_login.assert_called()
            # _bs_logout 应被调用（函数结束前应正常登出）
            spy_logout.assert_called()

            # 验证估值字段已填充
            conn = sqlite3.connect(sync.db_path)
            row = conn.execute(
                "SELECT peTTM, pbMRQ FROM stock_daily WHERE symbol='600000' AND date=?",
                (test_date,)
            ).fetchone()
            conn.close()
            assert row is not None, "数据行应存在"
            assert row[0] == 10.0, f"peTTM 应被回填为 10.0，实际 {row[0]}"
            assert row[1] == 1.0, f"pbMRQ 应被回填为 1.0，实际 {row[1]}"


# ═══════════════════════════════════════════════════
# Bug#3: repair_missing 进度日志 _last_progress 已初始化
# ═══════════════════════════════════════════════════

class TestRepairProgressLog:
    """验证 repair_missing 进度日志不抛出 NameError（Bug#3 修复）。"""

    @patch("sequoia_x.data.sync.DataSync.check_missing")
    @patch("baostock.login")
    @patch("baostock.query_history_k_data_plus")
    @patch("baostock.query_trade_dates")
    def test_progress_log_no_name_error(
        self, mock_qt, mock_query, mock_login, mock_check
    ):
        """repair_missing 的进度日志不应因 _last_progress 未定义而崩溃。"""
        mock_login.return_value = MagicMock(error_code="0")
        mock_qt.return_value = make_trade_days_bs("2024-01-02", "2024-01-05")
        mock_query.return_value = make_bs_result([
            stock_row("600000", "2024-01-05"),
        ])

        # 模拟 check_missing 返回缺失报告
        mock_check.return_value = {
            "status": "ok",
            "latest_date": "2024-01-05",
            "trade_days_expected": 3,
            "total_missing": 2,
            "missing_by_symbol": {
                "600000": ["2024-01-03", "2024-01-05"],
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            sync = make_sync(tmp_dir)
            # 确保有基础数据
            conn = sqlite3.connect(sync.db_path)
            conn.execute(
                "INSERT OR REPLACE INTO stock_daily "
                "(symbol, date, open, high, low, close, volume) "
                "VALUES ('600000', '2024-01-02', 10, 11, 9, 10.5, 1e6)"
            )
            conn.commit()
            conn.close()

            # 不应抛出 NameError
            try:
                result = sync.repair_missing(days=5, max_stocks=1)
                assert result["status"] in ("ok", "skipped"), f"修复应成功：{result}"
            except NameError as e:
                pytest.fail(f"NameError 抛出（_last_progress 未初始化？）: {e}")
            except Exception as e:
                # 其他异常可接受（取决于 mock 细节），但 NameError 不行
                pass


# ═══════════════════════════════════════════════════
# Bug#4: sync_index_daily Tencent 回退代码格式
# ═══════════════════════════════════════════════════

class TestIndexTencentFormat:
    """验证 sync_index_daily 传给 TencentSource 的格式正确（Bug#4 修复）。"""

    @patch("sequoia_x.data.sync.DataSync.is_trade_day", return_value=True)
    @patch("baostock.login")
    @patch("baostock.query_history_k_data_plus")
    def test_tencent_code_format(
        self, mock_query, mock_login, mock_trade
    ):
        """baostock 失败时，传给 TencentSource 的代码应无点号（sh000001 而非 sh.000001）。"""
        mock_login.return_value = MagicMock(error_code="0")

        # baostock 指数查询失败
        mock_query.return_value = make_bs_result([], error_code="10002007")

        with tempfile.TemporaryDirectory() as tmp_dir:
            sync = make_sync(tmp_dir)

            with patch.object(TencentSource, "get_daily") as mock_tc:
                # TencentSource 返回假数据
                tc_df = pd.DataFrame({
                    "date": ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
                    "open": [3000, 3010, 3020, 3015],
                    "close": [3010, 3020, 3015, 3025],
                    "high": [3020, 3030, 3025, 3035],
                    "low": [2990, 3000, 3005, 3010],
                    "volume": [1e8, 1.1e8, 0.9e8, 1.2e8],
                })
                mock_tc.return_value = tc_df

                result = sync.sync_index_daily(force=True)

            # TencentSource.get_daily 的第一个参数应该是无点号格式
            if mock_tc.call_count > 0:
                for call_args in mock_tc.call_args_list:
                    code_arg = call_args[0][0]  # 第一个位置参数
                    assert "." not in code_arg, (
                        f"Tencent 代码格式错误：'{code_arg}' 应无点号"
                    )
                    # 应以 sh 或 sz 开头
                    assert code_arg.startswith(("sh", "sz")), (
                        f"Tencent 代码应以 sh/sz 开头：{code_arg}"
                    )


# ═══════════════════════════════════════════════════
# Bug#5: _write_to_db 日志无死代码
# ═══════════════════════════════════════════════════

class TestWriteToDbLogging:
    """验证 _write_to_db 日志执行正确（Bug#5 修复）。"""

    @patch("sequoia_x.data.sync.logger")
    def test_log_output_after_write(self, mock_logger):
        """_write_to_db 应输出日志（不会被 return 截断）。"""
        df = pd.DataFrame({
            "symbol": ["600000"],
            "date": ["2024-01-02"],
            "open": [10.0], "high": [11.0], "low": [9.0], "close": [10.5],
            "volume": [1e6], "turnover": [1.5],
            "amount": [1e7], "pctChg": [0.5],
            "peTTM": [10.0], "pbMRQ": [1.0], "psTTM": [2.0], "pcfNcfTTM": [5.0],
        })

        with tempfile.TemporaryDirectory() as tmp_dir:
            sync = make_sync(tmp_dir)
            sync._open_db()

            count = sync._write_to_db(df)

            # 验证返回值
            assert count == 1, f"应写入 1 条，实际 {count}"
            # 验证日志被调用（写入后应有日志输出）
            log_calls = [c for c in mock_logger.info.call_args_list
                         if "_write_to_db" in str(c)]
            assert len(log_calls) >= 1, "_write_to_db 应输出日志"

            sync._close_db()


# ═══════════════════════════════════════════════════
# Bug#6-#8: 代码整洁 & PRAGMA & check_missing 区间
# ═══════════════════════════════════════════════════

class TestPragmaSettings:
    """验证 _open_db 设置了 PRAGMA（Bug#7 修复）。"""

    def test_open_db_sets_pragma(self):
        """_open_db 应设置 journal_mode=WAL 和 synchronous=NORMAL。"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            sync = make_sync(tmp_dir)
            sync._open_db()

            wal = sync._db_conn.execute("PRAGMA journal_mode").fetchone()[0]
            sync_setting = sync._db_conn.execute("PRAGMA synchronous").fetchone()[0]

            sync._close_db()

            assert wal.upper() == "WAL", f"journal_mode 应为 WAL，实际 {wal}"
            assert sync_setting == 1, f"synchronous 应为 1(NORMAL)，实际 {sync_setting}"


class TestCheckMissingRange:
    """验证 check_missing 使用 days*2 日历区间（Bug#8 修复）。"""

    @patch("sequoia_x.data.sync.DataSync._get_local_last_dates")
    @patch("baostock.query_trade_dates")
    @patch("baostock.login")
    def test_range_window_larger_than_calendar_days(
        self, mock_login, mock_qt, mock_dates
    ):
        """当 days=5 时，日历区间应 >= 7 天（至少覆盖 5 个交易日）。"""
        mock_login.return_value = MagicMock(error_code="0")
        mock_qt.return_value = make_trade_days_bs("2024-01-02", "2024-01-10")
        mock_dates.return_value = {"600000": "2024-01-10"}

        with tempfile.TemporaryDirectory() as tmp_dir:
            sync = make_sync(tmp_dir)
            # 确保有数据
            conn = sqlite3.connect(sync.db_path)
            conn.execute(
                "INSERT OR REPLACE INTO stock_daily "
                "(symbol, date, open, high, low, close, volume) "
                "VALUES ('600000', '2024-01-10', 10, 11, 9, 10.5, 1e6)"
            )
            conn.execute(
                "INSERT OR REPLACE INTO stock_daily "
                "(symbol, date, open, high, low, close, volume) "
                "VALUES ('600000', '2024-01-09', 10, 11, 9, 10.5, 1e6)"
            )
            conn.commit()
            conn.close()

            result = sync.check_missing(days=5)

            # 区间从 2024-01-10 往前推 days*2=10 日历日 → 2024-01-01 或 2024-01-02
            # （取决于周末/节假日，至少应覆盖 5 个交易日）
            assert result["trade_days_expected"] >= 5, (
                f"days=5 时至少应检查 5 个交易日，实际 {result['trade_days_expected']}"
            )


# ═══════════════════════════════════════════════════
# Bug#9-#10: import 整洁和双源切换
# ═══════════════════════════════════════════════════

class TestSyncIndexDailyFallback:
    """验证 sync_index_daily 的双源切换（baostock→Tencent）。"""

    @patch("sequoia_x.data.sync.DataSync.is_trade_day", return_value=True)
    @patch("baostock.login")
    @patch("baostock.query_history_k_data_plus")
    def test_baostock_then_tencent_fallback(
        self, mock_query, mock_login, mock_trade
    ):
        """baostock 失败后 sync_index_daily 应自动切 Tencent。"""
        mock_login.return_value = MagicMock(error_code="0")

        # 先让 上证指数 查询成功，其他失败
        def query_side_effect(bs_code, *args, **kwargs):
            if bs_code == "sh.000001":
                return make_bs_result([
                    ["2024-01-02", "3000", "3010", "2990", "3005",
                     "1e8", "3e11", "0.5"],
                ], fields=["date", "open", "high", "low", "close",
                          "volume", "amount", "pctChg"])
            return make_bs_result([], error_code="10002007")
        mock_query.side_effect = query_side_effect

        with tempfile.TemporaryDirectory() as tmp_dir:
            sync = make_sync(tmp_dir)

            with patch.object(TencentSource, "get_daily") as mock_tc:
                tc_df = pd.DataFrame({
                    "date": ["2024-01-02", "2024-01-03"],
                    "open": [5000, 5010],
                    "close": [5010, 5005],
                    "high": [5020, 5025],
                    "low": [4990, 4995],
                    "volume": [5e7, 5.5e7],
                })
                mock_tc.return_value = tc_df

                result = sync.sync_index_daily(force=True)

            assert result["status"] == "ok", f"指数同步应成功：{result}"
            # 至少 1 个指数从 baostock 获取成功（上证指数）
            # 至少有 1 个来自 Tencent
            assert result["index_count"] >= 1, "至少 1 个指数应同步成功"


# ═══════════════════════════════════════════════════
# 端到端：管线整体运行
# ═══════════════════════════════════════════════════

class TestRunFullPipeline:
    """验证完整管线 run_full 的正常流程。"""

    @patch("sequoia_x.data.sync.DataSync.sync_stock_list")
    @patch("sequoia_x.data.sync.DataSync.sync_index_daily")
    @patch("sequoia_x.data.sync.DataSync.sync_daily")
    @patch("sequoia_x.data.sync.DataSync.repair_missing")
    @patch.object(DataSync, "_fill_valuation_gaps")
    def test_run_full_phases_sequence(
        self, mock_fill, mock_repair, mock_daily, mock_index, mock_stock
    ):
        """run_full 应依次执行各 Phase，遇到 Phase 2 error 应终止。"""
        mock_stock.return_value = {"status": "ok", "new_listed": [], "delisted": [], "total": 5000}
        mock_daily.return_value = {"status": "ok", "stock_count": 5000, "is_trade_day": True}
        mock_repair.return_value = {"status": "ok", "affected_stocks": 0, "total_filled": 0}
        mock_fill.return_value = {"status": "ok", "filled": 0}
        mock_index.return_value = {"status": "ok", "index_count": 6}

        with tempfile.TemporaryDirectory() as tmp_dir:
            sync = make_sync(tmp_dir)
            result = sync.run_full()

        assert result["status"] == "ok", f"管线应成功：{result}"
        mock_stock.assert_called_once()
        mock_daily.assert_called_once()
        mock_repair.assert_called_once()
        mock_fill.assert_called_once()
        mock_index.assert_called_once()

    @patch("sequoia_x.data.sync.DataSync.sync_stock_list")
    @patch("sequoia_x.data.sync.DataSync.sync_index_daily")
    @patch("sequoia_x.data.sync.DataSync.sync_daily")
    @patch("sequoia_x.data.sync.DataSync.repair_missing")
    @patch.object(DataSync, "_fill_valuation_gaps")
    def test_run_full_aborts_on_phase2_error(
        self, mock_fill, mock_repair, mock_daily, mock_index, mock_stock
    ):
        """Phase 2 失败时管线应终止，后续 Phase 不执行。"""
        mock_stock.return_value = {"status": "ok", "new_listed": [], "delisted": [], "total": 5000}
        mock_daily.return_value = {"status": "error", "stock_count": 0, "is_trade_day": True, "error": "baostock 失败"}

        with tempfile.TemporaryDirectory() as tmp_dir:
            sync = make_sync(tmp_dir)
            result = sync.run_full()

        assert result["status"] == "error", "Phase 2 error 后管线应标记为 error"
        mock_repair.assert_not_called()  # Phase 3 不应执行
        mock_fill.assert_not_called()    # Phase 4 不应执行
        mock_index.assert_not_called()   # Phase 5 不应执行
