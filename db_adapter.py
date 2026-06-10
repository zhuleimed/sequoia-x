"""
统一数据访问适配器 — 从 Sequoia-X SQLite 数据库读取股票/ETF/指数日线数据。

供 015_indicator_scanner 和 016_etf_lstm_predict 项目使用，
替代原来的 CSV 文件读取和 baostock 直连模式。

用法:
    from data.db_adapter import get_stock_data, get_index_data
    df = get_stock_data("000001", start_date="2024-01-01")
    idx = get_index_data("sh.000300", start_date="2024-01-01")
"""

import sqlite3
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

# Sequoia-X 数据库路径
DB_PATH = "/public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x/data/sequoia_v2.db"


def _ensure_db() -> Path:
    """确保数据库文件存在。"""
    p = Path(DB_PATH)
    if not p.exists():
        raise FileNotFoundError(f"数据库文件不存在: {DB_PATH}")
    return p


def get_stock_data(
    stock_code: str,
    start_date: str = "",
    end_date: str = "",
    min_days: int = 240,
) -> Optional[pd.DataFrame]:
    """
    从 Sequoia-X 数据库获取单只股票日线数据。

    Parameters
    ----------
    stock_code : str
        6 位股票代码，如 "000001"。
    start_date : str
        开始日期 "YYYY-MM-DD"。为空则返回全部数据。
    end_date : str
        结束日期。为空则返回到最新。
    min_days : int
        最少交易天数（次新股过滤）。

    Returns
    -------
    pd.DataFrame 或 None
        包含列：date, open, high, low, close, volume, amount, pct_chg,
               cumulative_returns, stock_code
    """
    _ensure_db()

    query = """
        SELECT date, open, high, low, close, volume, amount, pctChg as pct_chg
        FROM stock_daily
        WHERE symbol = ?
    """
    params: list = [stock_code]

    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)

    query += " ORDER BY date ASC"

    try:
        conn = sqlite3.connect(str(DB_PATH))
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
    except Exception as e:
        print(f"  [db_adapter] {stock_code} 查询失败: {e}")
        return None

    if df.empty:
        print(f"  [db_adapter] {stock_code}: 无数据")
        return None

    # 日期转换
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # 数量过滤
    if len(df) < min_days:
        print(f"  [db_adapter] {stock_code}: 数据不足 ({len(df)} < {min_days})")
        return None

    # 计算收益率
    df["cumulative_returns"] = (1 + df["close"].pct_change()).cumprod()
    df.loc[0, "cumulative_returns"] = 1.0
    df["stock_code"] = stock_code

    return df.reset_index(drop=True)


def get_index_data(
    index_code: str,
    start_date: str = "",
    end_date: str = "",
) -> Optional[pd.DataFrame]:
    """
    从 Sequoia-X 数据库获取指数日线数据。

    Parameters
    ----------
    index_code : str
        baostock 指数代码，如 "sh.000300"。

    Returns
    -------
    pd.DataFrame 或 None
        包含列：date, open, high, low, close, volume, amount,
               benchmark_returns, benchmark_cumulative_returns
    """
    _ensure_db()

    query = """
        SELECT date, open, high, low, close, volume, amount, pctChg as pct_chg
        FROM index_daily
        WHERE symbol = ?
    """
    params: list = [index_code]

    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)

    query += " ORDER BY date ASC"

    try:
        conn = sqlite3.connect(str(DB_PATH))
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
    except Exception as e:
        print(f"  [db_adapter] {index_code} 查询失败: {e}")
        return None

    if df.empty:
        print(f"  [db_adapter] {index_code}: 无数据（可能需补充指数同步）")
        return None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    df["benchmark_returns"] = df["close"].pct_change()
    df["benchmark_cumulative_returns"] = (1 + df["benchmark_returns"]).cumprod()
    df.loc[0, "benchmark_cumulative_returns"] = 1.0

    return df.reset_index(drop=True)


def get_etf_data(
    etf_code: str,
    start_date: str = "",
    end_date: str = "",
    min_days: int = 50,
) -> Optional[pd.DataFrame]:
    """
    从 Sequoia-X 数据库获取 ETF 日线数据。

    ETF 存储在 stock_daily 表中（symbol 为 6 位代码），
    此函数封装 stock_daily 查询，返回格式与 get_stock_data 一致。

    Parameters
    ----------
    etf_code : str
        6 位 ETF 代码，如 "510300"。

    Returns
    -------
    pd.DataFrame 或 None
        包含列：date, open, high, low, close, volume, amount, pct_chg
    """
    df = get_stock_data(etf_code, start_date, end_date, min_days)
    if df is not None:
        # 移除 cumulative_returns（由调用方计算）
        if "cumulative_returns" in df.columns:
            df = df.drop(columns=["cumulative_returns"])
    return df


def get_stock_codes() -> list:
    """获取数据库中所有股票代码列表。"""
    _ensure_db()
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT symbol FROM stock_daily ORDER BY symbol")
        codes = [r[0] for r in cur.fetchall()]
        conn.close()
        return codes
    except Exception:
        return []
