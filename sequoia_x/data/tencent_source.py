#!/usr/bin/env python3
"""多数据源模块 — Tencent/Sina/Baostock 三轨数据源。

数据源层级（三轨平级）：
1. TencentSource — 腾讯证券 API (web.ifzq.gtimg.cn)
2. SinaSource   — 新浪财经 API (quotes.sina.cn)，独立连接，不受 Tencent 影响
3. Baostock     — baostock 全字段含估值（在 sync.py 中直接调用）

所有类返回统一格式 DataFrame: date, open, high, low, close, volume
"""
from __future__ import annotations

import time
import logging
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  工具函数
# ──────────────────────────────────────────────


def to_tencent_code(symbol: str) -> str:
    """将内部 symbol 转为腾讯 API 格式。

    Args:
        symbol: 如 '000001' 或 'sh.000001' 或 'sh600519'。

    Returns:
        腾讯格式，如 'sh000001' 或 'sz000001'。
    """
    # 已带市场前缀
    if '.' in symbol:
        market, code = symbol.split('.')
        return f"{market.lower()}{code}"
    # 纯数字，根据首位推断市场
    prefix = "sh" if symbol.startswith(('5', '6', '9')) else "sz"
    return f"{prefix}{symbol}"


def to_sina_code(symbol: str) -> str:
    """将内部 symbol 转为新浪 API 格式。

    Args:
        symbol: 如 '000001' 或 'sh.000001' 或 'sh600519'。

    Returns:
        新浪格式，如 'sh000001' 或 'sz000001'。
    """
    # 已带市场前缀
    if '.' in symbol:
        market, code = symbol.split('.')
        return f"{market.lower()}{code}"
    # 纯数字
    prefix = "sh" if symbol.startswith(('5', '6', '9')) else "sz"
    return f"{prefix}{symbol}"


# ──────────────────────────────────────────────
#  TencentSource（腾讯数据源，含源追踪）
# ──────────────────────────────────────────────


class TencentSource:
    """腾讯数据源 — 获取 A 股日线数据（前复权）。

    API:
      - 日 K 线: web.ifzq.gtimg.cn（腾讯证券，前复权）
      - 实时行情: qt.gtimg.cn（成交额、市盈率等）

    数据源内部追踪机制（参考 019 项目）：
      - active_source: 当前活跃源（"tencent" / "sina"）
      - source_count: 各源累计成功次数
      - 每 50 次 Sina 请求尝试恢复 Tencent
    """

    def __init__(self, request_interval: float = 0.15):
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        })
        self.request_interval = request_interval
        self._last_request = 0.0
        # 数据源追踪
        self.last_source: str | None = None
        self.source_count: dict[str, int] = {"tencent": 0, "sina": 0}
        self.active_source: str = "tencent"

    def _rate_limit(self) -> None:
        """请求间隔限制。"""
        now = time.time()
        elapsed = now - self._last_request
        if elapsed < self.request_interval:
            time.sleep(self.request_interval - elapsed)
        self._last_request = time.time()

    # ── 腾讯 K 线 ──

    def _tencent_kline(self, code: str, days: int = 5) -> Optional[pd.DataFrame]:
        """从腾讯获取前复权日线。

        Args:
            code: 股票代码，如 'sh600519'。
            days: 获取最近 N 条日线。

        Returns:
            DataFrame 或 None。
        """
        url = (
            f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/"
            f"get?param={code},day,,,{days},qfq"
        )
        try:
            self._rate_limit()
            r = self._session.get(url, timeout=10)
            data = r.json()
            rows = data.get("data", {}).get(code, {}).get("qfqday", [])
            if not rows:
                return None
            df = pd.DataFrame(
                rows, columns=["date", "open", "close", "high", "low", "volume"]
            )
            for col in ["open", "close", "high", "low", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            return df
        except Exception as e:
            logger.debug(f"Tencent kline error for {code}: {e}")
            return None

    # ── 新浪 K 线（TencentSource 内部保留） ──

    def _sina_kline(self, code: str, days: int = 5) -> Optional[pd.DataFrame]:
        """从新浪获取日线（备选方案，与 SinaSource 使用相同 API）。"""
        url = (
            f"https://quotes.sina.cn/cn/api/json_v2.php/"
            f"CN_MarketData.getKLineData?symbol={code}&scale=240&ma=no&datalen={days}"
        )
        try:
            self._rate_limit()
            r = self._session.get(url, timeout=10)
            rows = r.json()
            if not rows or not isinstance(rows, list):
                return None
            records = []
            for row in rows:
                records.append({
                    "date": row["day"][:10],
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                })
            return pd.DataFrame(records)
        except Exception as e:
            logger.debug(f"Sina kline error for {code}: {e}")
            return None

    # ── 实时行情 ──

    def get_realtime(self, code: str) -> Optional[dict]:
        """获取实时行情（含市盈率等）。

        Args:
            code: 腾讯格式代码，如 'sh600519'。

        Returns:
            dict 含 price, close_yest, open_today, volume, amount, pe 等。
        """
        url = f"https://qt.gtimg.cn/q={code}"
        try:
            self._rate_limit()
            r = self._session.get(url, timeout=10)
            text = r.text
            if not text or '="' not in text:
                return None
            parts = text.split('"')[1].split('~')
            if len(parts) < 40:
                return None
            return {
                "name": parts[1],
                "code": parts[2],
                "price": float(parts[3]) if parts[3] else 0,
                "close_yest": float(parts[4]) if parts[4] else 0,
                "open_today": float(parts[5]) if parts[5] else 0,
                "volume": float(parts[6]) if parts[6] else 0,
                "amount": float(parts[37]) if parts[37] else 0,
                "pe": float(parts[39]) if len(parts) > 39 and parts[39] else None,
                "high": float(parts[33]) if len(parts) > 33 and parts[33] else 0,
                "low": float(parts[34]) if len(parts) > 34 and parts[34] else 0,
            }
        except Exception as e:
            logger.debug(f"Tencent realtime error for {code}: {e}")
            return None

    # ── 主入口（带智能源切换） ──

    def get_daily(self, code: str, days: int = 5) -> Optional[pd.DataFrame]:
        """获取单只股票日线数据（源追踪版）。

        智能切换策略（参考 019）：
          - 活跃源为 Tencent → 优先用 Tencent，失败切 Sina
          - 活跃源为 Sina → 每 50 次尝试恢复 Tencent
          - 双源均失败 → 返回 None

        Args:
            code: 腾讯格式代码，如 'sh600519'。
            days: 获取最近 N 条日线。

        Returns:
            DataFrame 或 None。
        """
        if self.active_source == "tencent":
            df = self._tencent_kline(code, days)
            if df is not None and not df.empty:
                self.last_source = "tencent"
                self.source_count["tencent"] += 1
                return df
            # Tencent 失败 → 切 Sina
            logger.debug(f"Tencent 失败（{code}），切换 Sina")
            self.active_source = "sina"
            df = self._sina_kline(code, days)
            if df is not None and not df.empty:
                self.last_source = "sina"
                self.source_count["sina"] += 1
                return df
            return None

        # 活跃源为 Sina → 每 50 次试一次 Tencent 恢复
        if self.source_count["sina"] > 0 and self.source_count["sina"] % 50 == 0:
            df = self._tencent_kline(code, days)
            if df is not None and not df.empty:
                logger.info(f"Tencent 已恢复，切回主源（code={code}）")
                self.active_source = "tencent"
                self.last_source = "tencent"
                self.source_count["tencent"] += 1
                return df

        df = self._sina_kline(code, days)
        if df is not None and not df.empty:
            self.last_source = "sina"
            self.source_count["sina"] += 1
            return df

        # Sina 也失败 → 再试一次 Tencent 碰运气
        df = self._tencent_kline(code, days)
        self.last_source = "tencent" if df is not None else None
        if df is not None:
            self.source_count["tencent"] += 1
        return df

    def batch_get_daily(self, codes: list[str], days: int = 5) -> dict[str, pd.DataFrame]:
        """批量获取多只股票日线。

        Args:
            codes: 代码列表，如 ['sh600519', 'sz000001']。
            days: 获取最近 N 条。

        Returns:
            {code: DataFrame}，失败的不在结果中。
        """
        results: dict[str, pd.DataFrame] = {}
        for i, code in enumerate(codes):
            df = self.get_daily(code, days)
            if df is not None:
                results[code] = df
            if (i + 1) % 100 == 0:
                logger.info(f"TencentSource batch: {i+1}/{len(codes)} 完成")
        return results

    @staticmethod
    def to_baostock_code(sym: str) -> str:
        """将内部 symbol 转为腾讯格式（与 baostock 格式互通）。

        Args:
            sym: '000001' 或 'sh.000001'。

        Returns:
            'sh000001' 或 'sz000001'。
        """
        if '.' in sym:
            market, code = sym.split('.')
            return f"{market.lower()}{code}"
        prefix = "sh" if sym.startswith(('5', '6', '9')) else "sz"
        return f"{prefix}{sym}"


# ──────────────────────────────────────────────
#  SinaSource（独立新浪数据源）
# ──────────────────────────────────────────────


class SinaSource:
    """新浪数据源 — 独立获取 A 股日线数据（不依赖腾讯 API）。

    API: quotes.sina.cn/cn/api/json_v2.php/CN_MarketData.getKLineData

    独立会话连接，不受 TencentSource 状态影响。
    用于 sync_daily 三轨制中作为 Tencent 之后的第二个独立尝试源。

    Example:
        >>> sina = SinaSource()
        >>> df = sina.get_daily("sh600519", 5)
        >>> df[["date", "open", "close", "volume"]]
    """

    def __init__(self, request_interval: float = 0.15):
        """初始化 SinaSource。

        Args:
            request_interval: 请求间隔（秒），默认 0.15s。
        """
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        })
        self.request_interval = request_interval
        self._last_request = 0.0
        self.success_count: int = 0
        self.fail_count: int = 0

    def _rate_limit(self) -> None:
        """请求间隔限制。"""
        now = time.time()
        elapsed = now - self._last_request
        if elapsed < self.request_interval:
            time.sleep(self.request_interval - elapsed)
        self._last_request = time.time()

    def get_daily(self, code: str, days: int = 5) -> Optional[pd.DataFrame]:
        """获取单只股票日线（仅新浪 API，独立连接）。

        Args:
            code: 新浪格式代码，如 'sh600519'。
            days: 获取最近 N 条日线（最大约 800 条）。

        Returns:
            DataFrame 含 date, open, high, low, close, volume，失败返回 None。
        """
        url = (
            f"https://quotes.sina.cn/cn/api/json_v2.php/"
            f"CN_MarketData.getKLineData?symbol={code}&scale=240&ma=no&datalen={days}"
        )
        try:
            self._rate_limit()
            r = self._session.get(url, timeout=10)
            rows = r.json()
            if not rows or not isinstance(rows, list):
                self.fail_count += 1
                return None
            records = []
            for row in rows:
                records.append({
                    "date": row["day"][:10],
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                })
            self.success_count += 1
            return pd.DataFrame(records)
        except Exception as e:
            logger.debug(f"SinaSource error for {code}: {e}")
            self.fail_count += 1
            return None


# ──────────────────────────────────────────────
#  main 测试
# ──────────────────────────────────────────────


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== TencentSource ===")
    ts = TencentSource()
    code = ts.to_baostock_code("sh.600519")
    df = ts.get_daily(code, 5)
    if df is not None:
        print(f"TencentSource {code} ({len(df)} 条):")
        print(df.to_string(index=False))
    else:
        print("TencentSource FAILED")

    rt = ts.get_realtime("sh600519")
    if rt:
        print(f"\n实时: price={rt['price']} pe={rt['pe']} vol={rt['volume']}")

    print("\n=== SinaSource ===")
    sina = SinaSource()
    code = to_sina_code("sh.600519")
    df = sina.get_daily(code, 5)
    if df is not None:
        print(f"SinaSource {code} ({len(df)} 条):")
        print(df.to_string(index=False))
    else:
        print("SinaSource FAILED")
