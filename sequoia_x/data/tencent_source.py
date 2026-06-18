#!/usr/bin/env python3
"""Tencent/Sina data source module for Sequoia-X - dual-track with baostock."""
import time, logging, requests, pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)

class TencentSource:
    """腾讯数据源 - 获取A股日线数据（前复权）。

    API 说明:
    - 日线: web.ifzq.gtimg.cn (腾讯证券)
    - 实时: qt.gtimg.cn (腾讯行情)
    - 备用: hq.sinajs.cn (新浪财经)
    """

    def __init__(self, request_interval: float = 0.15):
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        })
        self.request_interval = request_interval
        self._last_request = 0.0

    def _rate_limit(self):
        """请求间隔限制."""
        now = time.time()
        elapsed = now - self._last_request
        if elapsed < self.request_interval:
            time.sleep(self.request_interval - elapsed)
        self._last_request = time.time()

    def _tencent_kline(self, code: str, days: int = 5) -> Optional[pd.DataFrame]:
        """从腾讯获取前复权日线.

        Args:
            code: 股票代码，如 'sh600519'
            days: 获取最近N条日线

        Returns:
            DataFrame 或 None
        """
        url = f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,,,{days},qfq"
        try:
            self._rate_limit()
            r = self._session.get(url, timeout=10)
            data = r.json()
            rows = data.get("data", {}).get(code, {}).get("qfqday", [])
            if not rows:
                return None
            df = pd.DataFrame(rows, columns=["date","open","close","high","low","volume"])
            for col in ["open","close","high","low","volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            return df
        except Exception as e:
            logger.debug(f"Tencent kline error for {code}: {e}")
            return None

    def _sina_kline(self, code: str, days: int = 5) -> Optional[pd.DataFrame]:
        """从新浪获取日线数据（备选方案）."""
        # 新浪 240=日线, datalen=N条
        url = f"https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketData.getKLineData?symbol={code}&scale=240&ma=no&datalen={days}"
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

    def get_daily(self, code: str, days: int = 5) -> Optional[pd.DataFrame]:
        """获取单只股票日线数据（腾讯 -> 新浪 自动切换）."""
        df = self._tencent_kline(code, days)
        if df is not None and not df.empty:
            return df
        # 腾讯失败，切到新浪
        logger.debug(f"Tencent failed for {code}, fallback to Sina")
        return self._sina_kline(code, days)

    def get_realtime(self, code: str) -> Optional[dict]:
        """获取实时行情（含市盈率等），返回dict."""
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

    def batch_get_daily(self, codes: list[str], days: int = 5) -> dict[str, pd.DataFrame]:
        """批量获取多只股票日线.

        Args:
            codes: 股票代码列表，如 ['sh600519', 'sz000001']
            days: 获取最近N条

        Returns:
            {code: DataFrame}
        """
        results = {}
        for i, code in enumerate(codes):
            df = self.get_daily(code, days)
            if df is not None:
                results[code] = df
            if (i + 1) % 100 == 0:
                logger.info(f"TencentSource: {i+1}/{len(codes)} done")
        return results

    @staticmethod
    def to_baostock_code(sym: str) -> str:
        """将内部symbol转为腾讯格式."""
        # 内部symbol: '000001' 或 'sh.000001'
        # 腾讯格式: 'sh000001'
        if '.' in sym:
            market, code = sym.split('.')
            return f"{market.lower()}{code}"
        return f"sh{sym}" if sym.startswith('6') else f"sz{sym}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ts = TencentSource()
    code = ts.to_baostock_code("sh.600519")
    df = ts.get_daily(code, 5)
    if df is not None:
        print(f"=== {code} 日线 ({len(df)}条) ===")
        print(df.to_string(index=False))
    else:
        print("FAILED")

    rt = ts.get_realtime("sh600519")
    if rt:
        print(f"\n=== 实时 ===")
        print(f"price={rt['price']} pe={rt['pe']} vol={rt['volume']} amt={rt['amount']}")
