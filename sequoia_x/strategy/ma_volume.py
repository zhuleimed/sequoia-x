"""均线+成交量选股策略：5日均线上穿20日均线且成交量放大。"""

import pandas as pd

from sequoia_x.core.logger import get_logger
from sequoia_x.strategy.base import BaseStrategy

logger = get_logger(__name__)


class MaVolumeStrategy(BaseStrategy):
    """均线+成交量选股策略。

    选股条件（全部向量化，严禁 iterrows）：
    1. 5日收盘均线上穿20日收盘均线（金叉）
    2. 当日成交量 > 20日均量的 1.5 倍（放量确认）

    排名分数 = 当日成交量 / 20日均量（放量越猛排名越高）。
    """

    webhook_key: str = "ma_volume"
    display_name: str = "均量线突破"

    def run(self) -> list[str]:
        symbols = self.stock_pool or self.engine.get_local_symbols()
        candidates: list[tuple[str, float]] = []

        for symbol in symbols:
            try:
                df = self.engine.get_ohlcv(symbol)
                if len(df) < 20:
                    continue

                df["ma5"] = df["close"].rolling(5).mean()
                df["ma20"] = df["close"].rolling(20).mean()
                df["vol_ma20"] = df["volume"].rolling(20).mean()

                last = df.iloc[-1]
                prev = df.iloc[-2]

                golden_cross = (
                    prev["ma5"] < prev["ma20"]
                    and last["ma5"] > last["ma20"]
                )
                volume_surge = last["volume"] > last["vol_ma20"] * 1.5

                if golden_cross and volume_surge:
                    # 分数 = 放量倍数（越高越好）
                    score = last["volume"] / last["vol_ma20"]
                    if self.stock_pool is None or symbol in self.stock_pool:
                        candidates.append((symbol, score))

            except Exception as exc:
                logger.warning(f"[{symbol}] 策略计算失败：{exc}")
                continue

        result = self._pick_top(candidates, self.top_n)
        logger.info(f"MaVolumeStrategy 选出 {len(result)} 只（候选{len(candidates)}只）")
        return result
