"""涨停洗盘策略：昨日涨停后今日放量收阴但不破昨收。"""

import pandas as pd

from sequoia_x.core.logger import get_logger
from sequoia_x.strategy.base import BaseStrategy

logger = get_logger(__name__)


class LimitUpShakeoutStrategy(BaseStrategy):
    """涨停洗盘策略。

    选股条件（向量化，严禁 iterrows）：
    1. 昨日涨停：昨日 close >= 前日 close * 1.095
    2. 今日收阴：今日 close < 今日 open
    3. 今日放量：今日 volume > 昨日 volume * 2.0
    4. 支撑不破：今日 low >= 昨日 close

    Attributes:
        webhook_key: 路由到 'shakeout' 专属飞书机器人。
    """

    webhook_key: str = "shakeout"
    display_name: str = "涨停洗盘"
    _MIN_BARS: int = 3  # 至少需要 3 根 K 线（前日、昨日、今日）

    def run(self) -> list[str]:
        symbols = self.stock_pool or self.engine.get_local_symbols()
        candidates: list[tuple[str, float]] = []

        for symbol in symbols:
            try:
                df = self.engine.get_ohlcv(symbol)
                if len(df) < self._MIN_BARS:
                    continue

                prev2 = df.iloc[-3]
                prev1 = df.iloc[-2]
                today = df.iloc[-1]

                limit_up_yesterday = prev1["close"] >= prev2["close"] * 1.095
                bearish_today = today["close"] < today["open"]
                volume_surge = today["volume"] > prev1["volume"] * 2.0
                support_hold = today["low"] >= prev1["close"]

                if limit_up_yesterday and bearish_today and volume_surge and support_hold:
                    # 分数 = 今日放量倍数 / 今日跌幅（放量越大、跌幅越小说明洗盘越充分）
                    score = (today["volume"] / prev1["volume"]) / max((prev1["close"] - today["close"]) / prev1["close"], 0.001)
                    candidates.append((symbol, score))

            except Exception as exc:
                logger.warning(f"[{symbol}] LimitUpShakeoutStrategy 计算失败：{exc}")
                continue

        result = self._pick_top(candidates, self.top_n)
        logger.info(f"LimitUpShakeoutStrategy 选出 {len(result)} 只（候选{len(candidates)}只）")
        return result
