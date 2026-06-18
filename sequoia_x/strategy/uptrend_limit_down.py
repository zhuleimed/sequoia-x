"""上升趋势跌停策略：趋势中放量跌停，捕捉错杀机会。"""

import pandas as pd

from sequoia_x.core.logger import get_logger
from sequoia_x.strategy.base import BaseStrategy

logger = get_logger(__name__)


class UptrendLimitDownStrategy(BaseStrategy):
    """上升趋势跌停策略。

    选股条件（向量化，严禁 iterrows）：
    1. 处于上升趋势：昨日20日均线 > 昨日60日均线
    2. 放量跌停：今日 close <= 昨日 close * 0.905
                且今日 volume > 20日均量的 2.0 倍

    Attributes:
        webhook_key: 路由到 'limit_down' 专属飞书机器人。
    """

    webhook_key: str = "limit_down"
    display_name: str = "上涨回调"
    _MIN_BARS: int = 60  # 至少需要 60 根 K 线（60日均线）

    def run(self) -> list[str]:
        symbols = self.stock_pool or self.engine.get_local_symbols()
        candidates: list[tuple[str, float]] = []

        for symbol in symbols:
            try:
                df = self.engine.get_ohlcv(symbol)
                if len(df) < self._MIN_BARS:
                    continue

                df["ma20"] = df["close"].rolling(20).mean()
                df["ma60"] = df["close"].rolling(60).mean()
                df["vol_ma20"] = df["volume"].rolling(20).mean()

                prev = df.iloc[-2]
                today = df.iloc[-1]

                if pd.isna(prev["ma20"]) or pd.isna(prev["ma60"]) or pd.isna(today["vol_ma20"]):
                    continue

                uptrend = prev["ma20"] > prev["ma60"]
                limit_down = today["close"] <= prev["close"] * 0.905
                volume_surge = today["volume"] > today["vol_ma20"] * 2.0

                if uptrend and limit_down and volume_surge:
                    # 分数 = 放量倍数 × 跌幅（放量大跌更可能被错杀，反弹潜力大）
                    drop_pct = (prev["close"] - today["close"]) / prev["close"]
                    score = (today["volume"] / today["vol_ma20"]) * drop_pct
                    candidates.append((symbol, score))

            except Exception as exc:
                logger.warning(f"[{symbol}] UptrendLimitDownStrategy 计算失败：{exc}")
                continue

        result = self._pick_top(candidates, self.top_n)
        logger.info(f"UptrendLimitDownStrategy 选出 {len(result)} 只（候选{len(candidates)}只）")
        return result
