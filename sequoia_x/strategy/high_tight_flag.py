"""高旗形整理策略：强动量后极度收敛缩量。"""

import pandas as pd

from sequoia_x.core.logger import get_logger
from sequoia_x.strategy.base import BaseStrategy

logger = get_logger(__name__)


class HighTightFlagStrategy(BaseStrategy):
    """高旗形整理策略。

    选股条件（向量化，严禁 iterrows）：
    1. 强动量：过去40天区间最高价 / 区间最低价 > 1.6（涨幅超60%）
    2. 极度收敛：最近10天区间最高价 / 区间最低价 < 1.15（振幅低于15%）
    3. 缩量：今日 volume < 过去20日 volume 均值的 0.6 倍

    排名分数 = 动量倍数 / 收敛幅度（强动量+窄幅整理 = 爆发潜力高）。
    """

    webhook_key: str = "flag"
    display_name: str = "高紧旗形突破"
    _MIN_BARS: int = 40

    def run(self) -> list[str]:
        symbols = self.stock_pool or self.engine.get_local_symbols()
        candidates: list[tuple[str, float]] = []

        for symbol in symbols:
            try:
                df = self.engine.get_ohlcv(symbol)
                if len(df) < self._MIN_BARS:
                    continue

                tail40 = df.tail(40)
                tail10 = df.tail(10)

                high40 = tail40["high"].max()
                low40 = tail40["low"].min()
                high10 = tail10["high"].max()
                low10 = tail10["low"].min()

                if low40 == 0 or low10 == 0:
                    continue

                momentum = high40 / low40 > 1.6
                consolidation = high10 / low10 < 1.15
                high_level = low10 >= high40 * 0.8
                vol_ma20 = df["volume"].iloc[-21:-1].mean()
                shrink = df["volume"].iloc[-1] < vol_ma20 * 0.6

                if momentum and consolidation and high_level and shrink:
                    # 分数 = 动量倍数 / 收敛幅度（值越高越好）
                    score = (high40 / low40) / (high10 / low10)
                    candidates.append((symbol, score))

            except Exception as exc:
                logger.warning(f"[{symbol}] HighTightFlagStrategy 计算失败：{exc}")
                continue

        result = self._pick_top(candidates, self.top_n)
        logger.info(f"HighTightFlagStrategy 选出 {len(result)} 只（候选{len(candidates)}只）")
        return result
