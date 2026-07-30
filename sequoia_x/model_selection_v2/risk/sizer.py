"""仓位调节模块：基于T3波动率预测的仓位缩放。

核心逻辑:
  1. 用 T3(CatBoost)预测每只股票的20日波动率
  2. 高波动股票 → 降仓位 (避免剧烈波动)
  3. 低波动股票 → 加仓位 (稳健收益)
  4. 整体仓位受风险预算约束

使用:
  from sequoia_x.model_selection_v2.risk import VolatilitySizer
  sizer = VolatilitySizer()
  sized = sizer.size_positions(signals, t3_predictions, initial_capital, top_n)
"""

import numpy as np

from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


class VolatilitySizer:
    """基于波动率的仓位调节器。

    参数:
      vol_neutral:       中性波动率水平(年化),默认0.25
      min_weight:         最小仓位权重,默认0.5 (半仓)
      max_weight:         最大仓位权重,默认1.5 (1.5倍仓)
      high_vol_threshold: 高波动阈值(相对中性水平的倍数),默认1.5
      low_vol_threshold:  低波动阈值(相对中性水平的倍数),默认0.5
    """

    def __init__(
        self,
        vol_neutral: float = 0.25,
        min_weight: float = 0.5,
        max_weight: float = 1.5,
        high_vol_threshold: float = 1.5,
        low_vol_threshold: float = 0.5,
    ):
        self.vol_neutral = vol_neutral
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.high_vol_threshold = high_vol_threshold
        self.low_vol_threshold = low_vol_threshold

    def size_positions(
        self,
        signals: list[dict],
        t3_predictions: dict[str, float],
        initial_capital: float = 500_000.0,
        top_n: int = 10,
        market_exposure: float = 1.0,
    ) -> list[dict]:
        """根据波动率调整每只股票的仓位。

        Args:
            signals: 选股信号 [{symbol, rank, ...}, ...]
            t3_predictions: {symbol: predicted_volatility, ...}
            initial_capital: 初始资金
            top_n: 选股数量
            market_exposure: 市场状态建议的仓位比例(来自RiskManager)

        Returns:
            调整后的信号, 新增字段: {weight, position_size, budget}
        """
        if not signals or not t3_predictions:
            # 无T3数据: 等权分配
            base_budget = initial_capital / max(len(signals), 1)
            for sig in signals:
                sig["weight"] = 1.0
                sig["budget"] = round(base_budget * market_exposure, 2)
            return signals

        # 1. 计算每只股票的波动率权重
        vols = []
        for sig in signals:
            vol = t3_predictions.get(sig["symbol"])
            if vol is not None and vol > 0:
                vols.append(vol)
            else:
                vols.append(self.vol_neutral)

        # 2. 波动率 → 仓位权重 (反比)
        # weight = clip(vol_neutral / vol, min_weight, max_weight)
        weights = np.clip(
            self.vol_neutral / np.array(vols, dtype=float),
            self.min_weight,
            self.max_weight,
        )

        # 3. 归一化权重(确保总权重 = top_n)
        total_weight = np.sum(weights)
        if total_weight > 0:
            weights = weights * len(signals) / total_weight

        # 4. 分配资金
        base_budget = initial_capital / top_n
        sized = []
        for sig, w in zip(signals, weights):
            sig["weight"] = round(float(w), 2)
            sig["budget"] = round(float(base_budget * w * market_exposure), 2)
            sig["vol_pred"] = round(float(vols[len(sized)]), 4)
            sized.append(sig)

        # 日志
        high_vol_count = sum(1 for v in vols if v > self.vol_neutral * self.high_vol_threshold)
        low_vol_count = sum(1 for v in vols if v < self.vol_neutral * self.low_vol_threshold)
        logger.debug(
            f"  仓位调节: {len(signals)}只, 高波动{high_vol_count}只(降仓), "
            f"低波动{low_vol_count}只(加仓), 均权={np.mean(weights):.2f}"
        )

        return sized
