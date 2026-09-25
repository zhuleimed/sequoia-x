"""海龟交易策略：20日新高突破 + 成交额过亿 + 动量阳线过滤。"""

import pandas as pd

from sequoia_x.core.logger import get_logger
from sequoia_x.strategy.base import BaseStrategy

logger = get_logger(__name__)


class TurtleTradeStrategy(BaseStrategy):
    """海龟交易策略（A股防诱多改良版）。

    选股条件（向量化，严禁 iterrows）：
    1. 突破新高：今日 close > 前20个交易日 high 的最大值
    2. 流动性：今日**成交额** > 1 亿元
    3. 防诱多过滤：今日必须是实体阳线（今日 close > 今日 open），且必须真涨（今日 close > 昨日 close）

    Attributes:
        webhook_key: 路由到 'turtle' 专属飞书机器人。

    ⚠️ 2026-09-25 修复（三处，详见 docs/2026-09_LLM模拟盘买入即跌归因诊断.md）：
      ① **流动性判据用错列**：原为 `turnover > 1e8`，但本库 `turnover` 是**换手率(%)**，
         拿去比 1 亿恒为 False —— 该策略因此**长期不产出任何选股**（存档中每日均为 0 只）。
         改用具**成交额**语义的 `amount` 列（本库约定单位=元），与 docstring 的"成交额过亿"一致。
      ② **排序依赖 baostock 且逐股调用**：原 `_get_market_caps` 对每个候选打一次 baostock
         （修好后日均 163 个候选 = 163 次网络调用/次运行），而 baostock 在部分股票上会卡死。
         改为**本地成交额排序**（成交额本身就是流动性度量，无需外加网络依赖）。
      ③ **排序在换手率缺失时退化为随机**：原实现要求 `turn > 0` 才算市值，而 2026-09-18 起
         `turnover` 缺失一律写 NULL（9 月非空率仅 5.6%），导致市值字典为空、所有分数为 0，
         `_pick_top` 实际按代码顺序取前 5 —— 等于随机选股。改用 amount 后该问题一并消除。

    ⚠️ 上线状态：**已修复但暂不进入实盘候选池**（2026-09-25 用户决策）——
       修复会让日均候选从 0 只变为约 163 只，是**改变线上选股行为的变更**，
       须先经 70 个月全样本回测确认其选股能力，再决定是否重新注册进 main.py。
    """

    webhook_key: str = "turtle"
    display_name: str = "海龟交易法则"
    _MIN_BARS: int = 21  # 至少需要 21 根 K 线（20日窗口 + 当日）
    _MIN_AMOUNT: float = 100_000_000.0  # 流动性门槛：成交额 ≥ 1 亿元

    def run(self) -> list[str]:
        """
        遍历全市场，返回满足海龟突破条件的股票代码列表。
        """
        symbols = self.stock_pool or self.engine.get_local_symbols()
        candidates: list[tuple[str, float]] = []

        for symbol in symbols:
            try:
                df = self.engine.get_ohlcv(symbol)
                if len(df) < self._MIN_BARS:
                    continue

                # 向量化：前20日 high 的滚动最大值（不含当日，shift(1) 后取 rolling(20)）
                df["high_20"] = df["high"].shift(1).rolling(20).max()

                last = df.iloc[-1]
                prev = df.iloc[-2]  # 获取昨日数据，用于对比

                if pd.isna(last["high_20"]):
                    continue

                # 核心条件 1：突破前 20 天最高点
                breakout = last["close"] > last["high_20"]
                # 核心条件 2：流动性过亿（2026-09-25 修复：成交额列是 amount，不是 turnover）
                amt = last.get("amount")
                liquid = pd.notna(amt) and float(amt) >= self._MIN_AMOUNT

                # 【新增防守条件】拒绝郑州煤电式的高开低走大阴线！
                is_yang = last["close"] > last["open"]   # 实体必须是阳线（红柱）
                is_up = last["close"] > prev["close"]    # 必须是真涨，不能是假阳线

                if breakout and liquid and is_yang and is_up:
                    candidates.append((symbol, float(amt)))

            except Exception as exc:
                logger.warning(f"[{symbol}] TurtleTradeStrategy 计算失败：{exc}")
                continue

        # 打分：成交额越大流动性越好（2026-09-25 修复：原用 baostock 逐股查流通市值，
        # 日均 163 个候选 = 163 次网络调用，且换手率缺失时退化为按代码顺序取前 5）
        result = self._pick_top(candidates, self.top_n)

        logger.info(f"TurtleTradeStrategy 选出 {len(result)} 只（候选{len(candidates)}只）")
        return result