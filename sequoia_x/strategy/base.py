"""策略基类模块：定义所有选股策略的抽象接口。"""

from abc import ABC, abstractmethod
from typing import Optional

from sequoia_x.core.config import Settings
from sequoia_x.data.engine import DataEngine


class BaseStrategy(ABC):
    """选股策略抽象基类。

    所有具体策略必须继承此类并实现 run() 方法。
    策略内部应使用 _pick_top() 按分数排序截取前 N 支。

    Attributes:
        webhook_key: 策略对应的推送标识。
        display_name: 策略中文名称，用于推送消息。
        stock_pool: 可选的基础股票池，策略仅在此范围内选股。
        top_n: 选股结果上限，按分数取前 N 支。
    """

    webhook_key: str = "default"
    display_name: str = ""
    top_n: int = 5  # 按分数取前 N 支

    def __init__(
        self,
        engine: DataEngine,
        settings: Settings,
        stock_pool: Optional[list[str]] = None,
    ) -> None:
        self.engine = engine
        self.settings = settings
        self.stock_pool = stock_pool

    # ── 基础工具方法 ──

    def _apply_pool(self, symbols: list[str]) -> list[str]:
        """对选股结果施加基础股票池过滤（不截取）。

        Args:
            symbols: 策略原始选股结果。

        Returns:
            仅保留在基础股票池中的股票。
        """
        if self.stock_pool is not None:
            pool_set = set(self.stock_pool)
            symbols = [s for s in symbols if s in pool_set]
        return symbols

    @staticmethod
    def _pick_top(candidates: list[tuple[str, float]], top_n: int = 5) -> list[str]:
        """按分数降序排序，取前 top_n 支股票。

        每个策略应在 run() 内部构造 (symbol, score) 候选列表，
        最后调用本方法选出得分最高的 N 支。

        Args:
            candidates: (股票代码, 分数) 二元组列表。
            top_n: 保留前 N 支。

        Returns:
            按分数降序排列的股票代码列表。
        """
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in candidates[:top_n]]

    @abstractmethod
    def run(self) -> list[str]:
        ...
