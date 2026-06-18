"""选股结果保存模块：每次策略运行后将结果保存为 JSON 文件。

被 main.py 调用，输出到 data/results/ 目录下。
WorkBuddy 07:30 自动化任务将读取此文件，
结合通达信 MCP 生成深度荐股分析报告。
"""

import json
import os
from datetime import date
from typing import Optional

from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)

_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "results",
)


def save_strategy_results(
    strategies_results: dict[str, list[str]],
    market_summary: Optional[str] = None,
) -> str:
    """将策略选股结果保存为 JSON 文件。

    Args:
        strategies_results: {策略名: [股票代码列表]} 的字典。
        market_summary: 大盘环境摘要（可选）。

    Returns:
        保存的文件路径。
    """
    os.makedirs(_RESULTS_DIR, exist_ok=True)

    today = date.today()
    filename = f"results_{today.strftime('%Y%m%d')}.json"
    filepath = os.path.join(_RESULTS_DIR, filename)

    # 获取股票名称
    all_codes = set()
    for symbols in strategies_results.values():
        all_codes.update(symbols)

    stock_names = {}
    for code in all_codes:
        try:
            from sequoia_x.analysis.analyst import _get_stock_name

            stock_names[code] = _get_stock_name(code)
        except Exception:
            stock_names[code] = code

    data = {
        "date": today.strftime("%Y-%m-%d"),
        "strategies": strategies_results,
        "stock_names": stock_names,
        "market_summary": market_summary or "",
        "total_stocks": len(all_codes),
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"选股结果已保存: {filepath}")
    return filepath
