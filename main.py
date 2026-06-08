"""Sequoia-X V2 主程序入口。

两种运行模式：
  python main.py               # 日常模式：增量补数据 + 跑策略 + LLM分析 + WxPusher推送
  python main.py --backfill    # 回填模式：baostock 拉全市场历史K线（首次/补数据用，约12分钟）
"""

import argparse
import sys
from dotenv import load_dotenv
load_dotenv()

from datetime import date

import socket
socket.setdefaulttimeout(10.0)

from sequoia_x.core.config import get_settings
from sequoia_x.core.logger import get_logger
from sequoia_x.data.engine import DataEngine
from sequoia_x.notify.wxpusher import WxPusherNotifier
from sequoia_x.strategy.base import BaseStrategy
from sequoia_x.strategy.high_tight_flag import HighTightFlagStrategy
from sequoia_x.strategy.limit_up_shakeout import LimitUpShakeoutStrategy
from sequoia_x.strategy.ma_volume import MaVolumeStrategy
from sequoia_x.strategy.turtle_trade import TurtleTradeStrategy
from sequoia_x.strategy.uptrend_limit_down import UptrendLimitDownStrategy
from sequoia_x.strategy.rps_breakout import RpsBreakoutStrategy
from sequoia_x.strategy.private_placement import PrivatePlacementStrategy
from sequoia_x.analysis.analyst import MarketAnalyst


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequoia-X V2 选股系统")
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="回填模式：通过 baostock 拉取全市场历史 K 线",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="跳过 LLM 多维度分析（仅策略选股+推送）",
    )
    args = parser.parse_args()

    try:
        # 1. 初始化配置
        settings = get_settings()

        # 2. 初始化日志
        logger = get_logger(__name__)
        logger.info("Sequoia-X V2 启动")

        # 3. 初始化数据引擎
        engine = DataEngine(settings)

        if args.backfill:
            # ── 回填模式 ──
            logger.info("进入回填模式...")
            all_symbols = engine.get_all_symbols()
            engine.backfill(all_symbols)
            logger.info("Sequoia-X V2 回填模式运行完成")
            return

        # ── 日常模式 ──
        logger.info("开始拉取最新快照...")
        count = engine.sync_today_bulk()
        logger.info(f"快照同步完成，写入 {count} 只股票")

        # 4. 基础股票池
        logger.info("获取基础股票池...")
        base_pool = engine.get_base_stock_pool()
        logger.info(f"基础股票池共 {len(base_pool)} 只股票")

        # 5. 策略选股
        strategies: list[BaseStrategy] = [
            MaVolumeStrategy(engine=engine, settings=settings, stock_pool=base_pool),
            TurtleTradeStrategy(engine=engine, settings=settings, stock_pool=base_pool),
            HighTightFlagStrategy(engine=engine, settings=settings, stock_pool=base_pool),
            LimitUpShakeoutStrategy(engine=engine, settings=settings, stock_pool=base_pool),
            UptrendLimitDownStrategy(engine=engine, settings=settings, stock_pool=base_pool),
            RpsBreakoutStrategy(engine=engine, settings=settings, stock_pool=base_pool),
            PrivatePlacementStrategy(engine=engine, settings=settings, stock_pool=base_pool),
        ]

        notifier = WxPusherNotifier(settings)

        # 收集各策略选股结果（内部使用，不单独推送）
        strategies_results: dict[str, list[str]] = {}

        for strategy in strategies:
            strategy_name = getattr(strategy, "display_name", type(strategy).__name__)
            logger.info(f"执行策略：{strategy_name}")

            selected: list[str] = strategy.run()

            # 日志记录策略结果，但不单独推送（由 LLM 统一研判后一次推送）
            if selected:
                logger.info(f"{strategy_name} 选出 {len(selected)} 只: {' '.join(selected)}")
            else:
                logger.info(f"{strategy_name} 无选股结果")
            strategies_results[strategy_name] = selected

        # 7. LLM 多维度深度分析 → 唯一推送
        if not args.skip_llm and settings.deepseek_api_key:
            logger.info("开始 LLM 多维度深度分析...")
            try:
                analyst = MarketAnalyst(
                    settings=settings,
                    api_key=settings.deepseek_api_key,
                    model=settings.deepseek_model,
                )
                report = analyst.analyze(strategies_results)

                # 仅推送 LLM 综合研判报告（包含全部策略结果 + AI 分析）
                _push_ai_report(settings, report)
                logger.info("LLM 综合研判报告已推送")
            except Exception as e:
                logger.warning(f"LLM 分析异常: {e}")
                # LLM 异常时降级：直接推送策略原始结果
                _push_fallback_results(notifier, strategies_results)
        elif not args.skip_llm and not settings.deepseek_api_key:
            logger.info("未配置 DeepSeek API Key，跳过 LLM 分析")
            # 无 LLM 时降级：直接推送策略原始结果
            _push_fallback_results(notifier, strategies_results)

    except Exception:
        try:
            _logger = get_logger(__name__)
            _logger.exception("主流程发生未捕获异常，程序终止")
        except Exception:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    logger.info("Sequoia-X V2 运行完成")


def _push_ai_report(settings, report: str) -> None:
    """将 AI 分析报告通过 WxPusher 推送。"""
    from wxpusher import WxPusher

    from sequoia_x.core.logger import get_logger
    logger = get_logger(__name__)

    try:
        result = WxPusher.send_message(
            content=report,
            token=settings.wxpusher_token,
            topic_ids=settings.wxpusher_topic_ids,
            content_type=1,
        )
        if result.get("code") == 1000:
            logger.info("AI 分析报告推送成功")
        else:
            logger.warning(f"AI 报告推送失败: {result}")
    except Exception as e:
        logger.warning(f"AI 报告推送异常: {e}")


def _push_fallback_results(notifier, strategies_results: dict[str, list[str]]) -> None:
    """LLM 不可用时的降级推送：直接推送策略原始结果。"""
    from sequoia_x.core.logger import get_logger
    logger = get_logger(__name__)
    logger.info("LLM 不可用，降级为策略原始结果推送")

    for name, symbols in strategies_results.items():
        if symbols:
            notifier.send(symbols=symbols, strategy_name=name, webhook_key="default")
        else:
            logger.info(f"{name}: 无选股结果")


if __name__ == "__main__":
    main()
