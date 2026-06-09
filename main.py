"""Sequoia-X V2 主程序入口。

运行模式：
  python main.py                    # 日常模式：检查数据完整性 → 策略选股 → LLM分析 → 推送
  python main.py --sync-only        # 同步模式（19:10）：数据同步+清洗，不选股
  python main.py --repair           # 修复模式：手动修补数据（对比baostock最新交易日补齐缺失数据）
  python main.py --backfill         # 回填模式：baostock 全量历史K线
  python main.py --skip-llm         # 日常模式但跳过 LLM 分析
"""

import argparse
import sys
import time
from dotenv import load_dotenv

load_dotenv()

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
        "--sync-only",
        action="store_true",
        help="同步模式（19:10）：仅执行数据同步+清洗，不选股不推送",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="跳过 LLM 多维度分析（仅策略选股+推送）",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="修复模式：手动修补数据，对比 baostock 最新交易日补齐缺失数据",
    )
    args = parser.parse_args()

    try:
        # 1. 初始化配置
        settings = get_settings()

        # 2. 初始化日志和引擎
        logger = get_logger(__name__)
        logger.info("Sequoia-X V2 启动")
        engine = DataEngine(settings)
        _main_start = time.monotonic()

        if args.backfill:
            # ── 回填模式 ──
            logger.info("进入回填模式...")
            all_symbols = engine.get_all_symbols()
            engine.backfill(all_symbols)
            logger.info("Sequoia-X V2 回填模式运行完成")
            return

        if args.repair:
            # ═══════════════════════════════════════════════
            #  修复模式（手动调用，无时间限制）
            # ═══════════════════════════════════════════════
            logger.info("=== 修复模式 ===")
            logger.info("对比 baostock 最新交易日 → 清理退市/发现新股/补齐缺失数据")
            repair_t0 = time.monotonic()
            result = engine.repair_data()
            repair_elapsed = time.monotonic() - repair_t0

            logger.info(
                f"修复完成: 状态={result['status']} "
                f"{result['before']} → {result['after']} "
                f"退市{result['delisted']} 新股{result['new_listed']} "
                f"补填{result['backfilled']}天 "
                f"耗时{repair_elapsed:.0f}秒"
            )

            if result["status"] == "success":
                _push_sync_summary(settings, {
                    "status": "success",
                    "stock_count": int(result["after"].split("(")[1].rstrip("只)")),
                    "delisted": result["delisted"],
                    "new_listed": result["new_listed"],
                    "backfilled": result["backfilled"],
                    "latest_date": result["after"].split("(")[0],
                }, repair_elapsed)
            _elapsed_total = time.monotonic() - _main_start
            logger.info(f"Sequoia-X V2 修复模式运行完成（总耗时 {_elapsed_total:.0f} 秒）")
            return

        if args.sync_only:
            # ═══════════════════════════════════════════════
            #  同步模式（19:10 执行）
            # ═══════════════════════════════════════════════
            logger.info("=== 同步模式 ===")
            sync_t0 = time.monotonic()
            result = engine.sync_and_clean()
            sync_elapsed = time.monotonic() - sync_t0

            logger.info(
                f"同步完成: 状态={result['status']} "
                f"股票{result['stock_count']} "
                f"退市清理{result['delisted']}只 "
                f"新股发现{result['new_listed']}只 "
                f"补填{result['backfilled']}天 "
                f"耗时{sync_elapsed:.0f}秒"
            )

            # 同步结果推送到微信便于确认
            if result["status"] == "success":
                _push_sync_summary(settings, result, sync_elapsed)
            else:
                _push_data_alert(settings, {
                    "is_complete": False,
                    "latest_trade_day": "",
                    "coverage": 0.0,
                    "total_stocks": 0,
                    "stocks_with_data": 0,
                    "last_sync_status": result["status"],
                    "error": result["error"],
                }, mode="sync")

            _elapsed_total = time.monotonic() - _main_start
            logger.info(f"Sequoia-X V2 同步模式运行完成（总耗时 {_elapsed_total:.0f} 秒）")
            return

        # ═══════════════════════════════════════════════
        #  日常模式（20:55 执行）
        # ═══════════════════════════════════════════════
        logger.info("=== 日常模式 ===")

        # 第一步：检查数据完整性
        logger.info("检查数据完整性...")
        completeness = engine.check_data_completeness()

        if not completeness["is_complete"]:
            # 数据不完整 → 推送告警，跳过选股
            coverage_pct = completeness["coverage"] * 100
            logger.warning(
                f"数据不完整（覆盖率 {coverage_pct:.1f}%），"
                f"推送告警并跳过选股"
            )
            _push_data_alert(settings, completeness, mode="strategy")
            _elapsed_total = time.monotonic() - _main_start
            logger.info(
                f"Sequoia-X V2 因数据不完整跳过选股（总耗时 {_elapsed_total:.0f} 秒）"
            )
            return

        # 第二步：数据完整，继续原流程
        coverage_pct = completeness["coverage"] * 100
        logger.info(
            f"数据完整性验证通过（覆盖率 {coverage_pct:.1f}%），"
            f"最新交易日: {completeness['latest_trade_day']}"
        )

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

        # 收集各策略选股结果
        strategies_results: dict[str, list[str]] = {}

        for strategy in strategies:
            strategy_name = getattr(strategy, "display_name", type(strategy).__name__)
            logger.info(f"执行策略：{strategy_name}")

            selected: list[str] = strategy.run()

            if selected:
                logger.info(f"{strategy_name} 选出 {len(selected)} 只: {' '.join(selected)}")
            else:
                logger.info(f"{strategy_name} 无选股结果")
            strategies_results[strategy_name] = selected

        # 6. LLM 多维度深度分析 → 唯一推送
        if not args.skip_llm and settings.deepseek_api_key:
            logger.info("开始 LLM 多维度深度分析...")
            try:
                analyst = MarketAnalyst(
                    settings=settings,
                    api_key=settings.deepseek_api_key,
                    model=settings.deepseek_model,
                )
                report = analyst.analyze(strategies_results)

                _push_ai_report(settings, report)
                logger.info("LLM 综合研判报告已推送")
            except Exception as e:
                logger.warning(f"LLM 分析异常: {e}")
                _push_fallback_results(notifier, strategies_results)
        elif not args.skip_llm and not settings.deepseek_api_key:
            logger.info("未配置 DeepSeek API Key，跳过 LLM 分析")
            _push_fallback_results(notifier, strategies_results)

    except Exception:
        try:
            _logger = get_logger(__name__)
            _logger.exception("主流程发生未捕获异常，程序终止")
        except Exception:
            import traceback

            traceback.print_exc()
        sys.exit(1)

    _elapsed = time.monotonic() - _main_start
    logger.info(f"Sequoia-X V2 运行完成（总耗时 {_elapsed:.0f} 秒）")


# ══════════════════════════════════════════════════
#  推送辅助函数
# ══════════════════════════════════════════════════


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


def _push_data_alert(
    settings, completeness: dict, mode: str = "strategy"
) -> None:
    """推送数据完整性告警。

    Args:
        settings: 配置对象
        completeness: check_data_completeness() 返回的字典
        mode: "strategy"（20:55 选股前检查失败）或 "sync"（19:10 同步失败）
    """
    from wxpusher import WxPusher
    from sequoia_x.core.logger import get_logger
    from datetime import date

    logger = get_logger(__name__)

    today_str = date.today().strftime("%m-%d")

    if mode == "strategy":
        title = f"Sequoia-X 选股取消 | {today_str}"
        body = (
            f"数据完整性检查未通过，今日选股已取消。\n\n"
            f"最新交易日: {completeness.get('latest_trade_day', '未知')}\n"
            f"覆盖率: {completeness.get('coverage', 0) * 100:.1f}%\n"
            f"有数据股票: {completeness.get('stocks_with_data', 0)} / "
            f"{completeness.get('total_stocks', 0)}\n"
            f"最后同步状态: {completeness.get('last_sync_status', '未知')}\n\n"
            f"可能原因：\n"
            f"1. 19:10 数据同步未成功运行\n"
            f"2. baostock 数据入库延迟\n"
            f"3. 今日可能非交易日\n\n"
            f"请检查服务器日志: logs/daily_{date.today().strftime('%Y%m%d')}.log"
        )
    else:
        title = f"Sequoia-X 数据同步失败 | {today_str}"
        body = (
            f"19:10 数据同步执行异常。\n\n"
            f"错误信息: {completeness.get('error', '未知错误')}\n"
            f"股票数: {completeness.get('total_stocks', 0)}\n"
            f"退市清理: {completeness.get('delisted', 0)} 只\n"
            f"新股发现: {completeness.get('new_listed', 0)} 只\n\n"
            f"请检查服务器日志: logs/daily_{date.today().strftime('%Y%m%d')}.log"
        )

    message = f"{title}\n\n{body}"

    try:
        result = WxPusher.send_message(
            content=message,
            token=settings.wxpusher_token,
            topic_ids=settings.wxpusher_topic_ids,
            content_type=1,
        )
        if result.get("code") == 1000:
            logger.info(f"数据告警推送成功 ({mode})")
        else:
            logger.warning(f"数据告警推送失败: {result}")
    except Exception as e:
        logger.warning(f"数据告警推送异常: {e}")


def _push_sync_summary(settings, result: dict, elapsed: float) -> None:
    """推送 19:10 同步完成摘要。"""
    from wxpusher import WxPusher
    from sequoia_x.core.logger import get_logger
    from datetime import date

    logger = get_logger(__name__)
    today_str = date.today().strftime("%m-%d")

    message = (
        f"Sequoia-X 数据同步完成 | {today_str}\n\n"
        f"状态: {'成功' if result['status'] == 'success' else '失败'}\n"
        f"当前股票数: {result['stock_count']}\n"
        f"退市清理: {result['delisted']} 只\n"
        f"新股发现: {result['new_listed']} 只\n"
        f"缺失补填: {result['backfilled']} 天\n"
        f"最新日期: {result['latest_date']}\n"
        f"耗时: {elapsed:.0f} 秒\n\n"
        f"20:55 将自动执行选股流程"
    )

    try:
        r = WxPusher.send_message(
            content=message,
            token=settings.wxpusher_token,
            topic_ids=settings.wxpusher_topic_ids,
            content_type=1,
        )
        if r.get("code") == 1000:
            logger.info("同步摘要推送成功")
    except Exception as e:
        logger.warning(f"同步摘要推送异常: {e}")


if __name__ == "__main__":
    main()
