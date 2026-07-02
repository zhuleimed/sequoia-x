"""买入信号管理：LLM 推荐解析 → 写入 / 查询 / 更新 sim_buy_signals 表。

从 LLM 报告文本中提取推荐股票的三种策略：
  1. 解析 "RECOMMEND: 600519,000858" 末行（最精确）
  2. 解析 "最优关注: ..." 行中 "股票名(代码)" 模式
  3. 回退：按多策略重叠频率取前 2 只

T+1 模型（同 019 ETF 模拟盘）：
  T 日：LLM 推荐 → save_llm_recommendations() 写入 pending 信号
  T+1 日：get_pending_signals() 获取 → engine 执行买入
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import date
from typing import Optional

from sequoia_x.core.logger import get_logger
from sequoia_x.simulation.models import (
    insert_buy_signals_batch,
    get_pending_signals as _get_pending,
    mark_signal_executed as _mark_exec,
    mark_signal_cancelled as _mark_cancel,
    init_sim_tables,
)

logger = get_logger(__name__)


# ════════════════════════════════════════════════════════════
#  LLM 报告解析
# ════════════════════════════════════════════════════════════


def _extract_stock_code(text: str) -> Optional[str]:
    """从文本中提取 6 位 A 股代码，优先匹配首位非 6 开头的（沪市），再试全部。

    Args:
        text: 如 "(600519)" 或 "600519" 或 "宁德时代(300750)"

    Returns:
        6 位数字字符串，如 "600519"，找不到返回 None。
    """
    # 先找括号内的
    m = re.search(r"\((\d{6})\)", text)
    if m:
        return m.group(1)
    # 再找裸数字
    m = re.search(r"\b(\d{6})\b", text)
    if m:
        return m.group(1)
    return None


def parse_llm_report(report: str) -> list[str]:
    """从 LLM 报告中解析推荐股票代码（最多 2 只）。

    解析策略（按优先级）：
    1. 查找末尾 "RECOMMEND: code1,code2" 行
    2. 查找 "最优关注:" 行中的股票代码
    3. 查找 "综合建议" 区段中的股票代码

    Args:
        report: LLM 返回的完整报告文本。

    Returns:
        最多 2 只股票代码的列表，如 ["600519", "000858"]。
    """
    # ── 策略 1：RECOMMEND 末行 ──
    lines = report.strip().split("\n")
    for line in reversed(lines):
        line_stripped = line.strip()
        if line_stripped.upper().startswith("RECOMMEND:"):
            codes_str = line_stripped[len("RECOMMEND:"):].strip()
            codes = re.findall(r"\d{6}", codes_str)
            if codes:
                logger.info(f"解析 LLM 推荐（RECOMMEND 行）: {codes}")
                return codes[:2]

    # ── 策略 2："最优关注" 行 ──
    for line in lines:
        if "最优关注" in line:
            codes = re.findall(r"\d{6}", line)
            if codes:
                logger.info(f"解析 LLM 推荐（最优关注）: {codes}")
                return codes[:2]

    # ── 策略 3：综合建议区段中的所有代码 ──
    in_section = False
    all_codes: list[str] = []
    for line in lines:
        if "综合建议" in line or "综合研判" in line:
            in_section = True
            continue
        if in_section:
            codes = re.findall(r"\d{6}", line)
            all_codes.extend(codes)
            # 遇到下一个 ### 或空行且已收集完毕
            if line.startswith("###") or (not line.strip() and all_codes):
                break
    if all_codes:
        logger.info(f"解析 LLM 推荐（综合建议区段）: {all_codes}")
        return all_codes[:2]

    logger.warning("LLM 报告解析失败：未找到推荐股票代码")
    return []


# ════════════════════════════════════════════════════════════
#  多策略频率回退（解析失败时使用）
# ════════════════════════════════════════════════════════════


def get_top_by_strategy_frequency(strategies_results: dict[str, list[str]],
                                   top_n: int = 2) -> list[str]:
    """按多策略选中频率降序取前 N 只。

    Args:
        strategies_results: {"策略名": ["600519", "000858", ...], ...}
        top_n: 取前几只（默认 2）。

    Returns:
        股票代码列表，如 ["600519", "000858"]。
    """
    counter: Counter = Counter()
    for symbols in strategies_results.values():
        counter.update(symbols)
    top = [sym for sym, _ in counter.most_common(top_n)]
    logger.info(f"多策略频率回退推荐: {top}")
    return top


# ════════════════════════════════════════════════════════════
#  信号持久化
# ════════════════════════════════════════════════════════════


def save_llm_recommendations(
    db_path: str,
    strategies_results: dict[str, list[str]],
    llm_report: Optional[str] = None,
    top_n: int = 2,
) -> int:
    """将 LLM 推荐结果写入买入信号表。

    先尝试解析 LLM 报告文本，失败则回退到多策略频率。

    Args:
        db_path: 数据库路径。
        strategies_results: 各策略选股结果。
        llm_report: LLM 返回的报告文本（可选）。
        top_n: 最多推荐几只。

    Returns:
        写入的信号数量。
    """
    today_str = date.today().isoformat()

    # 确保表存在
    init_sim_tables(db_path)

    # 获取推荐股票
    recommended: list[str] = []
    if llm_report:
        recommended = parse_llm_report(llm_report)
        # LLM 已运行但未推荐任何股票（0 支），尊重 LLM 判断，不回退
    if not recommended and not llm_report:
        # LLM 未运行（如未配置 API Key），用多策略频率作为后备
        recommended = get_top_by_strategy_frequency(strategies_results, top_n)

    # 构建信号列表
    signals: list[dict] = []
    for sym in recommended[:top_n]:
        # 找出该股票由哪些策略选出（用于 strategy_from）
        strategies = [name for name, symbols in strategies_results.items() if sym in symbols]
        strategy_from = ",".join(strategies) if strategies else "LLM推荐"
        signals.append({"symbol": sym, "strategy_from": strategy_from, "llm_score": None})

    if not signals:
        logger.info("save_llm_recommendations: 无推荐股票，跳过")
        return 0

    count = insert_buy_signals_batch(db_path, signals)
    logger.info(f"save_llm_recommendations: 写入 {count} 条买入信号（{today_str}）: "
                f"{' '.join(s['symbol'] for s in signals)}")
    return count


# ════════════════════════════════════════════════════════════
#  查询与状态更新（代理到 models）
# ════════════════════════════════════════════════════════════


def get_pending_signals(db_path: str) -> list[dict]:
    """获取待执行的买入信号。"""
    today_str = date.today().isoformat()
    return _get_pending(db_path, today_str)


def mark_signal_executed(db_path: str, signal_id: int,
                         buy_price: float, shares: int) -> None:
    """标记买入信号已执行。"""
    today_str = date.today().isoformat()
    _mark_exec(db_path, signal_id, buy_price, shares, today_str)


def mark_signal_cancelled(db_path: str, signal_id: int, reason: str) -> None:
    """标记买入信号已取消。"""
    _mark_cancel(db_path, signal_id, reason)


# ════════════════════════════════════════════════════════════
#  公开 API — 策略可直接调用
# ════════════════════════════════════════════════════════════


def submit_buy_signals(
    db_path: str,
    symbols: list[str],
    strategy_name: str = "自定义策略",
    top_n: int = 2,
) -> int:
    """【公开接口】任意策略直接向模拟盘提交买入信号。

    用法示例（在策略 run() 末尾或 main.py 中调用）：
        from sequoia_x.simulation.signals import submit_buy_signals

        selected = my_strategy.run()  # 得到 ["600519", "000858"]
        submit_buy_signals(
            db_path="data/sequoia_v2.db",
            symbols=selected,
            strategy_name="我的新策略",
            top_n=2,
        )

    写入的信号会进入 sim_buy_signals 表，次日的 pipeline --sim-update
    步骤会自动读取并执行（T+1 开盘买入），无需额外配置。

    Args:
        db_path: 数据库路径（settings.db_path）。
        symbols: 策略选出的股票代码列表。
        strategy_name: 策略名称（记录到信号中）。
        top_n: 最多提交几只（默认 2）。

    Returns:
        写入的信号数量。
    """
    from sequoia_x.simulation.models import init_sim_tables, insert_buy_signals_batch

    init_sim_tables(db_path)

    signals = [
        {"symbol": s, "strategy_from": strategy_name, "llm_score": None}
        for s in symbols[:top_n]
    ]

    count = insert_buy_signals_batch(db_path, signals)
    if count > 0:
        logger.info(
            f"submit_buy_signals: 策略 [{strategy_name}] 提交 {count} 只信号: "
            f"{' '.join(s['symbol'] for s in signals)}"
        )
    return count
