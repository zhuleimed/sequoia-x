"""卖出规则引擎：多因子评分系统。

每只持仓每日独立计算评分，各规则分值累加。
总分 ≥ SELL_THRESHOLD(60) 触发卖出。

规则列表（分值从高到低）：
  S1 硬止损 -8%         → 100（穿透，不经总分判断直接卖出）
  T1 移动止盈回落 8%    →  85
  D1 持有 >20日         →  75
  M  均线死叉确认       →  70
  SH 夏普 <-0.5         →  70
  R1 跑输指数 >5%      →  60
  T2 移动止盈回落 5%    →  50
  SH 夏普 <0           →  50
  D2 持有 >15日         →  40
  S2 止损预警 -5%       →  40
  M  均线死叉首次       →  40
  R2 跑输指数 >3%      →  30
  SH 夏普 <0.5         →  30
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from sequoia_x.simulation.config import (
    # 阈值
    HARD_STOP_LOSS,
    HARD_STOP_LOSS_WARN,
    TRAILING_ACTIVATE,
    TRAILING_T1,
    TRAILING_T2,
    MAX_HOLD_DAYS,
    MAX_HOLD_DAYS_WARN,
    MA_DEATH_CROSS_DAYS,
    SHARPE_WINDOW_15,
    SHARPE_WINDOW_10,
    RELATIVE_LOOKBACK,
    RELATIVE_WEAK_THRESHOLD,
    RELATIVE_WARN_THRESHOLD,
    SELL_THRESHOLD,
    # 分值
    SCORE_HARD_STOP,
    SCORE_HARD_STOP_WARN,
    SCORE_TRAIL_T1,
    SCORE_TRAIL_T2,
    SCORE_TIME_D1,
    SCORE_TIME_D2,
    SCORE_MA_CROSS_CONFIRM,
    SCORE_MA_CROSS_TODAY,
    SCORE_SHARPE_BAD,
    SCORE_SHARPE_NEG,
    SCORE_SHARPE_LOW,
    SCORE_RELATIVE_WEAK,
    SCORE_RELATIVE_WARN,
    INDEX_SYMBOL,
)
from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


# ════════════════════════════════════════════════════════════
#  结果类型
# ════════════════════════════════════════════════════════════


class ExitRuleResult:
    """卖出规则判断结果。"""

    def __init__(self, should_exit: bool = False, reason: str = "",
                 score: int = 0, breakdown: Optional[list[tuple[str, int]]] = None):
        self.should_exit = should_exit       # 是否卖出
        self.reason = reason                  # 卖出原因描述
        self.score = score                    # 总评分
        self.breakdown = breakdown or []      # [(规则名, 分值), ...]

    def __repr__(self) -> str:
        return (f"ExitRuleResult(exit={self.should_exit}, score={self.score}, "
                f"reason='{self.reason}')")


# ════════════════════════════════════════════════════════════
#  单规则检查函数
# ════════════════════════════════════════════════════════════


def _check_hard_stop(entry_price: float, current_price: float) -> tuple[int, str]:
    """S1/S2 硬止损检查。

    Returns:
        (score, reason)
    """
    if entry_price <= 0:
        return 0, ""
    pnl_pct = (current_price - entry_price) / entry_price

    if pnl_pct <= HARD_STOP_LOSS:
        return SCORE_HARD_STOP, f"硬止损: 成本{entry_price:.2f}×{(1+HARD_STOP_LOSS):.2f}={entry_price*(1+HARD_STOP_LOSS):.2f}, 当前{current_price:.2f}"

    if pnl_pct <= HARD_STOP_LOSS_WARN:
        return SCORE_HARD_STOP_WARN, f"止损预警: 亏损{pnl_pct:.1%}≥{HARD_STOP_LOSS_WARN:.0%}"

    return 0, ""


def _check_trailing_stop(entry_price: float, current_price: float,
                         highest_price: float) -> tuple[int, str]:
    """T1/T2 移动止盈检查。

    仅当持仓收益率 ≥ TRAILING_ACTIVATE 时才激活。
    highest_price 在 engine 中每日更新。

    Returns:
        (score, reason)
    """
    if entry_price <= 0 or highest_price <= 0:
        return 0, ""

    pnl_pct = (current_price - entry_price) / entry_price

    # 未达到激活条件
    if pnl_pct < TRAILING_ACTIVATE:
        return 0, ""

    # 计算从最高点的回落幅度
    drawback = (highest_price - current_price) / highest_price

    if drawback >= TRAILING_T1:
        return (SCORE_TRAIL_T1,
                f"移动止盈触发(T1): 最高{highest_price:.2f}回落{drawback:.1%}≥{TRAILING_T1:.0%}, "
                f"收益率{pnl_pct:.1%}")

    if drawback >= TRAILING_T2:
        return (SCORE_TRAIL_T2,
                f"移动止盈预警(T2): 最高{highest_price:.2f}回落{drawback:.1%}≥{TRAILING_T2:.0%}")

    return 0, ""


def _check_hold_days(hold_days: int) -> tuple[int, str]:
    """D1/D2 时间止损检查。"""
    if hold_days > MAX_HOLD_DAYS:
        return SCORE_TIME_D1, f"持有超{MAX_HOLD_DAYS}日: 已持{hold_days}日"
    if hold_days > MAX_HOLD_DAYS_WARN:
        return SCORE_TIME_D2, f"持有接近上限: 已持{hold_days}日/上限{MAX_HOLD_DAYS}日"
    return 0, ""


def _check_ma_death_cross(df: pd.DataFrame) -> tuple[int, str]:
    """M 均线死叉检查。

    计算最近 N 日中 MA5 < MA10 的连续天数。
    需连续 MA_DEATH_CROSS_DAYS 日才确认。

    Args:
        df: 股票日线数据（含 close 列，至少 20 行）。

    Returns:
        (score, reason)
    """
    if df is None or len(df) < 20:
        return 0, ""

    closes = df["close"]
    ma5 = closes.rolling(5).mean()
    ma10 = closes.rolling(10).mean()

    # 取最近 MA_DEATH_CROSS_DAYS 天的截面
    recent = df.tail(MA_DEATH_CROSS_DAYS)
    recent_ma5 = ma5.tail(MA_DEATH_CROSS_DAYS)
    recent_ma10 = ma10.tail(MA_DEATH_CROSS_DAYS)

    # 连续 N 日均 MA5 < MA10
    cross_count = sum(1 for a, b in zip(recent_ma5, recent_ma10) if a < b)

    if cross_count >= MA_DEATH_CROSS_DAYS:
        return (SCORE_MA_CROSS_CONFIRM,
                f"均线死叉确认: 连续{cross_count}日MA5<MA10")
    if cross_count >= 1:
        return (SCORE_MA_CROSS_TODAY,
                f"均线死叉初现: 近{MA_DEATH_CROSS_DAYS}日中有{cross_count}日MA5<MA10")

    return 0, ""


def _check_sharpe(closes: pd.Series) -> tuple[int, str]:
    """SH 夏普率恶化检查。

    计算 15 日和 10 日滚动夏普率。

    Args:
        closes: 收盘价序列（至少 20 个数据点）。

    Returns:
        (score, reason)
    """
    if closes is None or len(closes) < SHARPE_WINDOW_15 + 1:
        return 0, ""

    daily_ret = closes.pct_change().dropna()
    rf_daily = 0.03 / 252  # 无风险利率 3%

    # 15 日夏普
    if len(daily_ret) >= SHARPE_WINDOW_15:
        r15 = daily_ret.tail(SHARPE_WINDOW_15)
        std15 = r15.std()
        sharpe15 = ((r15.mean() - rf_daily) / std15 * math.sqrt(252)) if std15 > 1e-10 else 0.0

        if sharpe15 < -0.5:
            return SCORE_SHARPE_BAD, f"15日夏普率{sharpe15:.2f}<-0.5"
        if sharpe15 < 0.5:
            # 可能只需要 30 分，但先检查是否更严重
            pass

    # 10 日夏普
    if len(daily_ret) >= SHARPE_WINDOW_10:
        r10 = daily_ret.tail(SHARPE_WINDOW_10)
        std10 = r10.std()
        sharpe10 = ((r10.mean() - rf_daily) / std10 * math.sqrt(252)) if std10 > 1e-10 else 0.0

        if sharpe10 < 0:
            return SCORE_SHARPE_NEG, f"10日夏普率{sharpe10:.2f}<0"

    # 如果 -0.5 ≤ 15日夏普 < 0.5
    if len(daily_ret) >= SHARPE_WINDOW_15:
        r15 = daily_ret.tail(SHARPE_WINDOW_15)
        std15 = r15.std()
        sharpe15 = ((r15.mean() - rf_daily) / std15 * math.sqrt(252)) if std15 > 1e-10 else 0.0
        if sharpe15 < 0.5:
            return SCORE_SHARPE_LOW, f"15日夏普率{sharpe15:.2f}<0.5（偏低）"

    return 0, ""


def _check_relative_strength(symbol_close: pd.Series,
                              index_close: pd.Series) -> tuple[int, str]:
    """R 相对弱势检查：近 5 日跑输上证指数。

    Args:
        symbol_close: 个股收盘价序列。
        index_close: 指数收盘价序列。

    Returns:
        (score, reason)
    """
    if symbol_close is None or index_close is None:
        return 0, ""
    if len(symbol_close) < RELATIVE_LOOKBACK + 1 or len(index_close) < RELATIVE_LOOKBACK + 1:
        return 0, ""

    # 近5日个股涨幅
    stock_recent = symbol_close.tail(RELATIVE_LOOKBACK + 1)
    stock_ret = (stock_recent.iloc[-1] - stock_recent.iloc[0]) / stock_recent.iloc[0]

    # 近5日指数涨幅
    idx_recent = index_close.tail(RELATIVE_LOOKBACK + 1)
    idx_ret = (idx_recent.iloc[-1] - idx_recent.iloc[0]) / idx_recent.iloc[0]

    relative = stock_ret - idx_ret

    if relative <= RELATIVE_WEAK_THRESHOLD:
        return (SCORE_RELATIVE_WEAK,
                f"相对弱势: 近{RELATIVE_LOOKBACK}日个股{stock_ret:.1%}跑输指数{idx_ret:.1%} "
                f"达{abs(relative):.1%}≥{abs(RELATIVE_WEAK_THRESHOLD):.0%}")

    if relative <= RELATIVE_WARN_THRESHOLD:
        return (SCORE_RELATIVE_WARN,
                f"相对弱势预警: 近{RELATIVE_LOOKBACK}日个股{stock_ret:.1%}跑输指数{idx_ret:.1%} "
                f"达{abs(relative):.1%}≥{abs(RELATIVE_WARN_THRESHOLD):.0%}")

    return 0, ""


# ════════════════════════════════════════════════════════════
#  综合评分入口
# ════════════════════════════════════════════════════════════


def evaluate_exit(
    entry_price: float,
    current_price: float,
    highest_price: float,
    hold_days: int,
    symbol_df: Optional[pd.DataFrame] = None,
    index_df: Optional[pd.DataFrame] = None,
    today_opened: bool = False,
) -> ExitRuleResult:
    """对单只持仓进行全维度卖出评分。

    T+1 保护：今日买入（today_opened=True）不触发任何卖出。

    Args:
        entry_price: 持仓成本价。
        current_price: 当日收盘价。
        highest_price: 持仓期间最高价。
        hold_days: 已持有天数。
        symbol_df: 个股日线 DataFrame（含 close 列）。
        index_df: 指数日线 DataFrame（含 close 列，用于相对强弱）。
        today_opened: 今日是否新开仓（T+1 保护）。

    Returns:
        ExitRuleResult。
    """
    # T+1 保护：今日买入不可卖
    if today_opened:
        return ExitRuleResult()

    breakdown: list[tuple[str, int]] = []
    total_score = 0

    # ── S 硬止损 ──
    score, reason = _check_hard_stop(entry_price, current_price)
    if score > 0:
        breakdown.append((f"S(硬止损)", score))
        total_score += score

    # 硬止损穿透：不管总分，直接卖出
    if score == SCORE_HARD_STOP:
        logger.info(f"硬止损触发: {reason}")
        return ExitRuleResult(
            should_exit=True,
            reason=reason,
            score=score,
            breakdown=breakdown,
        )

    # ── T 移动止盈 ──
    score, reason = _check_trailing_stop(entry_price, current_price, highest_price)
    if score > 0:
        breakdown.append((f"T(移动止盈)", score))
        total_score += score

    # ── D 时间止损 ──
    score, reason = _check_hold_days(hold_days)
    if score > 0:
        breakdown.append((f"D(持仓天数)", score))
        total_score += score

    # ── M 均线死叉 ──
    if symbol_df is not None:
        score, reason = _check_ma_death_cross(symbol_df)
        if score > 0:
            breakdown.append((f"M(均线死叉)", score))
            total_score += score

    # ── SH 夏普率 ──
    if symbol_df is not None and not symbol_df.empty:
        score, reason = _check_sharpe(symbol_df["close"])
        if score > 0:
            breakdown.append((f"SH(夏普率)", score))
            total_score += score

    # ── R 相对弱势 ──
    if symbol_df is not None and index_df is not None:
        score, reason = _check_relative_strength(symbol_df["close"], index_df["close"])
        if score > 0:
            breakdown.append((f"R(相对弱势)", score))
            total_score += score

    should_exit = total_score >= SELL_THRESHOLD

    if should_exit:
        # 构建原因描述（取最高分的 2 条）
        top_reasons = sorted(breakdown, key=lambda x: -x[1])[:2]
        reason_str = "; ".join(f"{name}({sc}分)" for name, sc in top_reasons)
    else:
        reason_str = ""

    return ExitRuleResult(
        should_exit=should_exit,
        reason=reason_str,
        score=total_score,
        breakdown=breakdown,
    )
