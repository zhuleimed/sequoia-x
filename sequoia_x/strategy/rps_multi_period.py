"""多周期 RPS 信号识别选股策略。

基于 RPS（相对强弱指标）多周期信号识别方法，
将市场上所有股票按三个时间维度的涨幅排列百分位，
综合判断趋势强度与信号质量。

参考：《RPS实战教程》多周期信号识别方法
"""

import pandas as pd
import sqlite3

from sequoia_x.core.logger import get_logger
from sequoia_x.strategy.base import BaseStrategy

logger = get_logger(__name__)


class RpsMultiPeriodStrategy(BaseStrategy):
    """多周期 RPS 信号识别选股策略。

    核心思路（来自《RPS实战教程》）：
    ─────────────────────────────────────────
    三个时间维度，反映资金的不同意图：
      - 短期 RPS（20日）：代表眼下的情绪
      - 中期 RPS（120日）：代表近半年的趋势
      - 长期 RPS（250日）：代表一年的底色

    真突破信号条件：
      1. 长期 RPS > 70  —— 一年底色不差
      2. 中期 RPS > 80  —— 半年有资金干活
      3. 短期 RPS 刚启动（60~90）—— 处于起涨点附近

    三不碰原则（硬过滤）：
      1. 长期 RPS < 50  —— 一年都没跑赢半数股票，不碰
      2. 短期 RPS > 95 且 中期 RPS < 70 —— 烟花信号，不碰
      3. 三周期同时向下拐头 —— 共振下跌，不碰

    评分公式（百分制）：
      base = 中期 RPS × 0.50 + 长期 RPS × 0.30 + 短期 RPS × 0.20
      真突破形态加分 +10（长期>70 & 中期>80 & 短期60~90）
      烟花形态扣分 -20（短期>95 & 中期<70）
      极度过热扣分 -10（三周期均>95）

    性能优化：全向量化运算，不逐行迭代。
    """

    webhook_key: str = "rps_multi"
    display_name: str = "多周期RPS突破"

    # 三个 RPS 计算周期
    short_period: int = 20
    mid_period: int = 120
    long_period: int = 250

    # ── 真突破信号阈值（来自教程） ──
    LONG_RPS_MIN: float = 70.0       # 长期 RPS 底线
    MID_RPS_MIN: float = 80.0        # 中期 RPS 底线
    SHORT_RPS_LOW: float = 60.0      # 短期 RPS 下限
    SHORT_RPS_HIGH: float = 90.0     # 短期 RPS 上限（过热线）

    # ── 三不碰硬阈值 ──
    NO_TOUCH_LONG_MIN: float = 50.0  # 长期 RPS < 50 不碰
    FIREWORK_SHORT_MIN: float = 95.0  # 烟花信号：短期 ≥ 95
    FIREWORK_MID_MAX: float = 70.0    # 烟花信号：中期 < 70
    OVERHEAT_ALL: float = 95.0        # 三周期均 ≥ 95 为极度过热

    # ── 评分权重（来自教程：中期50%，长期30%，短期20%） ──
    WEIGHT_MID: float = 0.50
    WEIGHT_LONG: float = 0.30
    WEIGHT_SHORT: float = 0.20

    # ── 形态加分/扣分 ──
    BONUS_TRUE_BREAKOUT: float = 10.0
    PENALTY_FIREWORK: float = -20.0
    PENALTY_OVERHEAT: float = -10.0

    def run(self) -> list[str]:
        """执行多周期 RPS 选股（全向量化版本）。

        Returns:
            按信号质量排序的股票代码列表（前 top_n 只）。
        """
        # ── 1. 批量读取行情数据 ──
        try:
            with sqlite3.connect(self.engine.db_path) as conn:
                df = pd.read_sql(
                    "SELECT symbol, date, close FROM stock_daily "
                    "ORDER BY symbol, date",
                    conn,
                )
        except Exception as exc:
            logger.error(f"读取数据库失败: {exc}")
            return []

        if df.empty:
            return []

        df["date"] = pd.to_datetime(df["date"])
        latest_date = df["date"].max()
        logger.info(
            f"多周期RPS行情数据: {len(df)} 条, "
            f"股票 {df['symbol'].nunique()} 只, "
            f"最新日期 {latest_date.date()}"
        )

        # ── 2. 全向量化计算三周期涨幅（groupby + shift） ──
        # 在每组 symbol 内纵向计算 N 日前的收盘价
        sort_idx = df.groupby("symbol")["date"].transform(lambda x: x.is_monotonic_increasing)
        if not sort_idx.all():
            df = df.sort_values(["symbol", "date"])

        grouped = df.groupby("symbol")["close"]

        df["pct_short"] = grouped.transform(
            lambda x: x / x.shift(self.short_period) - 1
        )
        df["pct_mid"] = grouped.transform(
            lambda x: x / x.shift(self.mid_period) - 1
        )
        df["pct_long"] = grouped.transform(
            lambda x: x / x.shift(self.long_period) - 1
        )

        # ── 3. 只保留最新日期的数据 ──
        latest_df = df[df["date"] == latest_date].copy()
        latest_df = latest_df.dropna(subset=["pct_short", "pct_mid", "pct_long"])

        if latest_df.empty:
            logger.info("多周期RPS策略：最新日期无有效数据")
            return []

        logger.info(
            f"有效候选: {len(latest_df)} 只 "
            f"（剔除涨幅缺失/数据不足的股票）"
        )

        # ── 4. 全市场横向排名（一次性完成） ──
        latest_df["rps_short"] = latest_df["pct_short"].rank(pct=True) * 100
        latest_df["rps_mid"] = latest_df["pct_mid"].rank(pct=True) * 100
        latest_df["rps_long"] = latest_df["pct_long"].rank(pct=True) * 100

        logger.info(
            f"RPS计算完成: "
            f"短期中位数{latest_df['rps_short'].median():.0f}, "
            f"中期中位数{latest_df['rps_mid'].median():.0f}, "
            f"长期中位数{latest_df['rps_long'].median():.0f}"
        )

        # ── 5. 向量化评分（不再逐行迭代） ──
        rs = latest_df["rps_short"]
        rm = latest_df["rps_mid"]
        rl = latest_df["rps_long"]

        # 5a. 三不碰硬过滤（mask）
        mask_no_touch = pd.Series(True, index=latest_df.index)

        # 不碰 1：长期 RPS < 50
        mask_no_touch &= rl >= self.NO_TOUCH_LONG_MIN

        # 不碰 2：烟花信号（短期>95 且 中期<70）
        mask_no_touch &= ~((rs >= self.FIREWORK_SHORT_MIN) & (rm < self.FIREWORK_MID_MAX))

        # 不碰 3：三周期同时向下（简化：短期<40 且 中期<60）
        mask_no_touch &= ~((rs < 40) & (rm < 60))

        # 5b. 基础分
        raw_score = (
            rs * self.WEIGHT_SHORT
            + rm * self.WEIGHT_MID
            + rl * self.WEIGHT_LONG
        )

        # 5c. 真突破加分
        true_breakout_mask = (
            (rl >= self.LONG_RPS_MIN)
            & (rm >= self.MID_RPS_MIN)
            & (rs >= self.SHORT_RPS_LOW)
            & (rs <= self.SHORT_RPS_HIGH)
        )
        raw_score = raw_score.where(~true_breakout_mask, raw_score + self.BONUS_TRUE_BREAKOUT)

        # 5d. 烟花扣分
        firework_mask = (rs >= self.FIREWORK_SHORT_MIN) & (rm < self.FIREWORK_MID_MAX + 10)
        raw_score = raw_score.where(~firework_mask, raw_score + self.PENALTY_FIREWORK)

        # 5e. 极度过热扣分
        overheat_mask = (
            (rs >= self.OVERHEAT_ALL)
            & (rm >= self.OVERHEAT_ALL)
            & (rl >= self.OVERHEAT_ALL)
        )
        raw_score = raw_score.where(~overheat_mask, raw_score + self.PENALTY_OVERHEAT)

        # 5f. 应用硬过滤 mask，移除不满足条件的股票
        selected = latest_df[mask_no_touch].copy()
        selected["score"] = raw_score[mask_no_touch]

        if selected.empty:
            logger.info("多周期RPS策略：硬过滤后无满足条件的股票")
            return []

        true_breakout_count = true_breakout_mask[mask_no_touch].sum()
        logger.info(
            f"硬过滤后: {len(selected)} 只 "
            f"（真突破形态{true_breakout_count}只）"
        )

        # ── 6. 基础股票池过滤 ──
        pool_filtered = self._apply_pool(selected["symbol"].tolist())
        selected = selected[selected["symbol"].isin(pool_filtered)]

        if selected.empty:
            logger.info("多周期RPS策略：基础池过滤后无结果")
            return []

        # ── 7. 按得分排序取 TOP N ──
        selected = selected.sort_values("score", ascending=False)
        result = selected["symbol"].head(self.top_n).tolist()

        # ── 8. 日志输出 ──
        for _, r in selected.head(self.top_n).iterrows():
            logger.info(
                f"  [{r['symbol']}] "
                f"短期RPS={r['rps_short']:.1f} "
                f"中期RPS={r['rps_mid']:.1f} "
                f"长期RPS={r['rps_long']:.1f} "
                f"得分={r['score']:.1f}"
            )

        logger.info(
            f"RpsMultiPeriodStrategy 选出 {len(result)} 只 "
            f"（候选{len(selected)}只）"
        )
        return result
