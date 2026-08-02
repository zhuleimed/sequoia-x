"""风险管理模块：市场状态检测 + T1方向过滤 + 极端月份降仓位。

核心逻辑:
  1. 检测当前市场状态(牛市/熊市/震荡/极端)
  2. 极端月份(大盘20日跌幅>10% 或 波动率翻倍) → 仓位减半
  3. 正常月份 → T1方向过滤(prob_up>0.55)
  4. 方向不明确 → 全市场选股,不做过滤

使用:
  from sequoia_x.model_selection_v2.risk import MarketState, RiskManager
  ms = MarketState(engine)
  state = ms.detect(test_month)
  rm = RiskManager()
  signals = rm.adjust_signals(signals, state, t1_predictions)
"""

import sqlite3
from typing import Optional

import numpy as np
import pandas as pd

from sequoia_x.core.logger import get_logger
from sequoia_x.model_selection_v2.config import V2Config, get_config

logger = get_logger(__name__)


class MarketState:
    """市场状态检测器。

    基于沪深300的近期表现判断当前处于什么市场阶段。
    需要 HS300 指数日线数据 (index_daily 表)。
    """

    INDEX_SYMBOL = "sh.000300"
    NORMAL_VOL_LOOKBACK = 240  # 正常波动率参考窗口(交易日)
    SHORT_WINDOW = 20          # 短期窗口
    MID_WINDOW = 60            # 中期窗口

    # 极端市场阈值
    EXTREME_DRAWDOWN = -0.10   # 20日跌幅超过10%
    EXTREME_VOL_RATIO = 2.0    # 20日波动率 > 正常水平的2倍
    BULL_MA_CROSS = 0.02       # 5日均线高于20日均线2%

    def __init__(self, engine=None, cfg: V2Config | None = None):
        self.cfg = cfg or get_config()
        self._idx_data: pd.DataFrame | None = None
        self._load_index_data()

    def _load_index_data(self) -> None:
        """加载沪深300日线数据。"""
        try:
            conn = sqlite3.connect(self.cfg.db_path)
            df = pd.read_sql(
                "SELECT date, close FROM index_daily WHERE symbol=? ORDER BY date",
                conn, params=(self.INDEX_SYMBOL,)
            )
            conn.close()
            if not df.empty:
                df["ret"] = df["close"].pct_change()
                self._idx_data = df
        except Exception as e:
            logger.warning(f"加载指数数据失败: {e}")

    def detect(self, month: str) -> dict:
        """检测指定月份前一月末的市场状态。

        6 级市场状态（BACKTEST_PLAN §9.5）：
          极端熊市(0.3) > 高波动(0.5) > 熊市(0.5) > 偏弱(0.7) > 结构分化(0.7) > 牛市/震荡(1.0)

        Args:
            month: "2026-05" 格式的月份。

        Returns:
            {
                "month": str,
                "state": str,             # 市场状态标签
                "is_extreme": bool,       # 是否极端（向后兼容）
                "is_bear": bool,          # 是否熊市（向后兼容）
                "market_return_20d": float,
                "market_vol_20d": float,
                "market_drawdown_20d": float,
                "direction_score": int,   # 1-5, 5=强牛
                "advised_exposure": float, # 建议仓位比例 (0.3/0.5/0.7/1.0)
            }
        """
        result = {
            "month": month,
            "state": "震荡",
            "is_extreme": False,
            "is_bear": False,
            "market_return_20d": 0.0,
            "market_vol_20d": 0.0,
            "market_drawdown_20d": 0.0,
            "direction_score": 3,        # 默认: 中性
            "advised_exposure": 1.0,      # 默认: 满仓
        }

        if self._idx_data is None or self._idx_data.empty:
            return result

        # 找到月份前的最后一个交易日（上月最后一天）
        # 修复前视偏差：不能用 month+"-31"（包含了当月数据）
        from calendar import monthrange
        ym_y, ym_m = int(month[:4]), int(month[5:7])
        pm = ym_m - 1
        py = ym_y
        if pm <= 0:
            pm += 12
            py -= 1
        last_day = monthrange(py, pm)[1]
        prev_month_end = f"{py}-{pm:02d}-{last_day}"
        df = self._idx_data[self._idx_data["date"] <= prev_month_end]
        if len(df) < self.SHORT_WINDOW:
            return result

        closes = df["close"].values
        returns = df["ret"].values

        # 20日涨跌幅
        result["market_return_20d"] = float(closes[-1] / closes[-self.SHORT_WINDOW] - 1)

        # 20日波动率(年化)
        recent_ret = returns[-self.SHORT_WINDOW:]
        result["market_vol_20d"] = float(np.nanstd(recent_ret) * np.sqrt(252))

        # 正常波动率(240日)
        if len(returns) >= self.NORMAL_VOL_LOOKBACK:
            normal_vol = float(np.nanstd(returns[-self.NORMAL_VOL_LOOKBACK:]) * np.sqrt(252))
        else:
            normal_vol = result["market_vol_20d"] or 0.15

        # 20日最大回撤
        rolling_high = pd.Series(closes[-self.SHORT_WINDOW:]).cummax().values
        result["market_drawdown_20d"] = float(closes[-1] / rolling_high[-1] - 1)

        # 计算 60 日涨跌幅、60 日回撤（新增，用于细化状态）
        ret_60d = float(closes[-1] / closes[-self.MID_WINDOW] - 1) if len(closes) >= self.MID_WINDOW else 0.0
        if len(closes) >= self.MID_WINDOW:
            rolling_high_60 = pd.Series(closes[-self.MID_WINDOW:]).cummax().values
            dd_60d = float(closes[-1] / rolling_high_60[-1] - 1)
        else:
            dd_60d = result["market_drawdown_20d"]

        # 均线关系
        ma5 = np.mean(closes[-5:])
        ma20 = np.mean(closes[-self.SHORT_WINDOW:])
        ma60 = np.mean(closes[-self.MID_WINDOW:]) if len(closes) >= self.MID_WINDOW else ma20
        ma5_vs_ma20 = ma5 / ma20 - 1 if ma20 > 0 else 0.0
        ma20_vs_ma60 = ma20 / ma60 - 1 if ma60 > 0 else 0.0

        # 市场广度（上涨天数占比）
        up_ratio = float(np.mean(recent_ret > 0)) if len(recent_ret) > 0 else 0.5

        # ── 6 级市场状态判断（BACKTEST_PLAN §9.5）──
        ret_20d = result["market_return_20d"]
        vol_20d = result["market_vol_20d"]
        dd_20d = result["market_drawdown_20d"]

        if abs(dd_20d) > 0.10:
            # 大盘20日跌超10%
            result["state"] = "极端熊市"
            result["is_extreme"] = True
            result["is_bear"] = True
            result["direction_score"] = 1
            result["advised_exposure"] = 0.3
        elif normal_vol > 0 and vol_20d > normal_vol * 2.0:
            # 波动率翻倍（恐慌）
            result["state"] = "高波动"
            result["is_extreme"] = True
            result["is_bear"] = True
            result["direction_score"] = 2
            result["advised_exposure"] = 0.5
        elif ret_20d < -0.05:
            # 大盘跌超5%
            result["state"] = "熊市"
            result["is_bear"] = True
            result["direction_score"] = 2
            result["advised_exposure"] = 0.5
        elif ret_20d < -0.03:
            # 大盘小跌
            result["state"] = "偏弱"
            result["direction_score"] = 2
            result["advised_exposure"] = 0.7
        elif up_ratio < 0.35:
            # 不足35%股票上涨（二八分化）
            result["state"] = "结构分化"
            result["direction_score"] = 2
            result["advised_exposure"] = 0.7
        elif ret_20d > 0.03 and ma5 > ma20:
            # 大盘涨+均线多头
            result["state"] = "牛市"
            result["direction_score"] = 5
            result["advised_exposure"] = 1.0
        else:
            result["state"] = "震荡"
            result["direction_score"] = 3
            result["advised_exposure"] = 1.0

        logger.debug(
            f"[{month}] 市场状态={result['state']}: ret20d={ret_20d:+.2%} "
            f"vol20d={vol_20d:.2%} dd20d={dd_20d:+.2%} "
            f"up_ratio={up_ratio:.0%} 仓位={result['advised_exposure']:.0%}"
        )
        return result


class RiskManager:
    """风险控制器：结合市场状态和T1信号调整选股。

    三种模式:
      1. 极端市场 → 仓位减半(top_n // 2)
      2. 正常+T1可用 → T1方向过滤(prob_up > 0.55)
      3. 正常+T1不可用 → 全市场选股
    """

    def __init__(self, t1_auc_threshold: float = 0.52):
        """
        Args:
            t1_auc_threshold: T1近期AUC阈值，超过此值才启用方向过滤。
        """
        self.t1_auc_threshold = t1_auc_threshold

    def adjust_signals(
        self,
        signals: list[dict],
        market_state: dict,
        t1_data: dict | None = None,
    ) -> list[dict]:
        """根据市场状态和T1信号调整选股。

        Args:
            signals: 原始信号列表 [{symbol, rank_score, ...}, ...]
            market_state: MarketState.detect() 返回的状态
            t1_data: {"auc": float, "predictions": {symbol: prob_up, ...}}

        Returns:
            调整后的信号列表(可能减少数量, 可能增加过滤标记)。
        """
        if not signals:
            return signals

        adjusted = []
        effective_top_n = len(signals)

        # 1. 极端市场: 仓位减半
        if market_state.get("is_extreme"):
            original_n = effective_top_n
            effective_top_n = max(3, effective_top_n // 2)
            logger.info(
                f"  极端市场(仓位减半): TOP_N {original_n}→{effective_top_n} "
                f"(回撤={market_state.get('market_drawdown_20d', 0):+.2%})"
            )

        # 2. T1方向过滤
        use_t1 = (
            t1_data is not None
            and t1_data.get("auc", 0) > self.t1_auc_threshold
            and not market_state.get("is_extreme")
        )

        for sig in signals:
            # T1过滤
            if use_t1:
                prob = t1_data.get("predictions", {}).get(sig["symbol"], 0.5)
                if prob < self.cfg.min_buy_prob if hasattr(self, 'cfg') else 0.55:
                    continue  # T1看空,跳过

            sig["t1_filtered"] = use_t1
            adjusted.append(sig)
            if len(adjusted) >= effective_top_n:
                break

        # 3. 如果过滤后股票不够,放宽到全市场
        if len(adjusted) < max(3, effective_top_n // 2):
            logger.warning(f"  T1过滤后仅剩{len(adjusted)}只, 放宽至全市场")
            adjusted = [s for s in signals if not use_t1 or s.get("t1_filtered")]
            adjusted = [dict(s, t1_filtered=False) for s in adjusted]
            adjusted = adjusted[:effective_top_n]

        return adjusted
