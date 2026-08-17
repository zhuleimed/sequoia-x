"""月度调仓+日级别持仓管理的回测引擎。

数据流：
  每月末最后一个交易日:
    1. 训练窗口 [前12月] → 训练 T2/T4/T1/T3
    2. 全股票池特征 → T2+T4 Rank融合 → T1过滤(可选) → T3仓位(可选) → 选 TOP_N
  次月第一个交易日:
    3. 以开盘价买入（滑点+佣金+整手+涨跌停检查）
  每日:
    4. 逐只持仓 evaluate_exit() → 触发则次日开盘卖出
    5. 收盘价更新估值 → 记录净值
  月末:
    6. 强制清仓 → 计算本月收益 → 回到步骤 1

使用:
  from sequoia_x.model_selection_v2.backtest.monthly_engine import MonthlyBacktestEngine
  engine = MonthlyBacktestEngine(cfg, data_engine, top_n=10, risk_mode="M0")
  results = engine.run("2025-08", "2026-06")
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from sequoia_x.core.config import Settings
from sequoia_x.core.logger import get_logger
from sequoia_x.data.engine import DataEngine
from sequoia_x.model_selection_v2.config import V2Config, get_config

logger = get_logger(__name__)


# ════════════════════════════════════════════════════════════
#  数据结构
# ════════════════════════════════════════════════════════════

@dataclass
class Position:
    """单只持仓。"""
    symbol: str
    shares: int
    cost: float           # 买入总成本（含佣金+滑点）
    entry_price: float    # 买入价（调整后）
    buy_date: str
    highest_price: float  # 持仓期间最高收盘价
    hold_days: int = 0
    current_price: float = 0.0

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def pnl(self) -> float:
        return self.market_value - self.cost

    @property
    def pnl_pct(self) -> float:
        return self.pnl / self.cost if self.cost > 0 else 0.0


@dataclass
class MonthlyCycle:
    """一个月的回测周期。"""
    month: str                    # "2025-08"
    train_end_date: str           # 训练截止日（月末最后交易日）
    buy_date: str                 # 买入日（次月第一个交易日）
    trading_days: list[str]       # 持仓期所有交易日（含买入日和卖出日）
    sell_date: str                # 强制卖出日（月末最后交易日）

    @property
    def first_hold_day(self) -> str:
        """买入后的第一个持仓日（买入日本身）。"""
        return self.buy_date

    @property
    def last_hold_day(self) -> str:
        """最后一个持仓日。"""
        return self.sell_date


@dataclass
class Trade:
    """单笔交易记录。"""
    symbol: str
    trade_type: str     # "buy" | "sell"
    date: str
    price: float
    shares: int
    amount: float       # 成交金额（含费用）
    pnl: float = 0.0    # 仅卖出时有意义
    reason: str = ""    # 卖出原因


# ════════════════════════════════════════════════════════════
#  并行特征构建（模块级函数，供 multiprocessing 使用）
# ════════════════════════════════════════════════════════════

def _build_features_chunk(args: tuple) -> tuple:
    """在子进程中构建特征块（模块级函数，pickle 可序列化）。

    Args:
        args: (symbols_chunk, ref_date, db_path, cfg, include_extra) 的元组。

    Returns:
        (X_chunk, symbols_chunk): 特征数组和有效股票列表。
    """
    symbols_chunk, ref_date, db_path, cfg, include_extra = args
    from sequoia_x.core.config import Settings
    from sequoia_x.data.engine import DataEngine
    from sequoia_x.model_selection_v2.features import build_batch_features

    # 每个子进程创建独立的 DataEngine（SQLite 连接不能跨进程共享）
    local_settings = Settings()
    local_settings.db_path = db_path
    local_engine = DataEngine(local_settings)

    X, valid_symbols = build_batch_features(
        list(symbols_chunk), ref_date, local_engine, cfg,
        include_extra=include_extra,
    )
    return X, list(valid_symbols)


# ════════════════════════════════════════════════════════════
#  交易成本常量（与 simulation/config.py 一致）
# ════════════════════════════════════════════════════════════

COMMISSION_RATE = 0.00025   # 佣金 万2.5
STAMP_TAX_RATE = 0.001      # 印花税 千1（仅卖出）
SLIPPAGE = 0.0001           # 滑点 万1

# 涨跌停限制
LIMIT_UP = 0.10
LIMIT_DOWN = -0.10


# ════════════════════════════════════════════════════════════
#  IC 加权仓位调节器（BACKTEST_PLAN §9.6 方法三）
# ════════════════════════════════════════════════════════════

def ic_weighted_sizing(signals: list[dict], top_n: int) -> list[dict]:
    """按 Rank 融合排名线性递减分配权重。

    第 1 名权重 = TOP_N / sum(1..TOP_N)
    第 N 名权重 = 1 / sum(1..TOP_N)
    """
    if not signals:
        return signals
    n = len(signals)
    denom = sum(range(1, n + 1))
    for sig in signals:
        rank = sig.get("rank", 1)
        sig["weight"] = round((n - rank + 1) / denom * n, 2)
    return signals


# ════════════════════════════════════════════════════════════
#  最小方差仓位调节器（BACKTEST_PLAN §9.6 方法四）
# ════════════════════════════════════════════════════════════

def min_variance_sizing(
    signals: list[dict],
    engine: DataEngine,
    ref_date: str,
    lookback: int = 60,
) -> list[dict]:
    """基于历史日收益率的反比方差权重。

    对每只信号股，取最近 lookback 天的日收益率，
    计算方差，权重 ∝ 1/方差。
    """
    if not signals or len(signals) < 2:
        return signals

    vols = []
    for sig in signals:
        sym = sig["symbol"]
        df = engine.get_ohlcv(sym)
        if df is None or len(df) < lookback + 5:
            vols.append(1.0)
            continue
        df = df[df["date"] <= ref_date]
        if len(df) < lookback:
            vols.append(1.0)
            continue
        rets = df["close"].pct_change().dropna().tail(lookback).values
        var = np.var(rets) if len(rets) > 5 else 1.0
        vols.append(1.0 / max(var, 1e-8))

    vols = np.array(vols)
    total = vols.sum()
    if total > 0:
        weights = vols / total * len(signals)
    else:
        weights = np.ones(len(signals))

    for sig, w in zip(signals, weights):
        sig["weight"] = round(float(w), 2)

    return signals


# ════════════════════════════════════════════════════════════
#  月度回测引擎
# ════════════════════════════════════════════════════════════

class MonthlyBacktestEngine:
    """月度调仓 + 日级别持仓管理的回测引擎。

    参数:
        cfg: V2Config 配置。
        engine: DataEngine 实例（只读查询行情）。
        top_n: 每月选股数量 (10/15/20/25)。
        risk_mode: 风控模式 "M0"|"M1"|"M2"|"M3"|"M4"|"M5"。
        initial_capital: 初始资金。
        use_real_t4: True=真实训练T4, False=跳过T4（快速测试用）。
    """

    def __init__(
        self,
        cfg: V2Config | None = None,
        engine: DataEngine | None = None,
        top_n: int = 10,
        risk_mode: str = "M0",
        initial_capital: float = 500_000.0,
        use_real_t4: bool = True,
        max_pool_size: int = 0,  # 0=全量，>0=限制股票池大小（快速测试）
        prediction_cache: dict | None = None,  # 月度预测缓存，提供则跳过训练+预测
        fusion_method: str = "pred_std",  # "pred_std"=原启发式 | "ic_weighted"=滚动IC加权（§25 方案1）
        keep_survivors: bool = False,  # True=模式B：月末不清仓幸存者，次月只补空位（模拟盘当前行为）
    ):
        self.cfg = cfg or get_config()
        self.engine = engine or DataEngine(Settings())
        self.top_n = top_n
        self.risk_mode = risk_mode.upper()
        self.initial_capital = initial_capital
        self.use_real_t4 = use_real_t4
        self.max_pool_size = max_pool_size
        self.prediction_cache = prediction_cache  # {month: {symbols, t2, t1, t3}}
        self.fusion_method = fusion_method
        self.keep_survivors = keep_survivors
        self.rolling_ics: list[dict] = []  # 滚动 IC 历史 [{month, t2_ic, t4_ic}]

        # 解析风控模式
        self._use_t1_filter = self.risk_mode in ("M1", "M5")
        self._use_t3_sizing = self.risk_mode in ("M2", "M5")
        self._use_market_state = self.risk_mode in ("M3", "M5")
        self._use_ic_weight = self.risk_mode in ("M4", "M5")

        # 状态
        self.cash: float = initial_capital
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.daily_records: list[dict] = []
        self.monthly_returns: list[float] = []
        self.monthly_labels: list[str] = []  # 按月顺序记录

        # 数据库连接（只读查询用）
        self._db_path = self.cfg.db_path

    # ════════════════════════════════════════════════════════
    #  主入口
    # ════════════════════════════════════════════════════════

    def run(self, start_month: str, end_month: str) -> dict:
        """运行月度回测。

        Args:
            start_month: 起始月 "2025-08"。
            end_month: 结束月 "2026-06"。

        Returns:
            {total_return, annual_return, sharpe, max_drawdown, win_rate,
             monthly_returns, daily_records, trades, ...}
        """
        t0 = time.time()

        use_cache = self.prediction_cache is not None and len(self.prediction_cache) > 0

        logger.info("=" * 60)
        logger.info(f"月度回测启动: {start_month}~{end_month}")
        logger.info(f"TOP_N={self.top_n}, 风控={self.risk_mode}, "
                    f"资金={self.initial_capital:,.0f}, "
                    f"T4={'真实' if self.use_real_t4 else '跳过'}, "
                    f"缓存={'有(' + str(len(self.prediction_cache)) + '月)' if use_cache else '无(实时训练)'}")
        logger.info("=" * 60)

        # 1. 获取月度交易周期
        cycles = self._get_monthly_cycles(start_month, end_month)
        if not cycles:
            logger.error("无法确定月度交易周期，数据不足？")
            return {}
        logger.info(f"月度周期: {len(cycles)} 个月 ({cycles[0].month}~{cycles[-1].month})")

        # 2. 初始化风控模块（缓存模式和实时训练模式共用）
        ms_detector, risk_manager, vol_sizer = None, None, None
        if self._use_market_state:
            from sequoia_x.model_selection_v2.risk.manager import MarketState
            ms_detector = MarketState(self.engine, self.cfg)
            logger.info("市场状态检测器已初始化")
        if self._use_t1_filter:
            from sequoia_x.model_selection_v2.risk.manager import RiskManager
            risk_manager = RiskManager(t1_auc_threshold=0.58)
            logger.info("T1 方向过滤器已初始化")
        if self._use_t3_sizing:
            from sequoia_x.model_selection_v2.risk.sizer import VolatilitySizer
            vol_sizer = VolatilitySizer()
            logger.info("T3 波动率仓位调节器已初始化")

        # 3. 实时训练模式：加载数据（缓存模式跳过）
        if not use_cache:
            logger.info("加载全量训练数据集...")
            X_full, y1_full, y2_full, y3_full, dates_full = self._load_full_dataset()
            if len(X_full) == 0:
                logger.error("训练数据集为空")
                return {}
            logger.info(f"全量数据: X={X_full.shape}, {len(set(dates_full))} 个采样日期")
            dates_arr = np.array(dates_full)
            stock_pool = self.engine.get_base_stock_pool()
            logger.info(f"股票池: {len(stock_pool)} 只")

        for ci, cycle in enumerate(cycles):
            month_start = time.time()
            logger.info(f"\n{'─'*40}")
            logger.info(f"[{ci+1:2d}/{len(cycles)}] {cycle.month} "
                        f"训练截止={cycle.train_end_date} 买入日={cycle.buy_date}")

            if use_cache:
                # ── 缓存模式：从预测缓存读取，跳过训练+特征构建+预测 ──
                cache_entry = self.prediction_cache.get(cycle.month)
                if cache_entry is None:
                    logger.warning(f"  {cycle.month}: 缓存中无数据，跳过")
                    continue

                cached_symbols = cache_entry.get("symbols", [])
                cached_t2 = cache_entry.get("t2", [])
                cached_t1 = cache_entry.get("t1", [])
                cached_t3 = cache_entry.get("t3", [])

                if len(cached_symbols) < self.top_n:
                    logger.warning(f"  {cycle.month}: 缓存数据不足 ({len(cached_symbols)} < {self.top_n})")
                    continue

                # 组装为统一格式
                cached_t4 = cache_entry.get("t4", [])
                predictions = []
                for i, sym in enumerate(cached_symbols):
                    predictions.append({
                        "symbol": sym,
                        "t2_pred": cached_t2[i] if i < len(cached_t2) else 0.0,
                        "t4_pred": cached_t4[i] if i < len(cached_t4) else 0.0,
                        "t1_prob": cached_t1[i] if i < len(cached_t1) else 0.5,
                        "t3_vol": cached_t3[i] if i < len(cached_t3) else 0.25,
                    })
                logger.info(f"  缓存加载: {len(predictions)} 只预测")
            else:
                # ── 实时训练模式 ──
                # 4a. 构建本月训练数据（12月滚动窗口）
                t_train = time.time()
                X_tr, y_tr, X_tr_2d = self._extract_training_data(
                    X_full, y2_full, dates_arr, cycle.train_end_date,
                )
                if len(X_tr) == 0:
                    logger.warning(f"  {cycle.month}: 训练数据不足，跳过")
                    continue
                logger.info(f"  训练集: {len(X_tr)} 样本 (X_3d={X_tr.shape})")

                # 4b. 训练模型
                t2_model = self._train_t2(X_tr_2d, y_tr)
                t4_model = None
                if self.use_real_t4:
                    t4_model = self._train_t4(X_tr, y_tr, cycle.month)
                t1_model = None
                if self._use_t1_filter:
                    y1_tr = self._extract_y1(y1_full, dates_arr, cycle.train_end_date)
                    if len(y1_tr) > 0:
                        t1_model = self._train_t1(X_tr_2d, y1_tr)
                t3_model = None
                if self._use_t3_sizing:
                    y3_tr = self._extract_y3(y3_full, dates_arr, cycle.train_end_date)
                    if len(y3_tr) > 0:
                        t3_model = self._train_t3(X_tr_2d, y3_tr)
                logger.info(f"  训练耗时: {time.time()-t_train:.0f}s")

                # 4c. 全股票池预测
                t_pred = time.time()
                predictions = self._predict_full_pool(
                    stock_pool, cycle.train_end_date,
                    t2_model, t4_model, t1_model, t3_model,
                )
                logger.info(f"  预测耗时: {time.time()-t_pred:.0f}s ({len(predictions)} 只有效预测)")

            if len(predictions) < self.top_n:
                logger.warning(f"  {cycle.month}: 有效预测不足 ({len(predictions)} < {self.top_n})")
                continue

            # 4d. 市场状态检测（缓存模式和训练模式共用）
            market_state = {"is_extreme": False, "advised_exposure": 1.0, "state": "震荡"}
            if ms_detector is not None:
                market_state = ms_detector.detect(cycle.month)
                logger.info(f"  市场状态: {market_state.get('state', '?')} "
                            f"建议仓位={market_state['advised_exposure']:.0%}")

            # 4e. 生成月度信号
            signals = self._generate_monthly_signals(
                predictions, market_state, risk_manager, vol_sizer,
            )
            logger.info(f"  信号: {len(signals)} 只 (原始 TOP_N={self.top_n})")

            # 4f. 月末强制卖出全部持仓（模式 B: 幸存者不清仓，留给次月）
            if self.positions and not self.keep_survivors:
                self._sell_all_positions(cycle.sell_date, reason="月末强制清仓")

            # 4g. 次月开盘买入
            # 模式 B: 只买空位（top_n - 当前持仓），与实盘 SimEngine 仓位上限语义一致；
            # 模式 A: 月初空仓 → 空位=top_n，行为不变
            if signals:
                if self.keep_survivors:
                    slots = max(0, self.top_n - len(self.positions))
                    signals = signals[:slots]
                self._execute_monthly_buy(signals, cycle.buy_date)

            # 4h. 日级别持仓管理循环
            self._daily_loop(cycle)

            # 4i. 月末结算
            month_return = self._settle_month(cycle)
            self.monthly_returns.append(month_return)
            self.monthly_labels.append(cycle.month)

            # 4j. 滚动 IC 采集（§25 方案1：供下月 ic_weighted 融合使用；最后一个月无需算）
            if self.fusion_method == "ic_weighted" and ci < len(cycles) - 1:
                ic_rec = self._compute_month_ic(predictions, cycle.month)
                if ic_rec is not None:
                    self.rolling_ics.append(ic_rec)
                    logger.info(f"  IC采集: {cycle.month} T2 IC={ic_rec['t2_ic']:+.4f} "
                                f"T4 IC={ic_rec['t4_ic']:+.4f} (n={ic_rec['n']})")

            logger.info(f"  {cycle.month} 完成: 收益={month_return:+.2%} "
                        f"净值={self.cash + sum(p.market_value for p in self.positions.values()):,.0f} "
                        f"耗时={time.time()-month_start:.0f}s")

        # 5. 计算最终指标
        metrics = self._compute_final_metrics()
        elapsed_min = (time.time() - t0) / 60
        logger.info(f"\n回测完成: {len(cycles)}个月, {len(self.trades)}笔交易, "
                    f"耗时={elapsed_min:.1f}min")
        logger.info(f"总收益={metrics['total_return']:+.2%} "
                    f"夏普={metrics['sharpe']:.2f} "
                    f"最大回撤={metrics['max_drawdown']:+.2%}")
        return metrics

    # ════════════════════════════════════════════════════════
    #  月度交易周期定位
    # ════════════════════════════════════════════════════════

    def _get_monthly_cycles(
        self, start_month: str, end_month: str,
    ) -> list[MonthlyCycle]:
        """从数据库获取月度交易周期。

        每月周期: 上月最后交易日(训练截止) → 本月第一交易日(买入) →
                  本月全部交易日(持仓) → 本月最后交易日(卖出)。
        实际回测中: 训练截止日是 month-1 的月末，买入日是 month 的首日。
        """
        conn = sqlite3.connect(self._db_path)

        # 获取回测范围内的所有交易日
        # start_month 的前一个月需要用于训练，所以查询从 start_month - 1 个月开始
        start_ym = start_month[:7]
        start_year, start_m = int(start_ym[:4]), int(start_ym[5:7])
        start_m -= 1
        if start_m <= 0:
            start_m += 12
            start_year -= 1
        lookback_start = f"{start_year}-{start_m:02d}-01"
        start_date = lookback_start
        end_date = end_month + "-31"
        all_dates = pd.read_sql(
            "SELECT DISTINCT date FROM stock_daily "
            "WHERE date >= ? AND date <= ? ORDER BY date",
            conn, params=(start_date, end_date),
        )["date"].tolist()
        conn.close()

        if len(all_dates) < 30:
            return []

        # 按月份分组
        month_to_dates: dict[str, list[str]] = {}
        for d in all_dates:
            ym = d[:7]
            if ym not in month_to_dates:
                month_to_dates[ym] = []
            month_to_dates[ym].append(d)

        months = sorted(month_to_dates.keys())

        # 过滤到回测范围内的月份 [start_month, end_month]
        test_months = [m for m in months if start_month <= m <= end_month]
        if not test_months:
            return []

        cycles = []
        for mi, month in enumerate(test_months):
            month_idx = months.index(month)
            if month_idx == 0:
                # 第一个月：用当月第一天作为训练截止（极端情况，数据不足则跳过）
                prev_month_dates = month_to_dates[months[0]]
                train_end = prev_month_dates[0]
            else:
                prev_month = months[month_idx - 1]
                prev_month_dates = month_to_dates[prev_month]
                train_end = prev_month_dates[-1]

            curr_dates = month_to_dates[month]
            buy_date = curr_dates[0]
            sell_date = curr_dates[-1]

            cycles.append(MonthlyCycle(
                month=month,
                train_end_date=train_end,
                buy_date=buy_date,
                trading_days=curr_dates,
                sell_date=sell_date,
            ))

        return cycles

    # ════════════════════════════════════════════════════════
    #  训练数据
    # ════════════════════════════════════════════════════════

    def _load_full_dataset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                            np.ndarray, list[str]]:
        """加载全量训练数据集（2020~最新）。

        一次性构建所有采样日期的特征+标签，后续按月用日期掩码切分。
        """
        from sequoia_x.model_selection_v2.labels import build_training_dataset

        # 使用全量范围加载
        # 注意: build_training_dataset 使用 cfg.sample_start/end，
        #       需要确保范围覆盖回测所需的所有训练数据
        X, y1, y2, y3, dates = build_training_dataset(
            self.engine, self.cfg, n_workers=8,
        )
        return X, y1, y2, y3, dates

    def _extract_training_data(
        self,
        X: np.ndarray,
        y2: np.ndarray,
        dates_arr: np.ndarray,
        train_end_date: str,
        train_months: int = 12,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """从全量数据中提取某个月份的训练集。

        训练窗口: train_end_date 之前 train_months 个月的采样数据。

        Args:
            X: 全量特征 (n, window, n_features) — 3D for LSTM。
            y2: 全量标签（20日超额收益）。
            dates_arr: 全量日期数组。
            train_end_date: 训练截止日（月末最后交易日）。
            train_months: 训练窗口月数。

        Returns:
            (X_tr_3d, y_tr, X_tr_2d):
                X_tr_3d: (n_train, window, n_features) — LSTM 用。
                y_tr: (n_train,)。
                X_tr_2d: (n_train, window*n_features) — 树模型用。
        """
        # 计算训练开始日期（train_months 个月前的第一天）
        end_ym = train_end_date[:7]
        end_year, end_month = int(end_ym[:4]), int(end_ym[5:7])

        # 回溯 12 个月
        start_month = end_month - train_months
        start_year = end_year
        if start_month <= 0:
            start_month += 12
            start_year -= 1
        train_start = f"{start_year}-{start_month:02d}-01"

        # 日期掩码
        mask = (dates_arr >= train_start) & (dates_arr <= train_end_date)
        n_train = mask.sum()
        if n_train < 100:
            return np.array([]), np.array([]), np.array([])

        X_tr = X[mask]
        y_tr = y2[mask]
        # 树模型用 2D 格式
        X_tr_2d = X_tr.reshape(n_train, -1)

        return X_tr, y_tr, X_tr_2d

    def _extract_y1(self, y1: np.ndarray, dates_arr: np.ndarray,
                     train_end_date: str, train_months: int = 12) -> np.ndarray:
        """提取 T1 标签（y1: 5日方向）。"""
        end_ym = train_end_date[:7]
        end_year, end_month = int(end_ym[:4]), int(end_ym[5:7])
        start_month = end_month - train_months
        start_year = end_year
        if start_month <= 0:
            start_month += 12
            start_year -= 1
        train_start = f"{start_year}-{start_month:02d}-01"
        mask = (dates_arr >= train_start) & (dates_arr <= train_end_date)
        return y1[mask]

    def _extract_y3(self, y3: np.ndarray, dates_arr: np.ndarray,
                     train_end_date: str, train_months: int = 12) -> np.ndarray:
        """提取 T3 标签（y3: 20日波动率）。"""
        end_ym = train_end_date[:7]
        end_year, end_month = int(end_ym[:4]), int(end_ym[5:7])
        start_month = end_month - train_months
        start_year = end_year
        if start_month <= 0:
            start_month += 12
            start_year -= 1
        train_start = f"{start_year}-{start_month:02d}-01"
        mask = (dates_arr >= train_start) & (dates_arr <= train_end_date)
        return y3[mask]

    # ════════════════════════════════════════════════════════
    #  模型训练
    # ════════════════════════════════════════════════════════

    def _train_t2(self, X_2d: np.ndarray, y: np.ndarray):
        """训练 T2 LightGBM 回归器。search_optuna=False 使用默认参数快速训练。"""
        from sequoia_x.model_selection_v2.models.tree_reg import train_reg
        return train_reg(X_2d, y, self.cfg, search_optuna=False)

    def _train_t4(self, X_3d: np.ndarray, y: np.ndarray, model_id: str):
        """训练 T4 LSTM 回归器。"""
        from sequoia_x.model_selection_v2.models.deep_lstm import train_lstm
        # 月度训练：不跑 Optuna（太慢），用默认参数
        return train_lstm(X_3d, y, self.cfg, search_optuna=False,
                          model_id=f"monthly_{model_id}")

    def _train_t1(self, X_2d: np.ndarray, y1: np.ndarray):
        """训练 T1 XGBoost 分类器。"""
        from sequoia_x.model_selection_v2.models.tree_cls import train_cls
        return train_cls(X_2d, y1, self.cfg, search_optuna=False)

    def _train_t3(self, X_2d: np.ndarray, y3: np.ndarray):
        """训练 T3 CatBoost 波动率回归器。"""
        from sequoia_x.model_selection_v2.models.tree_vol import train_vol
        return train_vol(X_2d, y3, self.cfg, search_optuna=False)

    # ════════════════════════════════════════════════════════
    #  全股票池预测
    # ════════════════════════════════════════════════════════

    def _predict_full_pool(
        self,
        stock_pool: list[str],
        ref_date: str,
        t2_model,
        t4_model,
        t1_model=None,
        t3_model=None,
    ) -> list[dict]:
        """用全部模型对全股票池批量预测。

        Args:
            stock_pool: 股票代码列表。
            ref_date: 预测截止日（月末最后交易日）。
            t2_model, t4_model, t1_model, t3_model: 训练好的模型。

        Returns:
            预测结果列表 [{symbol, t2_pred, t4_pred, t1_prob, t3_vol}, ...]
        """
        from sequoia_x.model_selection_v2.features import build_batch_features
        from sequoia_x.model_selection_v2.models.tree_reg import predict_reg

        # 限制股票池大小（快速测试模式）
        pool = stock_pool
        if self.max_pool_size > 0 and len(pool) > self.max_pool_size:
            import random
            random.seed(42)
            pool = random.sample(pool, self.max_pool_size)
            logger.info(f"  预测池: {len(stock_pool)} → {len(pool)} (max_pool_size={self.max_pool_size})")

        # 批量构建特征
        n_total = len(pool)
        n_workers = min(8, (os.cpu_count() or 4) - 2, n_total)
        use_parallel = n_workers >= 2 and n_total >= 200

        logger.info(f"  构建预测特征: {n_total} 只股票 (ref_date={ref_date})"
                    f"{', 并行=' + str(n_workers) if use_parallel else ''}...")
        t_feat = time.time()

        if use_parallel:
            # 多进程并行构建特征
            from multiprocessing import Pool
            chunks = np.array_split(list(pool), n_workers)
            task_args = [
                (list(c), ref_date, self._db_path, self.cfg)
                for c in chunks
            ]
            with Pool(n_workers) as p:
                chunk_results = p.map(_build_features_chunk, task_args)

            X_list, sym_list = [], []
            for X_chunk, syms_chunk in chunk_results:
                if len(X_chunk) > 0:
                    X_list.append(X_chunk)
                    sym_list.extend(syms_chunk)
            X_pred = np.concatenate(X_list) if X_list else np.array([]).reshape(0, self.cfg.window, 0)
            valid_symbols = sym_list
        else:
            X_pred, valid_symbols = build_batch_features(
                pool, ref_date, self.engine, self.cfg,
            )
        logger.info(f"  特征构建完成: {len(valid_symbols)}/{n_total} 有效 "
                    f"({time.time()-t_feat:.0f}s)")
        if len(X_pred) == 0:
            return []

        n_valid = len(X_pred)
        X_pred_2d = X_pred.reshape(n_valid, -1)

        # T2 预测
        pred_t2 = predict_reg(t2_model, X_pred_2d).flatten()

        # T4 预测
        pred_t4 = np.zeros(n_valid)
        if t4_model is not None:
            from sequoia_x.model_selection_v2.models.deep_lstm import predict_lstm
            pred_t4 = predict_lstm(t4_model, X_pred).flatten()

        # T1 预测
        pred_t1 = np.zeros(n_valid)
        if t1_model is not None:
            from sequoia_x.model_selection_v2.models.tree_cls import predict_cls
            pred_t1 = predict_cls(t1_model, X_pred_2d).flatten()

        # T3 预测
        pred_t3 = np.zeros(n_valid)
        if t3_model is not None:
            from sequoia_x.model_selection_v2.models.tree_vol import predict_vol
            pred_t3 = predict_vol(t3_model, X_pred_2d).flatten()

        # 组装结果
        results = []
        for i, sym in enumerate(valid_symbols):
            results.append({
                "symbol": sym,
                "t2_pred": float(pred_t2[i]),
                "t4_pred": float(pred_t4[i]),
                "t1_prob": float(pred_t1[i]),
                "t3_vol": float(pred_t3[i]),
            })

        return results

    # ════════════════════════════════════════════════════════
    #  信号生成
    # ════════════════════════════════════════════════════════

    def _compute_month_ic(self, predictions: list[dict], month: str) -> dict | None:
        """计算当月全部股票的 T2/T4 Rank IC（预测 vs 实际 20 日超额收益）。

        §25 方案1 的滚动 IC 数据源：预测来自缓存/实时训练，实际 y2 用
        采样日（该月最后交易日）→ 未来 20 个交易日的超额收益（沪深300 基准）。

        Returns:
            {"month", "t2_ic", "t4_ic", "n"} 或 None（数据不足）。
        """
        import sqlite3
        from scipy.stats import spearmanr

        try:
            conn = sqlite3.connect(self._db_path)
            # 采样日：该月最后一个交易日
            last_date = conn.execute(
                "SELECT MAX(date) FROM stock_daily WHERE date LIKE ?", (month + "%",)
            ).fetchone()[0]
            if last_date is None:
                conn.close()
                return None

            symbols = [p["symbol"] for p in predictions]
            pred_t2 = np.array([p["t2_pred"] for p in predictions])
            pred_t4 = np.array([p["t4_pred"] for p in predictions])

            # 批量加载采样日起的价格（未来 20 交易日足够）
            ph = ",".join("?" * len(symbols))
            prices = pd.read_sql(
                f"SELECT symbol, date, close FROM stock_daily "
                f"WHERE symbol IN ({ph}) AND date >= ? ORDER BY symbol, date",
                conn, params=symbols + [last_date],
            )
            idx_df = pd.read_sql(
                "SELECT date, close FROM index_daily "
                "WHERE symbol='sh.000300' AND date >= ? ORDER BY date",
                conn, params=(last_date,),
            )
            conn.close()

            # 采样日收盘 → 未来第 20 个交易日收盘
            t_close, t20_close = {}, {}
            for sym, g in prices.groupby("symbol"):
                g = g.sort_values("date")
                row_t = g[g["date"] == last_date]
                if row_t.empty:
                    continue
                future = g[g["date"] > last_date]
                if len(future) >= 20:
                    t_close[sym] = float(row_t["close"].iloc[0])
                    t20_close[sym] = float(future["close"].iloc[19])

            idx_t = idx_df[idx_df["date"] == last_date]
            idx_future = idx_df[idx_df["date"] > last_date]
            if idx_t.empty or len(idx_future) < 20:
                return None
            idx_ret = float(idx_future["close"].iloc[19]) / float(idx_t["close"].iloc[0]) - 1

            # 实际 y2（与训练标签同款定义）
            y2s, t2v, t4v = [], [], []
            for i, sym in enumerate(symbols):
                if sym in t_close and sym in t20_close:
                    stock_ret = t20_close[sym] / t_close[sym] - 1
                    y2s.append(np.clip(stock_ret - idx_ret, -0.5, 0.5))
                    t2v.append(pred_t2[i])
                    t4v.append(pred_t4[i])

            if len(y2s) < 50:
                return None
            y2_arr = np.array(y2s)
            t2_ic, _ = spearmanr(t2v, y2_arr)
            t4_ic, _ = spearmanr(t4v, y2_arr)
            return {"month": month, "t2_ic": float(t2_ic), "t4_ic": float(t4_ic),
                    "n": len(y2s)}
        except Exception as e:
            logger.warning(f"  IC计算失败 ({month}): {e}")
            return None

    def _generate_monthly_signals(
        self,
        predictions: list[dict],
        market_state: dict,
        risk_manager=None,
        vol_sizer=None,
    ) -> list[dict]:
        """Rank 融合 → 选股 → 风控管线。

        Returns:
            信号列表 [{symbol, rank_score, rank, weight, t2_pred, t4_pred, ...}]
        """
        n = len(predictions)
        symbols = [p["symbol"] for p in predictions]
        pred_t2 = np.array([p["t2_pred"] for p in predictions])
        pred_t4 = np.array([p["t4_pred"] for p in predictions])

        # 1. Rank 融合权重计算
        from sequoia_x.model_selection_v2.integration import rank_fusion

        if self.fusion_method == "ic_weighted":
            # ── §25 方案1：滚动 IC 动态加权 ──
            # 用过去最多 6 个月的真实月度 Rank IC 决定 T2/T4 权重
            #（IC<0 取 0：负 IC 表示模型反向，不给权重；历史不足时回退 pred_std）
            hist = [x for x in self.rolling_ics if x.get("t2_ic") is not None][-6:]
            if len(hist) >= 2:
                ic_t2 = float(np.mean([max(x["t2_ic"], 0.0) for x in hist]))
                ic_t4 = float(np.mean([max(x["t4_ic"], 0.0) for x in hist]))
                if ic_t2 + ic_t4 > 1e-9:
                    w_t2 = float(np.clip(ic_t2 / (ic_t2 + ic_t4), 0.3, 0.7))
                else:
                    w_t2 = 0.5
                w_t4 = 1.0 - w_t2
                logger.info(
                    f"  IC加权: 近{len(hist)}月 IC T2={ic_t2:.4f} T4={ic_t4:.4f} "
                    f"→ T2权重={w_t2:.2f} T4权重={w_t4:.2f}"
                )
            else:
                # 滚动历史不足（<2 个月）：回退 pred_std 启发式
                t4_std = float(np.std(pred_t4))
                t4_quality = min(t4_std / 0.02, 1.0)
                w_t4 = 0.25 + 0.25 * t4_quality
                w_t2 = 1.0 - w_t4
                logger.info(f"  IC加权: 历史不足({len(hist)}月)，回退 pred_std")
        else:
            # ── 原逻辑：T4 预测离散度启发式 ──
            # 用 T4 预测标准差判断信号质量：std<0.01→信号弱→降权
            t4_std = float(np.std(pred_t4))
            t4_quality = min(t4_std / 0.02, 1.0)  # 归一化到 [0, 1]
            w_t4 = 0.25 + 0.25 * t4_quality  # 范围 [0.25, 0.50]
            w_t2 = 1.0 - w_t4
            if w_t4 < 0.40:
                logger.debug(f"  T4信号弱(std={t4_std:.4f}), T2权重={w_t2:.2f} T4权重={w_t4:.2f}")
        # 加权排名
        rank_t2 = rankdata(-pred_t2, method="average")
        rank_t4 = rankdata(-pred_t4, method="average")
        rank_scores = w_t2 * rank_t2 + w_t4 * rank_t4

        # 1b. T2 预测分布预警：最优 100 只的 T2 预测均值 < -5% → 系统性看空，本月空仓
        top100_idx = np.argsort(-pred_t2)[:min(100, n)]
        top100_t2_mean = float(np.mean(pred_t2[top100_idx]))
        if top100_t2_mean < -0.05:
            logger.warning(
                f"  T2分布预警: top100均值={top100_t2_mean:.4f} < -0.05, "
                f"系统性看空→本月空仓"
            )
            return []

        # 排序取 Top N（初步）
        order = np.argsort(rank_scores)
        initial_top_n = self.top_n

        # 市场状态降仓
        effective_top_n = initial_top_n
        if self._use_market_state and market_state.get("is_extreme"):
            effective_top_n = max(3, initial_top_n // 2)
            logger.info(f"  极端市场降仓: TOP_N {initial_top_n}→{effective_top_n}")

        # 2. 构建信号
        signals = []
        for rank_i, idx in enumerate(order):
            sym = symbols[idx]
            signals.append({
                "symbol": sym,
                "rank_score": float(rank_scores[idx]),
                "rank": rank_i + 1,
                "t2_pred": float(pred_t2[idx]),
                "t4_pred": float(pred_t4[idx]),
                "t1_prob": predictions[idx].get("t1_prob", 0.5),
                "t3_vol": predictions[idx].get("t3_vol", 0.25),
            })

        # 3. T1 方向过滤
        # ⚠️ 2026-08-02 实测：T1 真实 AUC=0.499（70 个月均值，无预测能力），
        # 过滤器应保持关闭（auc 0.5 < 阈值 0.58 → use_t1=False）。
        # 未来若 T1 改造（换目标/模型）后 AUC 显著 >0.58，将真实 AUC 存入
        # prediction_cache 的 t1_auc 字段，回测从这里读取。
        if self._use_t1_filter and risk_manager is not None:
            t1_data = {
                "auc": 0.5,  # T1 无预测能力（实测 0.499），保持过滤关闭
                "predictions": {s["symbol"]: s["t1_prob"] for s in signals},
            }
            signals = risk_manager.adjust_signals(signals, market_state, t1_data)
            logger.info(f"  T1过滤后: {len(signals)} 只")

        # 4. 截断到 effective_top_n
        signals = signals[:effective_top_n]

        # 5. T3 波动率仓位调节
        if self._use_t3_sizing and vol_sizer is not None and signals:
            t3_preds = {s["symbol"]: s["t3_vol"] for s in signals}
            signals = vol_sizer.size_positions(
                signals, t3_preds, self.initial_capital, effective_top_n,
                market_exposure=market_state.get("advised_exposure", 1.0),
            )

        # 6. IC 加权仓位
        if self._use_ic_weight and signals:
            signals = ic_weighted_sizing(signals, effective_top_n)

        # 7. 默认等权
        for sig in signals:
            if "weight" not in sig:
                sig["weight"] = 1.0

        return signals

    # ════════════════════════════════════════════════════════
    #  交易执行
    # ════════════════════════════════════════════════════════

    def _execute_monthly_buy(self, signals: list[dict], buy_date: str) -> None:
        """在买入日以开盘价执行买入。

        考虑: 滑点、佣金、整手（100股）、涨跌停。
        不买: 涨停板股票（open ≈ prev_close * 1.10）。
        """
        if not signals:
            return

        n_signal = len(signals)
        budget_per_stock = self.cash / n_signal

        for sig in signals:
            sym = sig["symbol"]
            # 模式 B: 幸存者可能再次被选中 → 已在持仓中的跳过（与 SimEngine"已在持仓中"取消语义一致）
            if sym in self.positions:
                continue
            weight = sig.get("weight", 1.0)

            open_price = self._get_price(sym, buy_date, price_col="open")
            prev_close = self._get_price(sym, buy_date, price_col=None)  # 用 close 判断涨跌停
            if prev_close is None:
                prev_close = self._get_prev_close(sym, buy_date)

            if open_price is None or open_price <= 0:
                continue

            # 涨跌停检查：开盘价触及涨停板则跳过
            if prev_close and prev_close > 0:
                limit_up_price = prev_close * (1 + LIMIT_UP)
                limit_down_price = prev_close * (1 + LIMIT_DOWN)
                if open_price >= limit_up_price * 0.999:
                    logger.debug(f"  {sym} 涨停跳过: open={open_price:.2f} >= {limit_up_price:.2f}")
                    continue
                if open_price <= limit_down_price * 1.001:
                    logger.debug(f"  {sym} 跌停跳过: open={open_price:.2f} <= {limit_down_price:.2f}")
                    continue

            # 调整买入价（滑点）
            buy_price = open_price * (1 + SLIPPAGE)

            # 计算可买股数（整手）
            available = budget_per_stock * weight
            shares = int(available / buy_price / 100) * 100
            if shares < 100:
                continue

            # 计算总成本（含佣金）
            cost = shares * buy_price * (1 + COMMISSION_RATE)
            if cost > self.cash:
                # 资金不足：按可用资金重算
                shares = int(self.cash / (buy_price * (1 + COMMISSION_RATE)) / 100) * 100
                if shares < 100:
                    continue
                cost = shares * buy_price * (1 + COMMISSION_RATE)

            self.cash -= cost
            self.positions[sym] = Position(
                symbol=sym,
                shares=shares,
                cost=cost,
                entry_price=buy_price,
                buy_date=buy_date,
                highest_price=buy_price,
                current_price=buy_price,
            )

            self.trades.append(Trade(
                symbol=sym, trade_type="buy", date=buy_date,
                price=buy_price, shares=shares, amount=cost,
            ))

        logger.info(f"  买入: {sum(1 for t in self.trades if t.date == buy_date)} 只, "
                    f"剩余现金={self.cash:,.0f}")

    def _sell_position(self, sym: str, sell_date: str,
                        reason: str = "") -> float:
        """卖出一只持仓（以开盘价执行）。返回净收入。"""
        pos = self.positions.pop(sym)

        open_price = self._get_price(sym, sell_date, price_col="open")
        if open_price is None or open_price <= 0:
            # 无行情数据：按最后已知价格处理
            open_price = pos.current_price or pos.entry_price
            logger.warning(f"  {sym} 无{sell_date}开盘价，用最后已知价={open_price:.2f}")

        sell_price = open_price * (1 - SLIPPAGE)
        proceeds = pos.shares * sell_price
        commission = proceeds * COMMISSION_RATE
        stamp_tax = proceeds * STAMP_TAX_RATE
        net = proceeds - commission - stamp_tax
        pnl = net - pos.cost

        self.cash += net
        self.trades.append(Trade(
            symbol=sym, trade_type="sell", date=sell_date,
            price=sell_price, shares=pos.shares, amount=net,
            pnl=pnl, reason=reason,
        ))

        return net

    def _sell_all_positions(self, date: str, reason: str = "") -> None:
        """强制卖出全部持仓。"""
        for sym in list(self.positions.keys()):
            self._sell_position(sym, date, reason=reason)

    # ════════════════════════════════════════════════════════
    #  日级别持仓管理循环
    # ════════════════════════════════════════════════════════

    def _daily_loop(self, cycle: MonthlyCycle) -> None:
        """逐日推进：卖出检查 → 估值班 → 记录。

        卖出逻辑: 当日评估 → 如果触发卖出，在次日开盘执行。
        但简化为：当日收盘评估 → 次日开盘卖出（如果次日是最后一天，
        则在最后一天卖出）。

        实际上: 每日跑 evaluate_exit，如果触发则以当日收盘价标记卖出，
        次日开盘价执行。简化处理：触发即以下一个交易日的开盘价卖出。
        """
        from sequoia_x.simulation.rules import evaluate_exit

        # 构建指数 DataFrame（用于相对弱势检查）
        idx_df = self._get_index_df()

        # 记录买入日的估值
        if cycle.trading_days:
            self._mark_to_market(cycle.trading_days[0])
            self._record_daily(cycle.trading_days[0])

        pending_sells: set[str] = set()  # 待次日开盘卖出的股票

        for di, today in enumerate(cycle.trading_days):
            if di == 0:
                continue  # 买入日不卖出（T+1保护）

            # 执行前一日触发的卖出
            for sym in list(pending_sells):
                if sym in self.positions:
                    self._sell_position(sym, today, reason="规则触发")
            pending_sells.clear()

            # 逐只持仓检查卖出规则
            for sym, pos in list(self.positions.items()):
                if sym not in self.positions:
                    continue  # 可能已卖出

                # 获取个股日线数据
                symbol_df = self.engine.get_ohlcv(sym)
                if symbol_df is None or symbol_df.empty:
                    continue
                symbol_df = symbol_df[symbol_df["date"] <= today]

                # 卖出规则检查
                # 双轨止损（2026-08-12，与实盘 SimEngine 口径一致）：
                # pos.current_price 为前一交易日收盘（本日估值前的最后估值），
                # 其开盘价（symbol_df 倒数第2行）也参与硬止损判定；
                # symbol_df 已过滤 date<=today，无未来数据。
                prev_bar = symbol_df.iloc[-2] if len(symbol_df) >= 2 else None
                result = evaluate_exit(
                    entry_price=pos.entry_price,
                    current_price=pos.current_price,
                    highest_price=pos.highest_price,
                    hold_days=pos.hold_days,
                    symbol=sym,
                    symbol_df=symbol_df.tail(60) if len(symbol_df) >= 20 else None,
                    index_df=idx_df.tail(60) if idx_df is not None and not idx_df.empty else None,
                    today_opened=(di == 0),
                    day_open=float(prev_bar["open"]) if prev_bar is not None else None,
                )

                if result.should_exit:
                    # 不是最后一天 → 次日开盘卖；最后一天 → 当日收盘卖
                    if today != cycle.sell_date:
                        pending_sells.add(sym)
                        logger.debug(f"  {today} {sym} 触发卖出: {result.reason} (分={result.score})")
                    else:
                        self._sell_position(sym, today, reason=result.reason)
                        logger.debug(f"  {today} {sym} 月末清仓卖出: {result.reason}")

            # 日终估值（按今日收盘价）
            self._mark_to_market(today)
            self._record_daily(today)

        # 月末：处理最后一日仍未执行的待卖出；模式 A 额外强制清仓所有剩余持仓
        for sym in list(pending_sells):
            if sym in self.positions:
                self._sell_position(sym, cycle.sell_date, reason="规则触发(月末)")
        if not self.keep_survivors:
            self._sell_all_positions(cycle.sell_date, reason="月末强制清仓")

    def _settle_month(self, cycle: MonthlyCycle) -> float:
        """月末结算：记录月度收益率。"""
        # 模式 A: 持仓应已全部清仓（全部为现金）；
        # 模式 B: 月末仍有幸存持仓，按总资产（现金+市值）计算
        total_value = self.cash + sum(
            p.market_value for p in self.positions.values()
        ) if self.keep_survivors else self.cash
        # 计算本月收益
        if len(self.daily_records) > 0:
            month_start_value = self._get_month_start_value(cycle)
            if month_start_value > 0:
                month_return = (total_value - month_start_value) / month_start_value
            else:
                month_return = 0.0
        else:
            month_return = 0.0

        return month_return

    def _get_month_start_value(self, cycle: MonthlyCycle) -> float:
        """获取本月起始净值（从 daily_records 中查找）。"""
        # 找到本月买入日之前的最后一条记录
        for rec in reversed(self.daily_records):
            if rec["date"] <= cycle.buy_date:
                return rec["total_value"]
        return self.initial_capital

    # ════════════════════════════════════════════════════════
    #  估值与记录
    # ════════════════════════════════════════════════════════

    def _mark_to_market(self, date_str: str) -> None:
        """按当日收盘价更新所有持仓估值。"""
        for sym, pos in self.positions.items():
            close = self._get_price(sym, date_str, price_col="close")
            if close is not None and close > 0:
                pos.current_price = close
                if close > pos.highest_price:
                    pos.highest_price = close
            pos.hold_days += 1

    def _record_daily(self, date_str: str) -> None:
        """记录每日净值。"""
        stock_value = sum(p.market_value for p in self.positions.values())
        total = self.cash + stock_value
        self.daily_records.append({
            "date": date_str,
            "cash": round(self.cash, 2),
            "stock_value": round(stock_value, 2),
            "total_value": round(total, 2),
            "positions": len(self.positions),
        })

    # ════════════════════════════════════════════════════════
    #  数据查询
    # ════════════════════════════════════════════════════════

    def _get_price(self, symbol: str, date_str: str,
                    price_col: str = "open") -> float | None:
        """获取指定日期的价格。"""
        try:
            conn = sqlite3.connect(self._db_path)
            row = conn.execute(
                f"SELECT {price_col} FROM stock_daily "
                "WHERE symbol=? AND date=?",
                (symbol, date_str),
            ).fetchone()
            conn.close()
            return float(row[0]) if row and row[0] else None
        except Exception:
            return None

    def _get_prev_close(self, symbol: str, date_str: str) -> float | None:
        """获取前一交易日的收盘价（用于涨跌停判断）。"""
        try:
            conn = sqlite3.connect(self._db_path)
            row = conn.execute(
                "SELECT close FROM stock_daily "
                "WHERE symbol=? AND date < ? ORDER BY date DESC LIMIT 1",
                (symbol, date_str),
            ).fetchone()
            conn.close()
            return float(row[0]) if row and row[0] else None
        except Exception:
            return None

    def _get_index_df(self) -> pd.DataFrame:
        """获取沪深300指数日线数据（用于相对弱势检查）。"""
        try:
            conn = sqlite3.connect(self._db_path)
            df = pd.read_sql(
                "SELECT * FROM index_daily WHERE symbol='sh.000300' ORDER BY date",
                conn,
            )
            conn.close()
            return df if not df.empty else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    # ════════════════════════════════════════════════════════
    #  最终指标计算
    # ════════════════════════════════════════════════════════

    def _compute_final_metrics(self) -> dict:
        """从每日记录计算回测指标。"""
        n_days = len(self.daily_records)
        if n_days < 5:
            return {
                "total_return": 0.0, "annual_return": 0.0,
                "sharpe": 0.0, "max_drawdown": 0.0,
                "win_rate": 0.0, "n_months": 0, "n_trades": 0,
            }

        values = np.array([r["total_value"] for r in self.daily_records])
        initial = self.initial_capital

        # 总收益
        total_return = float(values[-1] / initial - 1)

        # 年化收益（按交易日折算）
        annual_return = float((1 + total_return) ** (252 / n_days) - 1)

        # 夏普比率（日收益率 → 年化）
        daily_rets = np.diff(values) / values[:-1]
        mean_daily = np.mean(daily_rets)
        std_daily = np.std(daily_rets)
        sharpe = float((mean_daily - 0.03 / 252) / std_daily * np.sqrt(252)) \
            if std_daily > 1e-10 else 0.0

        # 最大回撤
        cuml = values / values[0]
        running_max = np.maximum.accumulate(cuml)
        drawdown = (cuml - running_max) / running_max
        max_dd = float(drawdown.min())

        # 月胜率
        if self.monthly_returns:
            win_rate = float(np.mean(np.array(self.monthly_returns) > 0))
        else:
            win_rate = 0.0

        # 换手率（月均）
        buys = [t for t in self.trades if t.trade_type == "buy"]
        sells = [t for t in self.trades if t.trade_type == "sell"]
        n_months = len(self.monthly_returns)
        avg_turnover = len(buys) / n_months if n_months > 0 else 0.0

        # 交易统计
        win_trades = [t for t in sells if t.pnl > 0]
        lose_trades = [t for t in sells if t.pnl <= 0]
        avg_win = np.mean([t.pnl for t in win_trades]) if win_trades else 0.0
        avg_loss = np.mean([t.pnl for t in lose_trades]) if lose_trades else 0.0

        return {
            "total_return": round(total_return, 4),
            "annual_return": round(annual_return, 4),
            "sharpe": round(sharpe, 2),
            "max_drawdown": round(max_dd, 4),
            "win_rate": round(win_rate, 4),
            "n_months": n_months,
            "n_trades": len(self.trades),
            "n_buys": len(buys),
            "n_sells": len(sells),
            "avg_turnover": round(avg_turnover, 1),
            "win_trade_pct": round(len(win_trades) / len(sells), 4) if sells else 0.0,
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "final_value": round(self.cash, 2),
            "daily_records": self.daily_records,
            "monthly_returns": [round(r, 4) for r in self.monthly_returns],
            "_monthly_labels": self.monthly_labels,
            "trades": [
                {"symbol": t.symbol, "type": t.trade_type, "date": t.date,
                 "price": round(t.price, 4), "shares": t.shares,
                 "amount": round(t.amount, 2), "pnl": round(t.pnl, 2),
                 "reason": t.reason}
                for t in self.trades
            ],
        }
