"""model_selection_v2 - 全局配置模块。"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class V2Config:
    """V2 多任务树模型全局配置。"""

    # ── 路径 ──
    db_path: str = "data/sequoia_v2.db"
    model_dir: str = "data/models/v2_selection"
    output_dir: str = "output/backtest_v2"

    # ── 时间窗口 ──
    window: int = 120          # 时序窗口（交易日）
    predict_horizon_t1: int = 5   # T1: 短期方向预测窗口
    predict_horizon_t2: int = 20  # T2: 中期收益预测窗口
    predict_horizon_t3: int = 20  # T3: 波动率预测窗口

    # ── 采样 ──
    sample_start: str = "2020-01-01"
    sample_end: str = "2026-07-20"
    samples_per_month: int = 2    # 每月采样天数（月初+月中）

    # ── 训练 ──
    random_seed: int = 42
    test_ratio: float = 0.15
    early_stop_rounds: int = 50

    # ── Optuna ──
    optuna_n_trials: int = 50
    optuna_timeout: int = 7200   # 2h per model

    # ── 模型超参搜索范围 ──
    xgb_params: dict = field(default_factory=lambda: {
        "max_depth": (3, 12),
        "learning_rate": (0.01, 0.3),
        "subsample": (0.6, 1.0),
        "colsample_bytree": (0.6, 1.0),
        "reg_alpha": (1e-3, 10.0),
        "reg_lambda": (1e-3, 10.0),
        "min_child_weight": (1, 20),
    })
    lgbm_params: dict = field(default_factory=lambda: {
        "num_leaves": (15, 255),
        "learning_rate": (0.01, 0.3),
        "subsample": (0.6, 1.0),
        "colsample_bytree": (0.6, 1.0),
        "reg_alpha": (1e-3, 10.0),
        "reg_lambda": (1e-3, 10.0),
        "min_child_samples": (10, 100),
    })
    cat_params: dict = field(default_factory=lambda: {
        "depth": (3, 10),
        "learning_rate": (0.01, 0.3),
        "l2_leaf_reg": (0.1, 10.0),
        "random_strength": (0.1, 10.0),
    })

    # ── 硬件 ──
    n_jobs: int = 8

    # ── Walk-Forward ──
    purge_gap: int = 22          # 训练/测试间隔（交易日）

    # ── 回测 ──
    initial_capital: float = 500_000.0
    per_stock_budget: float = 50_000.0
    max_positions: int = 10
    top_n_buy_per_day: int = 2
    commission_rate: float = 0.00025
    stamp_tax_rate: float = 0.001
    slippage: float = 0.0001
    min_buy_prob: float = 0.55   # T1 买入概率阈值

    # ── 特征 ──
    feature_ma_periods: tuple = (5, 10, 20, 60, 120)
    feature_rsi_period: int = 14
    feature_atr_period: int = 14
    feature_adx_period: int = 14
    feature_macd_fast: int = 12
    feature_macd_slow: int = 26
    feature_macd_signal: int = 9
    feature_boll_period: int = 20
    feature_vol_periods: tuple = (5, 10, 20)
    feature_beta_period: int = 20

    @property
    def model_dir_path(self) -> Path:
        p = Path(self.model_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def output_dir_path(self) -> Path:
        p = Path(self.output_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def optuna_dir_path(self) -> Path:
        p = Path("data/models/v2_selection/optuna")
        p.mkdir(parents=True, exist_ok=True)
        return p


_config: V2Config | None = None


def get_config() -> V2Config:
    """获取全局配置单例。"""
    global _config
    if _config is None:
        _config = V2Config()
    return _config
