"""LSTM-Transformer 模型选股策略 — 配置模块。

统一管理模型架构、训练、预测、回测、模拟盘的所有可配置参数。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LSTMConfig:
    """LSTM-Transformer 股票收益率预测全局配置。"""

    # ── 路径 ──
    db_path: str = "data/sequoia_v2.db"           # 主数据库（行情数据）
    sim_db_path: str = "data/sim_lstm.db"          # LSTM 策略独立模拟盘 DB
    model_dir: str = "data/models/lstm_selection"  # 模型保存目录
    output_dir: str = "output/sim_lstm"            # 模拟盘状态输出

    # ── 时间窗口 ──
    window: int = 60             # 时序窗口（交易日）
    predict_horizon: int = 5     # 预测未来 N 日收益率

    # ── 模型架构（默认值，Optuna 会覆盖） ──
    lstm_units: int = 128
    lstm_units2: int = 64
    num_heads: int = 4
    ff_dim: int = 256
    num_transformers: int = 2
    dropout_rate: float = 0.2
    dense_units: int = 128
    learning_rate: float = 0.001

    # ── 训练参数 ──
    train_sample_stocks: int = 200   # 训练时抽样股票数（市值分层）
    test_ratio: float = 0.15        # 测试集比例
    val_ratio: float = 0.15         # 验证集比例
    random_seed: int = 42
    early_stop_patience: int = 20
    reduce_lr_patience: int = 8
    reduce_lr_factor: float = 0.5
    min_learning_rate: float = 1e-6

    # ── Optuna 搜索参数 ──
    optuna_n_trials: int = 50
    optuna_n_jobs: int = 6
    optuna_timeout: int = 259200     # 72 小时

    # ── Optuna 搜索范围 ──
    lstm_units_range: tuple = (64, 320)
    lstm_units2_range: tuple = (32, 192)
    num_heads_range: tuple = (2, 12)
    ff_dim_range: tuple = (64, 768)
    num_transformers_range: tuple = (1, 4)
    dropout_range: tuple = (0.1, 0.6)
    dense_units_range: tuple = (32, 384)
    learning_rate_range: tuple = (1e-6, 3e-2)
    batch_size_options: tuple = (16, 32, 64, 128)

    # ── 增量学习参数 ──
    incremental_lr: float = 1e-5
    incremental_epochs: int = 10
    incremental_lookback: int = 60  # 近 N 个交易日

    # ── 每周刷新参数 ──
    weekly_epochs: int = 100
    weekly_lookback: int = 252      # 近 1 年

    # ── 每日预测参数 ──
    min_pred_return: float = 0.01   # 最低预测收益率阈值
    top_n_buy_per_day: int = 2      # 每天最多买入数

    # ── 模拟盘参数 ──
    initial_capital: float = 500_000.0
    per_stock_budget: float = 50_000.0
    max_positions: int = 10
    commission_rate: float = 0.00025
    stamp_tax_rate: float = 0.001
    slippage: float = 0.0001
    sell_threshold: int = 60
    strategy_name: str = "LSTM-Transformer选股"

    # ── 特征工程参数 ──
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


# 全局单例
_config: LSTMConfig | None = None


def get_config() -> LSTMConfig:
    global _config
    if _config is None:
        _config = LSTMConfig()
    return _config
