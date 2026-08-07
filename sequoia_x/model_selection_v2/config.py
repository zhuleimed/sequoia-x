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
    sample_end: str = "2026-07-28"
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

    # ── T4 LSTM-Transformer ──
    #   模型架构：LSTM → TransformerBlock × N → LSTM → Dense → 回归
    #   搜索 6 个核心参数，其余固定为经验最优值（与 V1 对齐）：
    #     lstm_units2  = lstm_units // 2
    #     ff_dim        = lstm_units * 2
    #     dense_units   = lstm_units2
    #     num_heads     = 4（固定）
    #     optimizer     = "adam"（固定）
    #     huber_delta   = 0.1（固定，适合±10%以内的超额收益）
    #     gradient_clip = 1.0（固定）
    lstm_units: int = 128                # LSTM1 单元数（默认值，Optuna 覆盖）
    lstm_units2: int = 64               # LSTM2 单元数（= units // 2）
    lstm_num_heads: int = 4             # MultiHeadAttention 头数
    lstm_ff_dim: int = 256              # Transformer FFN 隐藏维度（= units * 2）
    lstm_num_transformers: int = 0      # Transformer 层数（2026-07-28 修复: 2→0, Transformer 稀释 LSTM 信号）
    lstm_dropout_rate: float = 0.285    # Dropout 比率（2026-07-28: Optuna 最优值）
    lstm_dense_units: int = 128         # 中间 Dense 单元数（= lstm_units2）
    lstm_learning_rate: float = 0.0096  # 学习率（2026-07-28: Optuna 最优值）
    lstm_l2_reg: float = 0.0             # L2 正则化强度（2026-07-28 修复: 1e-4→0, L2杀死LSTM input kernel）
    lstm_huber_delta: float = 0.1       # Huber loss delta（±10% 内 MSE，之外 MAE）
    lstm_gradient_clip_norm: float = 1.0  # 全局梯度范数裁剪
    lstm_batch_size: int = 64           # 批次大小（默认值，Optuna 覆盖）
    lstm_epochs: int = 200              # 全量训练最大轮数
    lstm_optuna_epochs: int = 100       # Optuna 每 trial 最大轮数
    lstm_early_stop_patience: int = 25  # 早停耐心
    lstm_reduce_lr_patience: int = 10   # ReduceLROnPlateau 耐心
    lstm_min_lr: float = 1e-6           # 最低学习率
    lstm_optuna_n_trials: int = 18      # Optuna 搜索 trial 数（18: 20h内完成，Hyperband均匀分配）
    lstm_optuna_timeout: int = 86400    # Optuna 超时 24h（实际 ~6-10h）
    lstm_tf_intraop_threads: int = 16   # TF 单个 op 内部并行线程数
    lstm_tf_interop_threads: int = 8    # TF 独立 op 间并行线程数
    lstm_omp_num_threads: int = 10      # BLAS/MKL 数值计算线程数

    # ── 硬件 ──
    n_jobs: int = 8              # 树模型内部并行线程（T1+T2+T3 并行时 3×8=24核）

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
    # 2026-08-07: 88+33=121 维扩展特征拼接（fund_flow/finance/holders/consensus/news/xdxr/forecast）
    #   True=启用（8 月首次月度重训起, 2026-08-07 定稿: 月末自动链按此重建 121 维缓存）
    #   数据不全时自动回退 88 维（4 层机制, 见 V3 文档 §19.2）——不会因启用而中断月度流程
    #   仅作用于树模型链路（T2/T1/T3, include_market_state=True）；T4 LSTM 保持 80 维不拼
    extra_features: bool = True
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
