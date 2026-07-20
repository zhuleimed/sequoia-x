"""LSTM 策略回测 — 配置模块。"""

# ── 回测参数 ──
INITIAL_CAPITAL: float = 500_000.0
PER_STOCK_BUDGET: float = 50_000.0
MAX_POSITIONS: int = 10
TOP_N_BUY_PER_DAY: int = 2
MIN_PRED_RETURN: float = 0.01

# ── 交易成本 ──
COMMISSION_RATE: float = 0.00025
STAMP_TAX_RATE: float = 0.001
SLIPPAGE: float = 0.0001

# ── 回测时间范围 ──
START_DATE: str = "2024-01-01"
END_DATE: str = ""

# ── 模型重训频率 ──
RETRAIN_MONTHLY: bool = True  # 每月末重训模型

# ── 输出 ──
OUTPUT_DIR: str = "output/backtest_lstm"
