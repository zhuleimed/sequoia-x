"""model_selection_v2 - 回测参数。"""
MAX_POSITIONS: int = 10
TOP_N_BUY_PER_DAY: int = 2
PER_STOCK_BUDGET: float = 50_000.0
INITIAL_CAPITAL: float = 500_000.0
MIN_PRED_RETURN: float = 0.0
COMMISSION_RATE: float = 0.00025
STAMP_TAX_RATE: float = 0.001
SLIPPAGE: float = 0.0001
MIN_BUY_PROB: float = 0.55
