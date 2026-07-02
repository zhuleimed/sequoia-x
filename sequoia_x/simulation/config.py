"""模拟盘配置参数：账户、交易成本、卖出规则阈值。"""

# ═══════════════════════════════════════════
#  账户参数
# ═══════════════════════════════════════════

INITIAL_CAPITAL: float = 1_000_000.0   # 初始总资金（100万）
PER_STOCK_BUDGET: float = 50_000.0     # 单只股票初始分配资金（5万）
MAX_POSITIONS: int = 20                 # 最大持仓数

# ═══════════════════════════════════════════
#  交易成本（A 股）
# ═══════════════════════════════════════════
# 佣金买入卖出双向收取，印花税仅卖出收取

COMMISSION_RATE: float = 0.00025       # 佣金 万2.5
STAMP_TAX_RATE: float = 0.001           # 印花税 千1（卖出）
SLIPPAGE: float = 0.0001                # 滑点 万1

# ═══════════════════════════════════════════
#  卖出评分规则阈值
# ═══════════════════════════════════════════
# 评分 ≥ SELL_THRESHOLD(60) 则触发卖出

SELL_THRESHOLD: int = 60               # 卖出触发线

# ── S: 硬止损 ──
HARD_STOP_LOSS: float = -0.08          # S1 硬止损 -8%（分值 100，穿透）
HARD_STOP_LOSS_WARN: float = -0.05     # S2 止损预警 -5%（分值 40）
SCORE_HARD_STOP: int = 100
SCORE_HARD_STOP_WARN: int = 40

# ── T: 移动止盈（激活条件 +15%） ──
TRAILING_ACTIVATE: float = 0.15         # 持仓收益率≥15%才激活
TRAILING_T1: float = 0.08              # T1: 从高点回落≥8%（分值 85，强卖）
TRAILING_T2: float = 0.05              # T2: 从高点回落≥5%（分值 50，预警）
SCORE_TRAIL_T1: int = 85
SCORE_TRAIL_T2: int = 50

# ── D: 时间止损 ──
MAX_HOLD_DAYS: int = 20                # D1 持有>20日（分值 75）
MAX_HOLD_DAYS_WARN: int = 15           # D2 持有>15日（分值 40）
SCORE_TIME_D1: int = 75
SCORE_TIME_D2: int = 40

# ── M: 均线死叉（需连续 N 日） ──
MA_DEATH_CROSS_DAYS: int = 3           # 需连续3日 MA5<MA10 才确认
SCORE_MA_CROSS_CONFIRM: int = 70       # 已确认死叉（分值 70）
SCORE_MA_CROSS_TODAY: int = 40         # 首次出现（分值 40）

# ── SH: 夏普率恶化 ──
SHARPE_WINDOW_15: int = 15             # 15日夏普
SHARPE_WINDOW_10: int = 10             # 10日夏普
SCORE_SHARPE_BAD: int = 70             # 15日夏普<-0.5（分值 70）
SCORE_SHARPE_NEG: int = 50             # 10日夏普<0（分值 50）
SCORE_SHARPE_LOW: int = 30             # 15日夏普<0.5（分值 30）

# ── R: 相对弱势 ──
SCORE_RELATIVE_WEAK: int = 60           # 近5日跑输指数>5%（分值 60）
SCORE_RELATIVE_WARN: int = 30           # 近5日跑输指数>3%（分值 30）
RELATIVE_LOOKBACK: int = 5              # 回看天数
RELATIVE_WEAK_THRESHOLD: float = -0.05  # 跑输 5%
RELATIVE_WARN_THRESHOLD: float = -0.03  # 跑输 3%

# ── 参考指数 ──
INDEX_SYMBOL: str = "sh.000001"         # 上证指数
