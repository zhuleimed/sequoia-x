# Sequoia-X: 王者回归 | The King Returns

> **A 股量化选股系统 V2** — 三阶层选股 + WxPusher 微信推送，每日收盘后自动运行

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 目录

- [三阶层选股架构](#-三阶层选股架构)
- [内置策略与排名](#-内置策略与排名)
- [快速开始](#-快速开始)
- [推送配置（WxPusher）](#-推送配置wxpusher)
- [Cron 定时任务](#-cron-定时任务)
- [目录结构](#-目录结构)
- [数据说明](#-数据说明)
- [新增策略指南](#-新增策略指南)
- [许可证](#-许可证)

---

## 🏛️ 三阶层选股架构

每次选股严格走以下三步，确保结果安全、精准、可排序：

### 第一步：基础股票池
`DataEngine.get_base_stock_pool()` 统一过滤所有 A 股：
| 过滤条件 | 规则 | 剔除对象 |
|---------|------|---------|
| 🚫 板块剔除 | 688/689/300/301/4xx/8xx | 科创板、创业板、北交所 |
| 🚫 质量剔除 | 名称含 ST/*ST/退 | ST股、退市风险股 |
| 🚫 次新剔除 | 上市日期 < 1年 | 次新股 |
| 🚫 低价剔除 | 最新收盘价 < 2元 | 低价股 |

### 第二步：策略选股 + 打分
每个策略只在 **基础股票池** 范围内运行，每个候选股票附带 **分数（score）**。

### 第三步：按分数取前 5 支
`_pick_top(candidates, top_n=5)` 按分数降序取前 5，确保推送的是策略最看好的股票。

---

## 📊 内置策略与排名

| 策略 | 中文名 | 排名依据 |
|------|-------|---------|
| **MaVolumeStrategy** | 均量线突破 | **放量倍数**（今日量/20日均量） |
| **TurtleTradeStrategy** | 海龟交易法则 | **流通市值**（市值越大流动性越好） |
| **HighTightFlagStrategy** | 高紧旗形突破 | **动量倍数/收敛幅度**（爆发潜力） |
| **LimitUpShakeoutStrategy** | 涨停洗盘 | **放量/跌幅比**（洗盘充分度） |
| **UptrendLimitDownStrategy** | 上涨回调 | **放量倍 × 跌幅**（错杀反弹潜力） |
| **RpsBreakoutStrategy** | RPS动量突破 | **RPS分数**（全市场动量百分位） |
| **PrivatePlacementStrategy** | 定增公告监控 | **公告日期**（最新优先） |

---

## 🚀 快速开始

### 环境要求

- Python >= 3.10

### 1. 安装依赖

```bash
# 推荐使用 uv（快速包管理器）
uv sync

# 或使用 pip
pip install -r requirements.txt

# 安装 WxPusher 推送库
uv pip install wxpusher
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填写 WxPusher Token
# WXPUSHER_TOKEN=your_app_token
# WXPUSHER_TOPIC_IDS=["39277"]
```

### 3. 首次回填历史数据

```bash
python main.py --backfill
```

约 12~60 分钟完成 ~5200 只 A 股历史后复权日 K 数据回填（视网络与服务器性能而定）。

### 4. 日常运行

```bash
python main.py
```

日常模式自动完成：增量补数据 → 基础股票池 → 7策略选股 → LLM综合研判 → 一次性推送。

---

## 🔔 推送配置（WxPusher）

本项目使用 [WxPusher](https://wxpusher.zjiecode.com) 替代原飞书推送，将选股结果推送至微信。

### 获取 WxPusher Token

1. 前往 [WxPusher 官网](https://wxpusher.zjiecode.com) 注册
2. 创建应用，获取 AppToken
3. 创建 Topic，获取 Topic ID
4. 扫码关注，即可接收消息

### 配置

```ini
# .env
WXPUSHER_TOKEN=AT_xxxxxxxxxxxxxxxxxxxxxxxx
WXPUSHER_TOPIC_IDS=["39277"]
```

### 消息示例

```
📊 Sequoia-X AI 综合研判 | 2026-06-08

📈 大盘环境
上证指数: 3050.23 (-0.68%) | 深证成指: 9250.18 (-1.25%)

🔍 个股深度分析
**1. 怡球资源 (601388) — 海龟交易法则** ⭐3.8/5
→ 一季报净利同比+179.68%，盈利拐点明确
...（由 DeepSeek 实时生成）

🏆 综合建议
- 最优关注: 怡球资源、电投水电
- 操作建议: 逢低分批建仓，设-5%止损
```

---

## ⏰ Cron 定时任务

建议每个交易日收盘后自动执行（配合 baostock 数据更新时间 17:30，推荐 19:00 后运行）：

```cron
# 每个交易日（周一至周五）20:55 自动选股推送
55 20 * * 1-5 cd /path/to/Sequoia-X && /path/to/python main.py >> logs/daily_$(date +\%Y\%m\%d).log 2>&1
```

### 回填数据加速

当使用 `--backfill` 回填全市场历史数据时，系统采用 **多进程并行** 模式：
- 每批 200 只股票，9 进程并发
- 每只股票间隔 200ms（防止 baostock 封 IP）
- 单线程约 60 分钟 → 并行约 10~15 分钟

---

## 📁 目录结构

```
Sequoia-X/
├── main.py                      # 入口：argparse 分发日常/回填模式
├── pyproject.toml               # 依赖声明 + 测试配置
├── .env.example                 # 环境变量模板
├── .env                         # 环境变量（不入 git）
├── data/                        # SQLite 数据库（运行时生成，不入 git）
├── logs/                        # 运行日志（不入 git）
├── sequoia_x/
│   ├── __init__.py
│   ├── core/
│   │   ├── config.py            # Pydantic-settings 配置管理
│   │   └── logger.py            # rich 结构化日志
│   ├── data/
│   │   └── engine.py            # 数据引擎（回填 + 增量同步 + 基础池）
│   ├── strategy/
│   │   ├── base.py              # 策略抽象基类（含 _pick_top 排序）
│   │   ├── turtle_trade.py      # 海龟交易策略
│   │   ├── ma_volume.py         # 均线放量策略
│   │   ├── high_tight_flag.py   # 高窄旗形策略
│   │   ├── limit_up_shakeout.py # 涨停洗盘策略
│   │   ├── uptrend_limit_down.py# 上升跌停策略
│   │   ├── rps_breakout.py      # RPS 突破策略
│   │   └── private_placement.py # 定增公告监控
│   └── notify/
│       └── wxpusher.py          # WxPusher 微信推送
└── tests/                       # 属性测试（pytest + hypothesis）
    ├── test_config.py
    ├── test_data_engine.py
    ├── test_feishu.py
    ├── test_logger.py
    ├── test_main.py
    └── test_strategy.py
```

---

## 📦 数据说明

- **数据源**：[baostock](http://baostock.com)（免费、无需注册、无限流）
- **复权方式**：后复权（hfq）— 历史价格不变，适合增量存储
- **存储**：本地 SQLite（`data/sequoia_v2.db`），可直接拷贝使用
- **增量更新**：`sync_today_bulk()` 多进程并行补数据，2~3 分钟完成
- **数据更新时间**：日 K 线每个交易日 17:30 入库，建议 19:00 后运行

---

## ➕ 新增策略指南

新增策略必须遵守三阶层选股架构：

```python
from sequoia_x.strategy.base import BaseStrategy

class MyNewStrategy(BaseStrategy):
    webhook_key: str = "my_new"
    display_name: str = "新策略名称"
    top_n: int = 5

    def run(self) -> list[str]:
        symbols = self.stock_pool or self.engine.get_local_symbols()
        candidates: list[tuple[str, float]] = []  # (代码, 分数)

        for symbol in symbols:
            # ... 选股逻辑 ...
            if 满足条件:
                score = ...  # 策略核心信号强度
                candidates.append((symbol, score))

        return self._pick_top(candidates, self.top_n)  # 按分排序取前N
```

**铁律**：
1. ✅ 用 `self.stock_pool`（而非全量遍历）
2. ✅ 构造 `(symbol, score)` 
3. ✅ 用 `_pick_top()` 截取
4. ✅ 在 `main.py` 的 `strategies` 列表注册

---

## 📄 许可证

MIT
