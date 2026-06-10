# Sequoia-X: 王者回归 | The King Returns

> **A 股量化选股系统 V2** — 三阶层选股 + WxPusher 微信推送 + DeepSeek LLM 多维度研判，每日收盘后自动运行

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 目录

- [三阶层选股架构](#-三阶层选股架构)
- [数据同步模块（DataSync）](#-数据同步模块datasync)
- [内置策略与排名](#-内置策略与排名)
- [每日选股流程](#-每日选股流程)
- [快速开始](#-快速开始)
- [推送配置（WxPusher）](#-推送配置wxpusher)
- [DeepSeek LLM 配置](#-deepseek-llm-配置)
- [Cron 定时任务](#-cron-定时任务)
- [参考文档](#-参考文档)

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

## 🤖 数据同步模块（DataSync）

> 基于 baostock 的 A 股日线全量/增量自动同步，职责从 `DataEngine` 独立为 `DataSync` 类。

### 四阶段同步管线 (`run_full()`)

```
Phase 1: sync_stock_list()       → baostock 全量列表对比，检测上市/退市
Phase 2: sync_daily(force=False) → 增量日线（交易日判断 + 17:30 时间门控）
Phase 3: repair_missing(days=5)  → 诊断缺失 + 自动补填（含2轮指数退避重试）
Phase 4: sync_index_daily()      → 6大指数日线（stock_daily 隔离存储）
```

### 核心特性

| 特性 | 说明 |
|------|------|
| 批量写入优化 | 500条/批，持久化SQLite连接，避免逐只提交开销 |
| 幂等写入 | INSERT OR REPLACE 基于 UNIQUE(symbol,date) |
| 断连恢复 | 连续50次错误自动重连 + 指数退避重试 |
| 交易日判断 | baostock query_trade_dates API（fail-open） |
| 新字段支持 | pctChg, peTTM, pbMRQ, psTTM, pcfNcfTTM |
| pandas 3.0 兼容 | _bs_get_data() 手动拼接，避免 baostock 内部 df.append() |

### 命令速查

| 命令 | 用途 | 耗时 |
|------|------|------|
| `python main.py --sync-only` | 日常自动同步（四阶段） | ~5-8 min |
| `python main.py --repair --all` | 全量修复缺失 | 按缺失量 |
| `python main.py --repair` | 快速修复前50只 | < 1 min |
| `python main.py --backfill` | 首次历史全量回填 | ~50 min |

详细文档：[股票及指数日线数据拉取模块使用指南.md](./股票及指数日线数据拉取模块使用指南.md)

---

系统每日分两个时段自动运行：

### 时间线

```
17:45  数据同步（--sync-only）
    ↓
├─ Phase 1: 股票列表同步（上市/退市检测）
├─ Phase 2: 增量日线同步（交易日判断 + 时间门控）
├─ Phase 3: 缺失补填（诊断 + 2轮重试）
├─ Phase 4: 6大指数日线同步
└─ 推送同步摘要到微信

20:55  选股策略运行
    ↓
├─ 1. 检查数据完整性 → check_missing(days=5)
│   ├─ 覆盖率 > 90% → ✅ 继续进行
│   └─ 覆盖率 ≤ 90% → ❌ 推送告警，跳过选股
├─ 2. 基础股票池过滤（~2800 只）
├─ 3. 7 策略独立选股 + 按分数取前 5
├─ 4. DeepSeek LLM 综合研判
├─ 5. WxPusher 推送到微信
└─ 预计耗时 3~5 分钟
```

### 退市股与新股处理

每次 `--sync-only` 运行时自动执行：

| 操作 | 判断依据 | 处理方式 |
|------|---------|---------|
| **退市股清理** | baostock 当前上市列表 vs 本地数据库 | 自动删除 `stock_daily` 中对应记录 |
| **新股发现** | baostock 有新代码 | 记录日志，下次增量同步会自动拉取 |
| **缺失交易日** | 对比交易日历和本地数据 | 自动回填缺失交易日数据 |
| **同步记录** | 每次同步后写入 `sync_log` 表 | 用于 20:55 检查完整性 |

### 命令速查

| 命令 | 用途 | 执行时间 |
|------|------|---------|
| `python main.py --sync-only` | 数据同步+清洗 | 17:45（cron） |
| `python main.py` | 完整选股+推送 | 20:55（cron） |
| `python main.py --skip-llm` | 选股但跳过 LLM 分析 | 手动 |
| `python main.py --backfill` | 首次回填/补充数据 | 手动 |

### 日志说明

- `logs/sync_YYYYMMDD.log` — 17:45 同步日志（含退市/新股/补填信息）
- `logs/daily_YYYYMMDD.log` — 20:55 选股日志

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

日常模式自动完成：增量补数据 → 获取基础股票池 → 7 策略选股 → WxPusher 微信推送。

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
📈 Sequoia-X 选股播报 | 均量线突破
日期: 2026-06-08
选股数量: 3

选股列表:
- 平安银行 (000001)
- 万科A (000002)
- 贵州茅台 (600519)
```

---

## ⏰ Cron 定时任务

系统在交易日自动运行两个阶段，cron 配置如下：

```cron
# ===== Sequoia-X 数据同步 (17:45) =====
# 退市清理 → 新股发现 → 缺失补填 → 增量拉取 → 推送同步摘要到微信
10 19 * * 1-5 cd /path/to/Sequoia-X && /path/to/python main.py --sync-only >> logs/sync_$(date +\%Y\%m\%d).log 2>&1

# ===== Sequoia-X 选股推送 (20:55) =====
# 检查数据完整性 → 基础池过滤 → 7策略选股 → LLM研判 → 推送结论到微信
55 20 * * 1-5 cd /path/to/Sequoia-X && /path/to/python main.py >> logs/daily_$(date +\%Y\%m\%d).log 2>&1
```

### 推送消息速览

| 时间 | 触发条件 | 推送内容 | 样式 |
|:---:|---------|---------|:----:|
| 17:45 同步成功 | `run_full()` 正常返回 | 股票数、退市清理数、新股发现数、补填天数、耗时 | 📊 数据同步完成 |
| 17:45 同步失败 | `run_full()` 异常 | 错误信息、股票数 | ⚠️ 数据同步失败 |
| 20:55 选股正常 | 数据覆盖率 > 85% | LLM 综合研判报告（含大盘/个股/财务/舆情分析） | 📈 AI 选股研判 |
| 20:55 数据不足 | 覆盖率 ≤ 85% | 覆盖率、有数据/总股票数、可能原因 | ❌ 选股已取消 |

### 回填数据加速

当使用 `--backfill` 回填全市场历史数据时，系统采用 **多进程并行** 模式：
- 每批 200 只股票，9 进程并发
- 每只股票间隔 200ms（防止 baostock 封 IP）
- 单线程约 60 分钟 → 并行约 10~15 分钟

### 日志文件说明

| 日志文件 | 对应模式 | 生产时间 |
|---------|---------|---------|
| `logs/sync_YYYYMMDD.log` | `--sync-only` | 17:45 |
| `logs/daily_YYYYMMDD.log` | 常规模式 | 20:55 |

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

## 📚 参考文档

- [股票及指数日线数据拉取模块使用指南.md](./股票及指数日线数据拉取模块使用指南.md) — DataSync 详细使用手册
- [004_Sequoia-X量化选股系统开发部署运行指南.md](./004_Sequoia-X量化选股系统开发部署运行指南.md) — 系统部署运维指南

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
