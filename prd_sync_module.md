# PRD：Sequoia-X 数据同步模块独立化

## 1. 项目信息

| 字段 | 内容 |
|------|------|
| Language | 中文 |
| Programming Language | Python 3.12+ |
| Project Name | `sequoia_x` |
| 关联模块 | `sequoia_x/data/sync.py`（新建） |
| 原始需求 | 将数据同步逻辑从 `engine.py` 独立为专用模块，融合旧脚本有用特性，完善交易日判断、上市/退市检测、运行日志等能力 |

---

## 2. 产品定义

### 2.1 Product Goals

1. **解耦同步与查询**：`DataEngine` 只保留数据查询能力（`get_ohlcv` / `get_base_stock_pool` 等），同步逻辑全部移入 `sequoia_x/data/sync.py`，使两个关注点各司其职、可独立测试。

2. **健壮的数据同步管线**：基于 baostock 交易日历精确判断是否交易日、数据是否缺失；用 `query_stock_basic` 实时对比上市/退市变化（替代 30 日间隔 + 文件对比方案）；支持单日增量、多日补填、首次回填三种场景。

3. **可观测与可运维**：每次运行输出清晰的中间状态（是否为交易日、API 连通性、同步进度、新股/退市数量、错误信息），支持手动运行和定时调度两种模式，同步结果写入 `sync_log` 可追溯。

### 2.2 User Stories

- **As a** 量化策略开发者，**I want** 在非交易时段（如周末）运行同步程序时自动跳过，**so that** 避免无效 API 调用和错误日志。
- **As a** 系统运维者，**I want** 数据同步模块可单独通过命令行手动执行（含补填历史数据），**so that** 在初次部署或数据异常时能快速修复而无需启动整个系统。
- **As a** 选股策略依赖方，**I want** 同步完成后自动校验最近交易日的覆盖率（>85% 视为完整），**so that** 选股前可确认数据质量，避免因数据缺失导致选股结果偏差。
- **As a** 015/016 项目维护者，**I want** 新同步模块稳定运行后停用旧脚本并切换数据源到本项目数据库，**so that** 统一数据入口、消除多源不一致风险。

---

## 3. 技术规范

### 3.1 Requirements Pool

#### P0 — 必须实现

| 编号 | 需求 | 说明 |
|------|------|------|
| P0-1 | **新建独立同步模块** `sequoia_x/data/sync.py` | 包含类 `DataSync`，接受 `Settings` 初始化，内置所有同步逻辑。模块可通过 `python -m sequoia_x.data.sync` 单独运行。 |
| P0-2 | **交易日精确判断** | 使用 baostock `query_trade_dates` 判断当日是否为交易日。非交易日自动跳过数据拉取，输出明确日志："今日非交易日，跳过同步"。 |
| P0-3 | **上市/退市实时检测** | 每次同步前调用 `bs.query_stock_basic` 获取全量上市股票列表，与本地 `stock_daily` 中 DISTINCT symbol 对比。退市股从本地删除，新股加入待拉取队列。替代旧的 30 日间隔检查方案。 |
| P0-4 | **增量日线同步** (`sync_daily`) | 遍历本地所有 symbol，对比 `MAX(date)` 与最新交易日，拉取缺失日期数据。逻辑从 `sync_today_bulk` 移植。保留 17:30 时间门控（baostock 日线入库时间），`force=True` 可跳过。 |
| P0-5 | **缺失数据回填** (`backfill_missing`) | 检测最近 N=15 个交易日中覆盖率 < 90% 的日期，自动回填。逻辑从 `sync_and_clean` 步骤 4 移植。 |
| P0-6 | **历史数据回填** (`backfill`) | 单进程顺序拉取指定 symbol 列表从 `start_date` 至今的全部日线。支持续传（已有数据自动跳过）。逻辑从 `DataEngine.backfill` 移植。 |
| P0-7 | **数据完整性检查** (`check_completeness`) | 检查最近交易日覆盖率，返回 `{is_complete, coverage, latest_trade_day, ...}`。逻辑从 `check_data_completeness` 移植。阈值统一为 85%（当前正式标准）。 |
| P0-8 | **运行中间日志** | 每次同步输出：今日是否为交易日、API 登录状态、股票总数、退市数、新股数、逐批进度（每 1000 只汇报）、最终耗时、sync_log 写入状态。 |
| P0-9 | **保留 engine.py 查询能力** | 从 `DataEngine` 中删除 `sync_and_clean` / `sync_today_bulk` / `repair_data` / `backfill` / `check_data_completeness` 方法。保留 `get_ohlcv` / `get_base_stock_pool` / `get_all_symbols` / `get_local_symbols` / `get_trade_calendar`（后两个也可迁入 sync.py，视架构决策而定）。 |

#### P1 — 应该实现

| 编号 | 需求 | 说明 |
|------|------|------|
| P1-1 | **CLI 入口 & 子命令** | `python -m sequoia_x.data.sync daily`（增量）、`python -m sequoia_x.data.sync repair`（修复/补填）、`python -m sequoia_x.data.sync backfill --symbols sh.600000,sz.000001`（历史回填）、`python -m sequoia_x.data.sync check`（完整性检查）。 |
| P1-2 | **sync_log 完整性增强** | sync_log 新增字段：`is_trade_day` (bool)、`api_status` (str)、`coverage` (float)。 |
| P1-3 | **错误隔离** | 单只股票拉取失败不影响其他股票，记录到 `sync_errors` 表或日志，最终汇总："成功 X 只 / 失败 Y 只"。 |
| P1-4 | **连接复用优化** | 避免每次登录/登出（当前 sync_and_clean 中存在多次 login/logout），改为在整个同步会话中保持一个连接，每 N=200 只重连一次（backfill 模式已采用此策略）。 |

#### P2 — 可以后续实现

| 编号 | 需求 | 说明 |
|------|------|------|
| P2-1 | **定时调度集成** | 提供适配 Windows 任务计划程序 / cron 的配置建议文档。不内置调度器，但确保 CLI 接口可直接被调度工具调用。 |
| P2-2 | **旧脚本迁移适配层** | 015/016 项目切换数据源时所需的数据库读取工具函数或连接配置文档。 |
| P2-3 | **同步速率自适应** | 根据 baostock API 响应时间动态调整请求间隔，当前固定 50-200ms 已够用。 |

### 3.2 模块结构草案

```
sequoia_x/data/
├── __init__.py
├── engine.py          # 保留：DataEngine（仅数据查询方法）
├── sync.py            # 新建：DataSync（所有同步逻辑）
└── schema.py          # 可选：SQL 建表语句提取
```

**DataSync 类方法概要：**

| 方法 | 来源 | 说明 |
|------|------|------|
| `__init__(settings)` | 新建 | 初始化 db_path、start_date、logger |
| `is_trade_day(date)` | 增强 `get_trade_calendar` | 判断指定日是否为交易日 |
| `get_active_stocks()` | 新建（融合旧脚本逻辑） | 调用 `query_stock_basic` 获取全量上市 A 股，过滤指数 |
| `sync_stock_list()` | 替代旧 30 日检查 | 对比 `query_stock_basic` 与本地，处理上市/退市 |
| `sync_daily(force)` | 移植 `sync_today_bulk` | 增量日线同步 |
| `backfill_missing()` | 移植 `sync_and_clean` 步骤 4 | 检测并回填缺失交易日 |
| `backfill(symbols)` | 移植 `backfill` | 历史数据回填 |
| `repair()` | 移植 `repair_data` | 一键修复（退市清理 + 新股发现 + 缺失回填 + 增量拉取） |
| `check_completeness()` | 移植 `check_data_completeness` | 数据完整性检查 |
| `run_full()` | 替代 `sync_and_clean` | 完整同步管线（一日一次的主入口） |
| `(CLI) main()` | 新建 | argparse 命令行入口 |

### 3.3 CLI 使用方式草案

```bash
# 每日增量同步（含上市/退市检测 + 增量拉取）
python -m sequoia_x.data.sync daily

# 完整修复（强制模式，不限时间）
python -m sequoia_x.data.sync repair

# 历史回填（指定股票，或全部）
python -m sequoia_x.data.sync backfill --symbols sh.600000,sz.000001
python -m sequoia_x.data.sync backfill --all

# 数据完整性检查
python -m sequoia_x.data.sync check
```

### 3.4 数据库变更

sync_log 表增强（新增字段）：

```sql
ALTER TABLE sync_log ADD COLUMN is_trade_day INTEGER DEFAULT 1;
ALTER TABLE sync_log ADD COLUMN api_status TEXT DEFAULT '';
ALTER TABLE sync_log ADD COLUMN coverage REAL DEFAULT 0.0;
ALTER TABLE sync_log ADD COLUMN duration_seconds REAL DEFAULT 0.0;
```

### 3.5 Open Questions

| # | 问题 | 影响范围 |
|---|------|----------|
| Q1 | `get_trade_calendar` 方法保留在 `DataEngine` 还是迁入 `DataSync`？建议保留一份在 engine 供查询场景复用，sync.py 内部自行调用或引用 engine 版本。 | P0 范围界定 |
| Q2 | `get_all_symbols` / `get_local_symbols` 是否也从 engine 移除？当前 demand 只说删除 sync 方法。建议保留在 engine 中供选股策略使用。 | P0 范围界定 |
| Q3 | 旧脚本 03 的 CSV 输出功能是否需要在新模块中保留？建议不保留——新模块以 SQLite 为唯一数据源，CSV 输出由上层按需实现。 | 功能取舍 |
| Q4 | 旧脚本 04 的 6 个指数（sh.000001, sh.000300 等）日线是否需要纳入同步范围？当前 `stock_daily` 表中有这些指数数据。建议作为可选配置项。 | 数据范围 |
| Q5 | `backfill` 是否需要支持 `--start-date` 参数覆盖配置中的 `start_date`？建议支持，增加回填灵活性。 | P1 实现细节 |
| Q6 | `check_completeness` 覆盖率阈值从当前的 45%（手动演示模式）调整到 85%（正式标准），是否需要做成可配置？建议可配置，默认 85%。 | 运维灵活性 |

---

*文档版本: v1.0 | 创建日期: 2025-01-20 | 作者: Alice (Product Manager)*
