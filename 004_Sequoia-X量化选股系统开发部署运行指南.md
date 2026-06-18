# 项目四：Sequoia-X A股量化选股系统 — 完整开发部署手册

---

## 0. 项目基本信息

| 项目 | 值 |
|------|-----|
| 项目名称 | Sequoia-X A股量化选股系统 V2 |
| 项目简称 | sequoia-x |
| GitHub 仓库 | https://github.com/zhuleimed/sequoia-x |
| 项目目录（服务器） | `/public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x/` |
| 项目目录（本地） | `G:\中心同步盘_20251027\文档\学习\001_量化平台\013_sequoia-x\` |
| 服务器 | `zhulei@[2001:250:4400:89:aae6:63d8:a8e0:51dc]` |
| 服务器配置 | Ubuntu LTS / 36核CPU / 192GB内存 / 无GPU |
| Python 环境 | `zhulei_py312`（Python 3.12.13） |
| 数据库 | SQLite → `data/sequoia_v2.db`（约600MB） |
| 数据源 | baostock(主) + TencentSource(备) 双轨 |
| 通知通道 | WxPusher 微信推送 |
| LLM 分析 | DeepSeek API（deepseek-v4-flash） |

**API 密钥（.env 中配置）：**

| 密钥 | 值 |
|:----|:----|
| WxPusher Token | `AT_hKGG0UfwrCP7bpcsO8cbQkrc4bZ9G3RX` |
| WxPusher Topic ID | `39277` |
| DeepSeek API Key | `sk-abb7f3b79c0c4f868156cdf92f45e141` |
| 知兔API Token | `2C0E4763-3F63-4174-9CE1-806A10D58FC3` |

---

## 1. 项目文件清单

```
004_sequoia-x/
├── main.py                          # 主入口（argparse 分发同步/修复/日常模式）
├── fill_extra_fields.py             # 扩展字段补填脚本（amount/pctChg/peTTM/等，一次性）
├── pyproject.toml                   # 依赖声明 + ruff/pytest 配置
├── uv.lock                          # uv 锁定依赖版本
├── .env                             # 环境变量
├── .env.example                     # 环境变量模板
├── .gitignore                       # Git 忽略规则
├── README.md                        # GitHub 页面 README
├── 004_Sequoia-X量化选股系统开发部署运行指南.md  ← 本文档
├── 数据同步框架需求与运行指南.md         ← 同步模块详细文档
├── data/
│   └── sequoia_v2.db                # SQLite 数据库（运行时生成）
├── logs/                            # 运行日志（运行时生成）
├── sequoia_x/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py                # Pydantic-settings 配置管理
│   │   └── logger.py                # rich 结构化日志
│   ├── data/
│   │   ├── __init__.py
│   │   ├── engine.py                # 数据引擎（基础股票池过滤、行情查询）
│   │   ├── sync.py                  # DataSync 同步模块（5阶段管线）
│   │   ├── tencent_source.py        # TencentSource 腾讯/新浪双API数据源
│   │   └── save_results.py          # 选股结果保存模块
│   ├── strategy/
│   │   ├── __init__.py
│   │   ├── base.py                  # BaseStrategy 抽象基类
│   │   ├── ma_volume.py             # 均量线突破策略
│   │   ├── turtle_trade.py          # 海龟交易法则策略
│   │   ├── high_tight_flag.py       # 高紧旗形突破策略
│   │   ├── limit_up_shakeout.py     # 涨停洗盘策略
│   │   ├── uptrend_limit_down.py    # 上涨回调策略
│   │   ├── rps_breakout.py          # RPS动量突破策略
│   │   ├── rps_multi_period.py      # 多周期RPS突破策略
│   │   └── private_placement.py     # 定增公告监控策略
│   ├── analysis/
│   ├── notify/
│   └── ... (其他模块)
└── tests/
```

---

## 2. 整体架构

### 2.1 执行流程（双时段拆分）

```
=== 时段1: 数据同步 (18:10) ===
  │
  ├─ Phase 1: sync_stock_list() — baostock 全量列表对比(上市/退市检测)
  ├─ Phase 2: sync_daily() — 增量日线同步(baostock优先→Tencent回退)
  ├─ Phase 3: repair_missing(days=5) — 诊断缺失 + 自动补填(含Tencent回退)
  ├─ Phase 4: _fill_valuation_gaps(days=5) — baostock回填估值字段(跳过不卡死)
  ├─ Phase 5: sync_index_daily() — 6大指数日线同步(baostock→Tencent)
  └─ WxPusher 推送同步摘要到微信

=== 时段2: 策略选股 (19:00) ===
  │
  ├─ 第1层：check_missing(days=5) 数据完整性检查
  │    覆盖率>90%继续，否则告警跳过
  ├─ 第2层：get_base_stock_pool() 基础股票池过滤
  │    ├─ 剔除 科创板(688/689)、创业板(300/301)、北交所(4xx/8xx)
  │    ├─ 剔除 ST/*ST/退市股
  │    ├─ 剔除 上市不满1年的次新股
  │    ├─ 剔除 最新收盘价<2元的低价股
  │    └─ 输出 ≈ 2,500~3,000 只
  ├─ 第3层：7个策略独立选股 + 打分(取前5)
  ├─ 第4层：MarketAnalyst 数据采集(知兔API+新浪+本地DB)
  ├─ 第5层：DeepSeek LLM 综合研判
  ├─ 第6层：保存选股结果
  └─ 第7层：WxPusher 推送分析报告到微信
```

### 2.2 数据流架构

```
baostock API + TencentSource(腾讯/新浪)
    │
    ▼
DataSync（数据同步层，5阶段管线）
    ├─ sync_stock_list()    — 股票列表同步(上市/退市检测)
    ├─ sync_daily()         — 增量日线同步(双轨数据源)
    ├─ repair_missing()     — 缺失补填(含Tencent回退)
    ├─ _fill_valuation_gaps() — 估值字段回填
    └─ sync_index_daily()   — 6大指数日线同步
    │
    ▼
SQLite (stock_daily + index_daily + stock_list + sync_log)
    │
    ▼
7 × BaseStrategy.run() — 各自读取DB，独立选股
    │
    ▼
MarketAnalyst.analyze() — 实时行情 + LLM综合研判
    │
    ▼
WxPusher → 微信推送
```

---

## 3. 内置策略详解

(内容与之前相同，此处省略以节省篇幅，保持原样)

---

## 4. 部署步骤

### 前置条件

- 服务器：Ubuntu LTS，已安装 Python 3.10+ 和 Anaconda3
- WxPusher Token（已配置）
- DeepSeek API Key（已配置）

### 第1步：克隆代码

```bash
git clone https://github.com/zhuleimed/sequoia-x.git
cd sequoia-x
```

### 第2步：创建环境并安装依赖

```bash
conda create -n zhulei_py312 python=3.12 -y
conda activate zhulei_py312
pip install uv
uv sync
pip install wxpusher
```

### 第3步：配置环境变量

```bash
cp .env.example .env
# 编辑 .env 填入密钥
```

### 第4步：首次回填历史数据

```bash
python main.py --backfill
# 约 2~3 小时，单线程拉取 baostock
```

### 第5步：补全扩展字段（如需）

```bash
python -u fill_extra_fields.py >> fill_extra_fields.log 2>&1
# 补全 amount/pctChg/peTTM/pbMRQ/psTTM/pcfNcfTTM 从2024-01-01至今
# 预计耗时 ~70分钟
```

### 第6步：验证运行

```bash
python main.py --skip-llm    # 测试基础池获取 + 推送
python main.py               # 完整测试(含LLM)
```

---

## 5~7. 推送配置、LLM分析配置、定时任务

(章节内容保持，但更新cron时间)

### 7.1 执行层 — 服务器 Cron

**当前配置：**

```bash
# 同步 + 选股分拆为两个时段：

# ===== Sequoia-X 数据同步(18:10) =====
10 18 * * 1-5 cd /public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x \
  && /home/zhulei/anaconda3/envs/zhulei_py312/bin/python main.py --sync-only \
  >> logs/sync_$(date +\%Y\%m\%d).log 2>&1

# ===== Sequoia-X 策略选股(19:00) =====
0 19 * * 1-5 cd /public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x \
  && /home/zhulei/anaconda3/envs/zhulei_py312/bin/python main.py \
  >> logs/daily_$(date +\%Y\%m\%d).log 2>&1
```

**注意：** cron 中 `%` 必须转义为 `\%`

| 时间 | 项目 | 说明 |
|:----:|:----|:------|
| **18:10** | **Sequoia-X 数据同步** | 5阶段管线(含双轨数据源) |
| **19:00** | **Sequoia-X 策略选股** | 数据检查+7策略+LLM+推送 |
| 19:05 | 015 指标扫描模拟盘 | 独立项目 |
| 19:10 | 016 ETF LSTM 预测 | 独立项目 |

### 7.3 日志查看

```bash
# 查看同步日志
tail -50 logs/sync_$(date +\%Y\%m\%d).log

# 查看选股日志
tail -50 logs/daily_$(date +\%Y\%m\%d).log
```

---

## 8. 数据说明

### 8.1 baostock 数据更新时间

| 数据类型 | 更新时间 |
|---------|---------|
| 日K线 | 当前交易日 17:30~18:00 |
| 复权因子 | 当前交易日 18:00 |
| 财务数据 | 第二自然日 1:30 |

### 8.3 数据存储

- **复权方式**：前复权（adjustflag=2）— 全项目统一
- **存储路径**：`data/sequoia_v2.db`
- **数据库大小**：约 582 MB（全市场 5206 只 × 2.5年数据）
- **PRAGMA 优化**：journal_mode=WAL, synchronous=NORMAL

---

## 9. 命令行参数

```bash
python main.py --sync-only         # 仅数据同步(5阶段管线)
python main.py                     # 日常模式(检查数据→选股→LLM→推送)
python main.py --repair --all      # 缺失数据修复
python main.py --skip-llm          # 跳过 LLM 分析
python main.py --backfill          # 回填模式：全市场历史K线
python -u fill_extra_fields.py     # 补全扩展字段(一次性)
```

---

## 10. 版本历史

| 日期 | 版本 | 变更说明 |
|:----:|:----:|---------|
| 2026-06-08 | v1.0~v1.4 | 初始部署、WxPusher集成、7策略、LLM分析 |
| **2026-06-10** | **v2.0** | **重大重构**：数据同步模块独立为 DataSync 类；修复 start<today_str Bug；get_active_stocks 增加 type/status 过滤；连续错误阈值 50→10；_write_to_db 保留停牌数据、空值前向填充；新增 fill_extra_fields.py |
| **2026-06-12** | **v2.1** | 同步时间 17:45→18:10；请求间隔 0.05s→0.15s；重连逻辑增强(5次指数退避)；cron %转义修复；TencentSource 双轨初步集成 |
| **2026-06-18** | **v2.2** | **全面双轨化** |
| **2026-06-18** | **v2.3** | **is_trade_day三层判断：周末过滤→baostock→chinese_calendar+fail-open** |**：Phase 3 repair_missing 新增 Tencent 回退；Phase 4 新增 baostock 健康检查(跳过不卡死)；Phase 5 sync_index_daily 新增 Tencent 回退；日志降级(每只→DEBUG) + 进度日志(每30s)；SQLite PRAGMA 优化(WAL+NORMAL)；停牌数据填充逻辑完善；cron 分拆为同步(18:10)和选股(19:00) |

---

## 附录：关键配置速查

### .env 文件

```ini
DB_PATH=data/sequoia_v2.db
START_DATE=2024-01-01
WXPUSHER_TOKEN=AT_hKGG0UfwrCP7bpcsO8cbQkrc4bZ9G3RX
WXPUSHER_TOPIC_IDS=["39277"]
DEEPSEEK_API_KEY=sk-abb7f3b79c0c4f868156cdf92f45e141
DEEPSEEK_MODEL=deepseek-v4-flash
```

### 服务器 SSH

| 项目 | 值 |
|:----|:----|
| Host | `[2001:250:4400:89:aae6:63d8:a8e0:51dc]` |
| Port | 22 |
| Username | `zhulei` |
| Password | `zhulei@HPC88660159` |

### Python 环境

| 环境 | 路径 |
|:----|:-----|
| zhulei (旧) | `/home/zhulei/anaconda3/envs/zhulei/bin/python` |
| zhulei_py312 (主) | `/home/zhulei/anaconda3/envs/zhulei_py312/bin/python` |
