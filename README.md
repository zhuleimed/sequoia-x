# Sequoia-X: A 股量化选股系统 V2

> **全自动量化选股系统** — 双轨数据同步(baostock+腾讯) + 7策略选股 + WxPusher 微信推送 + DeepSeek LLM 多维度研判

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue)](https://python.org)

---

## 📋 目录

- [数据同步模块](#-数据同步模块)
- [每日执行流程](#-每日执行流程)
- [三阶层选股架构](#-三阶层选股架构)
- [内置策略](#-内置策略)
- [快速开始](#-快速开始)
- [数据源说明](#-数据源说明)
- [Cron 定时任务](#-cron-定时任务)
- [参考文档](#-参考文档)

---

## 🤖 数据同步模块

5206 只 A 股 + 6 个指数日线数据自动同步，双数据源抗故障。

### 六阶段同步管线

```
Phase 1:  sync_stock_list()         → 上市/退市检测（type="1" status="1" 过滤）
Phase 1b: _archive_delisted_stocks() → 退市数据归档至 stock_daily_archive
Phase 2:  sync_daily()              → 增量日线同步（Tencent主力→baostock后备）
Phase 2b: _fill_ohlcv_gaps()         → OHLCV历史缺失补填（Tencent）
Phase 3:  _fill_valuation_gaps()     → 估值字段补充（baostock，超2h跳过）
Phase 4:  sync_index_daily()        → 6大指数日线
```

### 双轨数据源

| 数据源 | 状态 | 字段 | 稳定性 |
|:------|:----|:-----|:-------|
| **TencentSource** | **主力** | OHLCV+成交量+成交额+pctChg(计算) | **稳定** 0.34s/只 |
| **Baostock** | 后备 | 全字段（含peTTM/pbMRQ等） | 有时断连 |

切换逻辑：Tencent(主力) → 失败切 baostock(后备) → 每 200 只尝试恢复 Tencent。pctChg 由 Tencent 拉取后计算。

### 核心特性

| 特性 | 说明 |
|:------|:------|
| 双轨抗故障 | Tencent 主力拉取，baostock 后备保障 |
| 停牌数据保全 | volume=0，价格沿用前值，不允许数据空洞 |
| 主动重连 | 每 1400 次请求主动重建连接(baostock)，避免限流 |
| 退市数据归档 | 退市股行情移至 stock_daily_archive，主表仅含活跃股 |
| 指数同步 | 6 大指数日线独立存储于 index_daily 表 |
| SQLite 优化 | WAL 模式 + synchronous=NORMAL |

---

## 📅 每日执行流程

```
=== 全自动管线（cron 唯一入口 18:10）===
  │
  ├─ Step 1: 数据同步（main.py --sync-only）
  │    6阶段管线(Phase 1~4) → 同步完成立即下一步
  │
  ├─ Step 2: 策略选股+LLM（main.py）
  │    数据检查→8策略选股→LLM→推送，同时保存LLM推荐到模拟盘买入信号
  │
  ├─ Step 3: 模拟盘更新（main.py --sim-update）
  │    T+1开盘买入→每日估值→多因子评分卖出→单股报告+组合日报推送
  │
  └─ 推送全管线汇总到微信
```

---

## 🏛️ 三阶层选股架构

| 步骤 | 方法 | 说明 |
|:----|:-----|:------|
| **第一步** | `DataEngine.get_base_stock_pool()` | 统一过滤基础池 |
| **第二步** | 各策略 `run()` | 策略选股，构造 `(symbol, score)` |
| **第三步** | `_pick_top(candidates)` | 按分数降序取前 5 |

---

## 📊 内置策略

| 策略 | 排名依据 |
|:-----|:---------|
| MaVolumeStrategy 均量线突破 | 放量倍数 |
| TurtleTradeStrategy 海龟交易法则 | 流通市值 |
| HighTightFlagStrategy 高紧旗形 | 动量/收敛比 |
| LimitUpShakeoutStrategy 涨停洗盘 | 放量/跌幅比 |
| UptrendLimitDownStrategy 上涨回调 | 放量倍×跌幅 |
| RpsBreakoutStrategy RPS动量突破 | RPS分数 |
| PrivatePlacementStrategy 定增公告 | 公告日期 |

---

## 🚀 快速开始

```bash
# 1. 创建环境
conda create -n zhulei_py312 python=3.12 -y
conda activate zhulei_py312
pip install uv && uv sync

# 2. 配置 .env
cp .env.example .env

# 3. 首次历史回填
python main.py --backfill

# 4. 运行数据同步
python main.py --sync-only

# 5. 运行选股
python main.py
```

---

## 📦 数据源说明

- **主数据源**：baostock（免费，端口 10070，有时不稳定）
- **备用数据源**：腾讯证券 API（web.ifzq.gtimg.cn，稳定）
- **复权方式**：前复权（adjustflag=2 / qfq）— 全项目统一
- **存储**：SQLite `data/sequoia_v2.db`（约 600MB）
- **数据覆盖**：5206 只 A 股 + 6 大指数，从 2024-01-01 至今
- **扩展字段**：pctChg, peTTM, pbMRQ, psTTM, pcfNcfTTM, amount（由 `fill_extra_fields.py` 补全）

---

## 📈 模拟盘交易模块

LLM 推荐股票自动进入 T+1 模拟盘交易，全程自动化。

| 特性 | 说明 |
|:------|:------|
| 初始资金 | 100 万，单只分配 5 万，最多 20 只 |
| 买入价 | T+1 日开盘价（检查涨停/停牌/仓位上限） |
| 卖出触发 | 多因子评分 ≥ 60（硬止损/移动止盈/时间/均线死叉/夏普/相对弱势） |
| 交易成本 | 佣金万2.5 + 印花税千1 + 滑点万1 |
| 报告推送 | 单股平仓报告 + 组合日报（微信） |
| 新策略接入 | `submit_buy_signals()` 一行代码即可 |

详细说明见：[模拟盘交易模块使用说明.md](./模拟盘交易模块使用说明.md)

---

## ⏰ Cron 定时任务

```cron
# ===== Sequoia-X 全自动管线（唯一入口，顺序执行 sync→strategy→simulation）=====
10 18 * * 1-5 cd /public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x && /home/zhulei/anaconda3/envs/zhulei_py312/bin/python pipeline/pipeline.py >> logs/pipeline_$(date +\%Y\%m\%d).log 2>&1
```

> 注意：cron 中 `%` 必须转义为 `\%`  
> 旧入口（main.py --sync-only / main.py）已由 pipeline/pipeline.py 统一编排

| 时间 | 任务 | 说明 |
|:----:|:------|:------|
| 18:10 每日 | 全自动管线启动 | 同步→选股→模拟盘 链式执行 |
| 20:00 月末 | 模拟盘月度报告 | 推送本月交易汇总（胜率/盈亏比/夏普/回撤） |

状态文件：`/public/home/hpc/zhulei/superman/quant/code/pipeline_status.json`（code 根目录）

### 新增项目

在 `pipeline/pipeline.py` 的 `STEPS` 列表加一项即可，无需改 cron。详见[数据同步框架指南](./数据同步框架需求与运行指南.md#九全自动管线pipeline)。

---

## 📚 参考文档

- [数据同步框架需求与运行指南.md](./%E6%95%B0%E6%8D%AE%E5%90%8C%E6%AD%A5%E6%A1%86%E6%9E%B6%E9%9C%80%E6%B1%82%E4%B8%8E%E8%BF%90%E8%A1%8C%E6%8C%87%E5%8D%97.md) — 同步模块详细文档
- [004_Sequoia-X量化选股系统开发部署运行指南.md](./004_Sequoia-X%E9%87%8F%E5%8C%96%E9%80%89%E8%82%A1%E7%B3%BB%E7%BB%9F%E5%BC%80%E5%8F%91%E9%83%A8%E7%BD%B2%E8%BF%90%E8%A1%8C%E6%8C%87%E5%8D%97.md) — 系统部署运维指南
- [模拟盘交易模块使用说明.md](./%E6%A8%A1%E6%8B%9F%E7%9B%98%E4%BA%A4%E6%98%93%E6%A8%A1%E5%9D%97%E4%BD%BF%E7%94%A8%E8%AF%B4%E6%98%8E.md) — 模拟盘模块详细文档
- [模拟盘交易逻辑与运行全流程说明.md](./%E6%A8%A1%E6%8B%9F%E7%9B%98%E4%BA%A4%E6%98%93%E9%80%BB%E8%BE%91%E4%B8%8E%E8%BF%90%E8%A1%8C%E5%85%A8%E6%B5%81%E7%A8%8B%E8%AF%B4%E6%98%8E.md) — 买卖/覆盖/换股完整逻辑

---

## 📄 许可证

MIT
