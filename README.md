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

### 五阶段同步管线

```
Phase 1: sync_stock_list()        → 上市/退市检测（type="1" status="1" 过滤）
Phase 2: sync_daily()             → 增量日线同步（baostock优先→TencentSource回退）
Phase 3: repair_missing(days=5)   → 缺失补填（双源回退 + 2轮重试）
Phase 4: _fill_valuation_gaps()   → 估值字段回填（baostock健康检查，失败跳过）
Phase 5: sync_index_daily()       → 6大指数日线（上证/深证/沪深300/上证50/中证500/深证综指）
```

### 双轨数据源

| 数据源 | 状态 | 字段 | 稳定性 |
|:------|:----|:-----|:-------|
| **Baostock** | 主数据源 | 全字段（含peTTM/pbMRQ等） | 有时断连 |
| **TencentSource** | 备用 | OHLCV（估值字段=None） | **稳定** 0.34s/只 |

切换逻辑：baostock 失败 → 自动切 Tencent → 每 50 只再试一次 baostock。

### 核心特性

| 特性 | 说明 |
|:------|:------|
| 双轨抗故障 | baostock 故障时自动切换腾讯/新浪 API |
| 停牌数据保全 | volume=0，价格沿用前值，不允许数据空洞 |
| 主动重连 | 每 1400 次请求主动重建连接，避免限流 |
| 指数同步 | 6 大指数日线独立存储于 index_daily 表 |
| SQLite 优化 | WAL 模式 + synchronous=NORMAL |

---

## 📅 每日执行流程

```
=== 时段1: 数据同步 (cron 18:10) ===
  │
  ├─ Phase 1~5: 完整同步管线
  └─ WxPusher 推送同步摘要到微信
       │
=== 时段2: 策略选股 (cron 19:00) ===
  │
  ├─ 第1层：check_missing 数据完整性检查（覆盖率>90%继续）
  ├─ 第2层：get_base_stock_pool 基础股票池过滤
  │        科创板/创业板/北交所/ST/次新/低价 → 排除
  ├─ 第3层：7策略独立选股 + 打分取前5
  ├─ 第4层：实时数据采集（知兔API + 新浪 + 本地DB）
  ├─ 第5层：DeepSeek LLM 综合研判
  └─ 第6层：WxPusher 推送分析报告到微信
       │
=== 时段3: 辅助项目 (cron 19:05~19:10) ===
  ├─ 19:05  015 指标扫描模拟盘
  └─ 19:10  016 ETF LSTM 预测
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

## ⏰ Cron 定时任务

```cron
# ===== Sequoia-X 数据同步 (18:10) =====
10 18 * * 1-5 cd /path/to/sequoia-x && conda run -n zhulei_py312 python main.py --sync-only >> logs/sync_$(date +\%Y\%m\%d).log 2>&1

# ===== Sequoia-X 策略选股 (19:00) =====
0 19 * * 1-5 cd /path/to/sequoia-x && conda run -n zhulei_py312 python main.py >> logs/daily_$(date +\%Y\%m\%d).log 2>&1
```

> 注意：cron 中 `%` 必须转义为 `\%`

| 时间 | 任务 |
|:----:|:------|
| 18:10 | Sequoia-X 数据同步 |
| 19:00 | Sequoia-X 策略选股 |
| 19:05 | 015 指标扫描模拟盘 |
| 19:10 | 016 ETF LSTM 预测 |

---

## 📚 参考文档

- [数据同步框架需求与运行指南.md](./%E6%95%B0%E6%8D%AE%E5%90%8C%E6%AD%A5%E6%A1%86%E6%9E%B6%E9%9C%80%E6%B1%82%E4%B8%8E%E8%BF%90%E8%A1%8C%E6%8C%87%E5%8D%97.md) — 同步模块详细文档
- [004_Sequoia-X量化选股系统开发部署运行指南.md](./004_Sequoia-X%E9%87%8F%E5%8C%96%E9%80%89%E8%82%A1%E7%B3%BB%E7%BB%9F%E5%BC%80%E5%8F%91%E9%83%A8%E7%BD%B2%E8%BF%90%E8%A1%8C%E6%8C%87%E5%8D%97.md) — 系统部署运维指南

---

## 📄 许可证

MIT
