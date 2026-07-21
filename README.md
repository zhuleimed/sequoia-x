# Sequoia-X: A 股量化选股系统 V2

> **全自动量化选股系统** — 三轨数据同步 + LLM 多维度研判 + LSTM-Transformer 深度学习选股 + 双账户模拟盘 + WxPusher 微信推送

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue)](https://python.org)

---

## 📋 目录

- [数据同步模块](#-数据同步模块)
- [每日执行流程](#-每日执行流程)
- [选股策略](#-选股策略)
- [模拟盘交易](#-模拟盘交易)
- [快速开始](#-快速开始)
- [Cron 定时任务](#-cron-定时任务)
- [参考文档](#-参考文档)

---

## 🤖 数据同步模块

~5200 只 A 股 + 6 个指数日线数据自动同步，三轨抗故障。

### 三轨数据源

| 优先级 | 数据源 | 用途 |
|:---:|------|------|
| 1 | **TencentSource** | OHLCV+成交量+成交额 (主力，0.34s/只) |
| 2 | **SinaSource** | 独立备份 (与Tencent解耦) |
| 3 | **Baostock** | 全字段（含估值指标 peTTM/pbMRQ 等） |

每只股票依次尝试 Tencent → Sina → baostock，源健康追踪自动降级/恢复。

### 股票名称查询（独立于日线同步）

本地 SQLite (`stock_list.name`) 优先 → 腾讯实时行情 API 回退并自动缓存。不再依赖 baostock。

---

## 📅 每日执行流程

```
=== 全自动管线（cron 唯一入口 18:10）===
  │
  ├─ Step 1: 数据同步 (--sync-only)
  │    三轨日线同步 → 完成立即下一步
  │
  ├─ Step 2: LLM 策略选股 (main.py)
  │    8策略选股 → LLM分析 → WeChat/Feishu推送
  │
  ├─ Step 2.5: LSTM 增量学习 (--incremental)
  │    加载最新模型 → 近60日数据微调10轮 (~2min)
  │
  ├─ Step 2.6: LSTM 预测+模拟盘
  │    预测2964只 → 选top2 → 写买入信号 → SimEngine → 日报推送 (~3min)
  │
  ├─ Step 3: LLM 模拟盘更新 (--sim-update)
  │    T+1开盘买入 → 估值更新 → 多因子卖出 → 日报推送
  │
  └─ Step 3.5: 双策略汇总推送
       LLM + LSTM 合并账户概览 → 微信
```

---

## 📊 选股策略

### LLM 策略 (8 规则)

| 策略 | 排名依据 |
|:-----|:---------|
| MaVolumeStrategy 均量线突破 | 放量倍数 |
| TurtleTradeStrategy 海龟交易法则 | 流通市值 |
| HighTightFlagStrategy 高紧旗形 | 动量/收敛比 |
| LimitUpShakeoutStrategy 涨停洗盘 | 放量/跌幅比 |
| UptrendLimitDownStrategy 上涨回调 | 放量倍×跌幅 |
| RpsBreakoutStrategy RPS动量突破 | RPS分数 |
| PrivatePlacementStrategy 定增公告 | 公告日期 |

### LSTM-Transformer 策略 (深度学习)

| 参数 | 值 |
|------|------|
| 模型架构 | LSTM → TransformerBlock → LSTM → Dense |
| 时序窗口 | 120 交易日 |
| 特征维度 | 62 维（价格/量能/均线/技术指标/波动率/大盘关联/价格形态） |
| 标签 | 超额收益 (stock_ret - index_ret)，Huber loss |
| 归一化 | 每特征 Z-score 沿时间轴标准化 |
| 模型版本 | v20260721_1323，Rank IC=0.1575 |

**三层训练调度**：

| 层级 | 命令 | 频率 | 耗时 |
|------|------|------|:---:|
| 月度全新 | `--full` (Optuna 100 trials + 300 epochs) | 每月15日 00:00 | 60-80h |
| 每周刷新 | `--weekly` (最佳参数 + 252日数据) | 每周六 00:00 | 2-3h |
| 每日增量 | `--incremental` (微调10轮) | 每交易日 18:10 | ~2min |

冲突防护：日/周训练入口 pgrep 检测 `--full` 进程，存在则自动跳过。

---

## 📈 模拟盘交易

双账户独立核算，共用同一套 SimEngine 和卖出规则。

| 账户 | 初始资金 | 最大持仓 | 选股来源 |
|------|:---:|:---:|------|
| LLM | 100 万 | 20 只 | 8 策略规则 + LLM 分析 |
| LSTM | 50 万 | 10 只 | LSTM-Transformer 预测 |

**卖出规则**：13 条多因子评分，总分 ≥ 60 触发 T+1 开盘卖出。

| 类别 | 规则 | 分值 |
|------|------|:---:|
| S 硬止损 | 亏损≥8% / ≥5% | 100(穿透) / 40 |
| T 移动止盈 | 盈利≥15%→回落≥8% / ≥5% | 85 / 50 |
| D 时间 | 持有>20日 / >15日 | 75 / 40 |
| M 均线死叉 | MA5<MA10 连续≥3日 / 首日 | 70 / 40 |
| SH 夏普率 | 15日<-0.5 / 10日<0 / 15日<0.5 | 70 / 50 / 30 |
| R 相对弱势 | 跑输大盘>5% / >3% | 60 / 30 |

**报告推送**：单策略日报 + 双策略汇总 + 月末月度报告（微信 WxPusher）。

---

## 🚀 快速开始

```bash
# 1. 环境
conda activate zhulei_py312

# 2. 配置
cp .env.example .env  # 编辑 WxPusher Token 等

# 3. 数据同步
python main.py --sync-only

# 4. LLM 选股
python main.py

# 5. LSTM 预测（手动）
python -m sequoia_x.model_selection.predict

# 6. LSTM 模拟盘（手动）
python -m sequoia_x.model_selection.simulation.daily
```

---

## ⏰ Cron 定时任务

| 时间 | 任务 |
|------|------|
| **18:10 每日** (Mon-Fri) | 全自动管线 (sync→LLM→LSTM增量→LSTM预测→LLM模拟→双策略汇总) |
| **00:00 每月15日** | LSTM 月度全新训练 (--full, 60-80h) |
| **00:00 每周六** | LSTM 每周刷新 (--weekly, 2-3h, --full冲突时跳过) |
| **18:30 月末** | LLM 模拟盘月度报告 |
| **18:35 月末** | LSTM 模拟盘月度报告 |

---

## 📚 参考文档

- [数据同步框架需求与运行指南.md](./数据同步框架需求与运行指南.md)
- [004_Sequoia-X量化选股系统开发部署运行指南.md](./004_Sequoia-X量化选股系统开发部署运行指南.md)
- [模拟盘交易模块使用说明.md](./模拟盘交易模块使用说明.md)
- [模拟盘交易逻辑与运行全流程说明.md](./模拟盘交易逻辑与运行全流程说明.md)
- [LSTM-Transformer 模型选股设计文档](docs/superpowers/specs/2026-07-20-lstm-model-selection-design.md)
