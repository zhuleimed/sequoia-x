# THS + Mootdx 迁移方案（第一层）— 彻底摆脱 baostock

> 版本：v1 | 日期：2026-08-20 | 状态：待 review
> 目标：赶在 **2026-08-31 月末链（19:00 启动）+ 09-01 03:00 重训** 前，完成估值/财务/维度换源，
> 从 9/1 起 V2 体系**完全脱离 baostock**。维度变更触发缓存重建 + 重训，与月末链合并成一次。

---

## 1. 背景与决策依据（实测）

2026-08-20 已完成一组只读探测，证据链完整：

| 项 | 结论 | 实测来源 |
|---|---|---|
| 同花顺估值快照 | 口径对齐 baostock（茅台 pe -1%）、**亏损股负值保留**、补 ps/pcf、**无限流**（百股×20批 0 触发 4001） | `probe_hithink_valuation.py` |
| 同花顺**不提供历史估值** | 官方契约明说"不提供历史估值"，只能补当日 | `llms-full.txt` §4991 |
| 同花顺财务指标 | `report` 多期**可回查**（2025-1/2024-3/2023-1/2022-4 均 OK），5 类 23 指标连续值 | `probe_hithink_dimensions.py` [A] |
| 同花顺财务报表 | income/balance 支持 `quarterly+start/end(ms)` **历史区间**，含 `report_date_ms`/`holder_equity_total`/`parent_holder_net_profit` | 同上 [B] |
| 同花顺短线维度 | 龙虎榜/涨停池/热榜/竞价/异动 **全部落地**（code=0） | 同上 [C] |
| **同花顺无业绩预告** | 契约 0 命中 → forecast 4 特征**无法替换** | `llms-full.txt` |
| **mootdx 无业绩预告** | Quotes 方法无 forecast/yjyb | 实测 |
| mootdx 已拉资产 | finance 5206/forecast 5178/fund_flow 5206/holders 5161/xdxr 5206/consensus 3322/news 5206 + **mootdx_finance 119期585列** | 本地盘点 [D] |

**核心判断**：forecast 4 维是唯一孤儿，且系统已自标"语义可缺失、缺失不影响训练"（`features_extra/README.md`）。
用同花顺财务指标(23)/龙虎榜/涨停等**高信息量连续/情绪维度替代**，信息量净增，彻底摆脱 baostock。

---

## 2. 迁移范围（第一层，8/31 前必须）

| # | 动作 | 类型 | 说明 |
|---|---|---|---|
| 1 | **同花顺财务重建 finance + 历史估值** | 换源 | 用 ths 财务指标/报表重建 finance 段；历史 pe/pb 用报表重建 |
| 2 | **砍 forecast 4 维** | 删除 | 121→117 基准；forecast parquet/采集器/特征函数移除 |
| 3 | **同花顺龙虎榜 + 涨停池** | 新增 | 2 个高确定性情绪维度接入 121 维 |
| 4 | **估值日常换同花顺快照** | 已做 | `HithinkValuationSource` Phase 3b 已实现验证（B 完成） |
| 5 | **交易日历/股票列表换源** | 换源 | is_trade_day 近期用 ths；list 用 ths tickers/list |
| 6 | **feature_version 3→4** | 触发重建 | labels.py 3 处 |
| 7 | **缓存重建 + 干跑 + 预演重训 + 回测验证** | 流程 | 8/31 月末链整合 |

**暂缓（第二层，9 月迭代）**：热榜/集合竞价/异动、mootdx_finance 585、mootdx DDE 增强。
理由：第一层信息量已足够，控制 8/31 前的风险。

---

## 3. 详细设计（每步文件/特征/对齐）

### 3.1 同花顺财务重建 finance + 历史估值（改 `scripts/collect_extra_features.py` + `build_extra_features.py`）

**当前**：`finance` 段 = 同花顺 akshare 10 列（`fetch_finance`，降级 mootdx finance 快照），
`_finance_features` 读 `finance` parquet 算 10 维（roe/gp_margin/np_margin/debt_ratio/rev_yoy/profit_yoy/chg/cf_quality/eps/bps），asof 用 `_disclose_date` 法定披露日。

**改为（ths 财务指标 23 重建）**：
- **新增采集** `fetch_hithink_finance(code)`：调 `/financials/indicators?report=<各期季报>` 逐期回查，
  存 `finance_ths` 子集 parquet，字段含 `index_id/value/report`。
- **`_finance_features` 改为读 `finance_ths`**（保留 `finance` 作降级后备），扩展输出列：
  - 原 10 维保留（roe/gp_margin 等映射自 ths 的 `index_weighted_avg_roe`/`sale_gross_margin`/...）
  - **新增**：`fin_rev_yoy`、`fin_asset_yoy`、`fin_current_ratio`、`fin_quick_ratio`、`fin_cash_ratio`、`fin_debt_hard`(资产负载率)、`fin_inv_turn`、`fin_ar_turn`、`fin_asset_turn`、`fin_ocf_np`(净利现金含量)... 从 23 指标可取 15+ 个连续特征
  - **asof 对齐不变**：`report` → `_disclose_date` 法定披露日，防 look-ahead 零成本
- **历史估值重建**（替代 baostock 历史 pe/pb）：
  - 新增 `rebuild_valuation_history.py`：per stock 拉 `income`（`basic_eps`）+ `balance`（`holder_equity_total`）+ 股本
    → `pe = price/eps`、`pb = price/(holder_equity/total_shares)`，按 `report_date_ms` asof 对齐写回 `stock_daily.peTTM/pbMRQ`（ps/pcf 用报表科目）。
  - **口径**：需确认 ths 的 `basic_eps` 是 TTM 还是单季/累计，以及股本来源（总股本/股本科目）。**这一步是关键风险点，先小样本对拍 baostock 历史值。**

### 3.2 砍 forecast（改 `collect_extra_features.py` + `build_extra_features.py` + `month_end_pull.py`）

- `collect_extra_features.py`：`SUBSETS` 去掉 `"forecast"`，删 `fetch_forecast`
- `build_extra_features.py`：`FEATURE_GROUPS` 删 `"forecast"`, 删 `_forecast_features`, 特征清单 `"forecast"` 项去掉；33 列契约 → 33 - 4 + 新增长度（见 3.1/3.3，总量变化见 §5）
- `month_end_pull.py`：`SOFT` 去掉 `"forecast"`（无需保留），`KEY/SOFT` 复查
- 现有 `forecast` parquet 保留不删（历史参考，不再读）

### 3.3 新增同花顺龙虎榜 + 涨停池（新采集器 + 新特征）

**采集**（新增 `fetch_hithink_sentiment.py` 或并入 collect_extra_features）：
- **龙虎榜** `/dragon-tiger-list?board_type=all&date=<交易日>`（按交易日取，asof 自然对齐）
- **涨停池** `/limit-up-pool?date=<交易日>`（含 continue_day_cnt 连板、seal_money 封单、price_change）
- 存 `dragon_tiger` 子集 parquet + `limit_up` 子集 parquet（per (code,date)）

**特征**（`build_extra_features.py` 新增两个 group）：
- `_dragon_tiger_features`：`dt_net_buy`(净买入) / `dt_net_rate`(净买入率) / `dt_hot_rank` / `dt_range_days`(上榜天数) → 事件日 asof（龙虎榜仅上榜日有值，非上榜 fillna0）
- `_limit_up_features`：`lu_lianban`(连板数) / `lu_seal`(封单额/亿) / `lu_days30`(近30日涨停次数) / `lu_break`(炸板标记) → 事件日 asof

**防 look-ahead**：龙虎榜/涨停按 **交易日** 对齐（当日盘后数据次日生效，asof 天然安全）。

### 3.4 估值日常换同花顺快照（已完成，B 交付物）

`HithinkValuationSource`（tencent_source.py）+ `_fill_valuation_gaps` Phase 3b 已实现验证。
**本轮无新增**，只确认月末链 `DO_ESTIMATION=1` 时走新 Phase 3b。

### 3.5 交易日历/股票列表换源（改 `sync.py`）

- **is_trade_day 第 2 层**：baostock → ths `/calendar/trading-days`（近期；历史回测用本地快照，另议）
- **get_active_stocks**：baostock → ths `/meta/tickers/list`（分页 1000/页）+ 本地差异对比
- 注意：股票池换源后 n_stocks/hash 可能变 → **缓存重建的 pool 一致性**（month_end_pull `_refresh_stock_pool` 同源）

### 3.6 feature_version 3→4（改 `labels.py` 3 处）

```python
"feature_version": 3,  →  4   # labels.py:283, 372, 558
```
**含义**：v4 = 特征层换源（finance 重建 + 砍 forecast + 新增龙虎榜/涨停）。缓存 hash 变更 → 全量重建。

---

## 4. 数据/缓存重建流程

### 4.1 依赖链
```
collect_extra_features(新源) → finance_ths/dragon_tiger/limit_up parquet
  → feature_version=4
  → rebuild_dataset_cache(全量, 121+88+80 维)   # 8/31 月末链
  → build_prediction_cache
  → v2_monthly_retrain(09-01 03:00)
```

### 4.2 月末链整合（关键：一次做）
8/31 19:00 `month_end_pull.py`：
1. `_refresh_stock_pool`（换源后池一致）
2. `collect_extra_features` 全量刷新（含 finance_ths/dragon_tiger/limit_up；删 forecast）
3. `rebuild_dataset_cache --extra`（feature_version=4 → **全量重建**，121/88/80）
4. `_verify_caches`（新增维度的 coverage 检查，KEY/SOFT 更新）
5. 干跑 `build_prediction_cache` → 9/1 03:00 `v2_monthly_retrain` 轮询

### 4.3 干跑预演（8/29 前）
- 用当前日期预演 `rebuild_dataset_cache` + `build_prediction_cache`，确保 feature_version=4 全链路通
- 抽查新特征 coverage（dragon_tiger 上榜日占比 / limit_up 连板分布），防维度全 0 陷阱

---

## 5. 维度变化清单（信息量净增）

| 数据面 | 当前 | 改后 | 变化 |
|---|---|---|---|
| fund_flow | 6 | 6 | 不变 |
| finance | 10 | **~15+**（ths 23 指标重建） | +5+ |
| holders | 2 | 2 | 不变 |
| consensus | 5 | 5 | 不变 |
| news | 3 | 3 | 不变 |
| xdxr | 3 | 3 | 不变 |
| **forecast** | 4 | **0** | **-4** |
| **dragon_tiger** | 0 | **4** | **+4** |
| **limit_up** | 0 | **4** | **+4** |
| **合计** | **33** | **~37+** | **净 +4+** |

→ 121 维 → **~125+ 维**，feature_version=4。

---

## 6. 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| **ths basic_eps/股本口径 ≠ baostock TTM** | 历史估值重建错 | 先 20 只对拍 baostock 历史值，确认口径；不符则 TTM 折算 |
| 龙虎榜/涨停 coverage 低（上榜股少） | 特征大量 0，模型忽略 | 事件型特征正常；coverage 记录，不设 hard gate（col SOFT） |
| ths 免费 Key 限流/波动 | 月末链拉取失败 | 重试 + failed 清单 + mootdx/TDX 降级链（保留） |
| feature_version=4 全量重建耗时 | 2-6h | 8/31 19:00 启动，02:00 前完成；断点续跑 marker |
| 换股票池源 hash 变 | 缓存重建 pool 不一致 | month_end_pull `_refresh_stock_pool` 与新 list 同源 |
| **9/1 前未完成** | 模型延期 | 第二层维度(热榜等)砍掉只保第一层；8/29 干跑卡点 |

---

## 7. 时间表（8/20 今天起）

| 天 | 动作 | 交付 | 卡点 |
|---|---|---|---|
| 8/20(今天) | 本方案 doc review | 确认范围 | 你 review |
| 8/21-22 | 3.1 财务重建 + 历史估值对拍 | finance_ths + 对拍报告 | 口径一致 |
| 8/23-24 | 3.3 龙虎榜/涨停采集+特征 | 新 parquet + build 扩展 | coverage 合理 |
| 8/25 | 3.2 砍 forecast + 3.5 换源 | 清理 + sync 改 | 无报错 |
| 8/26 | 3.6 feature_version=4 + 本地重建 | 全量重建通过 | 断点续跑 |
| 8/27-28 | 干跑: rebuild + prediction + 抽查 | 预演通过 | 无维度全 0 |
| 8/29 | 预演重训 + 回测验证 | 模型产出 + 回测对比 | 收益不劣化 |
| 8/30 | 缓冲/修复 | — | — |
| **8/31** | **月末链正式跑（新体系）** | 128+ 缓存 + 信号 | 02:00 前重建完 |
| **9/1** | **03:00 重训 + 选股** | **新体系首月推荐** | 微信推送 |

**关键里程碑**：8/29 干跑全链路通过 = 可上 8/31 月末链；否则退回第二层再说。

---

## 8. 需要你 review 拍板的点

1. **范围**：确认按"第一层 5+2 步"执行（财务重建/砍forecast/龙虎榜/涨停/换源/feature_version）？第二层（热榜/竞价/mootdx_finance/DDE）明确**9 月迭代**，对吗？
2. **finance 特征扩展**：从 ths 23 指标里选 15+ 个，还是保守只保留原 10 维（降低对口径依赖）？（我建议先 10 维对齐原版，确认口径后二轮再加，**降风险**）
3. **历史估值重建**：先用 20 只对拍 baostock 再定 TTM 折算；若对拍不符，是否接受"新增日用同花顺、历史库存保留 baostock 值（B1）"过渡？
4. **feature_version**：确认直接 3→4（会全量重建），不接受部分缓存复用。

请 review 给出意见，确认后 8/21 开始实施。
