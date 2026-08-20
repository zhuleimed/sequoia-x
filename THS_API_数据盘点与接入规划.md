# 同花顺 Financial-API 数据种类盘点 与 接入规划（第二层依据）

> 版本：v1 | 日期：2026-08-20 | 状态：**第二层规划依据**（第一层=V4 129维已落地，本文件规划第二层）
> 数据来源：同花顺官方契约（https://fuyao.aicubes.cn/llms-full.txt，2026-08-20 实测验证）
> API Key：`HITHINK_FINANCE_API_KEY`（环境变量）
> 相关：THS_MOOTDX_MIGRATION_PLAN.md（第一层）

---

## 1. 决策背景

- 用户确认：**后续 finance 数据源从 akshare『同花顺摘要』切换为『同花顺官方 API』**（官方更稳定、无限流、字段规范）
- 同花顺 API 数据种类非常丰富，需**系统盘点**找可利用的新种类
- 本文件为**第二层（9月迭代）**的规划依据；第一层（V4 129维）已完成，**不动**

---

## 2. 全 API 数据种类盘点（REST 端点）

### 2.1 A股核心域 `/api/a-share`

| 数据 | 端点 | 项目现状 | 可利用度 |
|---|---|---|---|
| **实时行情** | `prices/snapshot` | 腾讯/新浪 | 后备 |
| **历史K线** | `prices/historical`（10年窗口） | 腾讯/新浪 | 后备 |
| **利润表** | `financials/income-statements` | akshare | **切换①** |
| **资产负债表** | `financials/balance-sheets` | akshare | **切换①** |
| **现金流量表** | `financials/cash-flow-statements` | **无** | **新增②** |
| **财务指标23** | `financials/indicators` | 无/部分 | **新增③** |
| 估值快照 | `valuations/snapshot` | 已用 | ✅ 已接 |
| 龙虎榜 | `special-data/dragon-tiger-list` | 已用 | ✅ |
| 涨停池 | `special-data/limit-up-pool` | 已用 | ✅ |
| 跌停池/炸板池 | `limit-down-pool`/`limit-break-pool` | 无 | 新增④ |
| **连板天梯** | `limit-up-ladder` | 无 | 新增④ |
| 热榜(+历史) | `hot-stock-list(-history)` | 已砍(覆盖0.6%) | 评估 |
| **个股排名走势** | `hot-stock-rank-trend` | 无 | 新增 |
| 集合竞价快照+基准 | `auction/snapshot`+`short-term-benchmark` | 无 | 新增④ |
| 异动原因 | `anomaly-analysis-*` | 无 | 新增④ |
| 火箭股 | `skyrocket-list` | 无 | 评估 |
| 复权因子 | `corporate-actions/adjustment-factors` | mootdx xdxr | 后备 |
| 交易日历 | `calendar/trading-days` | 已用 | ✅ |

### 2.2 指数/板块域 `/api/a-share-index`
- 指数列表/成分股/行情快照/历史K线
- 项目：指数日线已用 baostock/mootdx → **可切同花顺**（更全，含沪深300成分股）

### 2.3 标的域 `/api/meta`
- `tickers/list`（全A股清单）+ `tickers/search`（检索）
- 项目：已用 list（get_active_stocks）✅

### 2.4 基金域 `/api/fund`（本次暂不接入，记录备用）
- 基金资料/持仓/业绩/经理/分红/NAV/重仓（股票持仓/债券/行业配置）等 30+ 端点
- **潜在价值**：基金重仓股（北向/机构抱团信号）、行业配置——可能新的 alpha 信号源

---

## 3. 高价值新数据（不在当前体系，第二层候选）

| 数据 | 端点 | 潜在信号 | 优先级 |
|---|---|---|---|
| **现金流量表** | `financials/cash-flow-statements` | 盈利真实性（经营现金流 vs 净利润） | 高 |
| **财务指标23** | `financials/indicators` | ROE/毛利率/负债率规范值 + 更准披露日 | 高 |
| **连板天梯** | `limit-up-ladder` | 情绪周期强度 | 中 |
| **个股排名走势** | `hot-stock-rank-trend` | 单股热度趋势（比热榜快照有用） | 中 |
| **集合竞价基准** | `auction/short-term-benchmark` | 开盘情绪风向标 | 中 |
| **异动原因(按股)** | `anomaly-analysis-stock` | 事件驱动标签 | 中 |
| 基金重仓 | `fund/portfolio/holdings` | 机构抱团/北向信号 | 低（量大） |

---

## 4. 关于 finance 切换官方 API（用户点①）——关键设计

### 4.1 现状 vs 官方
| | 现状（akshare同花顺摘要） | 官方 API |
|---|---|---|
| 字段 | 10列摘要 | `financials/indicators` 23指标 + 报表细目 |
| 披露日 | 法定披露日（保守推算） | **`report_date_ms`（真实披露日）** |
| asof 精度 | 保守（可能晚于真实披露，丢信息） | **精确（防 look-ahead 更准）** |
| 稳定性 | akshare 版本依赖/限流 | 官方稳定无限流 |

### 4.2 设计要点
1. **新建采集器** `fetch_hithink_finance(code)` → `finance_ths` 子集 parquet
   - `financials/indicators`（23指标）+ `financials/cash-flow-statements`（现金流量）+ 报表细目
   - asof 用 **`report_date_ms`**（真实披露日，比保守法定日更能防 look-ahead + 信息更早可用）
2. **对拍验证**：`finance_ths` 与现有 `finance`（10维）对拍，确认同口径值一致（ROE/毛利率/eps/bps），防破坏历史缓存
3. **维度扩展**：在保留原16维基础上，新增 现金流/研发/商誉/应收存货等质量维度
4. **feature_version**：口径变 → 又触发缓存全量重建（第二层，非现在）

### 4.3 时序
- **现在不做**（等70月回测完成后，第二层独立迭代）
- 避免与正在跑的 70 月回测（用 akshare finance 的 129维）冲突

---

## 5. 第二层（9月）接入规划（建议顺序）

| 排 | 动作 | 依赖 | 说明 |
|---|---|---|---|
| 1 | **finance 切官方 API**（23指标+现金流+精确披露日） | 等待回测完成 | 对拍验证后替换/并存 |
| 2 | **新增质量维度** | 同1 | 现金流/研发/商誉/应收存货 |
| 3 | **短线情绪扩维** | — | 连板天梯/排名走势/竞价基准/异动 |
| 4 | 指数换同花顺 | — | a-share-index 替代 baostock/mootdx |
| 5 | mootdx_finance 精选接入 | — | 585→20-30维（独立富矿） |
| 6 | 基金重仓探索 | — | 机构抱团信号（留待评估） |

---

## 6. 注意事项

1. **切换 finance 会触发 feature_version 变 + 缓存全量重建**——放第二层。
2. **对拍验证必须**：官方 23指标 vs 现有 10维，同口径值一致才安全替换。
3. **`report_date_ms` 精度提升**：比保守法定日更准，但需验证其可靠（第一层曾发现 THS income 近期 report_date_ms 有误标问题——`_disclose_date` 兜底仍要留）。
4. **基金/另类数据** 量大、需评估成本收益，暂缓。
5. **本文件是规划依据**，实施在回测完成 + 用户批准后。

---

*(关联：THS_MOOTDX_MIGRATION_PLAN.md、memory/hithink-finance-api-eval.md、本项目 CLUUDE.md)*
