# Sequoia-X V2 体系完整操作指南（最终版 v4.0，2026-08-11）

> 本文档是 V2 体系的**最终权威指南**，从数据同步到月末缓存重建、下月初月度重训的完整闭环，
> 涵盖原理、目的、意义、做法、步骤、流程、逻辑、功能与经验。旧版（v3.0）见
> `V2_OPERATION_GUIDE_v3_backup.md`；研究方向与实验记录见 `V3研究方向与实验研究记录.md`；
> 教训/规则见 memory/；RESEARCH_STATE.md 为脚本自动生成的实验状态快照（勿手改）。
>
> **版本要点（v4.0 新增/修订）**：
> - 复权口径铁律（全库不复权 + 计算层后复权，feature_version=3）
> - fund_flow 全历史切换（新浪源，2018 起）→ 121 维采样日 10 → 143 天
> - 月末缓存**增量复用**（同参数旧缓存复制 + 只构建新增月份，重建 2-6h → 30-40min）
> - 月度重训时间 00:00 → **03:00**（与月末链错开）
> - LLM 研判推送修复（DeepSeek 推理 max_tokens 4096 → 8192）
> - 月末链 workers 优化（--workers 16，双 job 32 进程）

---

# 第一部分：体系总览

## 1. 项目定位与体系架构

**Sequoia-X V2** 是 A 股量化选股与模拟盘系统：每日自动同步行情 → 特征/模型预测 → 策略选股
→ LLM 多维研判 → 双模拟盘（V2 + LLM）执行与风控 → 微信推送。每月自动完成"月末扩展维度
拉取 + 训练缓存重建 + 月初重训 + 选股信号入库"的完整闭环。

```
数据层 ──→ 特征/模型层 ──→ 信号层 ──→ 执行层 ──→ 推送层
 日线同步      88/121 维      策略选股       V2 模拟盘     微信
 扩展维度      T2/T1/T3        LLM 研判      LLM 模拟盘    日志
 (月末拉取)     T4 LSTM        信号入库      风控/月报
```

**核心目标**：每日产出可执行的选股信号（V2 每月 10 只 + LLM 每日 2 只），模拟盘 T+1 执行，
全程无人值守（cron 驱动），异常微信告警。

## 2. 数据流全景（一图看懂）

| 环节 | 触发 | 脚本 | 产出 |
|------|------|------|------|
| 日线同步 | 每日 18:10 | pipeline/pipeline.py → sync | stock_daily 更新到当日 |
| 日常选股 | 18:10 管线 | main.py 策略层 | results_YYYYMMDD.json |
| LLM 研判 | 18:10 管线 | MarketAnalyst | AI 研判推送 + 2 只买入信号 |
| 模拟盘 | 18:10 管线 | SimEngine | 持仓/日报 |
| 月末拉取 | 月末 19:00 | month_end_pull.py | 扩展维度全量刷新 |
| 缓存重建 | 月末（拉取后） | rebuild_dataset_cache.py | 121/88+80 维缓存（增量/全量） |
| 月度重训 | 1 日 03:00 | v2_monthly_retrain.py | 模型重训 + 10 只选股信号 |

## 3. 关键决策记录表（理解体系的钥匙）

| 决策 | 内容 | 定稿日 |
|------|------|--------|
| 复权口径 | 全库**不复权**实际价入库；复权只在计算层（后复权 adjust.py） | 2026-08-10 |
| 成交量单位 | 腾讯返回"手"×100 转"股"入库 | 2026-08-02 |
| 数据源四轨 | OHLCV: baostock/腾讯/新浪；估值: TDX/baostock；指数: baostock | 2026-07-24 |
| fund_flow | **新浪 MoneyFlow 全历史**（2018 起；东财仅 120 天） | 2026-08-11 |
| 特征版本 | feature_version=3（后复权补丁②） | 2026-08-10 |
| 模型 | T2/T1/T3 树模型 + T4 LSTM（纯 LSTM，L2=0，无 Transformer） | 2026-07-29 |
| 月末重建 | 全量（首次）→ **增量复用**（池不变月份） | 2026-08-11 |
| 重训时间 | 每月 1 日 **03:00**（由 00:00 调整，避开月末链 02:00 完成窗口） | 2026-08-10 |

---

# 第二部分：数据层

## 4. 数据源四轨制

| 数据 | 主力源 | 后备源 | 说明 |
|------|--------|--------|------|
| OHLCV 日线 | baostock（adjustflag="3" 不复权） | 腾讯（fq 留空）→ 新浪 | 三轨制健康度排序 |
| 估值 PE/PB/PS/PCF | TDX/mootdx | baostock | 实时估值 |
| 指数日线 | baostock | 腾讯 | 大盘环境特征 |
| 扩展维度（7 类） | 见 §8 | - | 仅月末拉取 |

## 5. 复权口径铁律（2026-08-10 实测固化）——最重要

**全库统一【不复权】实际成交价（除权日有完整跳空），复权只在计算层做**：

- **为什么**：前复权基准=拉取时点，每次除权后全历史都要重算——增量同步只拉当天、历史不重拉
  → 历史是"旧基准前复权"、当天是"新基准前复权" → 拼接假断层（10 送 10 假跌 50%）。
  不复权=实际价，永不漂移，任何时点自洽。
- **实测确认**（2026-08-10）：全库 100% 匹配腾讯 bfq / baostock adjustflag=3；79028 个除权事件验证一致。
- **所有拉取一律不复权**：baostock `adjustflag="3"`（sync.py 6 处）、腾讯 fq 参数留空读 `day`
  （tencent_source.py，原 qfq 会污染 repair/_fill_ohlcv_gaps 多行写库）、新浪 `getKLineData` 天然不复权。
- **复权在计算层**：特征/标签用 `model_selection_v2/adjust.py`（后复权模块，因子来自 xdxr 分红表），
  `feature_version=3` 为后复权版本。后复权基准=上市日，历史永不漂移，缓存一次永久有效。
- **执行层**（模拟盘/回测）必须用实际成交价，不走复权。
- 任何新增数据源/拉取路径：先确认复权口径（必须不复权）+ 成交量单位（手→股）。

## 6. 成交量单位规则（2026-08-02 实测固化）

腾讯接口返回"手"（lots），入库前 ×100 转"股"（`tencent_source.py::_tencent_kline` 已固化）；
baostock/Sina 返回"股"无需转换。新增数据源先确认单位。

## 7. 日线同步（三轨制，每日 18:10）

`DataSync.run_full()` 流程：
1. **股票列表**（stock_list）：全市场清单管理（退市清理/新股发现）
2. **日线增量**（daily_sync）：按"源健康度"排序尝试 baostock → 腾讯 → 新浪，每只股票拉最近
   5 天（增量），失败降级下一源；baostock 为 TCP 长连接必须单进程串行，连续失败 5 次降级
3. **缺口修复**（repair）：对比 baostock 最新交易日，补齐缺失（腾讯/新浪兜底，30 天窗口）
4. **估值补全**（valuation）：TDX 快照 → baostock 全史双保险
5. 同步完成推送摘要（股票数/退市/新股/补填/最新日期）

## 8. 扩展维度（7 类，仅月末拉取）

| 数据 | 主源 | 覆盖 | 用途 |
|------|------|------|------|
| fund_flow | **新浪 MoneyFlow**（sync_fund_flow_history.py） | 2018-05 起全历史 | 33 维中的 6 个资金流特征 |
| finance | 同花顺 akshare | 102 期全历史 | 10 个财务特征 |
| holders | 东财 datacenter | 2013 起 | 2 个股东结构特征 |
| consensus | 东财报告 | 近期快照 | 5 个一致预期特征 |
| news | 东财 search-api | 近期 | 3 个新闻特征 |
| xdxr | mootdx 通达信 | 全历史分红 | 3 个分红特征 + 后复权因子 |
| forecast | baostock | 全历史预告 | 4 个业绩预告特征 |

**fund_flow 切换说明（2026-08-11）**：东财 push2his 仅返回近 ~120 天（接口硬限制）→ 121 维
采样日被锁死 10 天。新浪 MoneyFlow 全历史（实测 2018-05 起 2000 行）→ 采样日可达 143 天。
**口径差异**：新浪 netamount=四档净额之和（全市场），东财"主力"=超大+大单——同步脚本已
**还原东财口径**（主力=r0_net+r1_net，各档占比=全口径占比×档净额/全净额恒等式换算），
超大单特征与东财可比；"主力"特征语义为全市场净流入（训练/预测口径一致即可）。
同步脚本：`scripts/sync_fund_flow_history.py`（断点续跑 + 多进程，5206 只 14min）；
月末链/重训增量走 `collect_extra_features.py::fetch_fund_flow`（同源）。

## 9. 数据同步框架（cron 时间线）

```
每日 18:10   pipeline.py → sync（增量，~50min 至 18:59）
月末 19:00   month_end_pull.py → 扩展维度强制全量（--refresh-days 0，4-7h）
次日 03:00   v2_monthly_retrain.py Step0 → 增量补刷（--refresh-days 40，文件新鲜则跳过）
```

---

# 第三部分：特征与模型

## 10. 特征体系（80 / 88 / 121 维）

| 版本 | 维度 | 内容 |
|------|------|------|
| 80 | 量价 | 收益/均线/量能/技术指标/波动率/大盘关联（T4 LSTM 用，LSTM 自学市场模式） |
| 88 | 80 + 8 | + 市场状态特征（指数涨跌/波动/回撤/均线/上涨占比，树模型用） |
| 121 | 88 + 33 | + 扩展维度（fund_flow 6 + finance 10 + holders 2 + consensus 5 + news 3 + xdxr 3 + forecast 4） |

**后复权补丁（feature_version=3）**：DB 为不复权价，特征计算前 `adjust.py::apply_adjust`
对 OHLCV 原地后复权（除权日假断层会污染收益/均线/技术指标）；extra 特征用原始价构建。
执行层（模拟盘/回测）不走此函数。

## 11. 标签体系（T1/T2/T3）

- **T1**：未来 1 日涨跌方向（分类）
- **T2**：未来 5 日超额收益（回归，主力信号）
- **T3**：未来 20 日收益（回归，长周期）

标签用后复权价计算（跨除权日收益不假跌）。y2 均值 2026 H1 -10.8%（市场风格切换背景）。

## 12. 模型体系

| 模型 | 结构 | 职责 |
|------|------|------|
| T2 | LGBM/XGB 树（Optuna 50 trials） | 5 日超额收益主信号 |
| T1 | 树模型 | 方向风控（2026-08-02 实证关闭） |
| T3 | 树模型 | 长周期辅助 |
| T4 | LSTM（纯 LSTM，128 units，L2=0，无 Transformer，dropout 0.285） | 时序信号 |

**融合**：T2+T4 Rank 融合为主（回测最优 M4+TOP_N=10，89.5% 夏普 3.35）。

## 13. 数据集缓存机制

- 路径：`data/cache/v2_dataset/<hash>/`（X.npy/y1-3.npy/dates.json/metadata.json）
- **hash key**：n_stocks + sample_start/sample_end + window + feature_version + market_state
  （+ extra_features）——参数变化自动换目录重建
- **采样日**：每月 2 天（5 日 + 15 日交易日），146 天计划（2020-08 ~ 2026-08）
- **增量复用（2026-08-11）**：新 hash 目录构建时，扫描同参数旧缓存（仅 sample_end 更早）→
  旧采样日样本直接复制（特征只依赖 ≤ref_date 数据，确定性已逐位验证）→ 只构建新增采样日
  （~21 天）→ 月末重建 2-6h → **30-40min**。前提：股票池不变（n_stocks 变 → hash 变 → 全量，
  正确行为）。metadata 含 params（新格式），旧格式缓存（无 params）不可复用。
- 断点续跑：`.rebuild_done_<date>_<dim>` marker（月末链）；缓存存在则秒级加载

---

# 第四部分：日常运行（每日 18:10 管线）

## 14. 每日管线流程（pipeline/pipeline.py，cron `10 18 * * 1-5`）

```
18:10 启动
 ├─ 步骤 [sync]      数据同步+清洗（~50min）
 ├─ 步骤 [strategy]  8 策略选股 + LLM 多维度研判（~3min）
 ├─ 步骤 [simulation] LLM 模拟盘更新（T+1 执行买入信号 + 风控）
 ├─ 步骤 [v2_simulation] V2 模拟盘日常操作（日报推送）
 └─ 步骤 [strategy_summary] 策略汇总推送（V2 vs LLM 收益对比）
18:59 完成 → 状态推送
```

**策略选股（8 个策略）**：均量线突破 / 海龟交易法则 / 高紧旗形突破 / 涨停洗盘 / 上涨回调 /
RPS 动量突破 / 多周期 RPS 突破 / 定增公告监控。选股结果保存 `data/results/results_YYYYMMDD.json`
（供 LLM 分析 + 次日复盘）。

**数据完整性检查**：覆盖率 ≥90% 才继续；否则推送告警并跳过选股（不推送假信号）。

## 15. LLM 研判与推送（DeepSeek，关键修复）

**流程**：策略选股（限流取前 10 只）→ 实时数据采集（知兔行情 + 新浪指数 + 本地情绪 + 东财公告）
→ DeepSeek 分析（deepseek-v4-flash）→ 解析 RECOMMEND 行（2 只推荐）→ 推送 + 写入模拟盘信号。

**⚠️ 推理模型事故（2026-08-10 修复，勿回退）**：
- 根因：deepseek-v4-flash 为推理模型，`reasoning_content` 计入 completion_tokens。
  `max_tokens=4096` 被推理耗尽时 **content 为空** → 推送失败（wxpusher code 1001 "content不能为空"）
  + RECOMMEND 解析失败回退"多策略频率推荐"（**非 LLM 真推荐**）。8-3/8-6/8-10 三天受影响；
  8-7 截断致 RECOMMEND 行丢失（推送成功但**信号缺失**）。
- 修复：`max_tokens` 4096 → **8192** + content 为空自动重试一次 + 超时 120→180s。
- 判定推送成功：看 `_push_ai_report` 的 messageId 日志（main.py 的"已推送"INFO 是无条件打印的坑）。
- 判定信号是否真 LLM 推荐：signals.py 日志分支——"RECOMMEND 行"=真；"多策略频率回退"=假。

**推送节奏**（用户明确）：LLM 每日 2 只荐股 → 次日 T+1 开盘买入；V2 每月 1 日重训后
一次 10 只（仅重训后首个交易日买入），非重训日唯一买卖=风控卖出；两套日报并存是常态。

## 16. 模拟盘（双模拟盘完全隔离）

| 项 | V2 模拟盘 | LLM 模拟盘 |
|----|-----------|-----------|
| 数据库 | sim_v2.db（独立） | sequoia_v2.db 的 sim_* 表 |
| 资金 | 100 万 | 100 万 |
| 持仓规则 | 上限 10 只 × 10 万 | 10 只 × 10 万（月买 2 只滚动） |
| 信号来源 | 月度重训选股（V2 策略） | 每日 LLM 研判（top_n=2） |
| 买入 | T+1 开盘价 | T+1 开盘价 |
| 卖出 | 13 条规则（风控总分≥60） | 同引擎 |

**执行引擎**：`SimEngine.run_daily()`——pending 信号 → 开盘价买入（held_symbols 循环内更新 +
IntegrityError 兜底，防重复信号崩溃）→ 持仓估值/止损止盈判定 → 日报。行情查主库
（settings.db_path），持仓/信号查模拟盘库。

---

# 第五部分：月末链（关键节点一，cron `0 19 * * 1-5`）

## 17. 月末链总览（month_end_pull.py）

**目的**：每月最后交易日收盘后，把扩展维度数据拉取完整 + 重建训练缓存 + 验证链路，
使下月 1 日 03:00 重训可直接运行（无人值守，全自动，异常微信告警）。

**为什么 19:00 启动**：避开 18:10 日线同步窗口（18:59 结束）；A 股 15:00 收盘后数据齐全。

```
19:00 启动
 ├─ 月末最后交易日判断（akshare 交易日历；非月末零成本退出；
 │   日历失败兜底：工作日放行/周末跳过）
 ├─ 7 类扩展维度强制全量拉取（--refresh-days 0，4-7h，timeout 12h）
 │   含 fund_flow（新浪全历史重拉 ~15min）+ 财联社快讯
 ├─ ① 覆盖率检查（关键面 ≥90%；不足 → 降级 88 维保底，不中止）
 ├─ 1.5 股票池月度刷新（新上市/ST/低价剔除；失败保留旧池不阻断）
 ├─ ② 训练缓存重建（rebuild_dataset_cache.py --workers 16，双 job 并行 32 进程）
 │   正常：121 维 + 80 维；降级：--no-extra → 88 维 + 80 维
 │   断点：.rebuild_done_<date>_<dim> marker（重跑跳过，省 3-4h）
 ├─ ③ 自检（metadata 维度 + 采样日覆盖月末）
 ├─ ④ 单月干跑验证（build_prediction_cache 临时输出，30-60min，断点 .dryrun_cache.json）
 └─ 微信推送"全链完成"（成功/降级/失败三态）
```

## 18. 缓存重建详解（全量 vs 增量）

**全量**（首次月末 / 股票池变化 / 特征参数变化）：146 天 × 2978 只全构建。
- 121 维：3-4h（每只读 7 个 parquet，IO 瓶颈，32 进程）
- 80 维（T4）：2-3h（纯量价，无 parquet）
- 双 job 并行（16 workers each = 32 进程，36 核机器合理；**勿用 32**——64 进程过载）

**增量**（2026-08-11 新功能，股票池不变时自动生效）：
- 原理：缓存 hash 含 sample_end → 每月新目录 → 原本全量重建（~95% 是旧采样日重复计算）。
  特征只依赖 ≤ref_date 数据（DB 不复权历史不漂移 / xdxr 后复权因子不漂移）→ 旧采样日
  样本**确定性成立**（已验证：增量 vs 全量 47 采样日逐位相等）。
- 做法：`_find_reusable_cache` 扫描同参数旧缓存（feature_version/window/market_state/extra
  相同、仅 sample_end 更早）→ 旧采样日 X/y/dates 直接复制 → 只构建新增采样日（~21 天）→ 拼接。
- 用时：**30-40 分钟**（vs 全量 3-4h）。
- 边界：股票池变化（n_stocks 变）→ hash 变 → 自动全量（正确行为，每月都可能，属正常设计）。
- metadata 需含 params（新格式）；旧格式缓存不可复用（8/31 首次月末链为全量，9/30 起增量）。

## 19. 覆盖率检查与 88 维降级（四层回退机制）

| 层 | 位置 | 行为 |
|----|------|------|
| ① 自动链降级 | month_end_pull 覆盖率不足 | 告警 + `rebuild --no-extra` 强制 88+80 维 → 推送"已回退 88 维" |
| ② 重训轮询降级 | v2_monthly_retrain wait_for_cache_ready | 121 缺失但 88 就绪 → 降级接受（微信告知） |
| ③ 构建端双保险 | build_prediction_cache | 配置 121 但缓存未就绪 → 自动降级 88 维 |
| ④ 最后防线 | 两缓存都未建 | 轮询 12h 超时 → 微信告警 + 重训中止（人工介入） |

**关键面**（fund_flow/finance/holders ≥90% 判定）与**语义可缺失面**（consensus/news/xdxr/
forecast 只报告不阻断）区分——consensus 64% 是真实覆盖水平（A 股约 36% 无研报），不得误判降级。

## 20. 自检与干跑

- **自检**（_verify_caches）：树模型缓存（121/88）+ T4 缓存（80）存在 + 维度正确 + 采样日覆盖月末。
- **干跑**（build_prediction_cache --start-month --end-month --skip-t4）：单月预测全链路验证
  （不污染生产 prediction_cache.json，输出到 .dryrun_cache.json），T2 预测 std 检查合理性。

---

# 第六部分：月度重训（关键节点二，cron `0 3 1 * *`）

## 21. 重训总览（v2_monthly_retrain.py，每月 1 日 03:00）

**目的**：用上月完整数据重训全部模型 → 产出本月 10 只选股信号 → 写入 V2 模拟盘 → 推送月报。
**为什么 03:00**：月末链 19:00 起（拉取 4-7h + 重建 2-6h + 干跑 1h）≈ 8h 至 02:00；
03:00 重训与其零重叠，留 1h 缓冲（2026-08-10 由 00:00 调整）。

```
03:00 启动
 ├─ Step0: 辅助维度增量刷新（--refresh-days 40，月末已全量则秒过）+ 覆盖率告警
 ├─ Step0.5: 缓存就绪轮询（每 5min，最长 12h；121→88 降级；超时告警中止）
 ├─ Step1: T2/T1/T3 预测缓存构建（build_prediction_cache，断点续跑）
 ├─ Step2: T4 LSTM 训练（t4_monthly_worker，追加预测到缓存）
 ├─ Step3: 选股（T2+T4 Rank 融合 → TOP_N=10）
 ├─ Step4: 信号入库 sim_v2.db（strategy_from="V2"）
 ├─ Step5: 荐股推送
 └─ Step6: V2 + LLM 模拟盘月度报告合并推送（上月数据）
```

## 22. 缓存就绪轮询（wait_for_cache_ready）

- 判定：hash 目录 metadata 存在 + 维度正确（121/88）+ 采样日覆盖上月最后交易日
- 121 缺失但 88 就绪 → 降级接受（微信告知）；都缺 → 轮询至 12h 超时 → 告警中止
- 与月末链衔接：重建完成写 marker → 重训轮询即时通过；重建未完成 → 轮询等待（最长 12h 兜底）

## 23. 重训步骤要点

- **Step1 预测缓存**：树模型（T2/T1/T3）月度预测，增量断点续跑（已完成月份跳过）
- **Step2 T4**：LSTM 训练 + 预测追加；失败可重跑续跑（.t4_tmp 原子更新）
- **Step3 选股**：Rank 融合取 TOP10（回测最优 M4+TOP_N=10 配置）
- **Step4 信号**：T+1 规则——1 日产生的信号，2 日（首个交易日）开盘买入
- **Step6 月报**：合并 V2（sim_v2.db）+ LLM（sequoia_v2.db）双模拟盘月度报告

## 24. 失败保护（不卡死原则）

每个 Step 失败：微信告警 + sys.exit(1)（可重跑续跑）——**不存在静默卡死**；
重跑幂等：断点续跑（marker/pending 过滤/原子更新）；Step0 失败不阻断（仅告警）。

---

# 第七部分：验证与回测（决策依据）

## 25. 验证历程（小三步 → 大三步 → 全量回测）

| 阶段 | 内容 | 结论 |
|------|------|------|
| 小三步（IC 层面） | 月度 WF / 88 vs 80 维 / T2+T4 Rank 融合 | 88 维 IC 提升；融合有效 |
| 大三步（收益层面） | T2+T4 vs 纯 T2 回测 + 四项修复 | 融合显著占优 |
| 72 组全量回测 | py312 最终口径（2026-08-02） | **M4 + TOP_N=10 最优** |
| 八时段扩展 | 各年度明细（2020-09 ~ 2026-06） | 震荡市/趋势市分时段确认 |
| 逐月 IC 分析 | 60 个月全貌 | 2025 后 IC 衰减（风格切换） |

**最终最优配置（70 个月，2020-09~2026-06）**：T2+T4 Rank 融合，M4 风控，TOP_N=10，
年化 89.5%，夏普 3.35。**核心结论**：2026 风格切换后（y2 均值 -10.8%），滚动窗口 +
近期数据纳入可逐步改善 IC（-0.258 → -0.156 → -0.093）。

## 26. T4 LSTM 调试史（最深刻一课）

**表象**：T4 预测恒等（pred_std=0）→ **四轮误诊**（数据/标签/线程）→ **真凶**：
L2=1e-4 正则杀死 LSTM kernel（norm 0.00）+ Transformer 层稀释信号 → **修复**：
`lstm_l2_reg=0.0` + `num_transformers=0`（纯 LSTM，kernel norm 恢复 15.78）。
**元教训**：kernel norm/pred_std 是模型健康的运行时指标，必须观测；改一个参数验证一个。

---

# 第八部分：经验教训与铁律

## 27. 铁律体系

| # | 铁律 | 内容 |
|---|------|------|
| 0 | 系统时间 | 预测"还要多久"前必须 `date` 取真实时间 |
| 一 | 详尽日志 | >1h 任务：跑到哪/还要多久/中间结果 + 运行时自检 |
| 二 | 断点续跑 | 每原子单元落盘；同命令续跑；恢复时验证有效性 |
| 三 | 修复验证 | 改完代码必须运行时验证生效（grep 配置/小批量跑/模拟中断） |
| 六 | 环境一致 | 所有运行必须 py312（base python3 回测漂移 25%） |
| 七 | 研究状态 | RESEARCH_STATE.md 单一事实源 + 四层信息架构 |

**KMP_AFFINITY 锁核**（两次事故）：启动 TF 脚本前 `env -u KMP_AFFINITY -u OMP_NUM_THREADS`，
脚本内 `tf.config.threading.set_*()` 显式设置（get_* 返回 0 是误导）。

**多进程 kill 铁律**（2026-08-11 新教训）：kill 多进程任务必须杀**进程树**（pkill -f 或全 PID），
只杀 bash 包装会让 python 主进程成为孤儿继续运行（本次 3 组 96 worker 抢 CPU + 混写日志的教训）。

## 28. 重大事故与教训索引

| 事故 | 日期 | 教训 |
|------|------|------|
| KMP_AFFINITY 锁核 ×2 | 08-01 | env -u 启动 |
| L2 杀死 LSTM kernel | 07-28 | 观测 kernel norm |
| 环境差异漂移 25% | 08-02 | 统一 py312 |
| 复权口径前复权污染 | 08-10 | 全库不复权 + 计算层后复权 |
| 腾讯 qfq 污染多行写库 | 08-10 | 腾讯 fq 留空读 day |
| LLM 推理挤占 content 空 | 08-10 | max_tokens 8192 + 空重试 |
| forecast 0 列破坏 33 列契约 | 08-11 | 固定列初始化 + 列数校验 |
| 孤儿进程混写日志 | 08-11 | kill 进程树 |

## 29. 监控与排障

**微信推送**（三态）：成功 / 降级（88 维）/ 失败告警——月末链与重训每个环节都推送。
**日志位置**：`logs/pipeline_YYYYMMDD.log`（日常）/ `month_end_pull_YYYYMM.log`（月末）/
`v2_retrain_YYYYMM.log`（重训）/ `rebuild_*_YYYYMMDD.log`（重建）/ `sync_fund_flow_history_*.log`。
**关键判定**：LLM 推送成功看 messageId 日志；信号真伪看 RECOMMEND/回退分支；
缓存就绪看 wait_for_cache_ready 日志；重建完成看 marker 文件。
**进程检查**：`ps -eo pid,etime,pcpu,args | grep -E "rebuild|month_end|v2_monthly|pipeline"`。

## 30. 常用命令速查

```bash
PY=/home/zhulei/anaconda3/envs/zhulei_py312/bin/python   # 铁律六：必须 py312

# ═══ 数据同步（cron 18:10 自动；手动触发）═══
$PY main.py --sync-only                # 同步模式
$PY main.py --repair --all             # 修复模式（对比 baostock 补缺失）

# ═══ fund_flow 全历史同步（新浪，断点续跑）═══
$PY scripts/sync_fund_flow_history.py --workers 24        # 全量（已有跳过）
$PY scripts/sync_fund_flow_history.py --test 5            # 测速

# ═══ 扩展维度（月末链 19:00 自动）═══
$PY scripts/collect_extra_features.py --codes scripts/all_a_codes.txt \
    --data fund_flow,finance,holders,consensus,xdxr,news,forecast --refresh-days 0

# ═══ 数据集缓存重建（月末自动；手动）═══
$PY scripts/rebuild_dataset_cache.py --only-88 --workers 32   # 仅 121/88 维（单 job 32）
$PY scripts/rebuild_dataset_cache.py --only-88 --workers 16 --no-extra  # 88 维（降级）
# 月末链自动: month_end_pull.py 传 --workers 16（双 job 32 进程）

# ═══ 月末链 / 月度重训（cron 自动；手动干跑）═══
$PY scripts/month_end_pull.py           # 月末链（非月末零成本退出）
$PY scripts/v2_monthly_retrain.py       # 月度重训（1 日 03:00）

# ═══ 日常管线 / 模拟盘 ═══
$PY pipeline/pipeline.py                # 完整管线（cron 18:10）
$PY main.py --sim-update                # LLM 模拟盘更新
$PY scripts/v2_simulation_daily.py      # V2 模拟盘日常

# ═══ 回测 / 分析 ═══
$PY backtest_v2.py                      # 回测
$PY scripts/analyze_monthly_ic.py       # 逐月 IC 分析
```

---

# 附录：关键文件与目录

| 路径 | 用途 |
|------|------|
| `data/sequoia_v2.db` | 主库（日线/估值/扩展维度索引/LLM 模拟盘） |
| `data/sim_v2.db` | V2 模拟盘独立库 |
| `data/extra_features/{subset}/{code}.parquet` | 扩展维度 7 类（fund_flow 新浪全历史） |
| `data/cache/v2_dataset/<hash>/` | 训练数据集缓存（含 params metadata） |
| `data/results/results_YYYYMMDD.json` | 每日策略选股结果 |
| `output/backtest_v2/.stock_pool.json` | 统一股票池（三处同源） |
| `output/backtest_v2/.rebuild_done_*` | 月末重建断点 marker |
| `output/backtest_v2/.dryrun_cache.json` | 月末干跑断点 |
| `scripts/sync_fund_flow_history.py` | 新浪资金流全历史同步（2026-08-11 新增） |
| `sequoia_x/model_selection_v2/adjust.py` | 后复权模块（2026-08-10 新增） |
| `V3研究方向与实验研究记录.md` | 研究方向/实验/ADR 决策记录 |
| `V2_OPERATION_GUIDE_v3_backup.md` | 本指南旧版（v3.0）备份 |

---

*文档终。修改任何体系行为（数据源/复权/时间/参数）后，请同步更新本文档与记忆。*
