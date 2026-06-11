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
| Python 环境 | `zhulei_py312`（Python 3.12.13，新建于 2026-06-08） |
| 数据库 | SQLite → `data/sequoia_v2.db` |
| 数据源 | baostock（免费、无需注册、无限流） |
| 通知通道 | WxPusher 微信推送 |
| LLM 分析 | DeepSeek API（deepseek-v4-flash） |
| WxPusher Token | `AT_hKGG0UfwrCP7bpcsO8cbQkrc4bZ9G3RX` |
| WxPusher Topic ID | `39277` |
| DeepSeek API Key | `sk-abb7f3b79c0c4f868156cdf92f45e141` |
| 知兔API Token | `2C0E4763-3F63-4174-9CE1-806A10D58FC3` |
| 定时执行 | 交易日 20:55（服务器 cron） |
| 监控报告 | 交易日 21:15（WorkBuddy 自动化） |

---

## 1. 项目文件清单

```
004_sequoia-x/
├── main.py                          # 主入口（argparse 分发日常/回填模式）
├── pyproject.toml                   # 依赖声明 + ruff/pytest 配置
├── uv.lock                          # uv 锁定依赖版本
├── .env                             # 环境变量（WxPusher Token, DeepSeek Key, 知兔Token）
├── .env.example                     # 环境变量模板
├── .gitignore                       # Git 忽略规则
├── README.md                        # GitHub 页面 README
├── 004_Sequoia-X量化选股系统开发部署运行指南.md  ← 本文档
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
│   │   └── engine.py                # 数据引擎（baostock 回填/增量/基础池过滤）
│   │   └── save_results.py          # 选股结果保存模块（供 WorkBuddy 自动化读取）
│   ├── strategy/
│   │   ├── __init__.py
│   │   ├── base.py                  # BaseStrategy 抽象基类（含 _pick_top 排序）
│   │   ├── ma_volume.py             # 均量线突破策略
│   │   ├── turtle_trade.py          # 海龟交易法则策略
│   │   ├── high_tight_flag.py       # 高紧旗形突破策略
│   │   ├── limit_up_shakeout.py     # 涨停洗盘策略
│   │   ├── uptrend_limit_down.py    # 上涨回调策略
│   │   ├── rps_breakout.py          # RPS动量突破策略
│   │   ├── rps_multi_period.py      # 多周期RPS突破策略
│   │   └── private_placement.py     # 定增公告监控策略
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── analyst.py               # LLM 多维度分析引擎（10大市场模块 + 知兔API/Sina/SQLite三源采集）
│   ├── notify/
└── tests/                           # 属性测试（pytest + hypothesis）
    ├── __init__.py
    ├── test_config.py
    ├── test_data_engine.py
    ├── test_feishu.py                # 已重写为 WxPusher 测试
    ├── test_logger.py
    ├── test_main.py
    └── test_strategy.py
```

---

## 2. 整体架构

### 2.1 执行流程

```
20:55 服务器 cron 启动
  │
  ├─ 第1层：baostock 增量拉取今日日K线（8进程并行，2~3分钟）
  │     ↓
  ├─ 第2层：get_base_stock_pool() 基础股票池过滤
  │     ├─ 剔除 科创板(688/689)、创业板(300/301)、北交所(4xx/8xx)
  │     ├─ 剔除 ST/*ST/退市股
  │     ├─ 剔除 上市不满1年的次新股
  │     ├─ 剔除 最新收盘价<2元的低价股
  │     └─ 输出 ≈ 2,500~3,000 只
  │     ↓
  ├─ 第3层：8个量化策略独立选股 + 打分（取每策略前5支）
  │     ├─ 均量线突破 — 放量倍数
  │     ├─ 海龟交易法则 — 流通市值
  │     ├─ 高紧旗形突破 — 动量倍数/收敛幅度
  │     ├─ 涨停洗盘 — 放量/跌幅比
  │     ├─ 上涨回调 — 放量倍×跌幅
  │     ├─ RPS动量突破 — RPS分数
  │     ├─ 多周期RPS突破 — 综合信号评分（new!）
  │     └─ 定增公告监控 — 公告日期
  │     ↓
  ├─ 第4层：MarketAnalyst 多维度数据采集
  │     ├─ 知兔API → 个股实时行情 + PE/PB/市值/60日涨幅
  │     ├─ 新浪行情API → 大盘指数实时点位 + 个股行情兜底
  │     ├─ 本地SQLite → 10大市场情绪模块 + PE/PB估值分布
  │     ├─ 本地index_daily → 大盘指数趋势（5日/20日）
  │     ├─ 东方财富公告API → 个股新闻公告
  │     └─ 全部实时数据打包进 Prompt
  │     ↓
  ├─ 第5层：DeepSeek LLM 综合研判
  │     ↓
  ├─ 第6层：保存选股结果到 data/results/results_YYYYMMDD.json
  │     ↓
  └─ 第7层：WxPusher 推送初版报告到微信
              ↓
       ┌──── 次日 07:30 (WorkBuddy 自动化) ────┐
       │  SSH 读取结果 → 通达信MCP 深度查询    │
       │  研报评级+资金流向+热点题材+财报        │
       │  → 生成深度荐股报告 → WxPusher推送     │
       └──────────────────────────────────────┘
```

### 2.2 三阶层选股架构（铁律）

每次选股必须走完以下三步，**新增策略也必须遵守**：

| 步骤 | 方法 | 说明 |
|------|------|------|
| **第一步** | `DataEngine.get_base_stock_pool()` | 基础股票池过滤（板块/ST/次新/低价） |
| **第二步** | 各策略的 `run()` | 策略选股，构造 `(symbol, score)` |
| **第三步** | `BaseStrategy._pick_top()` | 按分数降序取前 5 支 |

### 2.3 数据流架构

```
baostock API（免费日K数据）
    │
    ▼
DataEngine
    ├─ backfill()    — 全市场历史回填（多进程并行）
    ├─ sync_today_bulk()  — 每日增量更新（8进程并行）
    └─ get_base_stock_pool() — 基础股票池过滤
    │
    ▼
SQLite (stock_daily 表)
    │
    ▼
7 × BaseStrategy.run()  — 各自读取 DB，独立选股
    │
    ▼
MarketAnalyst.analyze()
    ├─ akshare（实时行情/新闻/资金流）
    ├─ 东方财富股吧（实时舆情）
    └─ DeepSeek API（综合研判）
    │
    ▼
WxPusher → 微信推送
```

---

## 3. 内置策略详解

### 3.1 策略一览

| 策略 | 中文名 | webhook_key | 排名依据 | 说明 |
|------|-------|-------------|---------|------|
| MaVolumeStrategy | 均量线突破 | ma_volume | 放量倍数 | 5日均线上穿20日均线 + 成交量>20日均量1.5倍 |
| TurtleTradeStrategy | 海龟交易法则 | turtle | 流通市值 | 20日新高突破 + 成交额过亿 + 阳线过滤 |
| HighTightFlagStrategy | 高紧旗形突破 | flag | 动量/收敛比 | 强动量(40天涨幅>60%) + 收敛(10天振幅<15%) + 缩量 |
| LimitUpShakeoutStrategy | 涨停洗盘 | shakeout | 放量/跌幅比 | 昨涨停 + 今收阴 + 放量 > 昨2倍 + 不破昨收 |
| UptrendLimitDownStrategy | 上涨回调 | limit_down | 放量倍×跌幅 | 上升趋势(20均>60均) + 放量跌停 |
| RpsBreakoutStrategy | RPS动量突破 | rps | RPS分数 | 120天RPS>90 + 价格接近120天最高价90%以上 |
| PrivatePlacementStrategy | 定增公告监控 | private_placement | 公告日期 | 最近7天定向增发公告 |

### 3.2 新增策略模板

必须遵守三阶层选股架构：

```python
from sequoia_x.strategy.base import BaseStrategy

class MyNewStrategy(BaseStrategy):
    webhook_key: str = "my_new"
    display_name: str = "新策略名称"
    top_n: int = 5

    def run(self) -> list[str]:
        symbols = self.stock_pool or self.engine.get_local_symbols()
        candidates: list[tuple[str, float]] = []

        for symbol in symbols:
            # ... 选股逻辑 ...
            if 满足条件:
                score = ...  # 策略核心信号强度
                candidates.append((symbol, score))

        return self._pick_top(candidates, self.top_n)
```

**铁律**：
1. ✅ 用 `self.stock_pool`（不用全量遍历）
2. ✅ 构造 `(symbol, score)` 带分数
3. ✅ 用 `_pick_top()` 截取
4. ✅ 在 `main.py` 的 `strategies` 列表注册

---

## 4. 部署步骤

### 前置条件

- 服务器：Ubuntu LTS，已安装 Python 3.10+ 和 Anaconda3
- 本地：Windows + Git + Python 3.12+（可选）
- WxPusher Token（已配置）
- DeepSeek API Key（已配置）

### 第1步：克隆代码

```bash
# 从 GitHub 克隆
git clone https://github.com/zhuleimed/sequoia-x.git
cd sequoia-x
```

### 第2步：创建虚拟环境并安装依赖

```bash
# 推荐使用 conda（服务器）
conda create -n zhulei_py312 python=3.12 -y
conda activate zhulei_py312
pip install uv
uv sync

# 安装 WxPusher
uv pip install wxpusher

# 安装测试依赖
uv pip install pytest hypothesis
```

### 第3步：配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填写以下内容：
#   WXPUSHER_TOKEN=AT_hKGG0UfwrCP7bpcsO8cbQkrc4bZ9G3RX
#   WXPUSHER_TOPIC_IDS=["39277"]
#   DEEPSEEK_API_KEY=sk-abb7f3b79c0c4f868156cdf92f45e141
#   DEEPSEEK_MODEL=deepseek-v4-flash
```

### 第4步：首次回填历史数据

```bash
# 回填全市场历史日K线（约 2~3 小时，单线程拉取 baostock）
python main.py --backfill
```

> **提速提示**：当前 `backfill()` 是单线程逐个请求。如需加速，可修改 `engine.py` 中的 `backfill()` 使用多进程。新版已预留此设计，但尚未完全并行化。

### 第5步：验证运行

```bash
# 测试基础池获取 + WxPusher 推送（非交易时段跳过增量同步）
python main.py --skip-llm

# 完整测试（含 LLM 分析）
python main.py
```

### 第6步：验证测试

```bash
# 运行全部测试
python -m pytest tests/ -v
```

---

## 5. 推送配置（WxPusher）

### 5.1 获取 WxPusher Token

1. 前往 [WxPusher 官网](https://wxpusher.zjiecode.com) 注册
2. 创建应用，获取 AppToken
3. 创建 Topic，获取 Topic ID
4. 扫码关注，即可接收消息

### 5.2 配置

```ini
# .env
WXPUSHER_TOKEN=AT_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
WXPUSHER_TOPIC_IDS=["39277"]
```

### 5.3 消息格式示例

```
📊 Sequoia-X AI 综合研判 | 2026-06-08

📈 大盘环境
上证指数: 3050.23 (-0.68%) | 深证成指: 9250.18 (-1.25%)

🔍 个股深度分析

**1. 怡球资源 (601388) — 海龟交易法则** ⭐3.8/5
→ 一季报净利同比+179.68%，盈利拐点明确...

🏆 综合建议
- 最优关注: 怡球资源、电投水电
- 操作建议: 逢低分批建仓，设-5%止损
```

---

## 6. DeepSeek LLM 分析配置

### 6.1 获取 API Key

- 官网：https://platform.deepseek.com/
- 模型：deepseek-v4-flash（当前使用）
- 价格：约 ￥0.14/百万输入 tokens，每次分析约 3000 tokens → 成本极低

### 6.2 多维分析数据源

| 维度 | 采集方式 | 来源 |
|------|---------|------|
| 大盘指数 | akshare → 东方财富 | 实时收盘数据 |
| 板块资金流 | akshare → 东方财富 | 实时数据 |
| 个股实时行情 | akshare → 全市场快照 | 最新价/涨跌幅/换手率/PE/PB |
| 个股基本面 | akshare → 个股信息 | 总市值/流通市值/营收/利润 |
| 个股新闻 | akshare → 东方财富新闻 | 当日相关新闻 |
| 股吧情绪 | 直爬东方财富 API | 实时热帖标题/阅读量/评论数 |

### 6.3 降级保护

LLM 分析异常时自动降级为策略原始结果推送：

```python
try:
    report = analyst.analyze(...)           # LLM 分析
    _push_ai_report(settings, report)       # 推送综合报告
except Exception:
    _push_fallback_results(...)             # 降级：直接推策略结果
```

---

## 7. 定时任务

### 7.1 执行层 — 服务器 Cron

```bash
# 编辑 crontab
crontab -e

# 添加以下条目（每个交易日 20:55 自动选股）
55 20 * * 1-5 cd /public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x \
  && /home/zhulei/anaconda3/envs/zhulei_py312/bin/python main.py \
  >> logs/daily_$(date +\%Y\%m\%d).log 2>&1
```

当前服务器已配置的完整 cron：

| 时间 | 项目 | 说明 |
|:----:|:----|:----|
| 20:55 | Sequoia-X 选股 | 增量数据 + 7策略 + LLM分析 + 微信推送 |
| 21:00 | 015 指标扫描模拟盘 | 已有项目 |
| 21:10 | 016 ETF预测 | 已有项目 |

### 7.2 监控层 — WorkBuddy 自动化

| 项目 | 值 |
|------|-----|
| 自动化名称 | Sequoia-X 选股监控 |
| 触发时间 | 交易日 21:15 |
| 任务 | SSH 登录服务器 → 检查运行日志 → 验证选股结果 → 推送到小程序 |
| 工作目录 | `G:\中心同步盘_20251027\文档\学习\001_量化平台\013_sequoia-x` |

### 7.3 日志查看

```bash
# 查看今日运行日志
cat logs/daily_$(date +\%Y\%m\%d).log

# 查看最近运行日志
tail -50 logs/daily_*.log

# 查看基础池规模
grep -i "基础股票池最终" logs/daily_*.log

# 查看各策略结果
grep -i "选出" logs/daily_*.log
```

---

## 8. 数据说明

### 8.1 baostock 数据更新时间

| 数据类型 | 更新时间 |
|---------|---------|
| 日K线 | 当前交易日 17:30 |
| 复权因子 | 当前交易日 18:00 |
| 分钟K线 | 当前交易日 20:00 |
| 财务数据 | 第二自然日 1:30 |
| 周K线 | 周六 17:30 |
| 月K线 | 每月1号 17:30 |
| 成分股信息 | 每周一下午 |

### 8.2 SQLite 数据库

```bash
# 查看数据库统计
cd /path/to/sequoia-x
source ~/anaconda3/bin/activate zhulei_py312
python3 -c "
import sqlite3
conn = sqlite3.connect('data/sequoia_v2.db')
c = conn.execute('SELECT COUNT(DISTINCT symbol) FROM stock_daily')
print(f'股票总数: {c.fetchone()[0]}')
c = conn.execute('SELECT COUNT(*) FROM stock_daily')
print(f'总K线记录: {c.fetchone()[0]:,}')
c = conn.execute('SELECT MIN(date), MAX(date) FROM stock_daily')
r = c.fetchone()
print(f'日期范围: {r[0]} ~ {r[1]}')
conn.close()
"
```

### 8.3 数据存储

- **复权方式**：前复权（adjustflag=2）（hfq）— 历史价格不变，适合增量存储
- **存储路径**：`data/sequoia_v2.db`
- **数据库大小**：约 386 MB（全市场 5207 只 × 2.5年数据）
- **数据可直接拷贝**到其他机器使用

---

## 9. 命令行参数

```bash
python main.py                      # 日常模式：增量同步 + 策略 + LLM + 推送
python main.py --skip-llm           # 跳过 LLM 分析（仅策略选股 + 推送）
python main.py --backfill           # 回填模式：全市场历史K线
```

---

## 10. 快速上手（5分钟）

```bash
# 1. 激活环境
source ~/anaconda3/bin/activate zhulei_py312

# 2. 进入项目
cd /public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x

# 3. 运行日常模式
python main.py
```

---

## 11. 代码更新与同步

### 11.1 从 GitHub 拉取最新代码

```bash
# 本地
cd "G:\中心同步盘_20251027\文档\学习\001_量化平台\013_sequoia-x"
git pull

# 上传到服务器
# 使用 SFTP 上传修改过的 .py 文件
```

### 11.2 推送本地修改到 GitHub

```bash
git add -A
git commit -m "修改说明"
git push
```

---

## 12. 常见问题故障排查

### Q1：增量同步时卡住，日志停在"启动多进程并行拉取"

**原因**：baostock 当日数据尚未入库（需等 17:30），或无新数据返回时 multiprocessing 的 recv 阻塞。

**解决**：
- 确保在交易日 19:00 后运行（15:00 收盘 + 17:30 入库 + 容错时间）
- cron 已设为 20:55，确保数据已入库

### Q2：基础股票池数量异常（比如 < 1000 只）

**原因**：数据库未回填完整，`get_base_stock_pool()` 的价格过滤步骤查询 DB 最新收盘价时失败。

**解决**：
```bash
# 检查回填是否完整
python3 -c "
import sqlite3
conn = sqlite3.connect('data/sequoia_v2.db')
c = conn.execute('SELECT COUNT(DISTINCT symbol) FROM stock_daily')
print(f'股票总数: {c.fetchone()[0]}')
conn.close()
"
# 应接近 5207。如不完整，重新回填：
python main.py --backfill
```

### Q3：WxPusher 推送失败

**原因**：Token 过期 / Topic ID 错误 / 网络问题。

**解决**：
```bash
# 手动测试推送
python3 -c "
from wxpusher import WxPusher
result = WxPusher.send_message(
    content='Sequoia-X 测试消息',
    token='AT_hKGG0UfwrCP7bpcsO8cbQkrc4bZ9G3RX',
    topic_ids=['39277'],
    content_type=1,
)
print(result)
"
```

### Q4：DeepSeek API 调用失败

**原因**：API Key 过期 / 余额不足 / 网络超时。

**解决**：
```bash
# 验证 API Key 是否有效
curl https://api.deepseek.com/v1/chat/completions \
  -H "Authorization: Bearer sk-abb7f3b79c0c4f868156cdf92f45e141" \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"hi"}],"stream":false}'
```

LLM 异常时系统会自动降级推送策略原始结果（参见第6.3节）。

### Q5：策略全部无选股结果

**原因**：当日市场条件不符合任何策略的触发条件（例如震荡市无突破、无量能放大）。

**解决**：
- 查看日志确认各策略均正常运行
- 这是正常现象，某些策略在某些交易日确实会空仓
- 可调整策略参数（如放宽条件）或在 `main.py` 中临时禁用某些策略

### Q6：回填过程中断，如何续传？

**解决**：
```bash
# 直接重新运行回填，已有数据的股票会自动跳过
python main.py --backfill
```

`backfill()` 中的 `_get_last_date(symbol)` 会自动识别已入库的数据。

### Q7：如何调试单个策略？

```bash
python3 -c "
import sys
sys.path.insert(0, '/path/to/sequoia-x')
from sequoia_x.core.config import get_settings
from sequoia_x.data.engine import DataEngine
from sequoia_x.strategy.ma_volume import MaVolumeStrategy

settings = get_settings()
engine = DataEngine(settings)
pool = engine.get_base_stock_pool()[:50]  # 只用前50只测试

strategy = MaVolumeStrategy(engine=engine, settings=settings, stock_pool=pool)
result = strategy.run()
print(f'选出 {len(result)} 只: {result}')
"
```

### Q8：修改代码后需要重启什么？

| 场景 | 操作 |
|------|------|
| 修改了策略逻辑 | 重新运行 `python main.py` 即可 |
| 修改了配置 | 保存后下次运行自动生效 |
| 修改了 Cron 定时 | `crontab -e` 后自动生效 |

### Q9：如何查看 DeepSeek API 调用消耗？

```bash
# 登录 DeepSeek 官网查看用量
# 或查看运行日志中的 tokens 统计
grep -i "tokens" logs/daily_*.log
```

### Q10：如何手动触发一次完整运行？

```bash
# SSH 登录服务器
ssh zhulei@[2001:250:4400:89:aae6:63d8:a8e0:51dc]

# 激活环境并运行
source ~/anaconda3/bin/activate zhulei_py312
cd /public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x
python main.py
```

---

## 13. 验证清单

部署完成后逐项验证：

- [ ] `python main.py` → 成功运行，无报错
- [ ] 基础股票池数量在 2000~3500 之间
- [ ] 7个策略均正常执行（日志中有 "选出 X 只"）
- [ ] WxPusher 微信推送成功
- [ ] DeepSeek API 分析报告推送成功
- [ ] `python main.py --backfill` → 全市场回填完成（5207只）
- [ ] `python -m pytest tests/ -v` → 测试通过
- [ ] Cron 定时 20:55 已配置
- [ ] WorkBuddy 自动化 21:15 已配置（监控报告）
- [ ] GitHub 仓库已同步最新代码

---

## 14. 环境对比

| 环境 | 路径 | Python | 用途 |
|------|------|--------|------|
| 服务器生产 | `/public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x/` | 3.12.13（zhulei_py312） | 每日 20:55 cron 生产运行 |
| 本地开发 | `G:\中心同步盘_20251027\文档\学习\001_量化平台\013_sequoia-x\` | 3.13.12（.venv） | 代码修改、测试、Git 管理 |
| GitHub | `https://github.com/zhuleimed/sequoia-x` | — | 代码版本管理 |

---

## 15. 版本历史

| 日期 | 版本 | 变更说明 |
|:----:|:----:|---------|
| 2026-06-08 | v1.0 | 首次克隆部署，飞书→WxPusher 改造 |
| 2026-06-08 | v1.1 | 新增三阶层选股架构（基础池+打分+排序） |
| 2026-06-08 | v1.2 | 新增策略中文名，并行化回填引擎 |
| 2026-06-08 | v1.3 | 新增 LLM 多维度分析模块（DeepSeek API） |
| 2026-06-08 | v1.4 | 改为统一推送（7策略→LLM研判→1条消息） |

---

## 16. 附录：关键配置速查

### .env 文件

```ini
DB_PATH=data/sequoia_v2.db
START_DATE=2024-01-01
WXPUSHER_TOKEN=AT_hKGG0UfwrCP7bpcsO8cbQkrc4bZ9G3RX
WXPUSHER_TOPIC_IDS=["39277"]
DEEPSEEK_API_KEY=sk-abb7f3b79c0c4f868156cdf92f45e141
DEEPSEEK_MODEL=deepseek-v4-flash
```

### 服务器 SSH 信息

| 项目 | 值 |
|------|-----|
| Host | `[2001:250:4400:89:aae6:63d8:a8e0:51dc]` |
| Port | 22 |
| Username | `zhulei` |
| Password | `zhulei@HPC88660159` |
| sudo | 有 |

### Python 路径

| 环境 | 路径 |
|------|------|
| zhulei（原环境） | `/home/zhulei/anaconda3/envs/zhulei/bin/python` |
| zhulei_py312（新环境） | `/home/zhulei/anaconda3/envs/zhulei_py312/bin/python` |
