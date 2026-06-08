# Sequoia-X 项目记忆

## 三阶层选股架构（铁律）

每次策略选股必须走完以下三步，**新增策略也必须遵守**：

### 第一步：基础股票池
`DataEngine.get_base_stock_pool()` 统一过滤：
- 剔除科创板(688/689)、创业板(300/301)、北交所(4xx/8xx)
- 剔除名称含 ST/*ST/退 的股票
- 剔除上市不满1年的次新股
- 剔除最新收盘价 < 2元的低价股

### 第二步：策略选股 + 打分
- 策略只在 `self.stock_pool` 范围内处理
- 每个满足条件的股票必须附带 **分数 (score)**
- 构造 `candidates: list[tuple[str, float]]`

### 第三步：按分数取前 N 支
- 调用 `self._pick_top(candidates, self.top_n)` 按分数降序取前5
- **禁止**使用 `symbols[:5]` 或 `list[:top_n]` 等无排序的截取方式
- 分数定义必须反映策略的核心信号强度（放量倍数、RPS值、动量比等）

## 推送配置
- 通知通道: WxPusher (非飞书)
- Token: AT_hKGG0UfwrCP7bpcsO8cbQkrc4bZ9G3RX
- Topic ID: 39277
