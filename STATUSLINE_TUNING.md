# Claude Code 状态栏调优完整记录（2026-08-02）

> **问题**：右下角状态栏显示 `deepseek-v4-flash[1m] | 6…`——上下文百分比不完整（显示 "6…" 而非 "62%"）
> **结果**：成功显示 `63% 374K $35.7 3.2h 🧠 flash[1m] …/004_sequoia-x`
> **涉及文件**：`~/.claude/statusline.py`、`~/.claude/settings.json`

---

## 一、问题背景

- 环境：Claude Code 2.1.220 对接 DeepSeek API 代理（`ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic`，模型 `deepseek-v4-flash[1m]`，1M 上下文）
- 用户配置了自定义状态栏（`~/.claude/settings.json` → `statusLine` → `python3 ~/.claude/statusline.py`），期望显示：左下角当前目录 + 右下角模型名称及剩余上下文百分比
- 实际显示：右下角 `deepseek-v4-flash[1m] | 6…`（百分比被截断），且始终如此

## 二、排查过程（含失误）

### 第 1 步：怀疑脚本崩溃（正确方向，但理解不完整）

手动运行 `python3 ~/.claude/statusline.py` 报错：`json.loads(raw)` 时 raw 为空（stdin 无输入）→ 误以为"脚本崩溃导致回退内置状态栏"。

**修复**：给脚本加容错（stdin 空/非 JSON 时用环境变量 `ANTHROPIC_MODEL` 兜底）+ 模型名缩短（去掉 `deepseek-v4-` 前缀）。

### 第 2 步：验证配置是否生效（关键实验 ✅）

临时把 `statusLine.command` 改为 `bash -c "date >> /tmp/statusline_called.log && echo TEST"`，25 秒内日志出现 4 次 → **证明 statusLine 配置本身是生效的**（每 10 秒调用一次），问题在脚本。

### 第 3 步：Agent 查询官方文档 + 二进制 schema（权威确认 ✅）

- 配置格式正确：`type: "command"` 是唯一合法值（Zod schema 验证）
- statusline 命令通过 **stdin 传 JSON**（字段：model/context_window/cost/workspace/rate_limits 等）
- 修改 settings.json 需重启会话；但**脚本文件修改即时生效**（每次调用重新执行）
- 内置状态栏渲染带 `wrap:"truncate"`——**超长文本从尾部截断加省略号**

### 失误 1（两犯）：python -c 字符串替换的 `\n` 转义坑

用 `python -c` 做字符串替换插入调试日志时，`'\n'` 被写入成**真实换行**（Python 字符串内真实换行 = `SyntaxError: unterminated string literal`）→ 脚本崩溃 → 状态栏回退旧内容。
- 第一次：08:15 加调试日志时
- 第二次：08:26 用 python -c 插诊断时（用 Edit 工具后修复）

**教训**：修改 .py 文件一律用 Edit/Write 工具（字面量精确），不要用 `python -c` 做字符串替换；改完必须 `py_compile` 验证。

### 第 4 步：百分比仍不完整 → 发现宽度问题（真正根因 ✅）

脚本正常后输出 `zhulei@node20:/public/home/.../004_sequoia-x | flash[1m] | 62%`（103+ 字符）→ 仍显示不完整。分析：

**根因**：Claude Code footer 的 statusline 区域**宽度有限**（实测约 27-40 字符），超长输出从尾部截断 → 百分比放在末尾必然被截成 "6…"。截断模型：`wrap:"truncate"`，从行尾截断加 `…`。

**验证依据**：内置显示 `deepseek-v4-flash[1m] | 6…`——模型名（22 字符）完整、百分比（尾部）被截 → 截断点 ≈ 27-30 字符。

### 第 5 步：最终修复（用户提示"位置向前移动" ✅）

按用户提示把**百分比移到最前面** + 压缩总长：

```python
# 布局：百分比 → 限流 → token → 费用 → 耗时 → ⚡ → 🧠 → 模型 → 目录
# 截断只影响尾部（目录），核心信息永远完整
parts = [ctx, rl_str, tok_str, cost_str, dur_str, fast_str, th_str,
         model_short, f"…/{dir_short}"]
line = " ".join(p for p in parts if p)
```

- 去掉右对齐 padding（非 tty 下宽度检测不可靠，padding 反而造成超宽截断）
- 目录仅保留最后 1 级（`…/004_sequoia-x`）
- 总长控制在 ~50 字符内

**结果**：显示 `65% flash[1m] …/004_sequoia-x`，百分比完整 ✅

### 第 6 步：扩展信息——DeepSeek 代理字段缺失的坑（✅）

用户要求加"限流余量 + Token 消耗"。加诊断日志后确诊：

```
raw_len=1309 rl5h=None tok=0
```

- **`rate_limits` 字段不存在**（DeepSeek 代理不返回限流信息）→ 自动隐藏
- **`cost` 字段无 token 统计**（只有 total_cost_usd / total_duration_ms / total_lines_added/removed）
- **token 数据实际在 `context_window` 里**：`total_input_tokens + total_output_tokens`（365K）→ 修正数据源后正常显示

### 第 7 步：费用 + 耗时 + ⚡ + 🧠（✅）

- 费用：`cost.total_cost_usd`（$35.7，DeepSeek 代理下按 Claude 定价估算）
- 耗时：`cost.total_duration_ms`（3.2h）
- Fast mode：`fast_mode`（默认 False 不显示；`/fast` 切换；DeepSeek 代理下可能无实际提速效果）
- Thinking：`thinking.enabled`（True 显示 🧠）

## 三、最终方案

### 最终显示效果

```
63% 374K $35.7 3.2h 🧠 flash[1m] …/004_sequoia-x
```

| 信息 | 示例 | 数据源 | 说明 |
|------|------|--------|------|
| 上下文剩余 | `63%` | `context_window.remaining_percentage` | 1M 上下文剩余百分比 |
| Token 消耗 | `374K` | `context_window.total_input_tokens + total_output_tokens` | 自动 M/K 格式化 |
| 会话费用 | `$35.7` | `cost.total_cost_usd` | 代理下为估算值 |
| 会话耗时 | `3.2h` | `cost.total_duration_ms` | 3.0h/45m/30s 自动格式 |
| Fast mode | `⚡` | `fast_mode` | 开启时显示，`/fast` 切换 |
| Thinking | `🧠` | `thinking.enabled` | 开启时显示 |
| 模型 | `flash[1m]` | `model.display_name` | 去掉 `deepseek-v4-` 前缀 |
| 目录 | `…/004_sequoia-x` | `workspace.current_dir` | 最后 1 级，超宽最先截断 |

### 核心设计原则

1. **重要信息放前面**：截断从尾部开始，所以按重要性排序（百分比 > token > 费用 > 耗时 > 标记 > 模型 > 目录）
2. **字段缺失自动隐藏**：`" ".join(p for p in parts if p)`——数据源不存在的字段不占位
3. **总长控制**：~50 字符内，保证前 6 项完整

## 四、经验教训（重要）

1. **Claude Code statusline 区域宽度有限**（~27-40 字符），超长输出从**尾部截断**——关键信息必须放前面，这是本问题的最根本根因
2. **statusline 配置本身通常没问题**——先验证配置是否被调用（把 command 改为 `echo TEST` 或写文件测试），再怀疑脚本内容
3. **脚本修改即时生效**（每 10 秒重新调用），**settings.json 修改需重启会话**
4. **DeepSeek 代理缺字段**：`rate_limits` 不存在、`cost` 无 token 统计（token 在 `context_window`）——用兜底逻辑，字段缺失不显示
5. **改 .py 文件用 Edit/Write 工具**，`python -c` 字符串替换会把 `'\n'` 写成真实换行导致语法错误（本次两犯）；改完必须 `py_compile` 验证（铁律三）
6. **内置状态栏显示格式**：auto-compact 开启时显示 `N% context used`（已用），关闭时显示 `N% until auto-compact`（剩余），整数取整

## 五、附录：stdin JSON 字段清单（DeepSeek 代理实测 2026-08-02）

顶层字段（实测 dump）：

```
context_window, cost, cwd, effort, exceeds_200k_tokens, fast_mode, model,
output_style, prompt_id, session_id, session_name, thinking,
transcript_path, version, workspace
```

**无**：`rate_limits`、`worktree`、`vim`、`agent`、`pr`、`remote`（Anthropic 端点可能有）

关键子字段：

```json
context_window: {
  "total_input_tokens": 365219, "total_output_tokens": 178,
  "context_window_size": 1000000,
  "current_usage": {"input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"},
  "used_percentage": 37, "remaining_percentage": 63
}
cost: {
  "total_cost_usd": 33.6, "total_duration_ms": 10960245,
  "total_api_duration_ms": 2816455, "total_lines_added": 440, "total_lines_removed": 146
}
```

## 六、最终文件

- `~/.claude/statusline.py` —— 完整可用的状态栏脚本（含全部容错）
- `~/.claude/settings.json` —— `statusLine: {"type": "command", "command": "python3 /home/zhulei/.claude/statusline.py", "refreshInterval": 10}`
