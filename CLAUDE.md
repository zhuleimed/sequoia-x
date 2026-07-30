# CLAUDE.md — Sequoia-X V2

本文件为 004_sequoia-x 项目的 Claude Code 工作指南。父级规则见 `../CLAUDE.md`。

## 当前状态（2026-07-30）

**所有模型验证已完成**。**综合回测计划已固化在 `BACKTEST_PLAN.md`**（72 组对比），待编写主程序执行。

## ⚠️ 首次读取指引

如果你是新启动的 Claude 会话，**请务必先阅读以下文件**:

1. **`BACKTEST_PLAN.md`** ← 最重要！项目全景 + 回测计划，约 19,000 字符
2. **`CLAUDE.md`**（本文件）← 快速参考
3. **记忆文件目录** `memory/` ← 历史决策和教训

## 关键发现（2026-07-29）

### T4 纯 LSTM 有效（L2=0, num_transformers=0）

| Fold | 测试期 | T4 Rank IC |
|------|--------|-----------|
| 3 | 2025 全年 | **+0.0712** |
| 4 | 2025 Q2-Q4 | **+0.1007** |
| 5 | 2026 H1 | **-0.2584** ← 市场风格切换 |
| 6 | 2026 Q2 | **-0.0909** ← 正在恢复 |

**根因链**: L2=1e-4 杀死 LSTM kernel → Transformer 层稀释信号 → 预测退化。
**修复**: `lstm_l2_reg=0.0`, `num_transformers=0`, KMP_AFFINITY 清除。

### 2026 年市场风格切换

2025 年 y2 均值 +1~2%，2026 H1 y2 均值 -10.8%，Q2 -21.7%。仅 20% 股票跑赢沪深 300。
滚动窗口测试证明：缩短训练窗口 + 纳入近期数据可逐步改善 IC（-0.258→-0.156→-0.093）。

### 待执行改进（详见 memory/v2-postmortem-improvements.md）

1. 月度 Walk-Forward（12月滚动→1月测试）
2. 市场状态特征（80→88维，新增 8 个大盘环境特征）
3. 双模型集成（短周期 6月+长周期 2年）
4. 数据同步 + Fold 7 扩展

## 关键配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `window` | 120 | 时序窗口 |
| `lstm_units` | 128 | LSTM 单元数（最佳参数） |
| `lstm_num_transformers` | 0 | Transformer 层数（已移除） |
| `lstm_l2_reg` | 0.0 | L2 正则（关键：1e-4会杀死kernel） |
| `lstm_dropout_rate` | 0.285 | Dropout |
| `lstm_learning_rate` | 0.0096 | 学习率 |
| `lstm_optuna_n_trials` | 18 | T4 Optuna 搜索 trial 数 |
| `lstm_optuna_timeout` | 86400 | 24h 超时 |
| `optuna_n_trials` | 50 | 树模型 Optuna trials |
| `optuna_timeout` | 7200 | 树模型 2h 超时 |
| `n_jobs` | 8 | 树模型内部线程 |
| `lstm_tf_intraop_threads` | 16 | TF 单 op 并行 |
| `lstm_tf_interop_threads` | 8 | TF op 间并行 |
| `lstm_omp_num_threads` | 10 | BLAS/MKL 线程 |
| `sample_end` | 2026-07-28 | 数据截止日期 |

## 重要：环境变量问题

**KMP_AFFINITY** 在 `.bashrc` 中设置，会锁定线程到特定核心，导致 TF 无法充分利用 CPU。
启动任何 TensorFlow 脚本前必须 `env -u KMP_AFFINITY -u OMP_NUM_THREADS`。

**TF 线程配置**：`deep_lstm.py` 模块顶部显式调用 `tf.config.threading.set_*()`，
确保 `get_*()` 返回实际值（而非误导性的 0）。

## 已知 Bug 与修复

### t4_pending 断点续跑 Bug（已修复 2026-07-26）
**位置**: `evaluate.py:100` | **修复**: 过滤 `t4_pending=true` 的 Fold

### T4 Optuna trials 截断（已修复 2026-07-26）
`lstm_optuna_n_trials` 60 → 18

### L2 正则化杀死 LSTM kernel（已修复 2026-07-28）
`lstm_l2_reg` 1e-4 → 0.0，kernel norm 从 0.00 恢复到 15.78

### Transformer 层信号退化（已修复 2026-07-28）
`num_transformers` 2 → 0，纯 LSTM pred_std 从 0 → 0.022

### KMP_AFFINITY 锁核（已修复 2026-07-28）
修复: 启动命令加 `env -u KMP_AFFINITY -u OMP_NUM_THREADS`

### TF 线程数显示为 0（已修复 2026-07-29）
`tf.config.threading.get_*()` 返回 0 是因为 env var 设置但未通过 Python API 调用。
修复: 显式调用 `tf.config.threading.set_*()`。

## 特征维度

| 版本 | 维度 | 说明 |
|------|------|------|
| v1 (旧) | 80 | 原始特征，padding 到 80 |
| v2 (新) | 88 | 新增 8 维市场状态特征（大盘涨跌/波动/回撤/均线/上涨占比），padding 到 88 |

缓存路径: `data/cache/v2_dataset/<hash>/`，特征版本变更自动重建（`feature_version` 在 hash key 中）。

## 断点续跑机制

- **数据缓存**: `data/cache/v2_dataset/<hash>/`，mmap 秒级加载
- **Fold 级 Checkpoint**: 每 Fold 完成后存 `walk_forward_results.json`
- **Optuna 复用**: 树模型 Study 跨 Fold 共享（`load_if_exists=True`），skip 优化
- **T4 best_params**: `best_params_t4_lstm.json` 存在则跳过 Optuna Phase 1

## 目录速查

| 路径 | 用途 |
|------|------|
| `BACKTEST_PLAN.md` | **综合回测计划书（最重要）** |
| `data/sequoia_v2.db` | 主 SQLite 数据库 |
| `data/cache/v2_dataset/` | 数据集磁盘缓存（80维/88维） |
| `data/models/v2_selection/` | 模型文件 + Walk-Forward 结果 |
| `logs/` | 运行日志 |
| `scripts/` | 测试/验证/回测脚本 |
| `memory/` | 项目记忆文件 |
