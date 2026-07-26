# CLAUDE.md — Sequoia-X V2

本文件为 004_sequoia-x 项目的 Claude Code 工作指南。父级规则见 `../CLAUDE.md`。

## 当前状态（2026-07-26）

**V2 Walk-Forward 评估运行中**（PID 3541766，日志 `logs/v2_evaluate_20260726_1811.log`）

## 关键配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `window` | 120 | 时序窗口 |
| `lstm_optuna_n_trials` | 18 | T4 LSTM Optuna 搜索 trial 数（60→18，避免24h超时截断） |
| `lstm_optuna_timeout` | 86400 | 24h 超时 |
| `optuna_n_trials` | 50 | 树模型 Optuna trials |
| `optuna_timeout` | 7200 | 树模型 2h 超时 |
| `n_jobs` | 8 | 树模型内部线程（T1∥T2∥T3 并行时 3×8=24 核） |
| `lstm_tf_intraop_threads` | 16 | TF 单 op 并行 |

## 已知 Bug 与修复

### t4_pending 断点续跑 Bug（已修复 2026-07-26）

**位置**: `evaluate.py:100`

**问题**: `completed_fold_numbers = {r["fold"] for r in all_results}` 包含了 `t4_pending=true` 的 Fold，导致 T4 崩溃后重启时该 Fold 被错误跳过。

**修复**: 改为 `{r["fold"] for r in all_results if not r.get("t4_pending")}`。

### T4 Optuna trials 截断问题（已修复 2026-07-26）

**问题**: 60 trials + 24h timeout → 实际只完成 8-15 个（13-25%），Hyperband 资源分配被 timeout 打断。

**修复**: `lstm_optuna_n_trials` 60 → 18，确保在 24h 内完整运行所有 trial。

## 断点续跑机制

- **数据缓存**: `data/cache/v2_dataset/<hash>/`，mmap 秒级加载（4.9GB X.npy）
- **Fold 级 Checkpoint**: 每 Fold T1-T3 完成后存 `walk_forward_results.json`（含 `t4_pending` 标记和 `_pred_t1/2/3`）
- **Optuna 复用**: 树模型 Study 跨 Fold 共享（`load_if_exists=True`）

## 目录速查

| 路径 | 用途 |
|------|------|
| `data/sequoia_v2.db` | 主 SQLite 数据库 |
| `data/cache/v2_dataset/` | 数据集磁盘缓存（4.9GB） |
| `data/models/v2_selection/` | 模型文件 + Walk-Forward 结果 |
| `data/models/v2_selection/optuna/` | 树模型 Optuna 搜索记录 |
| `data/models/v2_selection/optuna_t4_lstm.db` | T4 Optuna 搜索记录 |
| `logs/` | 运行日志 |
| `output/backtest_v2/` | V2 回测输出 |
