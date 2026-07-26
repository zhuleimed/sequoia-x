"""model_selection_v2 - Purged Rolling Walk-Forward 评估。

对每个扩展窗口：训练 → 评估（purge gap 隔开）→ 报告 IC/AUC/RMSE。
"""
from __future__ import annotations
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, mean_squared_error
from sequoia_x.core.config import Settings
from sequoia_x.data.engine import DataEngine
from sequoia_x.core.logger import get_logger
from sequoia_x.model_selection_v2.config import V2Config, get_config
from sequoia_x.model_selection_v2.labels import build_training_dataset
from sequoia_x.model_selection_v2.models.tree_cls import train_cls, predict_cls
from sequoia_x.model_selection_v2.models.tree_reg import train_reg, predict_reg
from sequoia_x.model_selection_v2.models.tree_vol import train_vol, predict_vol
from sequoia_x.model_selection_v2.models.deep_lstm import train_lstm, predict_lstm

logger = get_logger(__name__)


def _compute_rank_ic(pred: np.ndarray, actual: np.ndarray) -> float:
    """计算 Rank IC (Spearman correlation)。"""
    if len(pred) < 10:
        return 0.0
    ic, _ = spearmanr(pred, actual)
    return float(ic) if not np.isnan(ic) else 0.0


def run_walk_forward(
    engine: DataEngine | None = None,
    cfg: V2Config | None = None,
    search_optuna: bool = True,
) -> list[dict]:
    """运行 Purged Rolling Walk-Forward 评估。

    Folds:
      Fold 1: train 2020-2023 → test 2024
      Fold 2: train 2020-2024Q1 → test 2024Q2-Q4
      Fold 3: train 2020-2024 → test 2025
      Fold 4: train 2020-2025Q1 → test 2025Q2-Q4
      Fold 5: train 2020-2025 → test 2026H1
      Fold 6: train 2020-2026Q1 → test 2026Q2

    Purge: 训练集最后日期与测试集第一个日期间隔 >= cfg.purge_gap 个交易日。
    """
    if cfg is None:
        cfg = get_config()
    if engine is None:
        engine = DataEngine(Settings())

    # ── 启动 Banner：记录所有关键配置 ──
    logger.info("=" * 70)
    logger.info("V2 Walk-Forward 评估启动")
    logger.info(f"  特征: window={cfg.window}, purge_gap={cfg.purge_gap}")
    logger.info(f"  T1: XGBoost分类(y1=5日方向)  T2: LightGBM回归(y2=20日超额)")
    logger.info(f"  T3: CatBoost回归(y3=波动率)  T4: LSTM-Transformer(y2)")
    logger.info(f"  Optuna: 树模型{cfg.optuna_n_trials}trials/{cfg.optuna_timeout}s | "
                f"LSTM {cfg.lstm_optuna_n_trials}trials/{cfg.lstm_optuna_timeout}s")
    logger.info(f"  并行: T1∥T2∥T3(各{cfg.n_jobs}核) → T4(TF{cfg.lstm_tf_intraop_threads}核)")
    logger.info("=" * 70)

    # 构建全量数据集（16 workers 并行，利用多核加速）
    t_data_start = time.time()
    X, y1, y2, y3, dates = build_training_dataset(engine, cfg, n_workers=16)
    t_data_elapsed = time.time() - t_data_start
    if len(X) == 0:
        logger.error("无数据")
        return []
    logger.info(f"数据构建耗时: {t_data_elapsed:.0f}s ({t_data_elapsed/60:.1f}min)")

    # 获取样本日期的唯一排序列表，用于确定 Fold 边界
    unique_dates = sorted(set(dates))
    logger.info(f"Walk-Forward: {len(X)} 样本, {len(unique_dates)} 个采样日期")
    logger.info(f"日期范围: {unique_dates[0]} ~ {unique_dates[-1]}")

    # 定义 Fold 边界（按年份+半年度）
    fold_boundaries = [
        ("2020-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),    # Fold 1
        ("2020-01-01", "2024-03-31", "2024-04-01", "2024-12-31"),    # Fold 2
        ("2020-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),    # Fold 3
        ("2020-01-01", "2025-03-31", "2025-04-01", "2025-12-31"),    # Fold 4
        ("2020-01-01", "2025-12-31", "2026-01-01", "2026-06-30"),    # Fold 5
        ("2020-01-01", "2026-03-31", "2026-04-01", "2026-07-20"),    # Fold 6
    ]

    # ── 断点续跑：加载已有结果，跳过已完成 Fold ──
    save_path = cfg.model_dir_path / "walk_forward_results.json"
    all_results: list[dict] = []
    completed_fold_numbers: set[int] = set()
    if search_optuna and save_path.exists():
        try:
            with open(save_path) as f:
                all_results = json.load(f)
            completed_fold_numbers = {r["fold"] for r in all_results if not r.get("t4_pending")}
            logger.info(f"断点续跑: 已加载 {len(all_results)} 个 Fold 结果 "
                        f"({sorted(completed_fold_numbers)})")
        except Exception as e:
            logger.warning(f"加载已有结果失败: {e}，从头开始")
            all_results = []

    t_pipeline_start = time.time()
    valid_folds_total = 0

    for fold_i, (train_start, train_end, test_start, test_end) in enumerate(fold_boundaries):
        fold_num = fold_i + 1

        # 断点续跑：跳过已完成 Fold
        if fold_num in completed_fold_numbers:
            logger.info(f"── Fold {fold_num}: 已完成，跳过 ──")
            continue

        # 内存监控：记录 Fold 开始时 RSS
        try:
            import resource
            rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            logger.info(f"── Fold {fold_num}: train {train_start}~{train_end}, "
                        f"test {test_start}~{test_end} [RSS={rss_mb:.0f}MB] ──")
        except Exception:
            logger.info(f"── Fold {fold_num}: train {train_start}~{train_end}, "
                        f"test {test_start}~{test_end} ──")
        t0 = time.time()

        # Purge: 找到训练集最后一个日期和测试集第一个日期
        train_dates = [d for d in unique_dates if train_start <= d <= train_end]
        test_dates = [d for d in unique_dates if test_start <= d <= test_end]
        if not train_dates or not test_dates:
            logger.warning(f"Fold {fold_num}: 无数据，跳过")
            continue

        # 找到有至少 purge_gap 间隔的切分点
        train_mask = np.array([d in train_dates for d in dates])
        test_mask = np.array([d in test_dates for d in dates])

        if train_mask.sum() < 100 or test_mask.sum() < 50:
            logger.warning(f"Fold {fold_num}: 样本不足（train={train_mask.sum()}, test={test_mask.sum()}），跳过")
            # 首个有效 Fold 时推算总数（数据从2024-08起，Fold 1-2无数据，有效Fold=4）
            if valid_folds_total == 0:
                valid_folds_total = len(fold_boundaries) - fold_num + 1
                logger.info(f"  有效 Fold 总数: {valid_folds_total} (Folds {fold_num}-{len(fold_boundaries)})")
            continue

        # 训练 4 个模型：T1/T2/T3 树模型 + T4 LSTM-Transformer
        X_train, X_test = X[train_mask], X[test_mask]
        y1_train, y1_test = y1[train_mask], y1[test_mask]
        y2_train, y2_test = y2[train_mask], y2[test_mask]
        y3_train, y3_test = y3[train_mask], y3[test_mask]

        # ── 断点续跑：检查该 Fold 是否已有树模型结果（T4 崩溃恢复）──
        existing_partial = next((r for r in all_results
                                 if r.get("fold") == fold_num and r.get("t4_pending")), None)

        if existing_partial:
            # 从 T4 断点恢复：加载已保存的树模型预测，只重做 T4
            logger.info(f"  Fold {fold_num}: 从 T4 断点恢复（树模型已完成）")
            t_tree_elapsed = 0  # 不计入本次耗时
            pred_t1 = np.array(existing_partial["_pred_t1"])
            pred_t2 = np.array(existing_partial["_pred_t2"])
            pred_t3 = np.array(existing_partial["_pred_t3"])
            # 重建 fold_result 基础（不含 T4）
            fold_result = {k: v for k, v in existing_partial.items()
                          if not k.startswith("_") and k != "t4_pending"}
            all_results.remove(existing_partial)
        else:
            # ── 树模型并行训练：T1∥T2∥T3 ──
            logger.info(f"  Fold {fold_num}: T1+T2+T3 并行训练启动 "
                        f"(n_train={train_mask.sum()}, n_test={test_mask.sum()})")
            t_tree_start = time.time()
            with ThreadPoolExecutor(max_workers=3) as executor:
                f1 = executor.submit(train_cls, X_train, y1_train, cfg, search_optuna=search_optuna)
                f2 = executor.submit(train_reg, X_train, y2_train, cfg, search_optuna=search_optuna)
                f3 = executor.submit(train_vol, X_train, y3_train, cfg, search_optuna=search_optuna)
                model_t1 = f1.result()
                model_t2 = f2.result()
                model_t3 = f3.result()
            t_tree_elapsed = time.time() - t_tree_start
            logger.info(f"  Fold {fold_num}: T1+T2+T3 并行训练完成，耗时 "
                        f"{t_tree_elapsed:.0f}s ({t_tree_elapsed/60:.1f}min)")

            # 预测（树模型）
            pred_t1 = predict_cls(model_t1, X_test)
            pred_t2 = predict_reg(model_t2, X_test)
            pred_t3 = predict_vol(model_t3, X_test)

            # 构建 Fold 基础结果
            fold_result = {
                "fold": fold_num,
                "train_start": train_start, "train_end": train_end,
                "test_start": test_start, "test_end": test_end,
                "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
            }

            # T1-T3 评估指标
            try:
                fold_result["t1_auc"] = float(roc_auc_score(y1_test, pred_t1))
            except ValueError:
                fold_result["t1_auc"] = 0.5
            fold_result["t1_accuracy"] = float(((pred_t1 > 0.5) == y1_test).mean())
            fold_result["t2_rank_ic"] = _compute_rank_ic(pred_t2, y2_test)
            fold_result["t2_rmse"] = float(np.sqrt(mean_squared_error(y2_test, pred_t2)))
            fold_result["t3_rmse"] = float(np.sqrt(mean_squared_error(y3_test, pred_t3)))

            # ── T1-T3 Checkpoint: 在危险的 T4 之前保存部分结果 ──
            checkpoint = {
                **fold_result,
                "t4_pending": True,
                "_pred_t1": pred_t1.tolist(),
                "_pred_t2": pred_t2.tolist(),
                "_pred_t3": pred_t3.tolist(),
            }
            all_results.append(checkpoint)
            with open(save_path, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            logger.info(f"  Fold {fold_num}: T1-T3 checkpoint 已保存")

        # T4: LSTM-Transformer 深度回归（单独训练，TF 独占多核）
        logger.info(f"  Fold {fold_num}: T4 LSTM 训练启动")
        t_lstm_start = time.time()
        model_t4 = train_lstm(X_train, y2_train, cfg, search_optuna=search_optuna,
                              model_id=f"fold{fold_num}")
        t_lstm_elapsed = time.time() - t_lstm_start
        logger.info(f"  Fold {fold_num}: T4 LSTM 训练完成，耗时 "
                    f"{t_lstm_elapsed:.0f}s ({t_lstm_elapsed/60:.1f}min)")

        # T4 预测与评估
        pred_t4 = predict_lstm(model_t4, X_test)
        fold_result["t4_rank_ic"] = _compute_rank_ic(pred_t4, y2_test)
        fold_result["t4_rmse"] = float(np.sqrt(mean_squared_error(y2_test, pred_t4)))

        # 方向胜率
        if len(pred_t1) > 0:
            buy_mask = pred_t1 > 0.55
            if buy_mask.sum() > 0:
                fold_result["direction_win_rate"] = float(y1_test[buy_mask].mean())
            else:
                fold_result["direction_win_rate"] = 0.0

        elapsed = time.time() - t0
        fold_result["elapsed"] = elapsed
        # 移除 T1-T3 checkpoint（含 t4_pending），追加完整结果
        all_results = [r for r in all_results if r.get("fold") != fold_num]
        all_results.append(fold_result)

        logger.info(
            f"Fold {fold_num}: "
            f"T1_AUC={fold_result.get('t1_auc', 0):.3f}, "
            f"T2_RankIC={fold_result.get('t2_rank_ic', 0):.4f}, "
            f"T4_RankIC={fold_result.get('t4_rank_ic', 0):.4f}, "
            f"方向胜率={fold_result.get('direction_win_rate', 0):.2%}, "
            f"总耗时={elapsed:.0f}s (树={t_tree_elapsed:.0f}s T4={t_lstm_elapsed:.0f}s)"
        )

        # ── 每 Fold 完成后立即保存（断点续跑安全保障）──
        with open(save_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"  Fold {fold_num} 结果已保存: {save_path}")

        # 总体进度 + ETA
        total_elapsed = time.time() - t_pipeline_start
        completed_folds = len(all_results)
        remaining_folds = valid_folds_total - completed_folds if valid_folds_total > 0 else 4 - completed_folds
        if remaining_folds > 0:
            eta = total_elapsed / completed_folds * remaining_folds
            logger.info(f"  总进度: {completed_folds}/{valid_folds_total if valid_folds_total > 0 else '?'} Folds, "
                        f"已耗时 {total_elapsed:.0f}s ({total_elapsed/3600:.1f}h), "
                        f"预计剩余 {eta:.0f}s ({eta/3600:.1f}h)")

    # 汇总
    if all_results:
        t2_rank_ics = [r.get("t2_rank_ic", 0) for r in all_results]
        t4_rank_ics = [r.get("t4_rank_ic", 0) for r in all_results]
        aucs = [r.get("t1_auc", 0.5) for r in all_results]
        logger.info("=" * 60)
        logger.info(f"Walk-Forward 汇总 ({len(all_results)} Folds):")
        logger.info(f"  T2 Rank IC (LightGBM):     mean={np.mean(t2_rank_ics):.4f}, "
                     f"min={np.min(t2_rank_ics):.4f}, "
                     f"std={np.std(t2_rank_ics):.4f}, "
                     f">0比例={sum(1 for ic in t2_rank_ics if ic>0)/len(t2_rank_ics):.0%}")
        logger.info(f"  T4 Rank IC (LSTM-Trans):   mean={np.mean(t4_rank_ics):.4f}, "
                     f"min={np.min(t4_rank_ics):.4f}, "
                     f"std={np.std(t4_rank_ics):.4f}, "
                     f">0比例={sum(1 for ic in t4_rank_ics if ic>0)/len(t4_rank_ics):.0%}")
        logger.info(f"  T1 AUC: mean={np.mean(aucs):.4f}")
        logger.info("=" * 60)

        logger.info(f"全部结果已保存: {save_path}")

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="V2 Walk-Forward 评估")
    parser.add_argument("--no-optuna", action="store_true", help="跳过 Optuna")
    args = parser.parse_args()
    run_walk_forward(search_optuna=not args.no_optuna)


if __name__ == "__main__":
    main()
