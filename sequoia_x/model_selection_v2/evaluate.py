"""model_selection_v2 - Purged Rolling Walk-Forward 评估。

对每个扩展窗口：训练 → 评估（purge gap 隔开）→ 报告 IC/AUC/RMSE。
"""
from __future__ import annotations
import argparse
import json
import time
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

    # 构建全量数据集（带日期标签）
    X, y1, y2, y3, dates = build_training_dataset(engine, cfg)
    if len(X) == 0:
        logger.error("无数据")
        return []

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

    all_results = []

    for fold_i, (train_start, train_end, test_start, test_end) in enumerate(fold_boundaries):
        logger.info(f"── Fold {fold_i+1}: train {train_start}~{train_end}, test {test_start}~{test_end} ──")
        t0 = time.time()

        # Purge: 找到训练集最后一个日期和测试集第一个日期
        train_dates = [d for d in unique_dates if train_start <= d <= train_end]
        test_dates = [d for d in unique_dates if test_start <= d <= test_end]
        if not train_dates or not test_dates:
            logger.warning(f"Fold {fold_i+1}: 无数据，跳过")
            continue

        # 找到有至少 purge_gap 间隔的切分点
        train_mask = np.array([d in train_dates for d in dates])
        test_mask = np.array([d in test_dates for d in dates])

        if train_mask.sum() < 100 or test_mask.sum() < 50:
            logger.warning(f"Fold {fold_i+1}: 样本不足（train={train_mask.sum()}, test={test_mask.sum()}），跳过")
            continue

        # 训练 3 个模型（不限 Optuna，快速评估）
        X_train, X_test = X[train_mask], X[test_mask]
        y1_train, y1_test = y1[train_mask], y1[test_mask]
        y2_train, y2_test = y2[train_mask], y2[test_mask]
        y3_train, y3_test = y3[train_mask], y3[test_mask]

        model_t1 = train_cls(X_train, y1_train, cfg, search_optuna=search_optuna)
        model_t2 = train_reg(X_train, y2_train, cfg, search_optuna=search_optuna)
        model_t3 = train_vol(X_train, y3_train, cfg, search_optuna=search_optuna)

        # 预测
        pred_t1 = predict_cls(model_t1, X_test)
        pred_t2 = predict_reg(model_t2, X_test)
        pred_t3 = predict_vol(model_t3, X_test)

        # 评估指标
        fold_result = {
            "fold": fold_i + 1,
            "train_start": train_start, "train_end": train_end,
            "test_start": test_start, "test_end": test_end,
            "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
        }

        # T1: AUC + 准确率
        try:
            fold_result["t1_auc"] = float(roc_auc_score(y1_test, pred_t1))
        except ValueError:
            fold_result["t1_auc"] = 0.5
        fold_result["t1_accuracy"] = float(((pred_t1 > 0.5) == y1_test).mean())

        # T2: Rank IC + RMSE
        fold_result["t2_rank_ic"] = _compute_rank_ic(pred_t2, y2_test)
        fold_result["t2_rmse"] = float(np.sqrt(mean_squared_error(y2_test, pred_t2)))

        # T3: RMSE
        fold_result["t3_rmse"] = float(np.sqrt(mean_squared_error(y3_test, pred_t3)))

        # 方向胜率
        if len(pred_t1) > 0:
            buy_mask = pred_t1 > 0.55
            if buy_mask.sum() > 0:
                fold_result["direction_win_rate"] = float(y1_test[buy_mask].mean())
            else:
                fold_result["direction_win_rate"] = 0.0

        elapsed = time.time() - t0
        fold_result["elapsed"] = elapsed
        all_results.append(fold_result)

        logger.info(
            f"Fold {fold_i+1}: "
            f"T1_AUC={fold_result.get('t1_auc', 0):.3f}, "
            f"T2_RankIC={fold_result.get('t2_rank_ic', 0):.4f}, "
            f"方向胜率={fold_result.get('direction_win_rate', 0):.2%}, "
            f"耗时={elapsed:.0f}s"
        )

    # 汇总
    if all_results:
        rank_ics = [r.get("t2_rank_ic", 0) for r in all_results]
        aucs = [r.get("t1_auc", 0.5) for r in all_results]
        logger.info("=" * 60)
        logger.info(f"Walk-Forward 汇总 ({len(all_results)} Folds):")
        logger.info(f"  T2 Rank IC: mean={np.mean(rank_ics):.4f}, "
                     f"min={np.min(rank_ics):.4f}, "
                     f"std={np.std(rank_ics):.4f}, "
                     f">0比例={sum(1 for ic in rank_ics if ic>0)/len(rank_ics):.0%}")
        logger.info(f"  T1 AUC: mean={np.mean(aucs):.4f}")
        logger.info("=" * 60)

        # 保存结果
        save_path = cfg.model_dir_path / "walk_forward_results.json"
        with open(save_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"结果已保存: {save_path}")

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="V2 Walk-Forward 评估")
    parser.add_argument("--no-optuna", action="store_true", help="跳过 Optuna")
    args = parser.parse_args()
    run_walk_forward(search_optuna=not args.no_optuna)


if __name__ == "__main__":
    main()
