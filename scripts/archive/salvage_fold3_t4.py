"""Fold 3 T4 挽救脚本 — 修复 clear_session() 导致的常数预测 Bug。

用正确方式加载 checkpoint（不先 clear_session），重新预测 Fold 3 测试集，
计算正确的 T4 Rank IC，更新 walk_forward_results.json。
"""
from __future__ import annotations
import json
import sys
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

# 添加项目根目录
sys.path.insert(0, "/public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x")

from sequoia_x.core.config import Settings
from sequoia_x.data.engine import DataEngine
from sequoia_x.core.logger import get_logger
from sequoia_x.model_selection_v2.config import get_config
from sequoia_x.model_selection_v2.labels import build_training_dataset
from sequoia_x.model_selection_v2.models.deep_lstm import TransformerBlock

logger = get_logger(__name__)


def main():
    cfg = get_config()
    engine = DataEngine(Settings())

    # ── 加载数据集（利用磁盘缓存，秒级）──
    logger.info("加载数据集...")
    X, y1, y2, y3, dates = build_training_dataset(engine, cfg, n_workers=8)
    unique_dates = sorted(set(dates))
    logger.info(f"数据: {len(X)} 样本, {len(unique_dates)} 日期")

    # ── Fold 3 定义 ──
    train_start, train_end = "2020-01-01", "2024-12-31"
    test_start, test_end = "2025-01-01", "2025-12-31"

    train_dates = [d for d in unique_dates if train_start <= d <= train_end]
    test_dates = [d for d in unique_dates if test_start <= d <= test_end]

    train_mask = np.array([d in train_dates for d in dates])
    test_mask = np.array([d in test_dates for d in dates])

    X_test = X[test_mask]
    y2_test = y2[test_mask]

    logger.info(f"Fold 3: train={train_mask.sum()}, test={test_mask.sum()}")

    # ── 方法 1：直接 load_model（不先 clear_session）──
    checkpoint_path = cfg.model_dir_path / "t4_checkpoint_fold3.keras"
    logger.info(f"加载 checkpoint: {checkpoint_path}")

    import tensorflow as tf

    # 关键：不调用 clear_session()，直接加载
    model = tf.keras.models.load_model(
        str(checkpoint_path),
        custom_objects={"TransformerBlock": TransformerBlock},
    )

    # 验证加载是否成功：检查预测值是否有方差
    n_check = min(500, len(X_test))
    pred_check = model.predict(X_test[:n_check], verbose=0).flatten()
    pred_std = float(np.std(pred_check))
    pred_mean = float(np.mean(pred_check))

    logger.info(f"检查预测: mean={pred_mean:.6f}, std={pred_std:.6f}")
    logger.info(f"预测范围: [{pred_check.min():.6f}, {pred_check.max():.6f}]")

    if pred_std < 1e-8:
        logger.error("❌ 方法1失败：预测值仍为常数！尝试方法2...")

        # ── 方法 2：重建模型架构 + load_weights ──
        logger.info("方法2: 重建模型 + load_weights")
        from sequoia_x.model_selection_v2.models.deep_lstm import _create_lstm_model

        # 从 best_params 获取最优架构参数
        best_params_path = cfg.model_dir_path / "best_params_t4_lstm.json"
        with open(best_params_path) as f:
            best_params = json.load(f)
        logger.info(f"最优参数: {best_params}")

        units = best_params.get("lstm_units", 128)
        units2 = max(32, units // 2)

        model2 = _create_lstm_model(
            window=X.shape[1],
            n_features=X.shape[2],
            lstm_units=units,
            lstm_units2=units2,
            num_heads=cfg.lstm_num_heads,
            ff_dim=units * 2,
            num_transformers=best_params.get("num_transformers", cfg.lstm_num_transformers),
            dropout_rate=best_params.get("dropout_rate", cfg.lstm_dropout_rate),
            dense_units=units2,
            learning_rate=best_params.get("learning_rate", cfg.lstm_learning_rate),
            l2_reg=best_params.get("l2_reg", cfg.lstm_l2_reg),
            huber_delta=cfg.lstm_huber_delta,
            gradient_clip_norm=cfg.lstm_gradient_clip_norm,
        )

        # 编译模型使其能接受 weights
        model2.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss="huber",
        )
        # 先跑一次 predict 触发 build
        model2.predict(X_test[:1], verbose=0)
        # 加载权重
        model2.load_weights(str(checkpoint_path))

        pred_check2 = model2.predict(X_test[:n_check], verbose=0).flatten()
        pred_std2 = float(np.std(pred_check2))
        logger.info(f"方法2 检查: mean={np.mean(pred_check2):.6f}, std={pred_std2:.6f}")

        if pred_std2 < 1e-8:
            logger.error("❌ 方法2也失败！中止")
            return 1

        logger.info("✅ 方法2成功！使用重建模型")
        model = model2
        pred_std = pred_std2
    else:
        logger.info("✅ 方法1成功！load_model 无需 clear_session")

    # ── 全量预测 ──
    logger.info(f"全量预测 {len(X_test)} 样本...")
    batch_size = 1024
    all_preds = []
    for i in range(0, len(X_test), batch_size):
        batch = X_test[i:i + batch_size]
        preds = model.predict(batch, verbose=0).flatten()
        all_preds.append(preds)
    pred_t4 = np.concatenate(all_preds)

    # ── 计算 Rank IC ──
    ic, _ = spearmanr(pred_t4, y2_test)
    rank_ic = float(ic) if not np.isnan(ic) else 0.0
    rmse = float(np.sqrt(mean_squared_error(y2_test, pred_t4)))

    logger.info(f"")
    logger.info(f"{'='*60}")
    logger.info(f"Fold 3 T4 挽救结果:")
    logger.info(f"  T4 Rank IC: {rank_ic:.4f}")
    logger.info(f"  T4 RMSE:    {rmse:.4f}")
    logger.info(f"  Pred mean:  {np.mean(pred_t4):.6f}")
    logger.info(f"  Pred std:   {np.std(pred_t4):.6f}")
    logger.info(f"  Pred range: [{pred_t4.min():.6f}, {pred_t4.max():.6f}]")
    logger.info(f"  Y2 mean:    {np.mean(y2_test):.6f}")
    logger.info(f"  Y2 std:     {np.std(y2_test):.6f}")
    logger.info(f"{'='*60}")

    # ── 更新 walk_forward_results.json ──
    save_path = cfg.model_dir_path / "walk_forward_results.json"
    with open(save_path) as f:
        all_results = json.load(f)

    for r in all_results:
        if r.get("fold") == 3:
            r["t4_rank_ic"] = rank_ic
            r["t4_rmse"] = rmse
            r["t4_pred_mean"] = float(np.mean(pred_t4))
            r["t4_pred_std"] = float(np.std(pred_t4))
            r["t4_salvage_note"] = "2026-07-27 Bug修复: 跳过clear_session直接load_model"
            logger.info(f"已更新 Fold 3 的 T4 结果")
            break

    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"结果已保存: {save_path}")
    logger.info("✅ Fold 3 挽救完成！")
    return 0


if __name__ == "__main__":
    sys.exit(main())
