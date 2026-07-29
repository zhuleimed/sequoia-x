"""T4 常数预测 Bug 诊断脚本。

验证 4 个对照测试，定位 predict() 输出常数的根因。

测试:
  A: model.predict(X)           — TF 图追踪推理
  B: model(X, training=False)   — eager 模式推理
  C: model(X, training=True)    — 训练模式（dropout 开）
  D: model.evaluate(X, y)       — 和 fit 内部验证一致

预期:
  - 若 B 正常 A 异常 → TF 图追踪问题
  - 若 B 也异常 → TransformerBlock 推理逻辑问题
  - 若 C 正常 B 异常 → dropout 依赖问题
  - 若 D 异常 → 模型训练时也没学到，容量/数据问题
"""
from __future__ import annotations
import os, sys, json, time, resource
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "10"
os.environ["TF_NUM_INTRAOP_THREADS"] = "16"
os.environ["TF_NUM_INTEROP_THREADS"] = "8"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sequoia_x.core.config import Settings
from sequoia_x.data.engine import DataEngine
from sequoia_x.core.logger import get_logger
from sequoia_x.model_selection_v2.config import get_config
from sequoia_x.model_selection_v2.labels import build_training_dataset
from sequoia_x.model_selection_v2.models.deep_lstm import (
    _create_lstm_model, TransformerBlock,
)

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

logger = get_logger(__name__)


def main():
    cfg = get_config()
    engine = DataEngine(Settings())

    # ── 加载数据集（缓存秒级）──
    logger.info("加载数据集...")
    X, y1, y2, y3, dates = build_training_dataset(engine, cfg, n_workers=4)
    logger.info(f"数据: {X.shape}, y2 mean={y2.mean():.4f} std={y2.std():.4f}")

    # ── Fold 3 训练集（用尾部 5000 样本加速调试）──
    unique_dates = sorted(set(dates))
    train_dates = [d for d in unique_dates if "2020-01-01" <= d <= "2024-12-31"]
    train_mask = np.array([d in train_dates for d in dates])

    n_train = min(5000, train_mask.sum())
    X_train = X[train_mask][-n_train:]
    y_train = y2[train_mask][-n_train:]

    # TimeSeriesSplit
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=3)
    splits = list(tscv.split(X_train))
    train_idx, val_idx = splits[-1]
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    logger.info(f"训练集={len(X_tr)}, 验证集={len(X_val)}")
    logger.info(f"y_tr mean={y_tr.mean():.4f} std={y_tr.std():.4f}")
    logger.info(f"y_val mean={y_val.mean():.4f} std={y_val.std():.4f}")

    # 常量预测基准: 预测 y_val 均值时的 MSE
    baseline_mse = float(np.var(y_val))
    logger.info(f"常量预测基准 MSE (var(y_val)): {baseline_mse:.6f}")

    # ── 构建模型（用 best_params 的最优架构）──
    best_params_path = cfg.model_dir_path / "best_params_t4_lstm.json"
    with open(best_params_path) as f:
        best_params = json.load(f)
    logger.info(f"最优参数: {best_params}")

    units = best_params.get("lstm_units", 128)
    units2 = max(32, units // 2)

    model = _create_lstm_model(
        window=X_train.shape[1],
        n_features=X_train.shape[2],
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

    # ── 训练（15 epoch，不用早停）──
    t0 = time.time()
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                "/tmp/t4_debug_checkpoint.keras",
                monitor="val_loss", save_best_only=True, verbose=0,
            ),
        ],
        verbose=0,
    )
    train_time = time.time() - t0
    val_losses = history.history["val_loss"]
    logger.info(f"训练完成: {train_time:.0f}s, "
                f"val_loss: {val_losses[0]:.4f} → {val_losses[-1]:.4f}")

    # ── 取一小批验证数据做测试 ──
    X_test = X_val[:200]
    y_test = y_val[:200]

    # ════════════════════════════════════════════
    #  测试 A: model.predict()
    # ════════════════════════════════════════════
    logger.info("")
    logger.info("=" * 60)
    pred_a = model.predict(X_test, verbose=0).flatten()
    logger.info(f"测试 A (predict):     mean={pred_a.mean():.6f}, "
                f"std={pred_a.std():.6e}, "
                f"range=[{pred_a.min():.6f}, {pred_a.max():.6f}]")

    # ════════════════════════════════════════════
    #  测试 B: model(X, training=False) eager
    # ════════════════════════════════════════════
    pred_b = model(X_test, training=False).numpy().flatten()
    logger.info(f"测试 B (eager inf):    mean={pred_b.mean():.6f}, "
                f"std={pred_b.std():.6e}, "
                f"range=[{pred_b.min():.6f}, {pred_b.max():.6f}]")

    # ════════════════════════════════════════════
    #  测试 C: model(X, training=True) eager
    # ════════════════════════════════════════════
    pred_c = model(X_test, training=True).numpy().flatten()
    logger.info(f"测试 C (eager train):  mean={pred_c.mean():.6f}, "
                f"std={pred_c.std():.6e}, "
                f"range=[{pred_c.min():.6f}, {pred_c.max():.6f}]")

    # ════════════════════════════════════════════
    #  测试 D: model.evaluate()
    # ════════════════════════════════════════════
    eval_result = model.evaluate(X_val, y_val, verbose=0)
    logger.info(f"测试 D (evaluate):     loss={eval_result[0]:.6f} "
                f"(baseline MSE={baseline_mse:.6f})")

    # ════════════════════════════════════════════
    #  测试 E: 多次 predict 是否一致（排除随机性）
    # ════════════════════════════════════════════
    pred_e1 = model.predict(X_test[:10], verbose=0).flatten()
    pred_e2 = model.predict(X_test[:10], verbose=0).flatten()
    logger.info(f"测试 E (稳定性):      两次 predict 差异: "
                f"max|diff|={np.abs(pred_e1 - pred_e2).max():.2e}")

    # ════════════════════════════════════════════
    #  诊断
    # ════════════════════════════════════════════
    logger.info("")
    logger.info("=" * 60)
    logger.info("诊断结论:")

    if pred_a.std() < 1e-7 and pred_b.std() > 1e-7:
        logger.error("→ TF 图追踪问题: predict() 走图追踪路径，TransformerBlock 未正确追踪")
    elif pred_a.std() < 1e-7 and pred_b.std() < 1e-7 and pred_c.std() > 1e-7:
        logger.error("→ Dropout 依赖: 模型依赖 dropout 噪声产生变化，inference 时退化")
    elif pred_a.std() < 1e-7 and pred_b.std() < 1e-7 and pred_c.std() < 1e-7:
        logger.error("→ 模型本身的容量/数据问题: 训练中也没有学到有意义的模式")
        # 进一步检查：训练过程中 val_loss 是否显著低于 baseline MSE
        if min(val_losses) < baseline_mse * 0.95:
            logger.error("   但 val_loss 显著低于常量预测基准，矛盾！需要检查 evaluate()")
        else:
            logger.error("   val_loss 也未低于常量预测基准，模型确实没学到信号")
    else:
        logger.info(f"  一切正常！A std={pred_a.std():.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
