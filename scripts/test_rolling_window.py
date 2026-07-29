"""滚动窗口 vs 扩展窗口对比测试。

在当前 V2 评估 (PID 284198) 并行运行，用低线程数避免资源竞争。
测试假设：更短的训练窗口能否让 LSTM 适应 2026 年的市场风格切换。

Fold 设计：
  Roll-1: train [12个月] 2025-01~2025-12 → test 2026 H1
  Roll-2: train [9个月]  2025-07~2026-03 → test 2026 Q2
  Expand-A: train [扩展] 2024-08~2025-12 → test 2026 H1（对照 Fold 5）
  Expand-B: train [扩展] 2024-08~2026-03 → test 2026 Q2（对照 Fold 6）

预期：滚动窗口 IC 应优于或接近扩展窗口，因为丢弃了不相关的旧数据。
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ── 低线程配置，避免与主进程竞争 ──
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

import numpy as np

np.random.seed(42)
import tensorflow as tf

tf.random.set_seed(42)
# 显式设置线程数
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)

from scipy.stats import spearmanr
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sequoia_x.core.config import Settings
from sequoia_x.core.logger import get_logger
from sequoia_x.data.engine import DataEngine
from sequoia_x.model_selection_v2.config import get_config
from sequoia_x.model_selection_v2.labels import build_training_dataset

logger = get_logger(__name__)

# ── 配置 ──
BEST_PARAMS = {
    "lstm_units": 128,
    "num_transformers": 0,
    "dropout_rate": 0.285,
    "learning_rate": 0.0096,
    "l2_reg": 0.0,
    "batch_size": 32,
}
EPOCHS = 50
PATIENCE = 15
OUTPUT_FILE = Path("data/models/v2_selection/rolling_window_test.json")


def build_pure_lstm(window: int, n_features: int, params: dict) -> Model:
    """构建纯 LSTM 模型（与当前最佳配置一致）。"""
    inputs = Input((window, n_features))
    x = LSTM(params["lstm_units"], return_sequences=True, name="lstm_1")(inputs)
    x = LSTM(params["lstm_units"] // 2, return_sequences=False, name="lstm_2")(x)
    x = Dense(params["lstm_units"] // 2, name="dense_1")(x)
    x = LeakyReLU(0.1)(x)
    if params["dropout_rate"] > 0:
        x = Dropout(params["dropout_rate"], name="dropout")(x)
    outputs = Dense(1, name="output")(x)
    model = Model(inputs, outputs)
    opt = Adam(learning_rate=params["learning_rate"], clipnorm=1.0)
    model.compile(optimizer=opt, loss="huber")
    return model


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    params: dict,
    label: str,
) -> dict:
    """训练纯 LSTM 并返回 Rank IC。"""
    # TimeSeriesSplit 取训练/验证
    tscv = TimeSeriesSplit(n_splits=3)
    tr_idx, val_idx = list(tscv.split(X_train))[-1]
    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    logger.info(f"  [{label}] train={len(X_tr)}, val={len(X_val)}, test={len(X_test)}")
    logger.info(f"  [{label}] y_train mean={y_tr.mean():.4f} std={y_tr.std():.4f}")

    t0 = time.time()
    tf.keras.backend.clear_session()
    tf.random.set_seed(42)

    model = build_pure_lstm(X_train.shape[1], X_train.shape[2], params)
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=params["batch_size"],
        verbose=0,
        callbacks=[
            EarlyStopping(
                monitor="val_loss", patience=PATIENCE,
                restore_best_weights=True, min_delta=1e-4,
            ),
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=8,
                min_lr=1e-6, verbose=0,
            ),
        ],
    )
    elapsed = time.time() - t0

    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    best_val = float(np.min(history.history["val_loss"]))

    # 测试集预测
    pred = model.predict(X_test, verbose=0, batch_size=512).flatten()
    tf.keras.backend.clear_session()

    # 诊断
    pred_std = float(np.std(pred))
    ic, _ = spearmanr(pred, y_test)
    rank_ic = float(ic) if not np.isnan(ic) else 0.0

    result = {
        "label": label,
        "n_train": len(X_tr),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "y_test_mean": float(y_test.mean()),
        "y_test_std": float(y_test.std()),
        "best_epoch": best_epoch,
        "best_val": best_val,
        "pred_std": pred_std,
        "pred_mean": float(pred.mean()),
        "rank_ic": rank_ic,
        "elapsed_s": int(elapsed),
    }

    status = "✅" if abs(rank_ic) > 0.03 else ("⚠️边缘" if abs(rank_ic) > 0.01 else "❌失效")
    logger.info(
        f"  [{label}] {status} RankIC={rank_ic:.4f} | "
        f"pred_std={pred_std:.4f} | best_epoch={best_epoch}/{EPOCHS} val={best_val:.4f} | "
        f"{elapsed:.0f}s"
    )
    return result


def main():
    logger.info("=" * 60)
    logger.info("滚动窗口 vs 扩展窗口 LSTM 对比测试")
    logger.info(f"配置: epochs={EPOCHS}, patience={PATIENCE}, TF线程=4+2, OMP=4")
    logger.info(f"最佳参数: {BEST_PARAMS}")

    # 加载数据
    cfg = get_config()
    engine = DataEngine(Settings())
    logger.info("加载数据集...")
    X, y1, y2, y3, dates = build_training_dataset(engine, cfg, n_workers=8)
    unique_dates = sorted(set(dates))
    dates_arr = np.array(dates)
    logger.info(f"数据: {len(X)} 样本, {len(unique_dates)} 个日期, {unique_dates[0]}~{unique_dates[-1]}")

    # ── 定义测试 Folds ──
    folds = [
        # 滚动窗口：12个月训练
        {
            "label": "Roll-1(12m→H1)",
            "train_start": "2025-01-01", "train_end": "2025-12-31",
            "test_start": "2026-01-01", "test_end": "2026-06-30",
        },
        # 滚动窗口：9个月训练（包含 2026 Q1 熊市数据）
        {
            "label": "Roll-2(9m→Q2)",
            "train_start": "2025-07-01", "train_end": "2026-03-31",
            "test_start": "2026-04-01", "test_end": "2026-07-20",
        },
        # 扩展窗口对照
        {
            "label": "Expand-A(全→H1)",
            "train_start": "2020-01-01", "train_end": "2025-12-31",
            "test_start": "2026-01-01", "test_end": "2026-06-30",
        },
        # 扩展窗口对照
        {
            "label": "Expand-B(全→Q2)",
            "train_start": "2020-01-01", "train_end": "2026-03-31",
            "test_start": "2026-04-01", "test_end": "2026-07-20",
        },
    ]

    results = []

    for f in folds:
        train_mask = np.array([f["train_start"] <= d <= f["train_end"] for d in dates_arr])
        test_mask = np.array([f["test_start"] <= d <= f["test_end"] for d in dates_arr])

        if train_mask.sum() < 100 or test_mask.sum() < 50:
            logger.warning(f"  跳过 {f['label']}: 样本不足 (train={train_mask.sum()}, test={test_mask.sum()})")
            continue

        X_train, y_train = X[train_mask], y2[train_mask]
        X_test, y_test = X[test_mask], y2[test_mask]

        train_dates = sorted(set(dates_arr[train_mask]))
        test_dates = sorted(set(dates_arr[test_mask]))
        logger.info(f"\n── {f['label']} ──")
        logger.info(f"  train: {train_dates[0]}~{train_dates[-1]} ({len(train_dates)} dates, {len(X_train)} samples)")
        logger.info(f"  test:  {test_dates[0]}~{test_dates[-1]} ({len(test_dates)} dates, {len(X_test)} samples)")

        result = train_and_evaluate(X_train, y_train, X_test, y_test, BEST_PARAMS, f["label"])
        results.append(result)

        # 增量保存
        with open(OUTPUT_FILE, "w") as fp:
            json.dump(results, fp, indent=2, default=str)

    # ── 汇总 ──
    logger.info("\n" + "=" * 60)
    logger.info("汇总对比:")
    logger.info(f"{'Label':<25s} {'RankIC':>8s} {'y_test_mean':>12s} {'Epoch':>6s} {'耗时':>8s}")
    logger.info("-" * 65)
    for r in results:
        logger.info(
            f"{r['label']:<25s} {r['rank_ic']:>+8.4f} "
            f"{r['y_test_mean']:>+12.4f} {r['best_epoch']:>4d}/{EPOCHS} "
            f"{r['elapsed_s']:>5d}s"
        )

    # 关键对比
    roll_ics = [r["rank_ic"] for r in results if r["label"].startswith("Roll")]
    expand_ics = [r["rank_ic"] for r in results if r["label"].startswith("Expand")]
    if roll_ics and expand_ics:
        logger.info(f"\n滚动窗口 avg IC: {np.mean(roll_ics):+.4f}")
        logger.info(f"扩展窗口 avg IC: {np.mean(expand_ics):+.4f}")

    logger.info(f"\n结果已保存: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
