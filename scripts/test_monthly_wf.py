"""月度 Walk-Forward 验证：12个月滚动训练 → 1个月测试。

对比半年度 Fold（原方案），验证月度重训练能否适应 2026 年市场切换。
主进程已完成，用全量 CPU 加速。
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

# ── 全量 CPU 配置 ──
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "10"
os.environ["TF_NUM_INTRAOP_THREADS"] = "16"
os.environ["TF_NUM_INTEROP_THREADS"] = "8"

import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)
tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(8)

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

BEST_PARAMS = {
    "lstm_units": 128, "num_transformers": 0, "dropout_rate": 0.285,
    "learning_rate": 0.0096, "l2_reg": 0.0, "batch_size": 32,
}
EPOCHS = 50
PATIENCE = 15
TRAIN_MONTHS = 12
OUTPUT_FILE = Path("data/models/v2_selection/monthly_walk_forward.json")


def build_pure_lstm(window: int, n_features: int, params: dict) -> Model:
    inputs = Input((window, n_features))
    x = LSTM(params["lstm_units"], return_sequences=True, name="lstm_1")(inputs)
    x = LSTM(params["lstm_units"] // 2, return_sequences=False, name="lstm_2")(x)
    x = Dense(params["lstm_units"] // 2, name="dense_1")(x)
    x = LeakyReLU(0.1)(x)
    if params["dropout_rate"] > 0:
        x = Dropout(params["dropout_rate"], name="dropout")(x)
    outputs = Dense(1, name="output")(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=params["learning_rate"], clipnorm=1.0), loss="huber")
    return model


def train_and_evaluate(X_tr, y_tr, X_te, y_te, params, label):
    tscv = TimeSeriesSplit(n_splits=3)
    tr_idx, val_idx = list(tscv.split(X_tr))[-1]
    X_t, y_t = X_tr[tr_idx], y_tr[tr_idx]
    X_v, y_v = X_tr[val_idx], y_tr[val_idx]

    tf.keras.backend.clear_session()
    tf.random.set_seed(42)
    model = build_pure_lstm(X_tr.shape[1], X_tr.shape[2], params)
    t0 = time.time()
    history = model.fit(
        X_t, y_t, validation_data=(X_v, y_v),
        epochs=EPOCHS, batch_size=params["batch_size"], verbose=0,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=PATIENCE,
                          restore_best_weights=True, min_delta=1e-4),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6, verbose=0),
        ],
    )
    elapsed = time.time() - t0
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    best_val = float(np.min(history.history["val_loss"]))
    pred = model.predict(X_te, verbose=0, batch_size=512).flatten()
    tf.keras.backend.clear_session()

    ic, _ = spearmanr(pred, y_te)
    rank_ic = float(ic) if not np.isnan(ic) else 0.0
    return {
        "label": label, "n_train": len(X_t), "n_test": len(X_te),
        "y_test_mean": float(y_te.mean()), "y_test_std": float(y_te.std()),
        "best_epoch": best_epoch, "best_val": best_val,
        "pred_std": float(np.std(pred)), "rank_ic": rank_ic, "elapsed_s": int(elapsed),
    }


def main():
    logger.info("=" * 60)
    logger.info(f"月度 Walk-Forward: {TRAIN_MONTHS}月滚动训练 → 1月测试")
    logger.info(f"TF线程=16+8, OMP=10, epochs={EPOCHS}, patience={PATIENCE}")

    cfg = get_config()
    engine = DataEngine(Settings())
    logger.info("加载数据集...")
    X, y1, y2, y3, dates = build_training_dataset(engine, cfg, n_workers=8)
    unique_dates = sorted(set(dates))
    dates_arr = np.array(dates)
    logger.info(f"数据: {len(X)} 样本, {len(unique_dates)} 个日期, {unique_dates[0]}~{unique_dates[-1]}")

    # ── 按月份分组 ──
    month_groups: dict[str, list[str]] = defaultdict(list)
    for d in unique_dates:
        month_key = d[:7]  # "2024-08"
        month_groups[month_key].append(d)
    months = sorted(month_groups.keys())
    logger.info(f"月份: {len(months)} 个 ({months[0]} ~ {months[-1]})")

    # ── 定义月度 Folds（12月训练 → 1月测试）──
    folds = []
    for i in range(TRAIN_MONTHS, len(months)):
        test_month = months[i]
        train_start_month = months[i - TRAIN_MONTHS]
        train_end_month = months[i - 1]
        folds.append({
            "label": f"Monthly-{test_month}",
            "train_start": train_start_month + "-01",
            "train_end": month_groups[train_end_month][-1],  # last date of last train month
            "test_start": month_groups[test_month][0],        # first date of test month
            "test_end": month_groups[test_month][-1],         # last date of test month
            "months_train": f"{train_start_month}~{train_end_month}",
            "months_test": test_month,
        })

    logger.info(f"共 {len(folds)} 个月度 Fold ({folds[0]['months_test']} ~ {folds[-1]['months_test']})")

    results = []
    overall_t0 = time.time()

    for fi, f in enumerate(folds):
        train_mask = np.array([f["train_start"] <= d <= f["train_end"] for d in dates_arr])
        test_mask = np.array([f["test_start"] <= d <= f["test_end"] for d in dates_arr])

        if train_mask.sum() < 100 or test_mask.sum() < 50:
            logger.warning(f"  跳过 {f['label']}: train={train_mask.sum()}, test={test_mask.sum()}")
            continue

        # 进度
        elapsed_total = time.time() - overall_t0
        completed = len(results)
        eta = elapsed_total / max(completed, 1) * (len(folds) - completed) if completed > 0 else 0
        logger.info(f"\n── [{fi+1}/{len(folds)}] {f['label']} "
                    f"(train {f['months_train']} → test {f['months_test']}) "
                    f"[{elapsed_total/60:.0f}min elapsed, ETA {eta/60:.0f}min] ──")

        X_train, y_train = X[train_mask], y2[train_mask]
        X_test, y_test = X[test_mask], y2[test_mask]
        logger.info(f"  n_train={len(X_train)}, n_test={len(X_test)}, "
                    f"y_train_mean={y_train.mean():.4f}, y_test_mean={y_test.mean():.4f}")

        result = train_and_evaluate(X_train, y_train, X_test, y_test, BEST_PARAMS, f["label"])
        results.append(result)

        # 增量保存
        with open(OUTPUT_FILE, "w") as fp:
            json.dump(results, fp, indent=2, default=str)

        # 逐 Fold 汇报
        status = "✅" if result["rank_ic"] > 0.03 else ("⚠️" if result["rank_ic"] > 0 else "❌")
        logger.info(f"  {status} RankIC={result['rank_ic']:+.4f} | "
                    f"ep={result['best_epoch']}/{EPOCHS} | {result['elapsed_s']:.0f}s")

    # ── 汇总 ──
    logger.info("\n" + "=" * 60)
    logger.info("月度 Walk-Forward 汇总:")
    logger.info(f"{'Month':<20s} {'RankIC':>8s} {'y_mean':>8s} {'n_test':>7s}")
    logger.info("-" * 50)

    ics = []
    for r in results:
        ics.append(r["rank_ic"])
        logger.info(f"{r['label']:<20s} {r['rank_ic']:>+8.4f} {r['y_test_mean']:>+8.4f} {r['n_test']:>7d}")

    # 分段统计
    pre_2026 = [r for r in results if "2025-" in r["label"]]
    post_2026 = [r for r in results if "2026-" in r["label"]]

    logger.info(f"\n2025 年月度 IC: mean={np.mean([r['rank_ic'] for r in pre_2026]):+.4f}, "
                f">0比例={sum(1 for r in pre_2026 if r['rank_ic']>0)}/{len(pre_2026)}")
    logger.info(f"2026 年月度 IC: mean={np.mean([r['rank_ic'] for r in post_2026]):+.4f}, "
                f">0比例={sum(1 for r in post_2026 if r['rank_ic']>0)}/{len(post_2026)}")
    logger.info(f"全部月度 IC:   mean={np.mean(ics):+.4f}, >0比例={sum(1 for ic in ics if ic>0)}/{len(ics)}")

    # 对比原半年度
    logger.info(f"\n对比原半年度 Fold:")
    logger.info(f"  Fold 5 (2026 H1): T4 IC = -0.2584")
    logger.info(f"  Fold 6 (2026 Q2): T4 IC = -0.0909")
    if post_2026:
        logger.info(f"  月度 2026 平均:        {np.mean([r['rank_ic'] for r in post_2026]):+.4f}")

    logger.info(f"\n结果已保存: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
