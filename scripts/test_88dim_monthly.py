"""方案A: 88维市场状态特征月度验证，仅测2026年6个月。"""
import json, os, sys, time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)
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

logger = get_logger("88dim_test")
BEST_PARAMS = {"lstm_units": 128, "num_transformers": 0, "dropout_rate": 0.285,
               "learning_rate": 0.0096, "l2_reg": 0.0, "batch_size": 32}
EPOCHS, PATIENCE = 50, 15
OUTPUT = Path("data/models/v2_selection/test_88dim_monthly.json")


def build_lstm(w, f, p):
    inp = Input((w, f))
    x = LSTM(p["lstm_units"], return_sequences=True)(inp)
    x = LSTM(p["lstm_units"] // 2, return_sequences=False)(x)
    x = Dense(p["lstm_units"] // 2)(x); x = LeakyReLU(0.1)(x)
    if p["dropout_rate"] > 0: x = Dropout(p["dropout_rate"])(x)
    out = Dense(1)(x)
    m = Model(inp, out)
    m.compile(optimizer=Adam(learning_rate=p["learning_rate"], clipnorm=1.0), loss="huber")
    return m


def main():
    logger.info("=" * 50)
    logger.info("方案A: 88维市场状态特征-月度验证(仅2026)")
    cfg = get_config()
    engine = DataEngine(Settings())
    logger.info("加载88维数据集...")
    X, y1, y2, y3, dates = build_training_dataset(engine, cfg, n_workers=4)
    logger.info(f"X={X.shape}, dates={sorted(set(dates))[0]}~{sorted(set(dates))[-1]}")

    months = sorted(set(d[:7] for d in dates))
    # 仅2026年月份
    target_months = [m for m in months if m.startswith("2026")]
    logger.info(f"测试月份: {target_months}")

    results = []
    t0 = time.time()
    for mi, test_month in enumerate(target_months):
        month_idx = months.index(test_month)
        train_start = months[max(0, month_idx - 12)]
        train_end = months[month_idx - 1]
        dates_arr = np.array(dates)
        train_mask = np.array([train_start + "-01" <= d <=
            [d2 for d2 in sorted(set(dates)) if d2[:7] == train_end][-1]
            for d in dates_arr])
        test_mask = np.array([d[:7] == test_month for d in dates_arr])

        logger.info(f"\n[{mi+1}/{len(target_months)}] {test_month}: train {train_start}~{train_end} ({train_mask.sum()}样本) → test {test_month} ({test_mask.sum()}样本)")
        X_tr, y_tr = X[train_mask], y2[train_mask]
        X_te, y_te = X[test_mask], y2[test_mask]

        tscv = TimeSeriesSplit(n_splits=3)
        tri, vai = list(tscv.split(X_tr))[-1]
        tf.keras.backend.clear_session(); tf.random.set_seed(42)
        m = build_lstm(X_tr.shape[1], X_tr.shape[2], BEST_PARAMS)
        t1 = time.time()
        h = m.fit(X_tr[tri], y_tr[tri], validation_data=(X_tr[vai], y_tr[vai]),
                  epochs=EPOCHS, batch_size=32, verbose=0,
                  callbacks=[EarlyStopping(monitor="val_loss", patience=PATIENCE,
                             restore_best_weights=True, min_delta=1e-4),
                             ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6)])
        elapsed = time.time() - t1
        pred = m.predict(X_te, verbose=0).flatten()
        tf.keras.backend.clear_session()
        ic, _ = spearmanr(pred, y_te)
        r = {"month": test_month, "rank_ic": float(ic) if not np.isnan(ic) else 0.0,
             "best_epoch": int(np.argmin(h.history["val_loss"])) + 1,
             "y_mean": float(y_te.mean()), "y_std": float(y_te.std()),
             "n_train": len(X_tr), "n_test": len(X_te), "elapsed": int(elapsed)}
        results.append(r)
        with open(OUTPUT, "w") as f: json.dump(results, f, indent=2)
        status = "✅" if r["rank_ic"] > 0.03 else ("⚠️" if r["rank_ic"] > 0 else "❌")
        logger.info(f"  {status} RankIC={r['rank_ic']:+.4f} | ep={r['best_epoch']}/{EPOCHS} | {elapsed:.0f}s")

    # 汇总
    ics = [r["rank_ic"] for r in results]
    logger.info(f"\n88维月度汇总: IC mean={np.mean(ics):+.4f}, >0={sum(1 for ic in ics if ic>0)}/{len(ics)}")
    logger.info(f"结果: {OUTPUT}")


if __name__ == "__main__":
    main()
