"""T4 快速验证脚本 — 5 分钟测试 dropout_rate 对 predict 方差的影响。

用真实 Fold 3 训练数据 + 10 epoch，快速验证不同 dropout_rate 下
predict() 是否输出常数。避免每次等 3 小时。
"""
import os, sys, json, time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "36"
os.environ["TF_NUM_INTRAOP_THREADS"] = "16"
os.environ["TF_NUM_INTEROP_THREADS"] = "8"

sys.path.insert(0, ".")
import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)
from sklearn.model_selection import TimeSeriesSplit
from sequoia_x.core.config import Settings
from sequoia_x.data.engine import DataEngine
from sequoia_x.model_selection_v2.config import get_config
from sequoia_x.model_selection_v2.labels import build_training_dataset
from sequoia_x.model_selection_v2.models.deep_lstm import _create_lstm_model

cfg = get_config()
engine = DataEngine(Settings())

print("加载数据...")
X, y1, y2, y3, dates = build_training_dataset(engine, cfg, n_workers=8)

# Fold 3 训练数据
unique_dates = sorted(set(dates))
train_dates = [d for d in unique_dates if "2020-01-01" <= d <= "2024-12-31"]
train_mask = np.array([d in train_dates for d in dates])
X_tr = X[train_mask]
y_tr = y2[train_mask]

# 取尾部 8000 样本（足够做 TimeSeriesSplit + 10 epoch）
n = min(8000, len(X_tr))
X_tr = X_tr[-n:]
y_tr = y_tr[-n:]

# TimeSeriesSplit 取最后一个 fold
tscv = TimeSeriesSplit(n_splits=3)
train_idx, val_idx = list(tscv.split(X_tr))[-1]
X_train, y_train = X_tr[train_idx], y_tr[train_idx]
X_val, y_val = X_tr[val_idx], y_tr[val_idx]

print(f"训练={len(X_train)}, 验证={len(X_val)}, y_val mean={y_val.mean():.4f} std={y_val.std():.4f}")

# 测试不同 dropout_rate
dropout_rates = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
results = []

for dr in dropout_rates:
    print(f"\n--- dropout_rate={dr} ---")
    tf.keras.backend.clear_session()
    tf.random.set_seed(42)

    model = _create_lstm_model(
        window=X_train.shape[1], n_features=X_train.shape[2],
        lstm_units=128, lstm_units2=64,
        num_heads=4, ff_dim=256, num_transformers=3,
        dropout_rate=dr,
        dense_units=64, learning_rate=0.01,
        l2_reg=1e-4, huber_delta=0.1, gradient_clip_norm=1.0,
    )

    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10, batch_size=32, verbose=0,
    )
    train_time = time.time() - t0

    # 检查 predict 方差
    X_test = X_val[:200]
    pred = model.predict(X_test, verbose=0).flatten()
    pred_std = float(np.std(pred))
    val_loss = min(history.history["val_loss"])
    baseline_mse = float(np.var(y_val))

    status = "✅" if pred_std > 1e-6 else "❌ 退化"
    print(f"  {status} | {train_time:.0f}s | val_loss={val_loss:.4f} | "
          f"pred_std={pred_std:.6e} | baseline_MSE={baseline_mse:.4f}")

    results.append({
        "dropout_rate": dr,
        "pred_std": pred_std,
        "val_loss": val_loss,
        "status": "ok" if pred_std > 1e-6 else "degenerate",
    })

print("\n" + "=" * 60)
print("汇总:")
for r in results:
    print(f"  dropout={r['dropout_rate']:.2f} → pred_std={r['pred_std']:.2e} "
          f"val_loss={r['val_loss']:.4f} [{r['status']}]")

good = [r for r in results if r["status"] == "ok"]
if good:
    print(f"\n最大可用 dropout: {max(r['dropout_rate'] for r in good):.2f}")
else:
    print("\n❌ 所有 dropout 值都退化！问题不在 dropout！")
