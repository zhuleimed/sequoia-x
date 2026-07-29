"""纯 LSTM 在 Fold 3 测试集上的 Rank IC 测试"""
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "36"
os.environ["TF_NUM_INTRAOP_THREADS"] = "16"
os.environ["TF_NUM_INTEROP_THREADS"] = "8"
sys.path.insert(0, ".")
import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)
from scipy.stats import spearmanr
from sklearn.model_selection import TimeSeriesSplit
from sequoia_x.core.config import Settings
from sequoia_x.data.engine import DataEngine
from sequoia_x.model_selection_v2.config import get_config
from sequoia_x.model_selection_v2.labels import build_training_dataset
from tensorflow.keras.layers import LSTM, Dense, Input, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

cfg = get_config()
engine = DataEngine(Settings())
print("加载数据...")
X, y1, y2, y3, dates = build_training_dataset(engine, cfg, n_workers=8)
unique_dates = sorted(set(dates))

# Fold 3 划分
train_dates = [d for d in unique_dates if "2020-01-01" <= d <= "2024-12-31"]
test_dates = [d for d in unique_dates if "2025-01-01" <= d <= "2025-12-31"]
train_mask = np.array([d in train_dates for d in dates])
test_mask = np.array([d in test_dates for d in dates])
X_train, y_train = X[train_mask], y2[train_mask]
X_test, y_test = X[test_mask], y2[test_mask]

# TimeSeriesSplit 取验证集
tscv = TimeSeriesSplit(n_splits=3)
train_idx, val_idx = list(tscv.split(X_train))[-1]
X_tr, y_tr = X_train[train_idx], y_train[train_idx]
X_val, y_val = X_train[val_idx], y_train[val_idx]
print(f"训练={len(X_tr)}, 验证={len(X_val)}, 测试={len(X_test)}")

# 纯 LSTM 模型
def build_pure_lstm(lr=0.01):
    inputs = Input((X.shape[1], X.shape[2]))
    x = LSTM(128, return_sequences=True)(inputs)
    x = LSTM(64, return_sequences=False)(x)
    x = Dense(64)(x)
    x = LeakyReLU(0.1)(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=lr), loss="huber")
    return model

# 测试两个学习率
for lr, label in [(0.01, "lr=0.01"), (0.001, "lr=0.001")]:
    tf.keras.backend.clear_session(); tf.random.set_seed(42)
    model = build_pure_lstm(lr=lr)
    t0 = time.time()
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
              epochs=50, batch_size=32, verbose=0,
              callbacks=[EarlyStopping(monitor="val_loss", patience=15,
                                       restore_best_weights=True, min_delta=1e-4)])
    train_time = time.time() - t0

    # 测试集预测
    pred_test = model.predict(X_test, verbose=0).flatten()
    ic, _ = spearmanr(pred_test, y_test)
    rank_ic = float(ic) if not np.isnan(ic) else 0.0

    print(f"\n{label}:")
    print(f"  训练时间: {train_time:.0f}s")
    print(f"  pred_std={pred_test.std():.4f} mean={pred_test.mean():.4f}")
    print(f"  y_test std={y_test.std():.4f} mean={y_test.mean():.4f}")
    print(f"  Rank IC = {rank_ic:.4f}")
    print(f"{'✅ 有效' if abs(rank_ic) > 0.03 else '❌ 无效'} (|IC| > 0.03)")
