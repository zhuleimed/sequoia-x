"""T4 第二层问题诊断：为什么 L2=0 后 kernel 活着但预测仍退化？

测试项:
  A: 纯 LSTM (无 Transformer)
  B: 纯 LSTM + 高学习率 0.1
  C: 纯 LSTM + 去梯度裁剪
  D: 当前架构 (LSTM+3T) + 高学习率 0.05
  E: 当前架构 + 去梯度裁剪
"""
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

from sklearn.model_selection import TimeSeriesSplit
from sequoia_x.core.config import Settings
from sequoia_x.data.engine import DataEngine
from sequoia_x.model_selection_v2.config import get_config
from sequoia_x.model_selection_v2.labels import build_training_dataset
from sequoia_x.model_selection_v2.models.deep_lstm import _create_lstm_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2 as L2Reg

cfg = get_config()
engine = DataEngine(Settings())
print("加载数据...")
X, y1, y2, y3, dates = build_training_dataset(engine, cfg, n_workers=8)
unique_dates = sorted(set(dates))
train_dates = [d for d in unique_dates if "2020-01-01" <= d <= "2024-12-31"]
train_mask = np.array([d in train_dates for d in dates])
X_tr = X[train_mask][-5000:]
y_tr = y2[train_mask][-5000:]
tscv = TimeSeriesSplit(n_splits=3)
train_idx, val_idx = list(tscv.split(X_tr))[-1]
X_train, y_train = X_tr[train_idx], y_tr[train_idx]
X_val, y_val = X_tr[val_idx], y_tr[val_idx]
print(f"训练={len(X_train)}, 验证={len(X_val)}")

def build_pure_lstm(window, n_features, lstm_units=128, lr=0.01, clip=1.0):
    """纯 LSTM，无 Transformer"""
    inputs = Input((window, n_features))
    x = LSTM(lstm_units, return_sequences=True, name="lstm_1")(inputs)
    x = LSTM(lstm_units // 2, return_sequences=False, name="lstm_2")(x)
    x = Dense(lstm_units // 2, name="dense")(x)
    x = LeakyReLU(0.1)(x)
    outputs = Dense(1, name="pred")(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=lr, clipnorm=clip),
                  loss="huber", metrics=[])
    return model


def test_model(model, name, epochs=15):
    """训练并测试 predict 方差"""
    t0 = time.time()
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=32, verbose=0)
    elapsed = time.time() - t0
    pred = model.predict(X_val[:200], verbose=0).flatten()
    pred_std = float(np.std(pred))
    val_loss = min(history.history["val_loss"])
    # 检查 LSTM kernel
    k1 = np.linalg.norm(model.get_layer("lstm_1").get_weights()[0])
    status = "✅" if pred_std > 1e-5 else ("⚠️边缘" if pred_std > 1e-7 else "❌退化")
    print(f"  {status} | {elapsed:.0f}s | val={val_loss:.4f} | pred_std={pred_std:.2e} | lstm1_k={k1:.2f}")
    return {"name": name, "pred_std": pred_std, "val_loss": val_loss, "status": status}


results = []

# A: 纯 LSTM, 标准参数
print("\n=== A: 纯LSTM (无Transformer, lr=0.01, L2=0) ===")
tf.keras.backend.clear_session(); tf.random.set_seed(42)
m = build_pure_lstm(X_train.shape[1], X_train.shape[2], lr=0.01, clip=1.0)
results.append(test_model(m, "A-纯LSTM"))

# B: 纯 LSTM + 高学习率
print("\n=== B: 纯LSTM + 高lr=0.1 ===")
tf.keras.backend.clear_session(); tf.random.set_seed(42)
m = build_pure_lstm(X_train.shape[1], X_train.shape[2], lr=0.1, clip=1.0)
results.append(test_model(m, "B-纯LSTM高lr"))

# C: 纯 LSTM + 去梯度裁剪
print("\n=== C: 纯LSTM + 无梯度裁剪 ===")
tf.keras.backend.clear_session(); tf.random.set_seed(42)
m = build_pure_lstm(X_train.shape[1], X_train.shape[2], lr=0.01, clip=None)
# 重编译
m.compile(optimizer=Adam(learning_rate=0.01), loss="huber")
results.append(test_model(m, "C-纯LSTM无裁剪"))

# D: 当前架构 + 高学习率
print("\n=== D: 当前架构(LSTM+3T) + 高lr=0.05 ===")
tf.keras.backend.clear_session(); tf.random.set_seed(42)
m = _create_lstm_model(X_train.shape[1], X_train.shape[2],
    lstm_units=128, lstm_units2=64, num_heads=4, ff_dim=256, num_transformers=3,
    dropout_rate=0.0, dense_units=64, learning_rate=0.05,
    l2_reg=0.0, huber_delta=0.1, gradient_clip_norm=1.0)
results.append(test_model(m, "D-原架构高lr"))

# E: 当前架构 + 去梯度裁剪
print("\n=== E: 当前架构 + 无梯度裁剪 ===")
tf.keras.backend.clear_session(); tf.random.set_seed(42)
m = _create_lstm_model(X_train.shape[1], X_train.shape[2],
    lstm_units=128, lstm_units2=64, num_heads=4, ff_dim=256, num_transformers=3,
    dropout_rate=0.0, dense_units=64, learning_rate=0.01,
    l2_reg=0.0, huber_delta=0.1, gradient_clip_norm=None)
results.append(test_model(m, "E-原架构无裁剪"))

print("\n" + "=" * 60)
print("汇总:")
for r in results:
    print(f"  {r['name']:25s} {r['status']} pred_std={r['pred_std']:.2e} val={r['val_loss']:.4f}")
