"""验证 dropout 修复: 降低 dropout + 移除 attention dropout → inference 不再退化。"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "10"
os.environ["TF_NUM_INTRAOP_THREADS"] = "16"
os.environ["TF_NUM_INTEROP_THREADS"] = "8"

sys.path.insert(0, ".")
import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)
from sequoia_x.model_selection_v2.models.deep_lstm import _create_lstm_model

# 构造随机数据模拟训练
n_samples, window, n_features = 1000, 120, 80
X = np.random.randn(n_samples, window, n_features).astype(np.float32)
y = np.random.randn(n_samples).astype(np.float32) * 0.1 + 0.02
X_val = X[-200:]
y_val = y[-200:]

# 用修复后的代码构建模型 (dropout=0.1)
model = _create_lstm_model(
    window=window, n_features=n_features,
    lstm_units=128, lstm_units2=64,
    num_heads=4, ff_dim=256,
    num_transformers=3,
    dropout_rate=0.1,  # ← 关键修复
    dense_units=64,
    learning_rate=0.001,
    l2_reg=1e-4,
    huber_delta=0.1,
    gradient_clip_norm=1.0,
)

history = model.fit(
    X, y, validation_data=(X_val, y_val),
    epochs=10, batch_size=32, verbose=0,
)

print(f"val_loss: {history.history['val_loss'][0]:.4f} → {history.history['val_loss'][-1]:.4f}")

# 测试 A/B/C
pred_a = model.predict(X_val, verbose=0).flatten()
pred_b = model(X_val, training=False).numpy().flatten()
pred_c = model(X_val, training=True).numpy().flatten()

print(f"\nA (predict):         std={pred_a.std():.6e} mean={pred_a.mean():.6f}")
print(f"B (eager inf):       std={pred_b.std():.6e} mean={pred_b.mean():.6f}")
print(f"C (eager train):     std={pred_c.std():.6e} mean={pred_c.mean():.6f}")

if pred_a.std() > 1e-6:
    print("✅ 修复成功！Inference 输出有方差")
else:
    print("❌ 修复失败！Inference 仍退化")
