"""LSTM-Transformer 股票收益率预测模型。

架构: LSTM → TransformerBlock × N → LSTM → Dense → 回归输出
基于 DoubleColorBall/ssq_model/red_model.py 适配为收益率回归。

关键变化:
  - 输出从 6×Dense(33, softmax) 改为 1×Dense(1, linear)
  - Loss 从 sparse_categorical_crossentropy 改为 MSE
  - 移除 Beam Search 解码
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input, MultiHeadAttention,
    LayerNormalization, Add,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop

from sequoia_x.core.logger import get_logger
from sequoia_x.model_selection.config import LSTMConfig, get_config

logger = get_logger(__name__)


# ════════════════════════════════════════════════════════════
#  Training Logger 回调
# ════════════════════════════════════════════════════════════

class TrainingLogger(tf.keras.callbacks.Callback):
    """详细的训练过程日志记录器。"""

    def __init__(self, trial_name: str = "", log_every: int = 5):
        super().__init__()
        self.trial_name = trial_name
        self.log_every = log_every
        self.best_val_loss = float("inf")
        self.best_epoch = 0

    def on_train_begin(self, logs=None):
        logger.info(
            f"[{self.trial_name}] 训练开始 | "
            f"样本={self.params.get('samples', '?')} | "
            f"批次={self.params.get('batch_size', '?')} | "
            f"总轮数={self.params.get('epochs', '?')}"
        )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch += 1
        val_loss = logs.get("val_loss", float("inf"))
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
            self.best_epoch = epoch

        if epoch % self.log_every == 0 or is_best or epoch == 1:
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            best = " ★" if is_best else ""
            logger.info(
                f"[{self.trial_name}] Epoch {epoch:3d}/{self.params['epochs']}{best} | "
                f"loss={logs.get('loss',0):.4f} val={val_loss:.4f} | "
                f"mae={logs.get('mae',0):.4f} val_mae={logs.get('val_mae',0):.4f} | "
                f"lr={lr:.2e} | 最佳轮={self.best_epoch}"
            )

    def on_train_end(self, logs=None):
        logger.info(
            f"[{self.trial_name}] 训练结束 | "
            f"最佳轮={self.best_epoch} | 最佳val_loss={self.best_val_loss:.4f}"
        )


# ════════════════════════════════════════════════════════════
#  Transformer Block
# ════════════════════════════════════════════════════════════

class TransformerBlock(tf.keras.layers.Layer):
    """自注意力 Transformer 模块。"""

    def __init__(self, embed_dim: int, num_heads: int,
                 ff_dim: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config


# ════════════════════════════════════════════════════════════
#  模型构建
# ════════════════════════════════════════════════════════════

def create_stock_model(
    window: int = 60,
    n_features: int = 55,
    lstm_units: int = 128,
    lstm_units2: int = 64,
    num_heads: int = 4,
    ff_dim: int = 256,
    num_transformers: int = 2,
    dropout_rate: float = 0.2,
    dense_units: int = 128,
    optimizer: str = "adam",
    learning_rate: float = 0.001,
    l2_reg: float = 1e-4,
    huber_delta: float = 0.1,
    gradient_clip_norm: float = 1.0,
) -> Model:
    """构建 LSTM-Transformer 股票收益率预测模型。

    Args:
        window: 时间序列窗口长度。
        n_features: 每期特征维度。
        lstm_units: 第一层 LSTM 单元数。
        lstm_units2: 第二层 LSTM 单元数。
        num_heads: MultiHeadAttention 头数。
        ff_dim: Transformer FFN 隐藏维度。
        num_transformers: Transformer 层数。
        dropout_rate: Dropout 比率。
        dense_units: 中间 Dense 层单元数。
        optimizer: 优化器名称。
        learning_rate: 学习率。
        l2_reg: L2 正则化强度 (0 = 不启用)。
        huber_delta: Huber loss 的 delta 阈值——误差在此范围内用 MSE，
                     超出则用 MAE，对异常收益率更鲁棒。
        gradient_clip_norm: 全局梯度范数裁剪阈值，防止 LSTM 梯度爆炸。

    Returns:
        Keras Model，输入 (batch, window, n_features)，输出 (batch, 1)。
    """
    from tensorflow.keras import regularizers

    inputs = Input(shape=(window, n_features), name="stock_sequence")

    # L2 正则化：由 Optuna 搜索最佳强度。
    # l2_reg=0 时不启用（向后兼容已有 best_params.json）。
    reg = regularizers.l2(l2_reg) if l2_reg > 0 else None

    # LSTM(1) → 全序列输出
    x = LSTM(lstm_units, return_sequences=True,
             kernel_regularizer=reg, recurrent_regularizer=reg,
             name="lstm_1")(inputs)
    x = Dropout(dropout_rate, name="dropout_lstm1")(x)

    # Transformer × N
    for i in range(num_transformers):
        x = TransformerBlock(
            embed_dim=lstm_units,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate,
            name=f"transformer_{i}",
        )(x)

    # LSTM(2) → 压缩为向量
    x = LSTM(lstm_units2, return_sequences=False,
             kernel_regularizer=reg, recurrent_regularizer=reg,
             name="lstm_2")(x)
    x = Dropout(dropout_rate, name="dropout_lstm2")(x)

    # 共享 Dense
    x = Dense(dense_units, activation="relu",
              kernel_regularizer=reg, name="shared_dense")(x)
    x = Dropout(dropout_rate, name="dropout_dense")(x)

    # 回归输出
    output = Dense(1, activation="linear", name="predicted_return")(x)

    model = Model(inputs, output, name="stock_lstm_transformer")

    # 选择优化器 + 梯度裁剪
    opt_class = {"adam": Adam, "nadam": Nadam, "rmsprop": RMSprop}.get(
        optimizer.lower(), Adam)
    opt = opt_class(
        learning_rate=learning_rate,
        clipnorm=gradient_clip_norm,   # 全局梯度裁剪，防止 LSTM 梯度爆炸
    )

    # Huber loss：delta 控制 MSE→MAE 的切换阈值。
    # 超额收益通常在 ±10% 以内，delta=0.1 意味着异常涨跌停（>10%）用 MAE 处理。
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.Huber(delta=huber_delta),
        metrics=["mae"],
    )
    return model


# ════════════════════════════════════════════════════════════
#  模型训练
# ════════════════════════════════════════════════════════════

def train_model(
    X: np.ndarray,
    y: np.ndarray,
    cfg: LSTMConfig | None = None,
    trial_name: str = "",
    epochs: int = 200,
    **model_params,
) -> tuple[Model, dict]:
    """训练 LSTM-Transformer 模型。

    Args:
        X: 特征矩阵 (n_samples, window, n_features)。
        y: 标签向量 (n_samples,)。
        cfg: 配置对象。
        trial_name: 日志标识。
        epochs: 最大训练轮数。
        **model_params: 传递给 create_stock_model 的参数。

    Returns:
        (trained_model, history_dict)。
    """
    if cfg is None:
        cfg = get_config()

    # 时间顺序切分：用 TimeSeriesSplit 确保验证集时间在训练集之后
    # （数据按日期拼接，索引顺序即时间顺序，TS 切分尊重这一约束）
    n = len(X)
    if n >= 100:
        # 数据充足时：3-fold TS，用最后一折的 val_idx 作为验证集
        tscv = TimeSeriesSplit(n_splits=3)
        splits = list(tscv.split(X))
        train_idx, val_idx = splits[-1]  # 训练前 2/3，验证后 1/3
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        # 测试集：val 之后的所有数据
        test_start = val_idx[-1] + 1
        if test_start < n:
            X_test, y_test = X[test_start:], y[test_start:]
        else:
            X_test, y_test = X_val, y_val  # 退化：测试=验证
    else:
        # 数据太少：简单 80/20 切分
        split = int(n * 0.8)
        X_train, y_train = X[:split], y[:split]
        X_val, y_val = X[split:], y[split:]
        X_test, y_test = X_val, y_val

    # 分离训练参数（batch_size/window/n_features 不传递给 create_stock_model）
    # window 和 n_features 由数据 Shape 决定，不从外部 params 传入，
    # 防止 train_full() 中 best_params["window"] 与显式参数冲突。
    model_params = dict(model_params)
    batch_size = model_params.pop("batch_size", 64)
    model_params.pop("window", None)
    model_params.pop("n_features", None)

    model = create_stock_model(
        window=X.shape[1],
        n_features=X.shape[2],
        **model_params,
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=cfg.early_stop_patience,
            restore_best_weights=True,
            min_delta=1e-4,
            verbose=0,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=cfg.reduce_lr_factor,
            patience=cfg.reduce_lr_patience,
            min_lr=cfg.min_learning_rate,
            verbose=0,
        ),
        TrainingLogger(trial_name=trial_name),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )

    # 测试集评估
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)

    result = {
        "train_loss": history.history["loss"][-1],
        "val_loss": min(history.history["val_loss"]),
        "test_loss": test_loss,
        "test_mae": test_mae,
        "best_epoch": callbacks[0].stopped_epoch if callbacks[0].stopped_epoch else epochs,
    }

    logger.info(
        f"[{trial_name}] 训练完成 | "
        f"train_loss={result['train_loss']:.4f} "
        f"val_loss={result['val_loss']:.4f} "
        f"test_loss={result['test_loss']:.4f}"
    )
    return model, result


# ════════════════════════════════════════════════════════════
#  模型预测
# ════════════════════════════════════════════════════════════

def predict_returns(model: Model, X: np.ndarray) -> np.ndarray:
    """批量预测收益率。

    Args:
        model: 训练好的 Keras Model。
        X: 特征矩阵 (n_samples, window, n_features)。

    Returns:
        预测收益率 (n_samples,)。
    """
    preds = model.predict(X, verbose=0)
    return preds.flatten()


# ════════════════════════════════════════════════════════════
#  模型保存/加载
# ════════════════════════════════════════════════════════════

def save_model(model: Model, cfg: LSTMConfig, params: dict) -> str:
    """保存模型和参数。

    Args:
        model: Keras Model。
        cfg: 配置对象。
        params: 超参数字典。

    Returns:
        版本目录路径。
    """
    version = f"v{datetime.now().strftime('%Y%m%d_%H%M')}"
    save_dir = cfg.model_dir_path / version
    save_dir.mkdir(parents=True, exist_ok=True)

    model.save(str(save_dir / "model.keras"))
    with open(save_dir / "params.json", "w") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

    logger.info(f"模型已保存: {save_dir}")
    return str(save_dir)


def load_latest_model(cfg: LSTMConfig | None = None) -> tuple[Model, dict] | None:
    """加载最新版本的模型。

    Returns:
        (model, params_dict)，无模型时返回 None。
    """
    if cfg is None:
        cfg = get_config()

    model_dir = cfg.model_dir_path
    if not model_dir.exists():
        return None

    versions = sorted([d for d in model_dir.iterdir() if d.is_dir()], reverse=True)
    if not versions:
        return None

    latest = versions[0]
    model_path = latest / "model.keras"
    params_path = latest / "params.json"

    if not model_path.exists():
        return None

    model = tf.keras.models.load_model(
        str(model_path),
        custom_objects={"TransformerBlock": TransformerBlock},
    )
    params = {}
    if params_path.exists():
        with open(params_path) as f:
            params = json.load(f)

    logger.info(f"模型已加载: {latest}")
    return model, params
