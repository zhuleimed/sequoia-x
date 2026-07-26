"""model_selection_v2 - T4: LSTM-Transformer 回归器（20日超额收益率）。

架构: LSTM → TransformerBlock × N → LSTM → Dense → 回归输出

与树模型 T2（LightGBM）预测同一目标（y2=20日超额收益），但捕捉
树模型无法看到的时序演化模式（动量加速/减速、波动率聚集、趋势反转）。
最终通过 T2+T4 ensemble 提升信号质量。

从 V1 (model_selection/model.py) 移植，适配 V2Config 和 Walk-Forward 管线。

CPU-only 设计：TF 内部多线程并行，Optuna trial 串行（避免多模型内存爆炸）。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

# ── CPU-only + TF 线程配置（必须在 import tensorflow 之前）──
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import optuna
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input, LeakyReLU, MultiHeadAttention,
    LayerNormalization, Add,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from sequoia_x.core.logger import get_logger
from sequoia_x.model_selection_v2.config import V2Config, get_config

logger = get_logger(__name__)


# ════════════════════════════════════════════════════════════════
#  Transformer Block（从 V1 移植）
# ════════════════════════════════════════════════════════════════

class TransformerBlock(tf.keras.layers.Layer):
    """自注意力 Transformer 模块。

    标准 Pre-LN Transformer block：LayerNorm → Attention → Residual
    → LayerNorm → FFN → Residual。

    关键: 实现了 build() 确保 Keras 序列化/反序列化后子层连接正确恢复。
    缺少 build() 会导致模型加载后所有 Transformer 层短路→预测恒为常数。
    """

    def __init__(self, embed_dim: int, num_heads: int,
                 ff_dim: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        # 子层在 __init__ 创建但在 build() 中显式构建
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def build(self, input_shape):
        """显式构建所有子层，确保序列化/反序列化后连接正确。

        不实现此方法时，Keras 反序列化过程无法正确恢复
        MultiHeadAttention/FFN 等子层的 built 状态，导致整个
        Transformer block 短路 → 输出变为常数（仅 output bias 通过）。
        """
        # input_shape: (batch, seq_len, embed_dim)
        query_shape = tf.TensorShape((None, None, self.embed_dim))

        self.att.build(query_shape, query_shape)
        # FFN Sequential: 输入即 attention 输出
        self.ffn.build(query_shape)
        self.layernorm1.build(query_shape)
        self.layernorm2.build(query_shape)
        # Dropout 子层无需显式 build
        self.dropout1.build(query_shape)
        self.dropout2.build(query_shape)

        super().build(input_shape)

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


# ════════════════════════════════════════════════════════════════
#  模型构建（从 V1 移植，适配 V2Config）
# ════════════════════════════════════════════════════════════════

def _create_lstm_model(
    window: int,
    n_features: int,
    lstm_units: int = 128,
    lstm_units2: int = 64,
    num_heads: int = 4,
    ff_dim: int = 256,
    num_transformers: int = 2,
    dropout_rate: float = 0.3,
    dense_units: int = 64,
    learning_rate: float = 0.001,
    l2_reg: float = 1e-4,
    huber_delta: float = 0.1,
    gradient_clip_norm: float = 1.0,
) -> Model:
    """构建 LSTM-Transformer 股票收益率预测模型。

    Args:
        window: 时间序列窗口长度（=cfg.window=120）。
        n_features: 每期特征维度（=62）。
        lstm_units: 第一层 LSTM 单元数（核心容量参数，Optuna 搜索）。
        lstm_units2: 第二层 LSTM 单元数（=lstm_units//2）。
        num_heads: MultiHeadAttention 头数（固定 4）。
        ff_dim: Transformer FFN 隐藏维度（=lstm_units*2）。
        num_transformers: Transformer 层数（Optuna 搜索 1-3）。
        dropout_rate: Dropout 比率（Optuna 搜索）。
        dense_units: 中间 Dense 层单元数（=lstm_units2）。
        learning_rate: Adam 学习率（Optuna 搜索，log 尺度）。
        l2_reg: L2 正则化强度（Optuna 搜索，log 尺度），0=不启用。
        huber_delta: Huber loss 的 delta 阈值——误差在此范围内用 MSE，
                     超出则用 MAE，对异常收益率（涨停/跌停）更鲁棒。
        gradient_clip_norm: 全局梯度范数裁剪阈值，防止 LSTM 梯度爆炸。

    Returns:
        Keras Model，输入 (batch, window, n_features)，输出 (batch, 1)。
    """
    from tensorflow.keras import regularizers

    inputs = Input(shape=(window, n_features), name="stock_sequence")

    # L2 正则化：仅用于 Dense kernel，不用于 LSTM（避免 recurrent 权重被压死）
    # 原因：L2 对 LSTM recurrent weights 极其有害 —— 时序记忆被正则化
    # 压至 0 → lstm_2 输出微弱 → shared_dense bias 变负 → ReLU 全死。
    reg = regularizers.l2(l2_reg) if l2_reg > 0 else None

    # ── LSTM(1)：处理时序 → 全序列输出给 Transformer ──
    #   ★ 仅 kernel_regularizer，不正则化 recurrent（时序记忆需要保留）
    x = LSTM(lstm_units, return_sequences=True,
             kernel_regularizer=reg,
             name="lstm_1")(inputs)
    x = Dropout(dropout_rate, name="dropout_lstm1")(x)

    # ── Transformer × N：自注意力捕捉跨时间步依赖 ──
    for i in range(num_transformers):
        x = TransformerBlock(
            embed_dim=lstm_units,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate,
            name=f"transformer_{i}",
        )(x)

    # ── LSTM(2)：压缩全序列为单一向量 ──
    #   ★ 同上：仅 kernel_regularizer
    x = LSTM(lstm_units2, return_sequences=False,
             kernel_regularizer=reg,
             name="lstm_2")(x)
    x = Dropout(dropout_rate, name="dropout_lstm2")(x)

    # ── 共享 Dense（LeakyReLU 防死神经元）──
    #   ReLU 在 bias 变负后输出恒 0，整个模型坍缩为常数。
    #   LeakyReLU(α=0.1) 负半轴仍有 10% 信号通过，杜绝完全死亡。
    x = Dense(dense_units, kernel_regularizer=reg,
              name="shared_dense")(x)
    x = LeakyReLU(negative_slope=0.1)(x)
    x = Dropout(dropout_rate, name="dropout_dense")(x)

    # ── 回归输出：预测 20 日超额收益率 ──
    output = Dense(1, activation="linear", name="predicted_return")(x)

    model = Model(inputs, output, name="stock_lstm_transformer")

    opt = Adam(
        learning_rate=learning_rate,
        clipnorm=gradient_clip_norm,
    )

    # Huber loss：对极端涨跌停（>10%）更鲁棒
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.Huber(delta=huber_delta),
        metrics=["mae"],
    )
    return model


# ════════════════════════════════════════════════════════════════
#  Optuna 剪枝回调
# ════════════════════════════════════════════════════════════════

class _OptunaPruneCallback(tf.keras.callbacks.Callback):
    """每 epoch 向 Optuna 上报 val_loss，支持 HyperbandPruner 中间剪枝。

    差的 trial 在早期 epoch 就被终止，大幅节省搜索时间。
    """

    def __init__(self, trial: optuna.Trial):
        super().__init__()
        self._trial = trial

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs or {}
        val = logs.get("val_loss")
        if val is not None:
            self._trial.report(val, step=epoch)
            if self._trial.should_prune():
                raise optuna.TrialPruned(
                    f"Trial pruned at epoch {epoch}: val_loss={val:.4f}"
                )


# ════════════════════════════════════════════════════════════════
#  Optuna 目标函数
# ════════════════════════════════════════════════════════════════

def _build_objective(
    X: np.ndarray,
    y: np.ndarray,
    cfg: V2Config,
) -> callable:
    """构建 Optuna 目标函数：最小化 TimeSeriesSplit 3-fold 平均验证 loss。

    搜索 6 个核心超参数：
      lstm_units, num_transformers, dropout_rate,
      learning_rate, l2_reg, batch_size

    其余参数从核心参数推导或固定：
      lstm_units2  = lstm_units // 2
      ff_dim        = lstm_units * 2
      dense_units   = lstm_units2
      num_heads     = 4（固定）
      optimizer     = Adam（固定）
      huber_delta   = 0.1（固定）
    """

    def objective(trial: optuna.Trial) -> float:
        # ── 6 个核心搜索参数 ──
        units = trial.suggest_int("lstm_units", 64, 320, step=32)
        num_tf = trial.suggest_int("num_transformers", 1, 3)
        dropout = trial.suggest_float("dropout_rate", 0.2, 0.5)
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        l2 = trial.suggest_float("l2_reg", 1e-6, 1e-2, log=True)
        batch = trial.suggest_categorical("batch_size", [32, 64, 128])

        # ── 推导参数 ──
        units2 = max(32, units // 2)
        ff = units * 2
        dense = units2

        # ── 3-fold TimeSeriesSplit 交叉验证 ──
        tscv = TimeSeriesSplit(n_splits=3)
        val_losses = []

        for fold_i, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            model = _create_lstm_model(
                window=X.shape[1],
                n_features=X.shape[2],
                lstm_units=units,
                lstm_units2=units2,
                num_heads=cfg.lstm_num_heads,
                ff_dim=ff,
                num_transformers=num_tf,
                dropout_rate=dropout,
                dense_units=dense,
                learning_rate=lr,
                l2_reg=l2,
                huber_delta=cfg.lstm_huber_delta,
                gradient_clip_norm=cfg.lstm_gradient_clip_norm,
            )

            callbacks = [
                EarlyStopping(
                    monitor="val_loss",
                    patience=cfg.lstm_early_stop_patience // 2,  # Optuna 阶段更激进早停
                    restore_best_weights=True,
                    min_delta=1e-4,
                    verbose=0,
                ),
                _OptunaPruneCallback(trial),
            ]

            model.fit(
                X_tr, y_tr,
                validation_data=(X_va, y_va),
                epochs=cfg.lstm_optuna_epochs,
                batch_size=batch,
                callbacks=callbacks,
                verbose=0,
            )

            val_loss = model.evaluate(X_va, y_va, verbose=0)[0]
            val_losses.append(val_loss)
            tf.keras.backend.clear_session()

            # 如果任何 fold 的 loss 异常，提前退出
            if np.isnan(val_loss) or np.isinf(val_loss):
                return float("inf")

        return float(np.mean(val_losses))

    return objective


# ════════════════════════════════════════════════════════════════
#  Phase 2 训练回调：每 epoch 进度日志
# ════════════════════════════════════════════════════════════════

class _EpochProgressLogger(tf.keras.callbacks.Callback):
    """每 N 个 epoch 打印一次训练/验证 loss，追踪最佳 val_loss。"""

    def __init__(self, log_every: int = 10):
        super().__init__()
        self.log_every = log_every
        self.best_val = float("inf")
        self.best_epoch = 0

    def on_epoch_end(self, epoch: int, logs: dict | None = None):
        logs = logs or {}
        epoch += 1  # 1-indexed
        val = logs.get("val_loss", float("inf"))
        is_best = val < self.best_val
        if is_best:
            self.best_val = val
            self.best_epoch = epoch
        if epoch % self.log_every == 0 or is_best or epoch == 1:
            best_mark = " ★" if is_best else ""
            logger.info(
                f"  T4 Epoch {epoch:3d}/{self.params.get('epochs', '?')}{best_mark} | "
                f"loss={logs.get('loss', 0):.4f} val={val:.4f} | "
                f"最佳轮={self.best_epoch}({self.best_val:.4f})"
            )


# ════════════════════════════════════════════════════════════════
#  公开接口：train_lstm / predict_lstm
# ════════════════════════════════════════════════════════════════

def train_lstm(
    X: np.ndarray,
    y: np.ndarray,
    cfg: V2Config | None = None,
    search_optuna: bool = True,
    best_params: dict | None = None,
    model_id: str = "default",
) -> Model:
    """训练 LSTM-Transformer 回归器（20 日超额收益率）。

    Args:
        X: 特征张量 (n_samples, window, n_features)，3D 格式，无需 flatten。
        y: 标签向量 (n_samples,)，即 y2（20 日超额收益）。
        cfg: V2Config 配置。
        search_optuna: True=先 Optuna 搜索再全量训练，False=用默认参数直接训练。
        best_params: 预定义最佳参数，跳过搜索和默认值。
        model_id: 模型标识（如 "fold3"），用于断点续跑时区分不同 Fold 的 checkpoint。

    Returns:
        训练好的 Keras Model，可直接用于 predict_lstm()。
    """
    if cfg is None:
        cfg = get_config()

    # ── 配置 TF 线程（CPU-only 多核利用）──
    os.environ.setdefault("OMP_NUM_THREADS", str(cfg.lstm_omp_num_threads))
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(cfg.lstm_tf_intraop_threads))
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", str(cfg.lstm_tf_interop_threads))

    n_samples = X.shape[0]
    logger.info(f"T4 LSTM 训练开始 | 样本={n_samples}, "
                f"window={X.shape[1]}, features={X.shape[2]}, "
                f"y mean={y.mean():.4f} std={y.std():.4f}")

    # ── Phase 1: 超参数搜索（可跳过）──
    if best_params is not None:
        logger.info(f"T4 使用传入最佳参数: {best_params}")
    elif search_optuna:
        logger.info("T4 Phase 1: Optuna 超参数搜索 "
                    f"({cfg.lstm_optuna_n_trials} trials, timeout={cfg.lstm_optuna_timeout}s)")

        # 数据量太大时抽样加速搜索（取最近 20000 样本，保持时间顺序）
        if n_samples > 20000:
            X_search = X[-20000:]
            y_search = y[-20000:]
            logger.info(f"  Optuna 抽样: {n_samples} → {len(X_search)}（取尾部时间切片）")
        else:
            X_search = X
            y_search = y

        study_db = str(cfg.model_dir_path / "optuna_t4_lstm.db")
        storage_url = f"sqlite:///{study_db}"

        study = optuna.create_study(
            direction="minimize",
            # HyperbandPruner：自动决定哪些 trial 值得分配更多 epochs，
            # 差的早期终止，比 MedianPruner 更高效。
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=3,
                max_resource=cfg.lstm_optuna_epochs,
                reduction_factor=3,
            ),
            storage=storage_url,
            study_name="t4_lstm_reg",
            load_if_exists=True,
        )

        objective_func = _build_objective(X_search, y_search, cfg)

        t0 = time.time()
        for trial_num in range(cfg.lstm_optuna_n_trials):
            elapsed = time.time() - t0
            remaining = cfg.lstm_optuna_timeout - elapsed
            if remaining <= 0:
                logger.info(f"T4 Optuna 超时 ({cfg.lstm_optuna_timeout}s)，"
                            f"已停止于 trial {trial_num}")
                break
            study.optimize(objective_func, n_trials=1, n_jobs=1,
                           timeout=remaining, show_progress_bar=False)
            elapsed_t = time.time() - t0
            logger.info(
                f"  [T4 Optuna] Trial {trial_num+1}/{cfg.lstm_optuna_n_trials} | "
                f"当前值={study.trials[-1].value:.4f} | "
                f"全局最佳={study.best_value:.4f} | "
                f"耗时={elapsed_t:.0f}s"
            )

        best_params = dict(study.best_params)
        logger.info(f"T4 Optuna 最佳: loss={study.best_value:.4f}, params={best_params}")

        # 持久化最佳参数
        params_path = cfg.model_dir_path / "best_params_t4_lstm.json"
        with open(params_path, "w") as f:
            json.dump(best_params, f, indent=2, ensure_ascii=False)
        logger.info(f"T4 最佳参数已保存: {params_path}")
    else:
        # 默认参数（无 Optuna 时的 fallback）
        best_params = {
            "lstm_units": cfg.lstm_units,
            "num_transformers": cfg.lstm_num_transformers,
            "dropout_rate": cfg.lstm_dropout_rate,
            "learning_rate": cfg.lstm_learning_rate,
            "l2_reg": cfg.lstm_l2_reg,
            "batch_size": cfg.lstm_batch_size,
        }
        logger.info(f"T4 使用默认参数: {best_params}")

    # ── Phase 2: 全量训练（支持断点续跑）──
    checkpoint_path = str(cfg.model_dir_path / f"t4_checkpoint_{model_id}.keras")
    logger.info("T4 Phase 2: 全量训练")

    # 分离训练参数与模型架构参数
    batch_size = best_params.pop("batch_size", cfg.lstm_batch_size)

    # 推导参数
    units = best_params.get("lstm_units", cfg.lstm_units)
    units2 = max(32, units // 2)

    # TimeSeriesSplit：最后的 fold 作为验证集
    if n_samples >= 100:
        tscv = TimeSeriesSplit(n_splits=3)
        splits = list(tscv.split(X))
        train_idx, val_idx = splits[-1]
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
    else:
        split = int(n_samples * 0.8)
        X_train, y_train = X[:split], y[:split]
        X_val, y_val = X[split:], y[split:]

    logger.info(f"  训练集={len(X_train)}, 验证集={len(X_val)}")

    # ── 断点续跑：检查是否已有训练好的模型 ──
    if Path(checkpoint_path).exists():
        logger.info(f"  T4 Phase 2: 从 checkpoint 恢复 ({checkpoint_path})")
        model = tf.keras.models.load_model(
            checkpoint_path,
            custom_objects={"TransformerBlock": TransformerBlock},
        )
        # 快速验证 loss
        val_loss = model.evaluate(X_val, y_val, verbose=0)[0]
        logger.info(f"  T4 Phase 2: checkpoint 加载完成, val_loss={val_loss:.4f}")
        return model

    model = _create_lstm_model(
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

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=cfg.lstm_early_stop_patience,
            restore_best_weights=True,
            min_delta=1e-4,
            verbose=0,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=cfg.lstm_reduce_lr_patience,
            min_lr=cfg.lstm_min_lr,
            verbose=0,
        ),
        # 每 10 epoch 打印进度日志
        _EpochProgressLogger(log_every=10),
        # 每个 epoch 保存最佳模型到磁盘（断点续跑安全保障）
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]

    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.lstm_epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )

    elapsed = time.time() - t0
    best_epoch = (callbacks[0].stopped_epoch
                  if callbacks[0].stopped_epoch > 0
                  else len(history.history["loss"]))
    val_loss = min(history.history["val_loss"])

    logger.info(
        f"T4 训练完成 | best_epoch={best_epoch}/{cfg.lstm_epochs}, "
        f"val_loss={val_loss:.4f}, "
        f"耗时={elapsed:.0f}s ({elapsed/60:.1f}min), "
        f"checkpoint={checkpoint_path}"
    )

    # ── 清理 TF session 并 从 checkpoint 重新加载模型 ──
    # clear_session() 会销毁 TF 默认图，导致 custom layer（TransformerBlock）
    # 内部子层（MultiHeadAttention/FFN/LayerNorm）的连接断裂。
    # 断裂后 predict() 返回的所有 Transformer 层被短路，输出恒为 bias 常数。
    #
    # 解决：clear_session 后重新从磁盘 checkpoint 加载模型。
    # 此时 TransformerBlock.build() 已实现，加载过程会正确恢复子层连接。
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(
        checkpoint_path,
        custom_objects={"TransformerBlock": TransformerBlock},
    )

    return model


def predict_lstm(model: Model, X: np.ndarray) -> np.ndarray:
    """使用 LSTM-Transformer 预测 20 日超额收益率。

    Args:
        model: 训练好的 Keras Model。
        X: 特征张量 (n_samples, window, n_features)。

    Returns:
        预测收益率 (n_samples,)。
    """
    preds = model.predict(X, verbose=0)
    return preds.flatten()


# ════════════════════════════════════════════════════════════════
#  CLI：快速验证（小规模测试用）
# ════════════════════════════════════════════════════════════════

def main() -> None:
    """CLI: 小规模快速验证 LSTM-Transformer 能正常训练+预测。

    用法: python -m sequoia_x.model_selection_v2.models.deep_lstm
    """
    from sequoia_x.core.config import Settings
    from sequoia_x.data.engine import DataEngine
    from sequoia_x.model_selection_v2.labels import build_training_dataset

    cfg = get_config()
    # 限制范围加速测试
    cfg.lstm_optuna_n_trials = 5
    cfg.lstm_optuna_epochs = 30
    cfg.lstm_epochs = 50
    cfg.sample_end = "2022-12-31"  # 只用早期数据

    engine = DataEngine(Settings())
    pool = engine.get_base_stock_pool()[:100]  # 只取 100 只

    logger.info("=" * 60)
    logger.info("T4 LSTM-Transformer 快速验证")
    logger.info("=" * 60)

    X, y1, y2, y3, dates = build_training_dataset(engine, cfg, symbols=pool)
    logger.info(f"数据: X={X.shape}, y2={y2.shape}")

    if len(X) < 50:
        logger.error("数据不足，无法测试")
        return

    # 小规模 Optuna + 全量训练
    model = train_lstm(X, y2, cfg, search_optuna=True)

    # 预测 + 评估
    preds = predict_lstm(model, X[-200:])
    y_true = y2[-200:]

    from scipy.stats import spearmanr
    ic, _ = spearmanr(preds, y_true)
    rmse = float(np.sqrt(np.mean((preds - y_true) ** 2)))

    logger.info(f"测试结果 | Rank IC={ic:.4f}, RMSE={rmse:.4f}")
    logger.info("T4 验证完成！")


if __name__ == "__main__":
    main()
