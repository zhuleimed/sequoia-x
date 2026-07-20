"""LSTM-Transformer 模型训练入口。

三种模式:
  --full          月度完整 Optuna 搜索 + 最终训练 (每月15日)
  --incremental   每日增量学习 (管线中)
  --weekly        每周刷新 (每周六)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "8")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "4")

import numpy as np
import optuna
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sequoia_x.core.config import Settings
from sequoia_x.data.engine import DataEngine
from sequoia_x.core.logger import get_logger
from sequoia_x.model_selection.config import LSTMConfig, get_config
from sequoia_x.model_selection.features import build_batch_features
from sequoia_x.model_selection.model import (
    create_stock_model, train_model, save_model, load_latest_model,
)

logger = get_logger(__name__)


def _sample_stocks(engine: DataEngine, n: int = 200) -> list[str]:
    """简单随机抽样代表性股票（后续可改为市值分层抽样）。"""
    pool = engine.get_base_stock_pool()
    if len(pool) <= n:
        return pool

    # 简单随机抽样（后续可改为市值分层）
    rng = np.random.RandomState(42)
    indices = rng.choice(len(pool), size=n, replace=False)
    return [pool[i] for i in indices]


def _get_trade_dates(engine: DataEngine, lookback: int) -> list[str]:
    """获取最近 N 个交易日日期列表。"""
    import sqlite3
    conn = sqlite3.connect(engine.settings.db_path)
    rows = conn.execute(
        "SELECT DISTINCT date FROM stock_daily ORDER BY date DESC LIMIT ?",
        (lookback,)
    ).fetchall()
    conn.close()
    return [r[0] for r in reversed(rows)]


def _build_objective(engine: DataEngine, symbols: list[str],
                     ref_dates: list[str], cfg: LSTMConfig):
    """构建 Optuna 目标函数。"""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "lstm_units": trial.suggest_int("lstm_units",
                                             *cfg.lstm_units_range, step=32),
            "lstm_units2": trial.suggest_int("lstm_units2",
                                              *cfg.lstm_units2_range, step=16),
            "num_heads": trial.suggest_int("num_heads",
                                           *cfg.num_heads_range, step=2),
            "ff_dim": trial.suggest_int("ff_dim",
                                        *cfg.ff_dim_range, step=64),
            "num_transformers": trial.suggest_int("num_transformers",
                                                   *cfg.num_transformers_range),
            "dropout_rate": trial.suggest_float("dropout_rate",
                                                 *cfg.dropout_range),
            "dense_units": trial.suggest_int("dense_units",
                                              *cfg.dense_units_range, step=32),
            "learning_rate": trial.suggest_float("learning_rate",
                                                  *cfg.learning_rate_range, log=True),
            "optimizer": trial.suggest_categorical("optimizer",
                                                    ["adam", "nadam", "rmsprop"]),
            "batch_size": trial.suggest_categorical("batch_size",
                                                     list(cfg.batch_size_options)),
        }

        # 用最后一个日期构建特征
        ref_date = ref_dates[-1]
        X, y = build_batch_features(symbols, ref_date, engine, cfg)
        if len(X) < 50:
            return float("inf")

        # 3-fold TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=3)
        val_losses = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            model = create_stock_model(
                window=cfg.window,
                n_features=X.shape[2],
                **params,
            )

            model.fit(
                X_tr, y_tr,
                validation_data=(X_va, y_va),
                epochs=50,
                batch_size=params["batch_size"],
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss", patience=10,
                        restore_best_weights=True, min_delta=1e-4,
                    ),
                ],
                verbose=0,
            )
            val_loss = model.evaluate(X_va, y_va, verbose=0)[0]
            val_losses.append(val_loss)
            tf.keras.backend.clear_session()

        return np.mean(val_losses)

    return objective


def train_full(cfg: LSTMConfig | None = None):
    """完整训练：Optuna 搜索 + 最终训练。"""
    if cfg is None:
        cfg = get_config()

    settings = Settings()
    engine = DataEngine(settings)

    logger.info("=" * 60)
    logger.info(f"LSTM 模型完整训练 | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    logger.info("=" * 60)

    # 抽样股票
    symbols = _sample_stocks(engine, cfg.train_sample_stocks)
    logger.info(f"训练股票池: {len(symbols)} 只")

    # 获取最近交易日
    dates = _get_trade_dates(engine, cfg.window + cfg.predict_horizon + 30)
    logger.info(f"可用交易日: {len(dates)} 天")

    # Phase 1: Optuna 搜索
    logger.info("Phase 1: Optuna 超参数搜索")
    objective_func = _build_objective(engine, symbols, dates, cfg)

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )

    t0 = time.time()
    for trial_num in range(cfg.optuna_n_trials):
        elapsed = time.time() - t0
        remaining = cfg.optuna_timeout - elapsed
        if remaining <= 0:
            logger.info(f"Optuna 搜索超时 ({cfg.optuna_timeout}s)，已停止于 trial {trial_num}")
            break
        study.optimize(objective_func, n_trials=1, n_jobs=1,
                       timeout=remaining, show_progress_bar=False)
        logger.info(
            f"[Optuna] Trial {trial_num+1}/{cfg.optuna_n_trials} 完成 | "
            f"当前值={study.trials[-1].value:.4f} | "
            f"全局最佳={study.best_value:.4f} | "
            f"耗时={time.time()-t0:.0f}s"
        )

    logger.info(f"Optuna 搜索完成 | 最佳值={study.best_value:.4f}")
    logger.info(f"最佳参数: {study.best_params}")

    # Phase 2: 最终训练
    logger.info("Phase 2: 最终训练")
    ref_date = dates[-1]
    X, y = build_batch_features(symbols, ref_date, engine, cfg)
    logger.info(f"最终训练数据: X={X.shape}, y={y.shape}")

    best_params = study.best_params
    best_params["window"] = cfg.window
    best_params["n_features"] = X.shape[2]

    model, result = train_model(
        X, y, cfg,
        trial_name="最终训练",
        epochs=200,
        **best_params,
    )

    # 计算 IC
    from scipy.stats import spearmanr
    model_preds = model.predict(X[-int(len(X)*0.15):], verbose=0).flatten()
    actual = y[-int(len(y)*0.15):]
    ic = np.corrcoef(model_preds, actual)[0, 1]
    rank_ic = spearmanr(model_preds, actual)[0]
    logger.info(f"测试集 IC={ic:.4f} | Rank IC={rank_ic:.4f}")

    # 保存
    version_dir = save_model(model, cfg, best_params)
    logger.info(f"训练完成 | 模型版本: {version_dir}")


def train_incremental(cfg: LSTMConfig | None = None):
    """增量学习：加载最新模型，用近60日数据微调。"""
    if cfg is None:
        cfg = get_config()

    logger.info("LSTM 增量学习开始...")
    t0 = time.time()

    # 加载模型
    loaded = load_latest_model(cfg)
    if loaded is None:
        logger.warning("增量学习: 无已有模型，跳过")
        return
    model, params = loaded

    # 构建数据
    settings = Settings()
    engine = DataEngine(settings)
    symbols = _sample_stocks(engine, cfg.train_sample_stocks)
    dates = _get_trade_dates(engine, cfg.incremental_lookback + cfg.window + 10)
    if len(dates) < 30:
        logger.warning("增量学习: 数据不足，跳过")
        return

    ref_date = dates[-1]
    X, y = build_batch_features(symbols, ref_date, engine, cfg)
    if len(X) < 20:
        logger.warning("增量学习: 样本不足，跳过")
        return

    # 微调
    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg.incremental_lr),
        loss="mse",
        metrics=["mae"],
    )
    model.fit(X, y, epochs=cfg.incremental_epochs, batch_size=64, verbose=0)

    # 覆盖保存（用原版本目录）
    loaded_version_dir = cfg.model_dir_path / sorted(
        [d for d in cfg.model_dir_path.iterdir() if d.is_dir()]
    )[-1].name
    model.save(str(loaded_version_dir / "model.keras"))

    elapsed = time.time() - t0
    logger.info(f"增量学习完成 | 样本={len(X)} | 耗时={elapsed:.0f}s")


def train_weekly(cfg: LSTMConfig | None = None):
    """每周刷新：用最佳参数 + 近252日数据重新训练。"""
    if cfg is None:
        cfg = get_config()

    logger.info("LSTM 每周刷新开始...")

    # 加载最佳参数
    loaded = load_latest_model(cfg)
    params = {}
    if loaded is not None:
        _, params = loaded

    settings = Settings()
    engine = DataEngine(settings)
    symbols = _sample_stocks(engine, cfg.train_sample_stocks)
    dates = _get_trade_dates(engine, cfg.weekly_lookback + cfg.window + 30)
    ref_date = dates[-1] if dates else datetime.now().strftime("%Y-%m-%d")

    X, y = build_batch_features(symbols, ref_date, engine, cfg)
    if len(X) < 50:
        logger.error("每周刷新: 样本不足")
        return

    model_params = {
        "lstm_units": params.get("lstm_units", cfg.lstm_units),
        "lstm_units2": params.get("lstm_units2", cfg.lstm_units2),
        "num_heads": params.get("num_heads", cfg.num_heads),
        "ff_dim": params.get("ff_dim", cfg.ff_dim),
        "num_transformers": params.get("num_transformers", cfg.num_transformers),
        "dropout_rate": params.get("dropout_rate", cfg.dropout_rate),
        "dense_units": params.get("dense_units", cfg.dense_units),
        "learning_rate": params.get("learning_rate", cfg.learning_rate),
        "optimizer": params.get("optimizer", "adam"),
        "batch_size": params.get("batch_size", 64),
    }

    model, result = train_model(
        X, y, cfg,
        trial_name="每周刷新",
        epochs=cfg.weekly_epochs,
        **model_params,
    )

    version_dir = save_model(model, cfg, model_params)
    logger.info(f"每周刷新完成 | 版本: {version_dir} | val_loss={result['val_loss']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="LSTM-Transformer 模型训练")
    parser.add_argument("--full", action="store_true", help="完整 Optuna 搜索+训练")
    parser.add_argument("--incremental", action="store_true", help="增量学习")
    parser.add_argument("--weekly", action="store_true", help="每周刷新")
    parser.add_argument("--trials", type=int, help="Optuna 试验次数")
    parser.add_argument("--timeout", type=int, help="搜索超时（秒）")
    args = parser.parse_args()

    cfg = get_config()
    if args.trials:
        cfg.optuna_n_trials = args.trials
    if args.timeout:
        cfg.optuna_timeout = args.timeout

    if args.incremental:
        train_incremental(cfg)
    elif args.weekly:
        train_weekly(cfg)
    elif args.full:
        train_full(cfg)
    else:
        # 默认：增量学习
        train_incremental(cfg)


if __name__ == "__main__":
    main()
