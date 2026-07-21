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


# ════════════════════════════════════════════════════════════
#  Optuna 剪枝回调：让 HyperbandPruner 在训练中途淘汰差 trial
# ════════════════════════════════════════════════════════════

class _OptunaPruneCallback(tf.keras.callbacks.Callback):
    """每 epoch 向 Optuna 上报 val_loss，支持 HyperbandPruner 中间剪枝。"""

    def __init__(self, trial: optuna.Trial, monitor: str = "val_loss"):
        super().__init__()
        self._trial = trial
        self._monitor = monitor

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs or {}
        val = logs.get(self._monitor)
        if val is not None:
            self._trial.report(val, step=epoch)
            if self._trial.should_prune():
                raise optuna.TrialPruned(
                    f"Trial pruned at epoch {epoch}: {self._monitor}={val:.4f}"
                )


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
    conn = sqlite3.connect(engine.db_path)
    rows = conn.execute(
        "SELECT DISTINCT date FROM stock_daily ORDER BY date DESC LIMIT ?",
        (lookback,)
    ).fetchall()
    conn.close()
    return [r[0] for r in reversed(rows)]


def _preload_stock_cache(engine: DataEngine, symbols: list[str]) -> dict:
    """预加载所有训练股票的OHLCV数据到内存，避免反复查询SQLite。"""
    cache = {}
    for symbol in symbols:
        df = engine.get_ohlcv(symbol)
        if df is not None and len(df) > 0:
            cache[symbol] = df
    logger.info(f"数据缓存就绪: {len(cache)}/{len(symbols)} 只股票")
    return cache


def _build_batch_from_cache(
    symbols: list[str],
    ref_date: str,
    cache: dict,
    engine: DataEngine,
    cfg: LSTMConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """从内存缓存构建特征矩阵，避免查询数据库。

    与 build_batch_features 功能相同，但使用预加载的缓存数据。
    """
    from sequoia_x.model_selection.features import build_stock_features

    # 为缓存版本临时替换 get_ohlcv
    _orig_get_ohlcv = engine.get_ohlcv

    def _cached_get_ohlcv(symbol):
        return cache.get(symbol)

    engine.get_ohlcv = _cached_get_ohlcv  # type: ignore
    try:
        return build_batch_features(symbols, ref_date, engine, cfg)
    finally:
        engine.get_ohlcv = _orig_get_ohlcv  # type: ignore


def _build_objective(engine: DataEngine, symbols: list[str],
                     ref_dates: list[str], cache: dict, cfg: LSTMConfig):
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
            "l2_reg": trial.suggest_float("l2_reg",
                                           *cfg.l2_reg_range, log=True),
            "huber_delta": trial.suggest_float("huber_delta",
                                                *cfg.huber_delta_range),
            "gradient_clip_norm": trial.suggest_float("gradient_clip_norm",
                                                       0.1, 5.0, log=True),
        }

        # 分离训练参数与模型架构参数
        batch_size = params.pop("batch_size")

        # 从内存缓存快速构建多日期特征
        # 取最近 36 天中的每第 3 个 → ~12 个时间点 × 400 只 ≈ 4800 样本
        recent = ref_dates[-36:] if len(ref_dates) > 36 else ref_dates
        sample_dates = [recent[i] for i in range(0, len(recent), 3)]
        if len(sample_dates) < 3:
            sample_dates = [ref_dates[-1]]

        X_list, y_list = [], []
        for d in sample_dates:
            X_batch, y_batch = _build_batch_from_cache(symbols, d, cache, engine, cfg)
            if len(X_batch) > 0:
                X_list.append(X_batch)
                y_list.append(y_batch)

        if not X_list:
            return float("inf")

        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
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
                epochs=100,
                batch_size=batch_size,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss", patience=10,
                        restore_best_weights=True, min_delta=1e-4,
                    ),
                    # HyperbandPruner 中间剪枝：每个 epoch 上报 val_loss，
                    # 让 Optuna 在 trial 明显无望时提前终止，节省搜索时间。
                    _OptunaPruneCallback(trial),
                ],
                verbose=0,
            )
            val_loss = model.evaluate(X_va, y_va, verbose=0)[0]
            val_losses.append(val_loss)
            tf.keras.backend.clear_session()

        return np.mean(val_losses)

    return objective


def train_full(cfg: LSTMConfig | None = None, skip_optuna: bool = False):
    """完整训练：Optuna 搜索 + 最终训练。

    Args:
        cfg: 配置对象。
        skip_optuna: True=跳过 Optuna，从 best_params.json 读取参数直接 Phase 2。
    """
    if cfg is None:
        cfg = get_config()

    settings = Settings()
    engine = DataEngine(settings)

    logger.info("=" * 60)
    logger.info(f"LSTM 模型完整训练 | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    if skip_optuna:
        logger.info("模式: 跳过 Optuna，使用已有最佳参数")
    logger.info("=" * 60)

    # 抽样股票 + 预加载数据缓存
    symbols = _sample_stocks(engine, cfg.train_sample_stocks)
    logger.info(f"训练股票池: {len(symbols)} 只")
    cache = _preload_stock_cache(engine, symbols)

    # 获取最近交易日
    dates = _get_trade_dates(engine, cfg.window + cfg.predict_horizon + 30)
    logger.info(f"可用交易日: {len(dates)} 天")

    # ── Phase 1: Optuna 搜索（可跳过）──
    if skip_optuna:
        # 从 best_params.json 读取已有最佳参数
        import json
        params_path = cfg.model_dir_path / "best_params.json"
        if not params_path.exists():
            logger.error(f"最佳参数文件不存在: {params_path}，请先运行完整 Optuna 搜索")
            return
        with open(params_path) as f:
            best_params = json.load(f)
        # 移除元数据字段（以 _ 开头）
        best_params = {k: v for k, v in best_params.items() if not k.startswith("_")}
        logger.info(f"从 {params_path} 加载最佳参数: {best_params}")
    else:
        logger.info("Phase 1: Optuna 超参数搜索")
        objective_func = _build_objective(engine, symbols, dates, cache, cfg)

        # ── study 持久化：防止进程崩溃丢失搜索结果 ──
        study_db = str(cfg.model_dir_path / "optuna_study.db")
        storage_url = f"sqlite:///{study_db}"
        logger.info(f"Optuna study 存储: {storage_url}")

        study = optuna.create_study(
            direction="minimize",
            # HyperbandPruner：比 MedianPruner 更高效——自动决定
            # 哪些 trial 值得分配更多资源（epochs），差的提前终止。
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=3,         # 最少 3 步后开始剪枝
                max_resource=100,       # 每 fold 最多 100 epochs
                reduction_factor=3,     # 每轮淘汰 2/3 的 trial
            ),
            storage=storage_url,
            study_name="lstm_stock_selection",
            load_if_exists=True,
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

        # 持久化最佳参数到 JSON（双保险，即使 study.db 损坏也能恢复）
        best_params = dict(study.best_params)
        import json
        params_path = cfg.model_dir_path / "best_params.json"
        with open(params_path, "w") as f:
            json.dump(best_params, f, indent=2, ensure_ascii=False)
        logger.info(f"最佳参数已保存: {params_path}")

    # ── Phase 2: 最终训练（使用全部12个采样点+缓存）──
    logger.info("Phase 2: 最终训练")
    # 与 Optuna 相同的多日期采样
    recent = dates[-36:] if len(dates) > 36 else dates
    sample_dates = [recent[i] for i in range(0, len(recent), 3)]
    if len(sample_dates) < 3:
        sample_dates = [dates[-1]]

    X_list, y_list = [], []
    for d in sample_dates:
        X_batch, y_batch = _build_batch_from_cache(symbols, d, cache, engine, cfg)
        if len(X_batch) > 0:
            X_list.append(X_batch)
            y_list.append(y_batch)
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    logger.info(f"最终训练数据: X={X.shape}, y={y.shape}")

    best_params["window"] = cfg.window
    best_params["n_features"] = X.shape[2]

    model, result = train_model(
        X, y, cfg,
        trial_name="最终训练",
        epochs=300,
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
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=cfg.incremental_lr,
            clipnorm=cfg.gradient_clip_norm,
        ),
        loss=tf.keras.losses.Huber(delta=cfg.huber_delta),
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
    parser.add_argument("--skip-optuna", action="store_true",
                        help="跳过 Optuna，使用 best_params.json 直接进入最终训练")
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
    elif args.full or args.skip_optuna:
        train_full(cfg, skip_optuna=args.skip_optuna)
    else:
        # 默认：增量学习
        train_incremental(cfg)


if __name__ == "__main__":
    main()
