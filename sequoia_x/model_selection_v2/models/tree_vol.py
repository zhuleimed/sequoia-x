"""model_selection_v2 - T3: CatBoost 回归器（20日波动率）。"""
from __future__ import annotations
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import TimeSeriesSplit
from sequoia_x.core.logger import get_logger
from sequoia_x.model_selection_v2.config import V2Config, get_config

logger = get_logger(__name__)


def _objective(trial, X_2d: np.ndarray, y: np.ndarray, cfg: V2Config) -> float:
    """Optuna 目标：最小化验证集 RMSE。"""
    params = {
        "depth": trial.suggest_int("depth", *cfg.cat_params["depth"]),
        "learning_rate": trial.suggest_float("learning_rate", *cfg.cat_params["learning_rate"], log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", *cfg.cat_params["l2_leaf_reg"], log=True),
        "random_strength": trial.suggest_float("random_strength", *cfg.cat_params["random_strength"], log=True),
        "iterations": 500,
        "verbose": False,
        "thread_count": cfg.n_jobs,
        "random_seed": cfg.random_seed,
    }
    tscv = TimeSeriesSplit(n_splits=3)
    losses = []
    for train_idx, val_idx in tscv.split(X_2d):
        model = CatBoostRegressor(**params)
        model.fit(
            X_2d[train_idx], y[train_idx],
            eval_set=(X_2d[val_idx], y[val_idx]),
            early_stopping_rounds=50,
            verbose=False,
        )
        pred = model.predict(X_2d[val_idx])
        loss = np.sqrt(np.mean((pred - y[val_idx]) ** 2))
        losses.append(loss)
    return np.mean(losses)


def train_vol(
    X: np.ndarray, y: np.ndarray,
    cfg: V2Config | None = None,
    search_optuna: bool = True,
) -> CatBoostRegressor:
    """训练 CatBoost 波动率回归器。"""
    if cfg is None:
        cfg = get_config()

    n_samples = X.shape[0]
    X_2d = X.reshape(n_samples, -1)

    if search_optuna:
        import optuna
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
            storage=f"sqlite:///{cfg.optuna_dir_path}/t3_cat.db",
            study_name="t3_cat_vol",
            load_if_exists=True,
        )
        study.optimize(
            lambda trial: _objective(trial, X_2d, y, cfg),
            n_trials=cfg.optuna_n_trials,
            timeout=cfg.optuna_timeout,
            n_jobs=1,
            show_progress_bar=True,
        )
        best_params = study.best_params
        logger.info(f"T3 Optuna best: RMSE={study.best_value:.4f}, params={best_params}")
    else:
        best_params = {
            "depth": 6, "learning_rate": 0.1,
            "l2_leaf_reg": 3.0, "random_strength": 1.0,
        }

    tscv = TimeSeriesSplit(n_splits=3)
    splits = list(tscv.split(X_2d))
    train_idx, val_idx = splits[-1]

    model = CatBoostRegressor(
        **best_params,
        iterations=1000,
        verbose=False,
        thread_count=cfg.n_jobs,
        random_seed=cfg.random_seed,
        early_stopping_rounds=cfg.early_stop_rounds,
    )
    model.fit(
        X_2d[train_idx], y[train_idx],
        eval_set=(X_2d[val_idx], y[val_idx]),
        verbose=False,
    )

    logger.info(f"T3 训练完成: tree_count={model.tree_count_}")
    return model


def predict_vol(model: CatBoostRegressor, X: np.ndarray) -> np.ndarray:
    """预测 20 日年化波动率。"""
    n_samples = X.shape[0]
    X_2d = X.reshape(n_samples, -1)
    return model.predict(X_2d)
