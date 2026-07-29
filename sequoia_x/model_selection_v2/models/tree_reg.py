"""model_selection_v2 - T2: LightGBM 回归器（20日超额收益率）。"""
from __future__ import annotations
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sequoia_x.core.logger import get_logger
from sequoia_x.model_selection_v2.config import V2Config, get_config

logger = get_logger(__name__)


def _objective(trial, X_2d: np.ndarray, y: np.ndarray, cfg: V2Config) -> float:
    """Optuna 目标：最小化验证集 RMSE（仅负值不剪枝，直接返回验证 loss）。"""
    params = {
        "num_leaves": trial.suggest_int("num_leaves", *cfg.lgbm_params["num_leaves"]),
        "learning_rate": trial.suggest_float("learning_rate", *cfg.lgbm_params["learning_rate"], log=True),
        "subsample": trial.suggest_float("subsample", *cfg.lgbm_params["subsample"]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", *cfg.lgbm_params["colsample_bytree"]),
        "reg_alpha": trial.suggest_float("reg_alpha", *cfg.lgbm_params["reg_alpha"], log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", *cfg.lgbm_params["reg_lambda"], log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", *cfg.lgbm_params["min_child_samples"]),
        "verbosity": -1,
        "n_jobs": cfg.n_jobs,
        "random_state": cfg.random_seed,
    }
    tscv = TimeSeriesSplit(n_splits=3)
    losses = []
    for train_idx, val_idx in tscv.split(X_2d):
        train_data = lgb.Dataset(X_2d[train_idx], label=y[train_idx])
        val_data = lgb.Dataset(X_2d[val_idx], label=y[val_idx], reference=train_data)
        model = lgb.train(
            params, train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        pred = model.predict(X_2d[val_idx])
        loss = np.sqrt(np.mean((pred - y[val_idx]) ** 2))
        losses.append(loss)
    return np.mean(losses)


def train_reg(
    X: np.ndarray, y: np.ndarray,
    cfg: V2Config | None = None,
    search_optuna: bool = True,
    best_params: dict | None = None,
) -> lgb.Booster:
    """训练 LightGBM 回归器。"""
    if cfg is None:
        cfg = get_config()

    n_samples = X.shape[0]
    X_2d = X.reshape(n_samples, -1)

    if best_params is not None:
        pass  # 使用传入参数,跳过搜索和默认值
    elif search_optuna:
        import optuna
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
            storage=f"sqlite:///{cfg.optuna_dir_path}/t2_lgbm.db",
            study_name="t2_lgbm_reg",
            load_if_exists=True,
        )
        # 断点续跑：已有最优 trial 则跳过搜索
        if study.best_trial is not None:
            best_params = study.best_params
            logger.info(f"T2 Optuna: 跳过（已有 {len(study.trials)} trials, best={study.best_value:.4f}）")
        else:
            study.optimize(
                lambda trial: _objective(trial, X_2d, y, cfg),
                n_trials=cfg.optuna_n_trials,
                timeout=cfg.optuna_timeout,
                n_jobs=1,
                show_progress_bar=True,
            )
            best_params = study.best_params
            logger.info(f"T2 Optuna best: RMSE={study.best_value:.4f}, params={best_params}")
    else:
        best_params = {
            "num_leaves": 31, "learning_rate": 0.1, "subsample": 0.8,
            "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0,
            "min_child_samples": 20,
        }

    tscv = TimeSeriesSplit(n_splits=3)
    splits = list(tscv.split(X_2d))
    train_idx, val_idx = splits[-1]

    params = {
        **best_params,
        "objective": "huber",
        "alpha": 0.1,  # Huber delta
        "metric": "rmse",
        "verbosity": -1,
        "n_jobs": cfg.n_jobs,
        "random_state": cfg.random_seed,
    }

    train_data = lgb.Dataset(X_2d[train_idx], label=y[train_idx])
    val_data = lgb.Dataset(X_2d[val_idx], label=y[val_idx], reference=train_data)

    model = lgb.train(
        params, train_data,
        valid_sets=[val_data],
        num_boost_round=2000,
        callbacks=[lgb.early_stopping(cfg.early_stop_rounds), lgb.log_evaluation(0)],
    )

    logger.info(f"T2 训练完成: best_iteration={model.best_iteration}")
    return model


def predict_reg(model: lgb.Booster, X: np.ndarray) -> np.ndarray:
    """预测 20 日超额收益率。

    Returns:
        (n_samples,) 预测超额收益率
    """
    n_samples = X.shape[0]
    X_2d = X.reshape(n_samples, -1)
    return model.predict(X_2d)
