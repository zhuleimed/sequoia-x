"""model_selection_v2 - T1: XGBoost 二分类器（5日涨跌方向）。"""
from __future__ import annotations
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sequoia_x.core.logger import get_logger
from sequoia_x.model_selection_v2.config import V2Config, get_config

logger = get_logger(__name__)


def _objective(trial, X: np.ndarray, y: np.ndarray, cfg: V2Config) -> float:
    """Optuna 目标函数：最小化验证集 AUC 的负数（即最大化 AUC）。"""
    params = {
        "max_depth": trial.suggest_int("max_depth", *cfg.xgb_params["max_depth"]),
        "learning_rate": trial.suggest_float("learning_rate", *cfg.xgb_params["learning_rate"], log=True),
        "subsample": trial.suggest_float("subsample", *cfg.xgb_params["subsample"]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", *cfg.xgb_params["colsample_bytree"]),
        "reg_alpha": trial.suggest_float("reg_alpha", *cfg.xgb_params["reg_alpha"], log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", *cfg.xgb_params["reg_lambda"], log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", *cfg.xgb_params["min_child_weight"]),
        "n_estimators": 1000,
        "verbosity": 0,
        "n_jobs": cfg.n_jobs,
        "random_state": cfg.random_seed,
        "tree_method": "hist",
    }

    # 扁平化 X 为 2D
    n_samples = X.shape[0]
    X_2d = X.reshape(n_samples, -1)

    tscv = TimeSeriesSplit(n_splits=3)
    aucs = []
    for train_idx, val_idx in tscv.split(X_2d):
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_2d[train_idx], y[train_idx],
            eval_set=[(X_2d[val_idx], y[val_idx])],
            verbose=False,
        )
        from sklearn.metrics import roc_auc_score
        proba = model.predict_proba(X_2d[val_idx])[:, 1]
        try:
            auc = roc_auc_score(y[val_idx], proba)
            aucs.append(auc)
        except ValueError:
            aucs.append(0.5)

    return -np.mean(aucs)  # Optuna 最小化


def train_cls(
    X: np.ndarray, y: np.ndarray,
    cfg: V2Config | None = None,
    search_optuna: bool = True,
    best_params: dict | None = None,
) -> xgb.XGBClassifier:
    """训练 XGBoost 分类器。

    Args:
        X: (n_samples, window, n_features)
        y: (n_samples,) 二分类标签
        cfg: 配置
        search_optuna: True=Optuna搜索, False=默认参数
        best_params: 指定参数（跳过搜索和默认值）

    Returns:
        训练好的 XGBoost 模型
    """
    if cfg is None:
        cfg = get_config()

    n_samples = X.shape[0]
    X_2d = X.reshape(n_samples, -1)

    if best_params is not None:
        pass  # 使用传入参数
    elif search_optuna:
        import optuna
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
            storage=f"sqlite:///{cfg.optuna_dir_path}/t1_xgb.db",
            study_name="t1_xgb_cls",
            load_if_exists=True,
        )
        # 断点续跑：已有最优 trial 则跳过搜索
        if study.best_trial is not None:
            best_params = study.best_params
            logger.info(f"T1 Optuna: 跳过（已有 {len(study.trials)} trials, best={-study.best_value:.4f}）")
        else:
            study.optimize(
                lambda trial: _objective(trial, X_2d, y, cfg),
                n_trials=cfg.optuna_n_trials,
                timeout=cfg.optuna_timeout,
                n_jobs=1,
                show_progress_bar=True,
            )
            best_params = study.best_params
            logger.info(f"T1 Optuna best: AUC={-study.best_value:.4f}, params={best_params}")
    else:
        best_params = {
            "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8,
            "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0,
            "min_child_weight": 5,
        }

    # 最终训练（TimeSeriesSplit 最后 fold 做验证集）
    tscv = TimeSeriesSplit(n_splits=3)
    splits = list(tscv.split(X_2d))
    train_idx, val_idx = splits[-1]

    model = xgb.XGBClassifier(
        **best_params,
        n_estimators=1000,
        early_stopping_rounds=cfg.early_stop_rounds,
        verbosity=0,
        n_jobs=cfg.n_jobs,
        random_state=cfg.random_seed,
        tree_method="hist",
    )
    # 自动平衡正负样本权重
    neg_count = (y[train_idx] == 0).sum()
    pos_count = (y[train_idx] == 1).sum()
    scale_pos_weight = neg_count / max(pos_count, 1)
    model.set_params(scale_pos_weight=scale_pos_weight)

    model.fit(
        X_2d[train_idx], y[train_idx],
        eval_set=[(X_2d[val_idx], y[val_idx])],
        verbose=False,
    )

    # 特征重要性
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[-20:][::-1]
    logger.info(f"T1 训练完成: n_estimators={model.n_estimators}")
    logger.info(f"T1 Top-10 特征 idx: {top_indices[:10].tolist()}")
    logger.info(f"T1 Top-10 重要性: {importances[top_indices[:10]].round(4).tolist()}")

    return model


def predict_cls(model: xgb.XGBClassifier, X: np.ndarray) -> np.ndarray:
    """预测涨概率。

    Args:
        model: 训练好的 XGBoost 模型。
        X: (n_samples, window, n_features)

    Returns:
        (n_samples,) 涨概率 [0, 1]
    """
    n_samples = X.shape[0]
    X_2d = X.reshape(n_samples, -1)
    return model.predict_proba(X_2d)[:, 1]
