"""model_selection_v2 - 训练入口：协调3个模型训练 + 特征重要性分析。"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from sequoia_x.core.config import Settings
from sequoia_x.data.engine import DataEngine
from sequoia_x.core.logger import get_logger
from sequoia_x.model_selection_v2.config import V2Config, get_config
from sequoia_x.model_selection_v2.labels import build_training_dataset
from sequoia_x.model_selection_v2.models.tree_cls import train_cls
from sequoia_x.model_selection_v2.models.tree_reg import train_reg
from sequoia_x.model_selection_v2.models.tree_vol import train_vol

logger = get_logger(__name__)


def train_all(
    engine: DataEngine | None = None,
    cfg: V2Config | None = None,
    search_optuna: bool = True,
) -> dict:
    """训练全部 3 个模型。

    Args:
        engine: DataEngine 实例，None 则自动创建。
        cfg: 配置。
        search_optuna: True=Optuna超参搜索。

    Returns:
        {"t1_model": ..., "t2_model": ..., "t3_model": ..., "feature_importance": {...}}
    """
    if cfg is None:
        cfg = get_config()
    if engine is None:
        engine = DataEngine(Settings())

    logger.info("=" * 60)
    logger.info("V2 模型训练开始")
    logger.info("=" * 60)

    # 构建数据集
    t0 = time.time()
    X, y1, y2, y3, dates = build_training_dataset(engine, cfg)
    if len(X) == 0:
        logger.error("无训练数据")
        return {}
    n_total = len(X)
    logger.info(f"训练数据: X={X.shape}, {len(dates)} 个采样日期")

    # ═══════════════════════════════════════════════════
    #  Optuna 加速：抽样 2 万搜索 → 全量重训
    # ═══════════════════════════════════════════════════
    OPTUNA_SAMPLE = 20000
    if search_optuna and n_total > OPTUNA_SAMPLE:
        logger.info(f"Phase 1: Optuna 搜索（抽样 {OPTUNA_SAMPLE}/{n_total}）")
        idx = np.random.RandomState(cfg.random_seed).choice(
            n_total, OPTUNA_SAMPLE, replace=False)
        Xs, y1s, y2s, y3s = X[idx], y1[idx], y2[idx], y3[idx]

        # T1: XGBoost
        m1 = train_cls(Xs, y1s, cfg, search_optuna=True)
        bp1 = {k: m1.get_params()[k] for k in [
            "max_depth", "learning_rate", "subsample", "colsample_bytree",
            "reg_alpha", "reg_lambda", "min_child_weight"] if k in m1.get_params()}
        logger.info(f"T1 best_params: {bp1}")

        # T2: LightGBM
        m2 = train_reg(Xs, y2s, cfg, search_optuna=True)
        bp2 = {k: m2.params.get(k, cfg.lgbm_params.get(k, [0,1])[0]) for k in [
            "num_leaves", "learning_rate", "subsample", "colsample_bytree",
            "reg_alpha", "reg_lambda", "min_child_samples"] if k in m2.params}
        logger.info(f"T2 best_params: {bp2}")

        # T3: CatBoost
        m3 = train_vol(Xs, y3s, cfg, search_optuna=True)
        bp3 = {k: m3.get_params()[k] for k in [
            "depth", "learning_rate", "l2_leaf_reg", "random_strength"]
               if k in m3.get_params()}
        logger.info(f"T3 best_params: {bp3}")

        logger.info(f"Phase 2: 全量训练（{n_total} 样本，用最佳参数）")
        model_t1 = train_cls(X, y1, cfg, search_optuna=False, best_params=bp1)
        model_t2 = train_reg(X, y2, cfg, search_optuna=False, best_params=bp2)
        model_t3 = train_vol(X, y3, cfg, search_optuna=False, best_params=bp3)
    else:
        # 默认参数快速训练（或数据量小直接 Optuna）
        logger.info("── 训练 T1: XGBoost 分类器 ──")
        t1 = time.time()
        model_t1 = train_cls(X, y1, cfg, search_optuna=search_optuna)
        logger.info(f"T1 耗时: {time.time()-t1:.0f}s")

        logger.info("── 训练 T2: LightGBM 回归器 ──")
        t2 = time.time()
        model_t2 = train_reg(X, y2, cfg, search_optuna=search_optuna)
        logger.info(f"T2 耗时: {time.time()-t2:.0f}s")

        logger.info("── 训练 T3: CatBoost 回归器 ──")
        t3 = time.time()
        model_t3 = train_vol(X, y3, cfg, search_optuna=search_optuna)
        logger.info(f"T3 耗时: {time.time()-t3:.0f}s")

    elapsed = time.time() - t0
    logger.info(f"全部训练完成: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # 保存模型到磁盘（供回测加载）
    cfg.model_dir_path.mkdir(parents=True, exist_ok=True)
    model_t1.save_model(str(cfg.model_dir_path / "t1_xgb.json"))
    model_t2.save_model(str(cfg.model_dir_path / "t2_lgbm.txt"))
    model_t3.save_model(str(cfg.model_dir_path / "t3_cat.cbm"))
    logger.info(f"模型已保存: {cfg.model_dir_path}")

    # 汇总特征重要性（以 T2 LightGBM 的特征重要性为主）
    feature_importance = {
        "t1_xgb": model_t1.feature_importances_.tolist() if hasattr(model_t1, 'feature_importances_') else [],
        "t2_lgbm": model_t2.feature_importance(importance_type="gain").tolist(),
    }

    # 保存结果
    result = {
        "t1_model": model_t1,
        "t2_model": model_t2,
        "t3_model": model_t3,
        "feature_importance": feature_importance,
        "n_samples": len(X),
        "n_dates": len(set(dates)),
        "elapsed_seconds": elapsed,
    }

    # 持久化特征重要性
    importance_path = cfg.model_dir_path / "feature_importance.json"
    with open(importance_path, "w") as f:
        json.dump(feature_importance, f, indent=2)
    logger.info(f"特征重要性已保存: {importance_path}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="V2 多任务树模型训练")
    parser.add_argument("--no-optuna", action="store_true", help="跳过 Optuna 超参搜索")
    parser.add_argument("--symbols", type=int, default=0, help="限制训练股票数（0=全量）")
    args = parser.parse_args()

    cfg = get_config()
    engine = DataEngine(Settings())

    if args.symbols > 0:
        # 快速测试模式
        pool = engine.get_base_stock_pool()[:args.symbols]
        X, y1, y2, y3, dates = build_training_dataset(engine, cfg, symbols=pool)
        logger.info(f"测试数据: X={X.shape}, {len(set(dates))} 日期")
        model_t1 = train_cls(X, y1, cfg, search_optuna=False)
        model_t2 = train_reg(X, y2, cfg, search_optuna=False)
        model_t3 = train_vol(X, y3, cfg, search_optuna=False)
    else:
        train_all(engine, cfg, search_optuna=not args.no_optuna)


if __name__ == "__main__":
    main()
