"""方案B: T2(LightGBM)月度对比，作为LSTM的基准线。"""
import json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.stats import spearmanr
from sequoia_x.core.config import Settings
from sequoia_x.core.logger import get_logger
from sequoia_x.data.engine import DataEngine
from sequoia_x.model_selection_v2.config import get_config
from sequoia_x.model_selection_v2.labels import build_training_dataset
from sequoia_x.model_selection_v2.models.tree_reg import train_reg, predict_reg

logger = get_logger("t2_monthly")
OUTPUT = Path("data/models/v2_selection/test_t2_monthly.json")


def main():
    logger.info("=" * 50)
    logger.info("方案B: T2(LightGBM)月度Walk-Forward对比")
    cfg = get_config()
    engine = DataEngine(Settings())
    logger.info("加载数据集...")
    X, y1, y2, y3, dates = build_training_dataset(engine, cfg, n_workers=8)
    months = sorted(set(d[:7] for d in dates))
    logger.info(f"数据: {X.shape}, 月份: {months[0]}~{months[-1]}")

    results = []
    t0 = time.time()
    for mi, test_month in enumerate(months[12:], 1):  # 跳过前12个月
        month_idx = months.index(test_month)
        train_start = months[month_idx - 12]
        dates_arr = np.array(dates)
        # 训练：前12个月
        train_end_date = [d for d in sorted(set(dates)) if d[:7] == months[month_idx - 1]][-1]
        train_mask = np.array([train_start + "-01" <= d <= train_end_date for d in dates_arr])
        test_mask = np.array([d[:7] == test_month for d in dates_arr])
        if train_mask.sum() < 100 or test_mask.sum() < 50:
            continue

        X_tr, y_tr = X[train_mask], y2[train_mask]
        X_te, y_te = X[test_mask], y2[test_mask]

        t1 = time.time()
        # T2 LightGBM: skip Optuna, use existing best params
        model = train_reg(X_tr, y_tr, cfg, search_optuna=False)
        pred = predict_reg(model, X_te)
        elapsed = time.time() - t1
        ic, _ = spearmanr(pred, y_te)
        r = {"month": test_month, "rank_ic": float(ic) if not np.isnan(ic) else 0.0,
             "y_mean": float(y_te.mean()), "n_train": len(X_tr), "n_test": len(X_te),
             "elapsed": int(elapsed)}
        results.append(r)
        with open(OUTPUT, "w") as f: json.dump(results, f, indent=2)
        status = "✅" if r["rank_ic"] > 0.03 else ("⚠️" if r["rank_ic"] > 0 else "❌")
        logger.info(f"[{mi:2d}] {test_month} {status} RankIC={r['rank_ic']:+.4f} | {elapsed:.0f}s")

    ics = [r["rank_ic"] for r in results]
    pre = [r for r in results if r["month"] < "2026"]
    post = [r for r in results if r["month"] >= "2026"]
    logger.info(f"\nT2月度汇总: 全部IC mean={np.mean(ics):+.4f}")
    if pre: logger.info(f"  2025: mean={np.mean([r['rank_ic'] for r in pre]):+.4f}, >0={sum(1 for r in pre if r['rank_ic']>0)}/{len(pre)}")
    if post: logger.info(f"  2026: mean={np.mean([r['rank_ic'] for r in post]):+.4f}, >0={sum(1 for r in post if r['rank_ic']>0)}/{len(post)}")
    logger.info(f"结果: {OUTPUT}")


if __name__ == "__main__":
    main()
