#!/usr/bin/env python
"""V2 多任务树模型回测 — 三模型综合信号。

用法:
    python backtest_v2.py               # 完整流程: 训练 → 预测缓存 → 回测
    python backtest_v2.py --replay-only  # 复用缓存, 仅回放

相比V1的变化:
  - 3个模型(T1过滤→T2排序→T3调仓) 替代 单LSTM
  - 卖出规则复用 simulation/rules.py + V2特有T1/T2因子
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from sequoia_x.core.config import Settings
from sequoia_x.data.engine import DataEngine
from sequoia_x.core.logger import get_logger
from sequoia_x.model_selection_v2.config import get_config as get_v2_config
from sequoia_x.model_selection_v2.backtest.engine import V2BacktestEngine
from sequoia_x.model_selection_v2.backtest.reporter import save_results, print_comparison_table

logger = get_logger(__name__)

CACHE_PATH = Path("output/backtest_v2/predictions_cache.json")
OUTPUT_DIR = Path("output/backtest_v2")
MODEL_DIR = Path("data/models/v2_selection")

PERIODS: dict[str, tuple[str, str, str]] = {
    "2024": ("2024-01-01", "2024-12-31", "震荡市, HS300 +1.71%"),
    "2025": ("2025-01-01", "2025-12-31", "大牛市, HS300 +34.94%"),
    "2026": ("2026-01-01", "2026-07-20", "快牛, HS300 +24.25%"),
    "full": ("2024-01-01", "2026-07-20", "全周期"),
}


# ════════════════════════════════════════════════════════════
#  模型保存/加载
# ════════════════════════════════════════════════════════════

def save_models(model_t1, model_t2, model_t3) -> None:
    """保存3个模型到磁盘。"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_t1.save_model(str(MODEL_DIR / "t1_xgb.json"))
    model_t2.save_model(str(MODEL_DIR / "t2_lgbm.txt"))
    model_t3.save_model(str(MODEL_DIR / "t3_cat.cbm"))
    logger.info(f"模型已保存: {MODEL_DIR}")


def load_models() -> tuple:
    """从磁盘加载3个模型。"""
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostRegressor

    t1_path = MODEL_DIR / "t1_xgb.json"
    t2_path = MODEL_DIR / "t2_lgbm.txt"
    t3_path = MODEL_DIR / "t3_cat.cbm"

    if not t1_path.exists() or not t2_path.exists() or not t3_path.exists():
        raise FileNotFoundError(f"模型文件缺失: {MODEL_DIR}")

    model_t1 = xgb.XGBClassifier()
    model_t1.load_model(str(t1_path))

    model_t2 = lgb.Booster(model_file=str(t2_path))

    model_t3 = CatBoostRegressor()
    model_t3.load_model(str(t3_path))

    logger.info(f"模型已加载: {MODEL_DIR}")
    return model_t1, model_t2, model_t3


# ════════════════════════════════════════════════════════════
#  Phase 1: 预测缓存
# ════════════════════════════════════════════════════════════

def run_phase1(model_t1, model_t2, model_t3) -> None:
    """计算并保存每日预测（全周期）。"""
    logger.info("=" * 60)
    logger.info("Phase 1: 计算每日预测缓存（2024-2026）")
    logger.info("=" * 60)

    from sequoia_x.model_selection_v2.models.tree_cls import predict_cls
    from sequoia_x.model_selection_v2.models.tree_reg import predict_reg
    from sequoia_x.model_selection_v2.models.tree_vol import predict_vol
    from sequoia_x.model_selection_v2.features import build_prediction_features

    settings = Settings()
    engine = DataEngine(settings)
    cfg = get_v2_config()

    # 获取全周期交易日
    import sqlite3
    conn = sqlite3.connect(engine.db_path)
    dates = conn.execute(
        "SELECT DISTINCT date FROM stock_daily WHERE date >= '2024-01-01' "
        "AND date <= '2026-07-20' ORDER BY date"
    ).fetchall()
    conn.close()
    dates = [d[0] for d in dates]

    base_pool = engine.get_base_stock_pool()
    logger.info(f"股票池: {len(base_pool)} 只, 交易日: {len(dates)} 天")

    warmup = cfg.window
    cache: dict[str, list] = {}
    t0 = time.time()

    for idx, today in enumerate(dates):
        if idx < warmup:
            continue
        prev_date = dates[idx - 1]

        # 批量预测
        xs, symbols = [], []
        for symbol in base_pool:
            try:
                X = build_prediction_features(symbol, engine, cfg)
                if X is not None:
                    xs.append(X)
                    symbols.append(symbol)
            except Exception:
                continue

        if not xs:
            continue

        X_batch = __import__('numpy').vstack(xs)
        prob_up = predict_cls(model_t1, X_batch)
        excess_ret = predict_reg(model_t2, X_batch)
        volatility = predict_vol(model_t3, X_batch)

        day_preds = []
        for i, sym in enumerate(symbols):
            if __import__('numpy').isfinite(prob_up[i]):
                day_preds.append([sym, [
                    round(float(prob_up[i]), 6),
                    round(float(excess_ret[i]), 6),
                    round(float(volatility[i]), 6),
                ]])

        if day_preds:
            cache[prev_date] = day_preds

        if (idx - warmup + 1) % 60 == 0 or idx == warmup:
            elapsed = time.time() - t0
            remaining = len(dates) - idx - 1
            logger.info(
                f"  预测 {idx - warmup + 1}/{len(dates) - warmup} 天, "
                f"缓存 {len(cache)} 天, {elapsed:.0f}s, "
                f"预估剩余 {elapsed/(idx-warmup+1)*remaining:.0f}s"
            )

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)
    elapsed = time.time() - t0
    logger.info(f"Phase 1 完成: {len(cache)} 天缓存, {elapsed:.0f}s ({elapsed/3600:.1f}h)")


# ════════════════════════════════════════════════════════════
#  Phase 2: 回测回放
# ════════════════════════════════════════════════════════════

def run_phase2(model_t1, model_t2, model_t3, cache: dict) -> list[dict]:
    """用缓存预测回放4个期间。"""
    logger.info("=" * 60)
    logger.info("Phase 2: 回测回放（复用预测缓存）")
    logger.info("=" * 60)

    settings = Settings()
    engine = DataEngine(settings)
    all_results: list[dict] = []

    for period_name, (start, end, period_desc) in PERIODS.items():
        t0 = time.time()
        bt = V2BacktestEngine(engine, model_t1, model_t2, model_t3)
        metrics = bt.run(start, end, predictions_cache=cache)

        if not metrics:
            logger.warning(f"  {period_name}: 无数据, 跳过")
            continue

        all_daily = metrics.pop("daily_records", [])
        all_trades = metrics.pop("trade_records", [])
        elapsed = time.time() - t0

        metrics["period"] = period_name
        metrics["description"] = period_desc

        logger.info(
            f"  {period_name}: 收益={metrics.get('total_return',0):+.2%} "
            f"夏普={metrics.get('sharpe',0):.2f} "
            f"回撤={metrics.get('max_drawdown',0):.2%} "
            f"胜率={metrics.get('win_rate',0):.1%} "
            f"耗时={elapsed:.0f}s"
        )

        # 保存单期间结果
        save_results(
            [metrics], str(OUTPUT_DIR / period_name),
            daily_records=all_daily, trade_records=all_trades,
        )
        all_results.append(metrics)

    return all_results


# ════════════════════════════════════════════════════════════
#  主入口
# ════════════════════════════════════════════════════════════

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="V2 多任务树模型回测")
    parser.add_argument("--replay-only", action="store_true", help="仅回放,跳过预测")
    args = parser.parse_args()

    # 加载模型
    try:
        model_t1, model_t2, model_t3 = load_models()
    except FileNotFoundError:
        logger.error("模型文件缺失, 请先运行 train.py 训练模型")
        return

    if not args.replay_only:
        run_phase1(model_t1, model_t2, model_t3)

    # 加载缓存
    if not CACHE_PATH.exists():
        logger.error("预测缓存不存在, 请先跑 Phase 1")
        return

    with open(CACHE_PATH) as f:
        cache = json.load(f)
    logger.info(f"预测缓存已加载: {len(cache)} 天")

    all_results = run_phase2(model_t1, model_t2, model_t3, cache)
    print_comparison_table(all_results)

    # 保存汇总
    summary_path = OUTPUT_DIR / "comparison_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"对比汇总已保存: {summary_path}")


if __name__ == "__main__":
    main()
