"""LSTM-Transformer 每日预测入口。

对全量基础池股票（~2000只）预测未来 N 日收益率，按降序输出。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool
from typing import Optional

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sequoia_x.core.config import Settings
from sequoia_x.data.engine import DataEngine
from sequoia_x.core.logger import get_logger
from sequoia_x.model_selection.config import LSTMConfig, get_config
from sequoia_x.model_selection.features import build_prediction_features
from sequoia_x.model_selection.model import predict_returns

logger = get_logger(__name__)


def _predict_single(args: tuple) -> tuple[str, float | None]:
    """单只股票预测（供多进程使用）。

    Worker 内独立加载模型，避免跨进程 pickle Keras 模型。
    """
    symbol, engine, model_path, cfg, n_features = args
    try:
        # Load model inside worker (avoids pickle issue)
        import tensorflow as tf
        from sequoia_x.model_selection.model import TransformerBlock
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={"TransformerBlock": TransformerBlock},
        )
        X = build_prediction_features(symbol, engine, cfg)
        if X is None:
            return (symbol, None)
        # 确保特征维度匹配
        if X.shape[2] != n_features:
            return (symbol, None)
        pred = predict_returns(model, X)[0]
        return (symbol, float(pred))
    except Exception:
        return (symbol, None)


def predict_all(
    engine: DataEngine,
    model=None,
    cfg: LSTMConfig | None = None,
    n_workers: int = 28,
) -> list[tuple[str, float]]:
    """对全量基础池股票预测收益率。

    Args:
        engine: DataEngine 实例。
        model: 保留参数，不再使用（自动从磁盘加载）。
        cfg: 配置对象。
        n_workers: 并行进程数。

    Returns:
        [(symbol, pred_return), ...] 按收益率降序排列。
    """
    if cfg is None:
        cfg = get_config()

    # 从磁盘获取最新模型路径（避免跨进程 pickle TF 模型）
    model_dir = cfg.model_dir_path
    versions = sorted([d for d in model_dir.iterdir() if d.is_dir()], reverse=True)
    if not versions:
        logger.error("无可用模型，请先训练")
        return []
    model_save_path = str(versions[0] / "model.keras")

    # 从 params.json 读取 n_features（轻量，避免加载完整 TF 模型）
    import json
    params_path = versions[0] / "params.json"
    n_features = cfg.window  # fallback
    if params_path.exists():
        with open(params_path) as f:
            saved_params = json.load(f)
            n_features = saved_params.get("n_features", 62)
    logger.debug(f"n_features={n_features} (from params.json)")

    symbols = engine.get_base_stock_pool()
    logger.info(f"预测池: {len(symbols)} 只股票")

    # 多进程并行预测（每个 worker 独立加载模型）
    t0 = time.time()
    tasks = [(s, engine, model_save_path, cfg, n_features) for s in symbols]

    with Pool(processes=n_workers) as pool:
        results = pool.map(_predict_single, tasks)

    # 过滤失败和低置信度预测
    valid = [(s, r) for s, r in results if r is not None and not np.isnan(r)]
    valid.sort(key=lambda x: x[1], reverse=True)

    elapsed = time.time() - t0
    logger.info(
        f"预测完成 | 有效={len(valid)}/{len(symbols)} | "
        f"耗时={elapsed:.0f}s | "
        f"top5={[s for s,_ in valid[:5]]}"
    )
    return valid


def main():
    parser = argparse.ArgumentParser(description="LSTM-Transformer 每日预测")
    parser.add_argument("--top", type=int, default=10, help="输出前 N 只")
    parser.add_argument("--output", type=str, help="输出 CSV 文件路径")
    args = parser.parse_args()

    cfg = get_config()
    settings = Settings()
    engine = DataEngine(settings)

    # 预测（自动从磁盘加载最新模型）
    predictions = predict_all(engine, cfg=cfg)

    # 输出
    print(f"\n{'='*60}")
    print(f"LSTM-Transformer 收益率预测 Top {args.top}")
    print(f"{'='*60}")
    for i, (symbol, pred) in enumerate(predictions[:args.top], 1):
        # 获取股票名称
        try:
            import sqlite3
            conn = sqlite3.connect(settings.db_path)
            name = conn.execute(
                "SELECT name FROM stock_list WHERE symbol=?", (symbol,)
            ).fetchone()
            name_str = name[0] if name else ""
            conn.close()
        except Exception:
            name_str = ""
        print(f"  {i:2d}. {symbol} {name_str:8s} 预测5日收益: {pred:+.2%}")

    # 可选保存
    if args.output:
        import pandas as pd
        df = pd.DataFrame(predictions, columns=["symbol", "pred_return"])
        df.to_csv(args.output, index=False)
        logger.info(f"预测结果已保存: {args.output}")


if __name__ == "__main__":
    main()
