"""数据集缓存重建（2026-08-02：加入 2026-07 采样日，供 8 月 V2 重训）

背景：缓存采样日最新 2026-06-22，8 月重训需要 2026-07（7/30 最后交易日）样本。
缓存 key 不含采样日数量 → 不会自动过期 → 必须 force_rebuild 全量重建。
重建覆盖原目录（13132147f8e8=88维 / 62cf234c5440=80维），不影响 prediction_cache.json。

用法：
  python scripts/rebuild_dataset_cache.py              # 并行重建 80+88 维
  python scripts/rebuild_dataset_cache.py --only-88     # 仅 88 维（T2/T1/T3）
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from sequoia_x.core.config import Settings
from sequoia_x.data.engine import DataEngine
from sequoia_x.model_selection_v2.config import get_config
from sequoia_x.model_selection_v2.labels import build_training_dataset


def rebuild(include_market_state: bool, n_workers: int) -> None:
    """重建单个维度缓存。"""
    cfg = get_config()
    engine = DataEngine(Settings())
    dim = "88维" if include_market_state else "80维"
    print(f"开始重建 {dim} 缓存 (workers={n_workers})...")
    X, y1, y2, y3, dates = build_training_dataset(
        engine, cfg, n_workers=n_workers,
        force_rebuild=True, include_market_state=include_market_state,
    )
    n_dates = len(set(dates))
    print(f"✅ {dim} 缓存完成: X={X.shape}, 采样日={n_dates}")


def main() -> None:
    parser = argparse.ArgumentParser(description="数据集缓存重建")
    parser.add_argument("--only-88", action="store_true", help="仅重建 88 维")
    parser.add_argument("--only-80", action="store_true", help="仅重建 80 维")
    parser.add_argument("--workers", type=int, default=12, help="每进程 worker 数")
    args = parser.parse_args()

    if args.only_88:
        rebuild(True, args.workers)
    elif args.only_80:
        rebuild(False, args.workers)
    else:
        # 并行重建两个维度（各 workers 个进程，总 2×workers ≤ 36 核）
        import multiprocessing
        p1 = multiprocessing.Process(target=rebuild, args=(True, args.workers))
        p2 = multiprocessing.Process(target=rebuild, args=(False, args.workers))
        p1.start(); p2.start()
        p1.join(); p2.join()
        print("全部缓存重建完成")


if __name__ == "__main__":
    main()
