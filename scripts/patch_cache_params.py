"""补写缓存 metadata params（2026-08-11 任务 #9 的独立执行版）。

背景: 408031（8-11 全历史 121 维重建）fork 于增量实现之前, 保存缓存时 metadata
无 params → 8/31 月末链的增量复用扫描会跳过它（旧格式不可复用）→ 8/31 退化为全量。
本脚本为已完成的最新 121 维缓存补写 params（新格式），使 8/31 月末链可直接增量复用。

用法: py312 python scripts/patch_cache_params.py [缓存目录hash，默认取最新的]
      重复执行幂等（已有 params 则跳过）。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data/cache/v2_dataset"
POOL_PATH = ROOT / "output/backtest_v2/.stock_pool.json"


def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else None
    symbols = json.loads(POOL_PATH.read_text())

    if target:
        dirs = [CACHE_DIR / target]
    else:
        # 取最新的 121 维缓存（extra_features=True, market_state=True 的 params 或 X_shape[2]==121）
        dirs = sorted(CACHE_DIR.glob("*"), key=lambda d: d.stat().st_mtime, reverse=True)

    patched = 0
    for d in dirs:
        if not (d / "metadata.json").exists():
            continue
        meta = json.loads((d / "metadata.json").read_text())
        if "params" in meta:
            print(f"跳过（已有 params）: {d.name}")
            continue
        if meta.get("X_shape", [0, 0, 0])[2] != 121:
            print(f"跳过（非 121 维）: {d.name} X_shape={meta.get('X_shape')}")
            continue
        dates = json.loads((d / "dates.json").read_text())
        params = {
            "n_stocks": len(symbols),
            "sample_start": "2020-01-01",
            "sample_end": str(dates[-1]),
            "window": 120,
            "feature_version": 3,
            "market_state": True,
            "extra_features": True,
        }
        meta["params"] = params
        (d / "metadata.json").write_text(json.dumps(meta))
        print(f"✅ 已补写 params: {d.name} sample_end={params['sample_end']} "
              f"n_stocks={params['n_stocks']} X={meta['X_shape']}")
        patched += 1

    if patched == 0:
        print("未找到需补写的 121 维缓存（可能已全部带 params 或缓存目录为空）")
    else:
        print(f"补写完成: {patched} 个缓存目录")


if __name__ == "__main__":
    main()
