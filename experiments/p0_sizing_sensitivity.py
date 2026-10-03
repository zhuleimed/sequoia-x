#!/usr/bin/env python
"""P0 方向一：最小方差"参数敏感性"（上生产前最后一道验证，2026-10-03）。

为什么跑它
    `p0_sizing_compare.py` 已证：**最小方差 > 等权**（+373.7% vs +277.8%）。
    但那是**一组参数**（120 日窗口、单票上限 15%）的成绩。上生产前必须回答：
    换个窗口/上限，结论还成立吗？—— 若换个参数就翻车，说明是"参数拟合"而非真优势。

七臂（其余口径与 compare 完全一致：69 个月 tail+T4 缓存、E1 风控、M0 隔离）
    等权（基线）
    排名递减（回测 M4 现口径，第二候选）
    最小方差 L60 / L120 / L250      ← 换协方差窗口（60/120/250 交易日）
    最小方差 C20 / C25              ← 换单票上限（20%/25%；默认 15%）

判据：与「等权」比（收益更高 且 回撤劣化 ≤2pp）须在各参数下一致成立；
      并与 L120C15（已选最优）比，看是否明显更差。

用法（铁律六：py312）
    /home/zhulei/anaconda3/envs/zhulei_py312/bin/python experiments/p0_sizing_sensitivity.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.pop("KMP_AFFINITY", None)
os.environ["OMP_NUM_THREADS"] = "1"

from sequoia_x.core.config import Settings                     # noqa: E402
from sequoia_x.core.logger import get_logger                   # noqa: E402
from sequoia_x.data.engine import DataEngine                   # noqa: E402
from sequoia_x.model_selection_v2.backtest.monthly_engine import (  # noqa: E402
    MonthlyBacktestEngine,
)
from sequoia_x.model_selection_v2.config import get_config     # noqa: E402

logger = get_logger(__name__)

OUTPUT_DIR = Path("output/backtest_v2")
CACHE_PATH = OUTPUT_DIR / ".t4_70m_tail.json"
RESULT_PATH = OUTPUT_DIR / "p0_sizing_sensitivity.json"

TOP_N = 10
INITIAL_CAPITAL = 500_000.0
START_MONTH, END_MONTH = "2020-10", "2026-06"

# (展示名, sizing_method, sizing_params)
ARMS: list[tuple[str, str, dict]] = [
    ("等权",       "equal",       {}),
    ("排名递减",   "ic_weighted", {}),
    ("最小方差L60",  "min_variance", {"lookback": 60}),
    ("最小方差L120", "min_variance", {"lookback": 120}),
    ("最小方差L250", "min_variance", {"lookback": 250}),
    ("最小方差C20",  "min_variance", {"cap": 0.20}),
    ("最小方差C25",  "min_variance", {"cap": 0.25}),
]


def run_one(method: str, params: dict, cache: dict) -> dict:
    bt = MonthlyBacktestEngine(
        cfg=get_config(), engine=DataEngine(Settings()), top_n=TOP_N,
        risk_mode="M0", initial_capital=INITIAL_CAPITAL, use_real_t4=True,
        prediction_cache=cache, fusion_method="pred_std", keep_survivors=False,
        intra_exit_policy="hard_stop_only", hard_stop_pct=-0.12,
        eom_sell_price="close", sizing_method=method, sizing_params=params,
    )
    t0 = time.time()
    m = bt.run(START_MONTH, END_MONTH)
    m["_elapsed_min"] = round((time.time() - t0) / 60, 2)
    m.pop("daily_records", None)
    m.pop("trades", None)
    return m


def main() -> int:
    if not CACHE_PATH.exists():
        logger.error(f"预测缓存不存在: {CACHE_PATH}")
        return 1
    cache = json.loads(CACHE_PATH.read_text())
    logger.info(f"缓存 {CACHE_PATH.name}（{len(cache)} 个月）| 七臂敏感性测试")

    results: dict[str, dict] = {}
    for i, (name, method, params) in enumerate(ARMS, 1):
        logger.info(f"\n{'=' * 60}\n[{i}/{len(ARMS)}] 臂「{name}」 method={method} params={params}\n{'=' * 60}")
        try:
            results[name] = run_one(method, params, cache)
            r = results[name]
            logger.info(f"[{i}/{len(ARMS)}] 「{name}」完成: 总收益={r['total_return']:+.2%} "
                        f"夏普={r['sharpe']:.2f} 回撤={r['max_drawdown']:+.2%} "
                        f"（{r['_elapsed_min']}min）")
        except Exception as e:
            logger.exception(f"臂「{name}」失败: {e}")
            results[name] = {"_error": str(e)}

    print("\n" + "=" * 104)
    print("最小方差 参数敏感性（69 个月 2020-10~2026-06）")
    print("=" * 104)
    print(f"{'臂':<12}{'总收益':>10}{'年化':>9}{'夏普':>7}{'最大回撤':>10}"
          f"{'月胜率':>8}{'去最赚2只':>10}{'前2只占比':>10}")
    for name, *_ in ARMS:
        r = results.get(name, {})
        if "_error" in r:
            print(f"{name:<12} 失败: {r['_error'][:50]}")
            continue
        print(f"{name:<12}{r['total_return']:>10.2%}{r['annual_return']:>9.2%}"
              f"{r['sharpe']:>7.2f}{r['max_drawdown']:>10.2%}{r['win_rate']:>8.1%}"
              f"{r['return_ex_top2']:>10.2%}{r['top2_pnl_share']:>10.1%}")

    base = results.get("等权", {})
    if base and "_error" not in base:
        print(f"\n相对「等权」基线（收益更高 且 回撤劣化 ≤2pp）：")
        for name, *_ in ARMS:
            r = results.get(name, {})
            if name == "等权" or "_error" in r:
                continue
            dr = r["total_return"] - base["total_return"]
            dd = r["max_drawdown"] - base["max_drawdown"]
            print(f"  {name:<12} 收益 {dr:+.2%} | 回撤差 {dd:+.2%} → "
                  f"{'✅' if (dr > 0 and dd >= -0.02) else '❌'}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in results.items()},
        ensure_ascii=False, indent=2))
    print(f"\n结果已写入 {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
