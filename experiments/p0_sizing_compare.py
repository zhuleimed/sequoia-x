#!/usr/bin/env python
"""P0 方向一：五种"分钱方法"70 个月 A/B 对比（2026-10-03）。

要回答的问题（用大白话）
    现在选出的 TOP10 股票，**每只买多少钱**是平均分的（每只 1 份）。
    换成"波动大的少买点（反比方差）"、"每只贡献的风险一样大（风险平价）"、
    "让整组起伏最小（最小方差）"，会不会更赚 / 更稳？

口径（六臂只在"分钱方法"这一个变量上不同，其余完全一致）
    · 预测缓存：`output/backtest_v2/.t4_70m_tail.json`（69 个月，2020-10~2026-06，
      **tail 抽样 + 含 T4** = 生产口径，见 09-29 抽样定案）
    · 风控：E1（月内只留 -12% 硬止损）+ 月末收盘价清仓（模式 A）+ 等权融合 0.5
    · risk_mode 统一用 M0（关掉 T1过滤/T3波动率/市场状态/IC加权），
      再叠加 `sizing_method` —— 这样"风控"完全相同，只比较"分钱方法"
    · 额外第 6 臂 M4-对照：risk_mode="M4" + sizing_method=None（=**生产回测现口径**），
      用于校验"新开关没有改变原行为"（应与 ic_weighted 臂逐项一致）

判据（预先声明，先写再跑）
    以「等权」为基线（= 生产模拟盘今天的行为）。某方法要"值得上生产"须同时满足：
      ① 总收益更高  且  ② 最大回撤不劣化超过 2 个百分点
    只看"总收益高"不够 —— 还要看抗极端指标（去 top-N / 集中度 / 中位数），
    避免又出现"一半收益靠 2 只妖股"的情况（见 docs/2026-10-03_P0实施记录）。

用法（铁律六：必须 py312）
    /home/zhulei/anaconda3/envs/zhulei_py312/bin/python experiments/p0_sizing_compare.py
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

# KMP_AFFINITY 清除（.bashrc 绑核会让数值库抢同一核心）
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
RESULT_PATH = OUTPUT_DIR / "p0_sizing_compare.json"

TOP_N = 10
INITIAL_CAPITAL = 500_000.0
START_MONTH, END_MONTH = "2020-10", "2026-06"

# (展示名, risk_mode, sizing_method, 说明)
ARMS: list[tuple[str, str, str | None, str]] = [
    ("等权",       "M0", "equal",       "基线：每只 1 份（= 生产模拟盘现状）"),
    ("排名递减",   "M0", "ic_weighted", "第 1 名权重最高、依次递减（回测 M4 用的方法）"),
    ("反比方差",   "M0", "inv_var",     "波动越大买越少（只看单只波动，60 日）"),
    ("风险平价",   "M0", "risk_parity", "每只对组合风险的贡献相等（用相关性，120 日）"),
    ("最小方差",   "M0", "min_variance", "整组起伏最小 + 单票≤15% + 行业≤30%（120 日）"),
    ("M4-对照",    "M4", None,          "生产回测现口径（应 ≈ 排名递减，用于校验开关无损）"),
]

# 关键指标（列名, 表头, 格式）
COLS: list[tuple[str, str, str]] = [
    ("total_return", "总收益", ".2%"),
    ("annual_return", "年化", ".2%"),
    ("sharpe", "夏普", ".2f"),
    ("max_drawdown", "最大回撤", ".2%"),
    ("win_rate", "月胜率", ".1%"),
    ("n_trades", "交易笔数", "d"),
    ("avg_turnover", "月换手", ".1f"),
    ("return_ex_top2", "去最赚2只", ".2%"),
    ("top2_pnl_share", "前2只占比", ".1%"),
    ("median_trade_pct", "单笔中位", ".2%"),
    ("median_monthly_return", "月度中位", ".2%"),
    ("final_value", "期末资产", ",.0f"),
]


def run_one(name: str, risk_mode: str, sizing_method: str | None, cache: dict) -> dict:
    bt = MonthlyBacktestEngine(
        cfg=get_config(),
        engine=DataEngine(Settings()),
        top_n=TOP_N,
        risk_mode=risk_mode,
        initial_capital=INITIAL_CAPITAL,
        use_real_t4=True,
        prediction_cache=cache,
        fusion_method="pred_std",
        keep_survivors=False,
        intra_exit_policy="hard_stop_only",
        hard_stop_pct=-0.12,
        eom_sell_price="close",
        sizing_method=sizing_method,
    )
    t0 = time.time()
    m = bt.run(START_MONTH, END_MONTH)
    m["_elapsed_min"] = round((time.time() - t0) / 60, 2)
    m.pop("daily_records", None)
    m.pop("trades", None)
    return m


def main() -> int:
    if not CACHE_PATH.exists():
        logger.error(f"预测缓存不存在: {CACHE_PATH}（应为 69 个月 tail+T4 口径）")
        return 1
    cache = json.loads(CACHE_PATH.read_text())
    logger.info(f"预测缓存: {CACHE_PATH.name}（{len(cache)} 个月: "
                f"{min(cache)} ~ {max(cache)}）")
    logger.info(f"六臂: {[a[0] for a in ARMS]}")

    results: dict[str, dict] = {}
    for i, (name, risk_mode, sizing, note) in enumerate(ARMS, 1):
        logger.info(f"\n{'=' * 70}\n[{i}/{len(ARMS)}] 臂「{name}」—— {note}"
                    f"\n{'=' * 70}")
        try:
            results[name] = run_one(name, risk_mode, sizing, cache)
        except Exception as e:
            logger.exception(f"臂「{name}」失败: {e}")
            results[name] = {"_error": str(e)}
        if "_error" not in results[name]:
            logger.info(f"[{i}/{len(ARMS)}] 臂「{name}」完成: "
                        f"总收益={results[name]['total_return']:+.2%} "
                        f"夏普={results[name]['sharpe']:.2f} "
                        f"回撤={results[name]['max_drawdown']:+.2%} "
                        f"（耗时 {results[name]['_elapsed_min']}min）")

    # ── 汇总表 ─────────────────────────────────────────────
    print("\n" + "=" * 120)
    print("P0 方向一：五种分钱方法对比（69 个月 2020-10~2026-06，E1 风控，等权融合）")
    print("=" * 120)
    header = f"{'方法':<10}" + "".join(f"{h:>12}" for _, h, _ in COLS)
    print(header)
    for name, *_ in ARMS:
        r = results.get(name, {})
        if "_error" in r:
            print(f"{name:<10}  失败: {r['_error'][:60]}")
            continue
        row = f"{name:<10}"
        for key, _, fmt in COLS:
            v = r.get(key, float("nan"))
            row += f"{v:>12{fmt}}" if isinstance(v, (int, float)) else f"{str(v):>12}"
        print(row)

    # ── 对照校验：新开关是否改变原行为（M4-对照 vs 排名递减）──
    a, b = results.get("M4-对照", {}), results.get("排名递减", {})
    if a and b and "_error" not in a and "_error" not in b:
        diff = abs(a["total_return"] - b["total_return"])
        ok = diff < 1e-4
        print(f"\n对照校验: M4-对照 总收益 {a['total_return']:+.4%} vs "
              f"排名递减 {b['total_return']:+.4%} → {'✅ 逐位一致（开关无损）' if ok else f'⚠ 有差异 {diff:.4%}（需排查）'}")

    # ── 相对基线的结论行 ────────────────────────────────────
    base = results.get("等权", {})
    if base and "_error" not in base:
        print("\n相对「等权」基线（判据：总收益更高 且 回撤劣化 ≤2pp）：")
        for name, *_ in ARMS:
            r = results.get(name, {})
            if name in ("等权",) or "_error" in r:
                continue
            dr = r["total_return"] - base["total_return"]
            dd = r["max_drawdown"] - base["max_drawdown"]
            verdict = "✅ 双条件满足" if (dr > 0 and dd >= -0.02) else "❌ 不满足"
            print(f"  {name:<10} 总收益 {dr:+.2%} | 回撤差 {dd:+.2%} → {verdict}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in results.items()},
        ensure_ascii=False, indent=2))
    print(f"\n结果已写入 {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
