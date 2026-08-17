"""A/B 月末模式对比实验：月末清仓（模式 A）vs 幸存者不清仓+补空位（模式 B）。

背景（2026-08-17）：
  - 模式 A = 回测现状：每月末强制清仓全部持仓，次月买入新 TOP10（monthly_engine 默认）
  - 模式 B = 模拟盘现状：月末幸存者不清仓，次月重训后只补空位（keep_survivors=True）
对比两种模式在 70 个月全周期（2020-09 ~ 2026-06）的收益差异。

运行（铁律六：必须 py312）：
  /home/zhulei/anaconda3/envs/zhulei_py312/bin/python experiments/compare_eom_modes.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# KMP_AFFINITY 清除（铁律：.bashrc 绑核会让多进程/数值库抢同一核心）
os.environ.pop("KMP_AFFINITY", None)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

from sequoia_x.core.config import Settings
from sequoia_x.core.logger import get_logger
from sequoia_x.data.engine import DataEngine
from sequoia_x.model_selection_v2.backtest.monthly_engine import MonthlyBacktestEngine
from sequoia_x.model_selection_v2.config import get_config

logger = get_logger(__name__)

OUTPUT_DIR = Path("output/backtest_v2")
CACHE_PATH = OUTPUT_DIR / "prediction_cache.json"
START_MONTH = "2020-09"
END_MONTH = "2026-06"
TOP_N = 10
RISK_MODE = "M4"
INITIAL_CAPITAL = 500_000.0


def run_one(keep_survivors: bool, prediction_cache: dict) -> dict:
    """跑一组回测，返回指标 + 交易统计。"""
    bt = MonthlyBacktestEngine(
        cfg=get_config(),
        engine=DataEngine(Settings()),
        top_n=TOP_N,
        risk_mode=RISK_MODE,
        initial_capital=INITIAL_CAPITAL,
        use_real_t4=True,
        prediction_cache=prediction_cache,
        fusion_method="pred_std",
        keep_survivors=keep_survivors,
    )
    t0 = time.time()
    metrics = bt.run(START_MONTH, END_MONTH)
    elapsed = time.time() - t0

    # 交易统计
    sells = [t for t in bt.trades if t.trade_type == "sell"]
    n_rule = sum(1 for t in sells if t.reason == "规则触发" or t.reason == "规则触发(月末)"
                 or "分" in t.reason and "清仓" not in t.reason)
    n_eom = sum(1 for t in sells if "清仓" in t.reason)
    # 月末幸存持仓数（每个 cycle 结束时的持仓）→ 用 daily_records 每月最后一天 positions
    daily_by_month: dict[str, list[dict]] = {}
    for r in bt.daily_records:
        daily_by_month.setdefault(r["date"][:7], []).append(r)
    month_end_pos = []
    for m, recs in sorted(daily_by_month.items()):
        month_end_pos.append(recs[-1]["positions"])

    result = {
        "mode": "A(月末清仓)" if not keep_survivors else "B(不清仓+补空位)",
        "keep_survivors": keep_survivors,
        **{k: v for k, v in metrics.items() if k not in ("daily_records", "trades", "monthly_returns", "monthly_labels")},
        "n_trades": len(bt.trades),
        "n_sells": len(sells),
        "n_rule_sells": n_rule,
        "n_eom_sells": n_eom,
        "month_end_pos_avg": float(np.mean(month_end_pos)),
        "elapsed_min": round(elapsed / 60, 1),
    }
    logger.info(f"[{result['mode']}] 完成: 总收益={result['total_return']:+.2%} "
                f"年化={result['annual_return']:+.1%} 夏普={result['sharpe']:.2f} "
                f"回撤={result['max_drawdown']:+.2%} 耗时={result['elapsed_min']}min")
    return result


def main() -> None:
    logger.info(f"═══ A/B 月末模式对比实验 ═══  python={sys.executable}")
    logger.info(f"np={np.__version__} | 时段 {START_MONTH}~{END_MONTH} | "
                f"TOP_N={TOP_N} 风控={RISK_MODE} | 初始资金 {INITIAL_CAPITAL:,.0f}")

    if not CACHE_PATH.exists():
        logger.error(f"预测缓存不存在: {CACHE_PATH}")
        sys.exit(1)
    prediction_cache = json.loads(CACHE_PATH.read_text())
    logger.info(f"预测缓存加载: {CACHE_PATH} ({len(prediction_cache)} 个月)")

    results = []
    for keep in (False, True):  # 先 A 后 B（A 顺带验证复现现有结果）
        results.append(run_one(keep, prediction_cache))

    # 对比表
    print("\n" + "=" * 78)
    print(f"A/B 模式对比 | {START_MONTH}~{END_MONTH} | TOP_N={TOP_N} 风控={RISK_MODE}")
    print("=" * 78)
    headers = ["模式", "总收益", "年化", "夏普", "最大回撤", "月胜率", "交易数", "规则卖", "月末清仓", "月末持仓均"]
    print(f"{headers[0]:<16}{headers[1]:>10}{headers[2]:>8}{headers[3]:>7}{headers[4]:>10}{headers[5]:>8}{headers[6]:>8}{headers[7]:>8}{headers[8]:>10}{headers[9]:>10}")
    for r in results:
        print(f"{r['mode']:<16}{r['total_return']:>9.1%}{r['annual_return']:>7.1%}"
              f"{r['sharpe']:>7.2f}{r['max_drawdown']:>9.1%}{r['win_rate']:>7.1%}"
              f"{r['n_trades']:>8}{r['n_rule_sells']:>8}{r['n_eom_sells']:>10}{r['month_end_pos_avg']:>10.1f}")

    # 差异
    a, b = results
    print("-" * 78)
    print(f"差异 (B-A): 总收益 {(b['total_return']-a['total_return'])*100:+.1f}pp | "
          f"夏普 {b['sharpe']-a['sharpe']:+.2f} | 回撤 {(b['max_drawdown']-a['max_drawdown'])*100:+.1f}pp | "
          f"月末清仓笔数 {b['n_eom_sells']-a['n_eom_sells']:+d}")

    # 保存结果
    out_path = OUTPUT_DIR / "compare_eom_modes.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    logger.info(f"对比结果已保存: {out_path}")


if __name__ == "__main__":
    main()
