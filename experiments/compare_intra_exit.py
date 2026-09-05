"""A/B 月内卖出政策对比实验：全规则(all) vs 纯持有(none) vs 只留硬止损(hard_stop_only)。

背景（2026-09-05，详见记忆 min-hold-bug-and-month-end-hold + plan stateful-mixing-squirrel.md）：
  - 月度换仓月初买满 TOP10（模式A月末清仓），但月内 evaluate_exit 动量规则
    （死叉/负夏普/相对弱势等）读"买入前6周历史"，在月初新仓上第2~5天就自触发，
    MIN_HOLD 修复后又叠加"读入选日历史 → 入场即弱 → 快被洗出"，把仓位在月内削到接近空仓、
    现金闲置到月末。用户想改为"真·纯持有到月末(买满→月末统一清仓)"是否更优？由本 A/B 裁决。
  - intra_exit_policy 于 2026-08/09 加入 monthly_engine，用于在回测框架隔离"是否月内规则卖出"。
  - 三臂 = 同一 70 月(2020-09~2026-06)、M4/TOP_N10/500k/pred_std/模式A(keep_survivors=False)：
      * all             : 现状——月内跑全套卖出规则(现已含 MIN_HOLD 修复的当前代码基线)
      * none            : 纯持有——月内不评价任何规则，仅月末 _sell_all_positions 强清 (用户方案)
      * hard_stop_only  : 折中——保留 -8% 硬止损(真实资金下行护栏)，去掉死叉/负夏普等动量规则
运行（铁律六/铁律：必须 py312，且 env -u KMP_AFFINITY）：
  /home/zhulei/anaconda3/envs/zhulei_py312/bin/python experiments/compare_intra_exit.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# 确保仓库根目录在模块搜索路径上（脚本位于 experiments/ 子目录，sys.path[0] 是 experiments/ 而非仓库根）
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# KMP_AFFINITY 清除（.bashrc 绑核会让多进程/数值库抢同一核心）
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

POLICIES = ["all", "none", "hard_stop_only"]  # 臂按此顺序跑（all 顺带验证复现当前基线）
POLICY_LABEL = {
    "all": "A现状:月内全规则",
    "none": "B纯持有:月末清仓",
    "hard_stop_only": "C只留硬止损-8%",
}


def run_one(policy: str, prediction_cache: dict) -> dict:
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
        keep_survivors=False,            # 模式 A：月末强制清仓 = 月内出场的基线场景
        intra_exit_policy=policy,
    )
    t0 = time.time()
    metrics = bt.run(START_MONTH, END_MONTH)
    elapsed = time.time() - t0

    # 交易统计：规则触发卖出 vs 月末强制清仓（复用 compare_eom_modes 分桶思路）
    sells = [t for t in bt.trades if t.trade_type == "sell"]
    n_eom = sum(1 for t in sells if "清仓" in (t.reason or ""))

    # 月末幸存持仓数（用 daily_records 每月最后一天 positions）
    daily_by_month: dict[str, list[dict]] = {}
    for r in bt.daily_records:
        daily_by_month.setdefault(r["date"][:7], []).append(r)
    month_end_pos = []
    for m, recs in sorted(daily_by_month.items()):
        month_end_pos.append(recs[-1].get("positions", np.nan))

    result = {
        "policy": policy,
        "label": POLICY_LABEL[policy],
        **{k: v for k, v in metrics.items()
           if k not in ("daily_records", "trades", "monthly_returns", "_monthly_labels")},
        "n_trades": len(bt.trades),
        "n_sells": len(sells),
        "n_rule_sells": len(sells) - n_eom,   # 非"清仓"的卖出席位 = 月内规则卖出
        "n_eom_sells": n_eom,
        "month_end_pos_avg": float(np.mean([p for p in month_end_pos if not isinstance(p, float) or p == p])),
        "monthly_returns": bt.monthly_returns,
        "_monthly_labels": metrics.get("_monthly_labels", bt.monthly_labels),
        "elapsed_min": round(elapsed / 60, 1),
    }
    logger.info(f"[{result['label']}] 完成: 总收益={result['total_return']:+.2%} "
                f"年化={result['annual_return']:+.1%} 夏普={result['sharpe']:.2f} "
                f"回撤={result['max_drawdown']:+.2%} 月末持仓均={result['month_end_pos_avg']:.2f} "
                f"规则卖/清仓={result['n_rule_sells']}/{result['n_eom_sells']} 耗时={result['elapsed_min']}min")
    return result


def main() -> None:
    logger.info(f"═══ A/B 月内卖出政策对比 ═══ python={sys.executable}")
    logger.info(f"np={np.__version__} | 时段 {START_MONTH}~{END_MONTH} | "
                f"TOP_N={TOP_N} 风控={RISK_MODE}(模式A月末清仓) | 初始资金 {INITIAL_CAPITAL:,.0f}")

    if not CACHE_PATH.exists():
        logger.error(f"预测缓存不存在: {CACHE_PATH}")
        sys.exit(1)
    prediction_cache = json.loads(CACHE_PATH.read_text())
    logger.info(f"预测缓存加载: {CACHE_PATH} ({len(prediction_cache)} 个月)")

    results = [run_one(p, prediction_cache) for p in POLICIES]

    # 对比表
    print("\n" + "=" * 88)
    print(f"A/B 月内卖出政策对比 | {START_MONTH}~{END_MONTH} | TOP_N={TOP_N} 风控={RISK_MODE} (模式A)")
    print("=" * 88)
    headers = ["政策", "总收益", "年化", "夏普", "最大回撤", "月胜率", "交易", "规则卖", "月末清仓", "月末持均", "耗时min"]
    print(f"{headers[0]:<26}{headers[1]:>9}{headers[2]:>8}{headers[3]:>7}{headers[4]:>10}"
          f"{headers[5]:>8}{headers[6]:>7}{headers[7]:>8}{headers[8]:>10}{headers[9]:>9}{headers[10]:>8}")
    for r in results:
        print(f"{r['label']:<26}{r['total_return']:>8.1%}{r['annual_return']:>7.1%}"
              f"{r['sharpe']:>7.2f}{r['max_drawdown']:>9.1%}{r['win_rate']:>7.1%}"
              f"{r['n_trades']:>7}{r['n_rule_sells']:>8}{r['n_eom_sells']:>10}{r['month_end_pos_avg']:>9.1f}"
              f"{r['elapsed_min']:>8.1f}")

    # 相对 A 臂的差异 + 逐月收益 diff
    a = results[0]
    print("\n" + "-" * 88)
    for r in results[1:]:
        d_tr = (r["total_return"] - a["total_return"]) * 100
        print(f"差异({r['label']} - {a['label']}): 总收益 {d_tr:+.1f}pp | "
              f"夏普 {r['sharpe']-a['sharpe']:+.2f} | 回撤 {(r['max_drawdown']-a['max_drawdown'])*100:+.1f}pp | "
              f"规则卖出 {r['n_rule_sells']-a['n_rule_sells']:+d}笔")

    # 逐月收益 diff 表（B-A 与 C-A 的每月收益差）
    months = a.get("_monthly_labels", [])
    rA, rB, rC = results[0], results[1], results[2]
    print("\n逐月月度收益对比(%)（前30月节选） month | A现状 | B纯持有 | C硬止损 | Δ(B-A) | Δ(C-A)")
    for i, m in enumerate(months):
        ma = rA["monthly_returns"][i] * 100
        mb = rB["monthly_returns"][i] * 100
        mc = rC["monthly_returns"][i] * 100
        if i < 30 or abs(mb - ma) > 15 or abs(mc - ma) > 15:
            print(f"{m} | {ma:>7.1f} | {mb:>7.1f} | {mc:>7.1f} | {mb-ma:>+7.1f} | {mc-ma:>+7.1f}")
    # B-A 累计胜月数
    n_win_b = sum(1 for i in range(len(months)) if rB["monthly_returns"][i] > rA["monthly_returns"][i])
    n_win_c = sum(1 for i in range(len(months)) if rC["monthly_returns"][i] > rA["monthly_returns"][i])
    print(f"\nB 赢 A 的月数: {n_win_b}/{len(months)} | C 赢 A 的月数: {n_win_c}/{len(months)}")

    # 保存结果
    out_path = OUTPUT_DIR / "compare_intra_exit.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    logger.info(f"对比结果已保存: {out_path}")


if __name__ == "__main__":
    main()
