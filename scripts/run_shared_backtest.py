"""Phase 2: 72 组共享预测缓存回测 —— 一键运行。

流程:
  1. 加载/构建预测缓存 (T2+T1+T3 月度预测值)
  2. 循环 72 组 (4 TOP_N × 3 时段 × 6 风控)
  3. 每组仅运行模拟循环（秒级/组），共享同一份预测值
  4. 输出 CSV 汇总 + JSON 明细 + 最优配置

用法:
  # 完整流程（先构建缓存，再跑 72 组）
  python scripts/run_shared_backtest.py --all

  # 使用已有缓存
  python scripts/run_shared_backtest.py --all --cache output/backtest_v2/prediction_cache.json

  # 快速测试（3个月×500股票×3组）
  python scripts/run_shared_backtest.py --months 3 --max-stocks 500 --top-n 10 --mode M0

  # 仅构建缓存（不跑回测）
  python scripts/build_prediction_cache.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sequoia_x.core.config import Settings
from sequoia_x.core.logger import get_logger
from sequoia_x.data.engine import DataEngine
from sequoia_x.model_selection_v2.config import V2Config, get_config
from sequoia_x.model_selection_v2.backtest.monthly_engine import MonthlyBacktestEngine

logger = get_logger("shared_backtest")

# ── 常量 ──
OUTPUT_DIR = Path("output/backtest_v2")
CACHE_PATH = OUTPUT_DIR / "prediction_cache.json"

TOP_N_LIST = [10, 15, 20, 25]
PERIODS = [
    ("2025年", "2025-08", "2025-12"),
    ("2026年", "2026-01", "2026-06"),
    ("全周期", "2025-08", "2026-06"),
]
# 2026-08-20: V4 129维 70 个月完整回测（2020-09~2026-06，与旧 121 维 70 月同口径对比）。
# 用 `--period full` 选择此完整全周期段。
FULL_PERIODS = [
    ("全周期70月", "2020-09", "2026-06"),
]
RISK_MODES = [
    ("M0", "裸融合"),
    ("M1", "+T1过滤"),
    ("M2", "+T3仓位"),
    ("M3", "+大盘风控"),
    ("M4", "+IC加权"),
    ("M5", "全风控"),
]
INITIAL_CAPITAL = 500_000.0


def _cache_is_complete(cache: dict, test_months: list[str],
                       manifest_path: Path) -> tuple[bool, str]:
    """判定预测缓存是否满足本次请求。

    2026-10-08 修复（B 臂回测 exit=1 的根因）：
      旧判据是 `months_ok >= len(test_months)` —— **永远为假**：2020-09 因 purge 后
      训练样本为 0 被**合法跳过**，缓存永远只有 69/70 个月。
      后果：① 每次回测都白试一次重建；② 一旦数据集缓存缺失（例如数据同步把
      `sample_end` 推进到新的交易日 ⇒ 数据集缓存 key 变了），重建会直接
      `FileNotFoundError` 崩掉 —— 2026-10-08 19:06 B 臂回测就是这样 exit=1 的。

    新判据：优先读缓存旁边的 manifest（`build_cache` 落盘，记录 requested/produced/skipped）：
      **`requested - produced ⊆ skipped` ⇒ 视为完整**（缺的都是合法跳过的月份）。
      没有 manifest（旧产物）→ 退回旧判据，并在返回信息里说明**为何可能是"合法不全"**。
    """
    missing = [m for m in test_months if m not in cache]
    if not missing:
        return True, "全部月份齐备"
    if manifest_path.exists():
        try:
            mf = json.loads(manifest_path.read_text())
            skipped = set(mf.get("skipped", {}))
            unexcused = [m for m in missing if m not in skipped]
            if not unexcused:
                return True, (f"缺 {len(missing)} 个月，但均在 manifest 中登记为合法跳过 "
                              f"{sorted(missing)}")
            return False, f"缺 {unexcused}（manifest 未登记为跳过）"
        except Exception as e:
            return False, f"manifest 读取失败({type(e).__name__}: {e})；缺 {missing}"
    return False, (f"缺 {missing}（**无 manifest，无法判定是否为合法跳过**；"
                   f"若这是 2020-09 这类「样本不足被跳过」的月份，"
                   f"请用 build_cache 重建一次以生成 manifest）")


def load_or_build_cache(
    cfg: V2Config,
    engine: DataEngine,
    cache_path: Path,
    test_months: list[str],
    max_stocks: int = 0,
    force_rebuild: bool = False,
    skip_t4: bool = False,
) -> dict:
    """加载预测缓存，如不存在则自动构建。

    Returns:
        {month: {symbols, t2, t1, t3}, ...}
    """
    if not force_rebuild and cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
        _accept, _why = _cache_is_complete(cache, test_months,
                                          Path(str(cache_path) + ".manifest.json"))
        if _accept:
            logger.info(f"预测缓存已就绪: {len(cache)} 个月 (来自 {cache_path})；{_why}")
            return cache
        logger.info(f"缓存不完整（{_why}）→ 重新构建...")
    else:
        logger.info("构建预测缓存...")

    from scripts.build_prediction_cache import build_cache
    return build_cache(cfg, engine, test_months, max_stocks, output_path=cache_path, skip_t4=skip_t4)


def run_single_group(
    cfg: V2Config,
    engine: DataEngine,
    prediction_cache: dict,
    top_n: int,
    risk_mode: str,
    start_month: str,
    end_month: str,
    sizing_method: str | None = None,
    sizing_params: dict | None = None,
) -> dict:
    """运行单组回测（使用共享预测缓存）。

    Returns:
        {total_return, annual_return, sharpe, max_drawdown, win_rate, ...}
    """
    bt = MonthlyBacktestEngine(
        cfg=cfg,
        engine=engine,
        top_n=top_n,
        risk_mode=risk_mode,
        initial_capital=INITIAL_CAPITAL,
        use_real_t4=False,
        prediction_cache=prediction_cache,
        sizing_method=sizing_method,        # 2026-10-11：None=等权（历史行为）
        sizing_params=sizing_params or {},
    )
    metrics = bt.run(start_month, end_month)
    return metrics


def save_summary_csv(all_results: list[dict]) -> None:
    """生成汇总 CSV。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "summary_all.csv"
    fieldnames = [
        "风控模式", "TOP_N", "时段", "月数",
        "总收益率", "年化收益率", "夏普比率", "最大回撤",
        "月胜率", "交易笔数", "月均换手率", "胜率(笔)", "终值",
        # ── 2026-10-11 补：**抗极端指标**（引擎早在待办 #23 就算出来了，
        #   但因下面 `extrasaction="ignore"` + 这里没列 → **被静默丢掉** ✗）。
        #   为什么必须补：2026-10-11 复核发现 B 臂的"最高分"其实是
        #   **2024-07 单月 +26.93%（买入 9 只）** 撑起来的、且**换口径就变** ——
        #   而**汇总表里看不到这层信息**，导致"哪个配置最好"这类结论被假象误导。
        #   ★ `return_ex_topN` = **剔除最赚 N 只**后的总收益率；腰斩 ⇒ 个别妖股撑的
        #   ★ `top2_pnl_share` = 前 2 只占总盈亏比（集中度）；越高越可疑
        "return_ex_top1", "return_ex_top2", "return_ex_top3",
        "top2_pnl_share", "median_trade_pct", "median_monthly_return",
    ]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)
    logger.info(f"汇总表: {summary_path} ({len(all_results)} 行)")


def save_monthly_matrix(all_results: list[dict]) -> None:
    """生成月度收益矩阵 CSV。"""
    monthly_data: dict[str, dict[str, float]] = {}
    for r in all_results:
        key = f"{r['风控模式']}_T{r['TOP_N']}_{r['时段']}"
        labels = r.get("_monthly_labels", [])
        returns = r.get("_monthly_returns", [])
        if labels and returns and len(labels) == len(returns):
            monthly_data[key] = dict(zip(labels, returns))

    if not monthly_data:
        return

    all_months = sorted(set(m for d in monthly_data.values() for m in d))
    matrix_path = OUTPUT_DIR / "monthly_returns.csv"
    with open(matrix_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["配置"] + all_months)
        for key in sorted(monthly_data.keys()):
            row = [key] + [
                f"{monthly_data[key].get(m, 0):+.4f}" if m in monthly_data[key] else ""
                for m in all_months
            ]
            writer.writerow(row)
    logger.info(f"收益矩阵: {matrix_path}")


def compute_and_save_optimal(all_results: list[dict]) -> dict:
    """综合评分选出最优配置。"""
    valid = [r for r in all_results
             if r.get("夏普比率") is not None and r.get("最大回撤") is not None]
    if not valid:
        return {}

    n = len(valid)
    idx = list(range(n))
    by_sharpe = sorted(idx, key=lambda i: valid[i]["夏普比率"])
    by_ret = sorted(idx, key=lambda i: valid[i]["总收益率"])
    by_win = sorted(idx, key=lambda i: valid[i]["月胜率"])
    by_dd = sorted(idx, key=lambda i: -valid[i]["最大回撤"])

    s = np.zeros(n)
    for rank, i in enumerate(by_sharpe):
        s[i] += 0.4 * (rank + 1) / n
    for rank, i in enumerate(by_ret):
        s[i] += 0.2 * (rank + 1) / n
    for rank, i in enumerate(by_win):
        s[i] += 0.15 * (rank + 1) / n
    for rank, i in enumerate(by_dd):
        s[i] += 0.25 * (rank + 1) / n

    best_idx = int(np.argmax(s))
    optimal = valid[best_idx].copy()
    optimal["综合评分"] = round(float(s[best_idx]), 4)

    opt_path = OUTPUT_DIR / "optimal_config.json"
    opt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(opt_path, "w") as f:
        json.dump(optimal, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"最优配置: {opt_path}")

    return optimal


def get_test_months(start_month: str, end_month: str) -> list[str]:
    """生成月份列表。"""
    start_ym = (int(start_month[:4]), int(start_month[5:7]))
    end_ym = (int(end_month[:4]), int(end_month[5:7]))
    months = []
    y, m = start_ym
    while (y, m) <= end_ym:
        months.append(f"{y}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def main():
    # 2026-09-26：让 --output-dir 真正生效。此前它只被 argparse 收下、**从未被使用**
    # —— save_summary_csv / monthly_returns / optimal_config 全都写死到模块级 OUTPUT_DIR，
    # 于是换目录跑（如 V5 对照回测）会**静默覆盖** output/backtest_v2/summary_all.csv，
    # 把既有基准冲掉。声明须在 parser 之前（下方 `default=str(OUTPUT_DIR)` 会先读它）。
    global OUTPUT_DIR
    parser = argparse.ArgumentParser(description="Phase 2: 72 组共享预测缓存回测")
    parser.add_argument("--all", action="store_true", help="运行全部 72 组")
    parser.add_argument("--top-n", type=int, default=0, help="限制 TOP_N")
    parser.add_argument("--mode", type=str, default="", help="限制风控模式")
    parser.add_argument("--period", type=str, default="",
                        help="限制时段: 2025/2026/all")
    parser.add_argument("--cache", type=str, default=str(CACHE_PATH),
                        help="预测缓存路径")
    parser.add_argument("--months", type=int, default=0,
                        help="限制月份数（0=全部）")
    parser.add_argument("--max-stocks", type=int, default=0,
                        help="限制股票池大小（0=全量）")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="强制重建预测缓存")
    parser.add_argument("--skip-t4", action="store_true",
                        help="跳过 T4 LSTM 训练")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="输出目录")
    # ── 2026-10-11 新增：**持仓分配方式**（补"回测 ↔ 生产"口径缺口）──
    #   背景：生产模拟盘用 `min_variance`（最小方差、lookback 250、单票≤15%），
    #   而本脚本**从来没传过 sizing_method** ⇒ 引擎默认 `weight=1.0` = **等权** ✗
    #   ⇒ 于是 A/B/C/V6-* **全部是"等权"口径，与生产不是同一个策略**。
    #   实测：生产 2026-10-01 那批权重在 **0.21 ~ 1.5**（差 7 倍）= 非等权；
    #   而等权 TOP10 下"前 2 只占 89% 盈亏"（待办 #46）—— 生产有单票上限压着。
    #   ★ 默认 "" ⇒ **行为与以前逐位不变**（历史结果仍可比）。
    parser.add_argument("--sizing-method", type=str, default="",
                        help="持仓分配方式：equal/ic_weighted/inv_var/risk_parity/"
                             "min_variance；空=引擎默认（等权）=历史行为")
    parser.add_argument("--sizing-lookback", type=int, default=0,
                        help="协方差估计窗口（交易日）；0=引擎默认(120)。生产用 **250**")
    parser.add_argument("--match-production-sizing", action="store_true",
                        help="★ 直接读**生产配置**（sequoia_x/simulation/config.py 的 "
                             "BUY_WEIGHT_METHOD/PARAMS）⇒ 保证回测↔生产口径一致，"
                             "且不会因手抄参数而漂移")
    # ── 2026-10-13 新增：**单票上限覆盖**（"收紧单票上限"实验用；默认 0 = 不改 ⇒ 行为逐位不变）──
    #   背景：定位决策 C（低回撤持有）⇒ 主战场是组合/风控层，判据是**构造性**的
    #   （前2只占比↓、剔最赚N只后收益↑、回撤↓），不靠统计显著性（MDE ≈1.5%/月，效应量测不出）。
    #   ⚠️ 与 TOP_N 的耦合：cap < 1/TOP_N 时组合**装不满**（10 只 × cap 5% = 最多 50% 仓位 ⇒ 留现金）
    #      ⇒ 低于 1/TOP_N 的 cap 会引入现金拖累，解读时要区分"分散化收益"与"减仓效应"。
    parser.add_argument("--sizing-cap", type=float, default=0.0,
                        help="单票权重上限（如 0.10）；0 = 用引擎/生产默认（生产为 0.15）")
    args = parser.parse_args()

    # 解析持仓分配方式（★ 默认空 ⇒ 不动 ⇒ 历史行为不变）
    _sizing_method, _sizing_params = None, {}
    if args.match_production_sizing:
        try:
            from sequoia_x.simulation.config import BUY_WEIGHT_METHOD, BUY_WEIGHT_PARAMS
            _sizing_method = BUY_WEIGHT_METHOD or None
            _sizing_params = dict(BUY_WEIGHT_PARAMS or {})
            logger.info(f"持仓分配 = **对齐生产**: {_sizing_method} {_sizing_params}")
        except Exception as _e:
            logger.error(f"❌ --match-production-sizing 读取生产配置失败: {_e}")
            return 1
    elif args.sizing_method:
        _sizing_method = args.sizing_method
        _sizing_params = {"lookback": args.sizing_lookback} if args.sizing_lookback else {}
        logger.info(f"持仓分配 = {_sizing_method} {_sizing_params or '(引擎默认参数)'}")
    if _sizing_method is None:
        logger.info("持仓分配 = 引擎默认（**等权**，与历史 A/B 结果同口径）")
    if args.sizing_cap > 0:
        if _sizing_method is None:
            logger.error("❌ --sizing-cap 只在指定了分配方式（含 --match-production-sizing）时才有意义")
            return 1
        _sizing_params["cap"] = float(args.sizing_cap)
        logger.info(f"★ 单票上限被覆盖为 {args.sizing_cap:.1%}（其余参数不变）")

    # 重绑模块级 OUTPUT_DIR（三个保存函数都在调用时读它）→ --output-dir 生效
    OUTPUT_DIR = Path(args.output_dir)
    logger.info(f"输出目录: {OUTPUT_DIR}")

    # 确定运行范围
    if args.all:
        top_n_list = TOP_N_LIST
        # 2026-08-20: --period full 用完整 70 月段（V4 129维 与旧 121维 70月 同口径对比）
        periods = FULL_PERIODS if args.period == "full" else PERIODS
        modes = RISK_MODES
    elif args.top_n and args.mode:
        top_n_list = [args.top_n]
        # 2026-09-27 修：单组合分支原先**静默忽略 `--period full`** ——
        #   原写法只在 PERIODS（2025年/2026年/全周期11月）里按子串匹配，
        #   "full" 匹配不到 → `or PERIODS` 回退成**跑全部短时段**（约 11 个月而非 70 个月）。
        #   这也是记忆「V4回测70月口径铁律」的坑源之一。现在显式支持 full。
        if args.period == "full":
            periods = FULL_PERIODS
        else:
            periods = PERIODS if not args.period else [
                (p[0], p[1], p[2]) for p in PERIODS if args.period in p[0]
            ] or PERIODS
        modes = [(args.mode, args.mode)]
    else:
        parser.print_help()
        return

    total_groups = len(top_n_list) * len(periods) * len(modes)
    logger.info(f"\n{'='*60}")
    logger.info(f"共享预测缓存回测")
    logger.info(f"TOP_N: {top_n_list}")
    logger.info(f"时段: {[p[0] for p in periods]}")
    logger.info(f"风控: {[m[0] for m in modes]}")
    logger.info(f"总计: {total_groups} 组")
    logger.info(f"{'='*60}")

    # 初始化
    cfg = get_config()
    engine = DataEngine(Settings())

    # 收集所有需要的月份
    all_months_set = set()
    for _, p_start, p_end in periods:
        all_months_set.update(get_test_months(p_start, p_end))
    test_months = sorted(all_months_set)

    if args.months > 0:
        test_months = test_months[:args.months]

    # Phase 1: 加载/构建预测缓存
    cache = load_or_build_cache(
        cfg, engine, Path(args.cache), test_months,
        max_stocks=args.max_stocks,
        force_rebuild=args.rebuild_cache,
        skip_t4=args.skip_t4,
    )

    actual_months = sorted(set(test_months) & set(cache.keys()))
    if not actual_months:
        logger.error("预测缓存为空！请先运行 build_prediction_cache.py")
        return

    logger.info(f"缓存就绪: {len(actual_months)} 个月 ({actual_months[0]}~{actual_months[-1]})")

    # Phase 2: 运行所有组合
    all_results = []
    t_total = time.time()
    group_idx = 0

    for top_n in top_n_list:
        for period_name, start_month, end_month in periods:
            for mode_name, mode_desc in modes:
                group_idx += 1
                t_group = time.time()

                # 过滤该时段内的月份
                period_months = [m for m in get_test_months(start_month, end_month)
                                 if m in cache]

                logger.info(f"\n{'─'*50}")
                logger.info(f"[{group_idx}/{total_groups}] TOP_N={top_n} "
                            f"{period_name}({len(period_months)}月) {mode_name}({mode_desc})")

                try:
                    metrics = run_single_group(
                        cfg, engine, cache, top_n, mode_name,
                        start_month, end_month,
                        sizing_method=_sizing_method,
                        sizing_params=_sizing_params,
                    )
                except Exception as e:
                    logger.error(f"  ❌ 失败: {e}", exc_info=True)
                    metrics = {"total_return": None, "sharpe": None,
                               "max_drawdown": None, "error": str(e)}

                elapsed = time.time() - t_group
                row = {
                    # 2026-09-27 补：把逐月收益序列挂进汇总行 —— save_monthly_matrix 读的正是
                    #   这两个下划线键，但原先组装 row 时没带上 ⇒ `monthly_returns.csv`
                    #   **从未生成过**（静默 return），导致"回测差异是否显著"无法检验
                    #   （09-27 的 tail/random A/B 只能靠 σ 估算，做不了配对检验）。
                    "_monthly_labels": metrics.get("_monthly_labels", []),
                    "_monthly_returns": metrics.get("monthly_returns", []),
                    "风控模式": mode_name,
                    "TOP_N": top_n,
                    "时段": period_name,
                    "月数": metrics.get("n_months", 0),
                    "总收益率": metrics.get("total_return"),
                    "年化收益率": metrics.get("annual_return"),
                    "夏普比率": metrics.get("sharpe"),
                    "最大回撤": metrics.get("max_drawdown"),
                    "月胜率": metrics.get("win_rate"),
                    "交易笔数": metrics.get("n_trades", 0),
                    "月均换手率": metrics.get("avg_turnover", 0),
                    "胜率(笔)": metrics.get("win_trade_pct", 0),
                    "终值": metrics.get("final_value", 0),
                    # ── 2026-10-11 补：**抗极端指标**（说明见 save_summary_csv 的 fieldnames）──
                    #   ⚠️ 这里才是"最后一公里"：引擎早就算出了这些字段，
                    #   但**本行是逐字段手工映射的** ⇒ 不写进来 = 值被丢掉。
                    #   （只往 `fieldnames` 加列名**不够** —— 2026-10-11 实测踩过：
                    #     表头有了 6 列、**值全空** ✗）
                    "return_ex_top1": metrics.get("return_ex_top1", 0),
                    "return_ex_top2": metrics.get("return_ex_top2", 0),
                    "return_ex_top3": metrics.get("return_ex_top3", 0),
                    "top2_pnl_share": metrics.get("top2_pnl_share", 0),
                    "median_trade_pct": metrics.get("median_trade_pct", 0),
                    "median_monthly_return": metrics.get("median_monthly_return", 0),
                }
                all_results.append(row)

                ret_str = f"{metrics.get('total_return', 0):+.2%}" \
                    if metrics.get('total_return') is not None else "N/A"
                logger.info(f"  ✅ 收益={ret_str} "
                            f"夏普={metrics.get('sharpe', 'N/A')} "
                            f"耗时={elapsed:.0f}s ({'缓存' if cache else '实时'})")

    # ── 2026-10-09（待办 #25）：显式告警"M1 是空转臂" ──
    #   M1(+T1过滤) 与 M0(裸融合) 结果**逐字段完全相同**：T1 实测 AUC=0.499（无预测能力），
    #   过滤按设计关闭（`use_t1` 恒为 False）⇒ M1 ≡ M0。
    #   危害有二：① 白跑 1/4 机时；② **汇总表里那行 "M1" 会让人误以为"T1 过滤被测过且无效"**。
    #   此处显式告警（不删该臂 —— 那会改变 2x 组数、破坏与历史结果的可比性）。
    try:
        _by = {(r["风控模式"], r["TOP_N"]): r for r in all_results}
        _dup = []
        for tn in {r["TOP_N"] for r in all_results}:
            if ("M1", tn) in _by and ("M0", tn) in _by:
                a = {k: v for k, v in _by[("M0", tn)].items() if k != "风控模式"}
                b = {k: v for k, v in _by[("M1", tn)].items() if k != "风控模式"}
                if a == b:
                    _dup.append(tn)
        if _dup:
            logger.warning(
                f"⚠️ M1(+T1过滤) 与 M0(裸融合) 在 TOP_N={sorted(_dup)} 上**逐字段完全相同** "
                f"⇒ M1 是**空转臂**（T1 实测 AUC=0.499 无预测能力、过滤按设计关闭，见待办 #25）。"
                f"**汇总表里的 M1 行不代表「T1 过滤被测试过」，勿据此下结论。**")
    except Exception:
        pass

    total_elapsed = time.time() - t_total

    # Phase 3: 保存结果
    save_summary_csv(all_results)
    save_monthly_matrix(all_results)
    optimal = compute_and_save_optimal(all_results)

    # ── 2026-10-11（待办 #47b）：把本次的【持仓分配口径】烙印到输出目录 + 显式提醒 ──
    #   为什么必须做：**结论会随口径翻转，不是小数点差别** ——
    #     同一臂只换分配方式：A 臂 51.82%→70.29%、B 臂 81.23%→39.74%（B−A 从 +26.72p 变 −29.78p）。
    #   事后靠翻日志反推"这份结果是哪个口径"代价很高（#47 就是这么反推出来的）⇒ 让产物自带口径。
    #   ★ 本段只写文件 + 打日志，**不碰任何计算结果** ⇒ 默认行为逐位不变。
    _scope = ("生产口径（直读生产配置）" if args.match_production_sizing
              else ("显式指定口径" if args.sizing_method else "等权（引擎默认）"))
    # ★ 月份必须取**实际跑到的月份**（= 汇总行里的 _monthly_labels），不能用 actual_months：
    #   `--months` 只截断"缓存完整性检查"用的月份，真正的回测月份由 `--period` 决定
    #   ⇒ 用 actual_months 会写出"2 个月"而实际跑了 6 个月（2026-10-11 实测踩到）。
    _labels = sorted({m for r in all_results for m in r.get("_monthly_labels", [])})
    manifest = {
        "sizing_method": _sizing_method or "equal",
        "sizing_params": _sizing_params,
        "scope": _scope,
        "is_production_scope": bool(args.match_production_sizing),
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cache": str(args.cache),
        "months": (f"{_labels[0]}~{_labels[-1]}（{len(_labels)} 个月）" if _labels
                   else "（无月度序列）"),
        "n_groups": len(all_results),
        "note": "结论随持仓分配口径会翻转 ⇒ 跨口径比总收益 = 比错东西；"
                "要生产口径请重跑并加 --match-production-sizing。",
    }
    (OUTPUT_DIR / "sizing_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"口径烙印: {OUTPUT_DIR}/sizing_manifest.json ← {_scope} "
                f"{_sizing_method or 'equal'} {_sizing_params or ''}")
    if not args.match_production_sizing:
        logger.warning(
            f"⚠️ 本次持仓分配 = **{_scope}**，与生产模拟盘口径可能不一致"
            f"（生产用最小方差 min_variance，见 sequoia_x/simulation/config.py）；"
            f"历史教训（待办 #47b）：同一臂只换分配方式，A 臂 51.82%→70.29%、B 臂 81.23%→39.74%，"
            f"**结论会翻转** ⇒ 需要生产口径时请加 --match-production-sizing 重跑。")

    logger.info(f"\n{'='*60}")
    logger.info(f"回测完成: {total_groups} 组, 总耗时={total_elapsed:.0f}s "
                f"({total_elapsed/60:.1f}min)")
    logger.info(f"{'='*60}")

    if optimal:
        print(f"\n🏆 最优配置:")
        print(f"  风控: {optimal.get('风控模式')}  TOP_N: {optimal.get('TOP_N')}  "
              f"时段: {optimal.get('时段')}")
        print(f"  年化: {optimal.get('年化收益率', 0):+.2%}  "
              f"夏普: {optimal.get('夏普比率', 0):.2f}  "
              f"回撤: {optimal.get('最大回撤', 0):+.2%}  "
              f"评分: {optimal.get('综合评分', 0):.4f}")


if __name__ == "__main__":
    main()
