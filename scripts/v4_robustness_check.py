#!/usr/bin/env python3
"""V4 去前视 70月 绝对收益稳健性检验（2026-08-21）。

目的: 判断 V4(129维) M4_TOP10 70月年化 73.4% 是否含幸存者偏差/被暴涨股虚增。
方法:
  1. 用与 run_shared_backtest 同源的 MonthlyBacktestEngine + 去前视预测缓存, 重跑 M4_TOP10 70月
  2. 提取 monthly_returns / trades, 分析单月收益分布, 定位异常月份/暴涨股
  3. 稳健性: 去掉收益最高的 K 个月 与 单笔收益最大的个股, 重算年化, 看 73.4% 是否塌方
  4. 输出 data/v4_robustness_check.json

遵守铁律: 用系统时间(date)做时长分析; 只读数据/不写生产; py312 运行。
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from sequoia_x.core.config import Settings
from sequoia_x.data.engine import DataEngine
from sequoia_x.model_selection_v2.config import get_config
from sequoia_x.model_selection_v2.backtest.monthly_engine import MonthlyBacktestEngine

PRED_CACHE = PROJ / "output/backtest_v2/prediction_cache_v4_nolook.json"
OUT = PROJ / "data" / "v4_robustness_check.json"


def metric_year_from_monthly(monthly, n_days, initial):
    """复刻 _compute_final_metrics 的年化/夏普口径(按交易日)."""
    values = []
    v = float(initial)
    values.append(v)
    for r in monthly:
        v *= (1 + r)
        values.append(v)
    values = np.array(values)
    total = values[-1] / initial - 1
    # 年化: daily_records 有 n_days 个, values 有 n_days+1 个点. 用 n_days 交易日年化近似.
    # _compute_final_metrics 用 252/n_days; 这里我们没有逐日序列, 用每月序列近似月年化.
    n_months = len(monthly)
    # 用月收益算年化(月离散): (1+monthly)累计 → 按 12/n_months 年化
    annual = float(prod_monthly(monthly, n_months))
    return total, annual


def prod_monthly(monthly, n_months):
    total = np.prod([1 + r for r in monthly]) - 1
    if n_months <= 0:
        return 0.0
    return float((1 + total) ** (12 / n_months) - 1)


def main():
    t0 = time.time()
    print(f"[{datetime.now():%H:%M:%S}] 基准系统时间 | 加载去前视预测缓存...", flush=True)
    cache = json.loads(PRED_CACHE.read_text())

    cfg = get_config()
    engine = DataEngine(Settings())

    top_n, mode, start, end = 10, "M4", "2020-09", "2026-06"
    bt = MonthlyBacktestEngine(
        cfg=cfg, engine=engine, top_n=top_n, risk_mode=mode,
        initial_capital=500_000.0, use_real_t4=False, prediction_cache=cache,
    )
    print(f"[{datetime.now():%H:%M:%S}] 重跑 M4_TOP10 {start}~{end} (70月)...", flush=True)
    metrics = bt.run(start, end)
    monthly = metrics.get("monthly_returns", [])
    labels = bt.monthly_labels
    trades = metrics.get("trades", [])
    print(f"[{datetime.now():%H:%M:%S}] 完成: 月数={len(monthly)} 交易={len(trades)} "
          f"年化={metrics.get('annual_return')} 夏普={metrics.get('sharpe')}", flush=True)

    rep = {
        "generated": str(datetime.now()),
        "config": {"top_n": top_n, "risk_mode": mode, "range": f"{start}~{end}"},
        "metrics": {k: metrics.get(k) for k in
                    ("total_return", "annual_return", "sharpe", "max_drawdown", "win_rate",
                     "n_months", "n_trades")},
        "monthly_returns": monthly,
        "monthly_labels": labels,
    }

    # ── 1. 单月收益分布 ──
    mr = np.array(monthly)
    rep["monthly_stats"] = {
        "n": len(mr),
        "mean_monthly": float(mr.mean()),
        "best_month": {"label": labels[int(np.argmax(mr))], "ret": float(mr.max())},
        "worst_month": {"label": labels[int(np.argmin(mr))], "ret": float(mr.min())},
        "top5_months": [{"label": labels[i], "ret": float(mr[i])}
                        for i in np.argsort(-mr)[:5]],
    }

    # ── 2. 稳健性: 去掉收益最高 K 个月 重算月年化 ──
    print("\n══ 稳健性: 去掉收益最高 K 个月 ══", flush=True)
    strip_table = []
    for K in (0, 1, 3, 6):
        trimmed = np.sort(mr)[0:len(mr) - K] if K > 0 else mr
        n_trim = len(trimmed)
        annual = prod_monthly([float(r) for r in trimmed], n_trim)
        total = np.prod([1 + float(r) for r in trimmed]) - 1
        strip_table.append({"K_months_removed": K, "n_remain": n_trim,
                            "total": round(float(total), 3),
                            "annual_pct": round(annual * 100, 2)})
        print(f"  去 {K} 月: 剩余{n_trim}月 总收益={total:+.2%} 月化年化={annual:+.2%}", flush=True)
    rep["robustness_remove_months"] = strip_table

    # ── 3. 个股贡献: 每笔卖出的 pnl, 找贡献最大的股票 ──
    print("\n══ 单笔最大盈利交易 (可能的暴涨股/幸存者偏差) ══", flush=True)
    # trades 是 dict 列表, 字段为 type/symbol/date/price/pnl (type: buy|sell)
    def _tattr(t, n):
        if isinstance(t, dict):
            return t.get(n) or t.get("type") if n == "trade_type" else t.get(n)
        return getattr(t, n, None)
    def _typ(t):
        return t.get("type") if isinstance(t, dict) else getattr(t, "type", None)

    sells = [t for t in trades if _typ(t) == "sell"]
    sells_sorted = sorted(sells, key=lambda t: -(t.get("pnl") or 0))
    top_sells = []
    for t in sells_sorted[:10]:
        rec = {"symbol": t.get("symbol"), "date": t.get("date"),
               "pnl": t.get("pnl"), "price": t.get("price"),
               "return_pct": (t.get("pnl") / (t.get("price", 1) * t.get("shares", 1))) * 100
                            if t.get("price") and t.get("shares") else None}
        top_sells.append(rec)
        ret = f" ({rec['return_pct']:+.0f}%)" if rec['return_pct'] is not None else ""
        print(f"  {rec['symbol']} {rec['date']} pnl={rec['pnl']:,.0f}{ret} "
              f"(价{rec['price']})", flush=True)
    rep["top_pnl_sells"] = top_sells

    # ── 4. 去重股票后的集中度: 同一股票是否反复贡献 ──
    from collections import Counter
    sym_counts = Counter(_tattr(t, "symbol") for t in sells if _tattr(t, "pnl") > 0)
    # 重复盈利股: 若很多大盈利来自少数常客
    rep["win_sell_symbol_freq_top10"] = sym_counts.most_common(10)

    print(f"\n[{datetime.now():%H:%M:%S}] 写结果 -> {OUT}", flush=True)
    OUT.write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str))
    print(f"[{datetime.now():%H:%M:%S}] 总耗时 {(time.time()-t0)/60:.1f}min (系统时间)", flush=True)


if __name__ == "__main__":
    main()
