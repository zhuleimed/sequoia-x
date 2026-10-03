#!/usr/bin/env python
"""卖出档位回放的稳健性检验（2026-10-03）——回答"B（仅 -12% 硬止损）是不是真的更好"。

背景
    `sell_policy_replay.py` 四臂回放中 B（仅 -12% 硬止损）在期末资产 / 回撤 / 夏普
    三项上都略优于现状 A（+0.63pp）。但"四选一里最好"可能只是运气。本脚本做三组检验：

  ① 分期一致性：把 3 个月拆成 7/8/9 月，看 B 的优势是否来自单月
  ② 参数敏感性：硬止损 -8/-10/-12/-15% + "完全不止损"（none）——若排名随阈值乱跳，
     则 B 的领先是刀刃上的噪声；若"少卖=更好"单调成立，则是**市场状态效应**而非止损本身
  ③ 块自助（5 日块 × 2000 次）：A 与 B 的日收益差序列重抽样 → 累计差的 95% 区间

  另附：各臂按月买入笔数（检验 B 是否靠"7 月后停止买入"取胜）与期末仓位/现金结构。

用法（铁律六：py312）
    /home/zhulei/anaconda3/envs/zhulei_py312/bin/python \
        experiments/llm_sim_diag/sell_policy_robustness.py
"""
from __future__ import annotations

import json
import sqlite3
import statistics
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))

# 复用回放框架（导入即完成日期补丁 + 推送屏蔽；不会跑它的 main）
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sell_policy_replay import ARMS, OUT_DIR, _equity, load_signals, run_arm, trading_days  # noqa: E402

import numpy as np  # noqa: E402

# 稳健性扩展臂：仅硬止损，阈值 -8/-10/-12/-15；以及"完全不止损"
EXTRA_ARMS: dict[str, dict] = {
    "S_-08":     {"groups": {"S"}, "hard_stop_loss": -0.08, "note": "仅硬止损 -8%"},
    "S_-10":     {"groups": {"S"}, "hard_stop_loss": -0.10, "note": "仅硬止损 -10%"},
    "S_-12":     {"groups": {"S"}, "hard_stop_loss": -0.12, "note": "仅硬止损 -12%（=B）"},
    "S_-15":     {"groups": {"S"}, "hard_stop_loss": -0.15, "note": "仅硬止损 -15%"},
    "NONE":      {"groups": None,  "sell_rules_mode": "none", "note": "完全不止损（纯持有）"},
}
ALL_ARMS = {**{k: ARMS[k] for k in ("A_all",)}, **EXTRA_ARMS}

BLOCK, N_BOOT, SEED = 5, 2000, 20261003


def monthly_returns(db: Path) -> dict[str, float]:
    """按月收益（用每月最后一个交易日的总资产算）。"""
    eq = _equity(db)
    if not eq:
        return {}
    by_month: dict[str, float] = {}
    for d, v in eq:
        by_month[d[:7]] = v          # 同月后出现的覆盖前面的 → 月末值
    months = sorted(by_month)
    out, prev = {}, eq[0][1]
    for m in months:
        out[m] = by_month[m] / prev - 1
        prev = by_month[m]
    return out


def buys_by_month(db: Path) -> dict[str, int]:
    with sqlite3.connect(db) as c:
        rows = c.execute(
            "SELECT substr(buy_date,1,7), COUNT(*) FROM sim_buy_signals "
            "WHERE status='executed' GROUP BY 1 ORDER BY 1").fetchall()
    return dict(rows)


def block_bootstrap(diff_log: np.ndarray, block: int = BLOCK, n: int = N_BOOT,
                    seed: int = SEED) -> dict:
    """对**对数日收益差**序列做块自助，返回累计差的分布。

    统计量取 Σ(ln(1+r_b) − ln(1+r_a))，其点估计恰等于 ln(1+ret_b) − ln(1+ret_a)，
    故自助分布的中心与点估计一致；输出换算为**相对超额** expm1(Σdiff) = (1+ret_b)/(1+ret_a) − 1。
    近似说明：按 5 日块重抽样以保留序列相关，未做块边界重叠校正。
    """
    rng = np.random.default_rng(seed)
    T = len(diff_log)
    n_blocks = int(np.ceil(T / block))
    starts = np.arange(0, max(1, T - block + 1))
    totals = np.empty(n)
    for i in range(n):
        pick = rng.choice(starts, size=n_blocks, replace=True)
        totals[i] = np.concatenate([diff_log[s:s + block] for s in pick])[:T].sum()
    return {"p2.5": float(np.expm1(np.percentile(totals, 2.5))),
            "p50": float(np.expm1(np.percentile(totals, 50))),
            "p97.5": float(np.expm1(np.percentile(totals, 97.5))),
            "share_gt_0": float((totals > 0).mean())}


def daily_returns(db: Path) -> dict[str, float]:
    eq = _equity(db)
    return {eq[i][0]: eq[i][1] / eq[i - 1][1] - 1 for i in range(1, len(eq)) if eq[i - 1][1] > 0}


def main() -> int:
    days = trading_days()
    signals = load_signals()
    print(f"稳健性检验：{days[0]} ~ {days[-1]}（{len(days)} 个交易日）| 臂 {list(ALL_ARMS)}")

    res = {}
    for name, cfg in ALL_ARMS.items():
        print(f"\n═══ {name} —— {cfg['note']} ═══")
        res[name] = run_arm(name, cfg, days, signals)

    # ── 汇总表 ─────────────────────────────────────────────
    print("\n" + "=" * 92)
    print("① 各臂总览")
    print("=" * 92)
    print(f"{'臂':<9}{'期末总资产':>12}{'总收益':>9}{'最大回撤':>9}{'夏普':>7}{'平仓':>6}{'已买':>6}")
    for name, r in res.items():
        if "error" in r:
            print(f"{name:<9}{r['error']}")
            continue
        print(f"{name:<9}{r['final_value']:>12,.0f}{r['total_return']:>8.2%}"
              f"{r['max_drawdown']:>9.2%}{r['sharpe']:>7.2f}{r['n_trades']:>6}{r['n_executed']:>6}")

    print("\n" + "=" * 92)
    print("② 分期收益（检验是否单月驱动）")
    print("=" * 92)
    months = sorted({m for name in res for m in monthly_returns(OUT_DIR / f"replay_{name}.db")})
    print(f"{'臂':<9}" + "".join(f"{m:>11}" for m in months) + f"{'7月后买入笔数':>14}")
    for name in res:
        mr = monthly_returns(OUT_DIR / f"replay_{name}.db")
        bb = buys_by_month(OUT_DIR / f"replay_{name}.db")
        after_jul = sum(v for k, v in bb.items() if k > "2026-07")
        print(f"{name:<9}" + "".join(f"{mr.get(m, float('nan')):>10.2%} " for m in months)
              + f"{after_jul:>14}")

    # ── ③ 块自助：候选 vs 现状 ──────────────────────────────
    print("\n" + "=" * 92)
    print("③ 块自助（5 日块 × 2000 次）：候选臂 − 现状A 的累计收益差分布")
    print("=" * 92)
    ra = daily_returns(OUT_DIR / "replay_A_all.db")
    print(f"{'对比':<14}{'点估计':>10}{'2.5%':>10}{'中位':>10}{'97.5%':>10}{'P(优于A)':>10}")
    boot_out = {}
    for name in res:
        if name == "A_all":
            continue
        rb = daily_returns(OUT_DIR / f"replay_{name}.db")
        common = sorted(set(ra) & set(rb))
        diff_log = np.array([np.log1p(rb[d]) - np.log1p(ra[d]) for d in common])
        # 点估计：相对超额（1+retB)/(1+retA)−1（与自助统计量同口径）
        point = (1 + res[name]["total_return"]) / (1 + res["A_all"]["total_return"]) - 1
        bt = block_bootstrap(diff_log)
        boot_out[name] = {**bt, "point": point}
        print(f"{name + ' vs A':<14}{point:>10.2%}{bt['p2.5']:>10.2%}{bt['p50']:>10.2%}"
              f"{bt['p97.5']:>10.2%}{bt['share_gt_0']:>10.2f}")

    with open(OUT_DIR / "sell_policy_robustness.json", "w") as f:
        json.dump({"arms": res, "bootstrap": boot_out,
                   "monthly": {n: monthly_returns(OUT_DIR / f"replay_{n}.db") for n in res}},
                  f, ensure_ascii=False, indent=2)
    print(f"\n已写入 {OUT_DIR / 'sell_policy_robustness.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
