#!/usr/bin/env python
"""LLM 盘卖出风控档位 A/B 回放（2026-10-03）。

问题
    V2 盘已切 E1（hard_stop_only），LLM 盘仍是"全规则"。候选折中档：
    `hard_stop + 20日时间止损`（去掉 M/SH/R 动量规则）是否更优？

方法（忠实回放）
    用 LLM 盘**自己的历史买入信号**（主库 sim_buy_signals，122 条，2026-07-02~09-30）
    在**生产 SimEngine** 上逐日重放，仅切换卖出规则档位：
      A all              : 现状（全规则）            ← 必须与生产实际账户对拍验证
      B hard_stop_only   : 只留 -12% 硬止损（V2 的 E1 档，参考）
      C s_plus_d         : 硬止损 + 20日时间止损     ← 用户候选（去掉 T/M/SH/R）
      D s_t_d            : 硬止损 + 移动止盈 + 时间止损（去掉 M/SH/R，保留止盈）
    复用的是生产执行链（T+1、开盘价成交、双轨硬止损、LLM 同日推荐覆盖、6% 追高过滤后的
    真实信号流），差异只在卖出规则集合，故各臂可比。

判据（**预先声明**，先写再跑，避免事后找理由）
    候选档"更好" = ① 期末总资产更高 且 ② 最大回撤劣化 ≤ 2 个百分点。
    两者同时满足才建议切换；否则维持现状。

隔离与安全
    · 每臂独立临时库 experiments/llm_sim_diag/data/replay_<arm>.db（只含 sim_* 表）；
      行情仍读主库 settings.db_path，**不写主库 sim_* 表**
    · 全程 monkeypatch 掉微信推送（不会发消息）
    · patch datetime.date（同 scripts/redo_sim_20260923.py 手法）使引擎认为"今天是回放日"

用法（铁律六：必须 py312）
    /home/zhulei/anaconda3/envs/zhulei_py312/bin/python \
        experiments/llm_sim_diag/sell_policy_replay.py            # 跑全部四臂
    ... sell_policy_replay.py --arms A                # 只跑 A（用于对拍验证）
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import shutil
import sqlite3
import statistics
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))

MAIN_DB = PROJ / "data" / "sequoia_v2.db"                 # 行情 + 生产 LLM 账户（只读）
OUT_DIR = Path(__file__).resolve().parent / "data"
START, END = "2026-07-02", "2026-09-30"

# ── 日期补丁：必须在 import 应用模块之前替换 datetime.date（模块内 `from datetime import date`
#    在 import 时绑定，晚替换不生效）——同 scripts/redo_sim_20260923.py 手法
_REAL_DATE = _dt.date


class _FakeDate(_REAL_DATE):
    """today() 恒返回当前回放日；其余行为与 datetime.date 一致。"""

    _current: "_REAL_DATE" = _REAL_DATE(2026, 7, 2)

    @classmethod
    def today(cls) -> "_REAL_DATE":
        return cls._current


_dt.date = _FakeDate  # type: ignore[misc]

import sequoia_x.simulation.engine as _eng              # noqa: E402
from sequoia_x.core.config import get_settings          # noqa: E402
from sequoia_x.core.logger import get_logger            # noqa: E402
from sequoia_x.simulation import rules as _rules        # noqa: E402
from sequoia_x.simulation.engine import SimEngine       # noqa: E402
from sequoia_x.simulation.models import init_sim_tables  # noqa: E402

logger = get_logger(__name__)

# 四臂：名称 → 配置（groups=None 表示全规则；S 硬止损恒开，见 rules.evaluate_exit）
#   hard_stop_loss / sell_rules_mode 为 None 时用生产默认（-12% 硬止损、全规则）
ARMS: dict[str, dict] = {
    "A_all":          {"groups": None,            "note": "现状：全规则（硬止损/移动止盈/时间/死叉/夏普/相对弱势）"},
    "B_hard_only":    {"groups": {"S"},           "note": "只留 -12% 硬止损（V2 的 E1 档）"},
    "C_s_plus_d":     {"groups": {"S", "D"},      "note": "硬止损 + 20日时间止损（去掉 移动止盈/M/SH/R）"},
    "D_s_t_d":        {"groups": {"S", "T", "D"}, "note": "硬止损 + 移动止盈 + 时间止损（去掉 M/SH/R）"},
}

# ── 推送屏蔽（绝不发微信）────────────────────────────────
_eng.push_trade_report = lambda *a, **k: None
_eng.push_sim_alert = lambda *a, **k: None


def trading_days() -> list[str]:
    """回放区间内的交易日（取 index_daily 的日期序列，权威且轻量）。"""
    with sqlite3.connect(MAIN_DB) as c:
        rows = c.execute(
            "SELECT DISTINCT date FROM index_daily WHERE date BETWEEN ? AND ? ORDER BY date",
            (START, END),
        ).fetchall()
    return [r[0] for r in rows]


def load_signals() -> list[dict]:
    """主库里的 LLM 历史买入信号（原样：symbol/strategy_from/llm_score/buy_date）。"""
    with sqlite3.connect(MAIN_DB) as c:
        c.row_factory = sqlite3.Row
        rows = c.execute(
            "SELECT symbol, strategy_from, llm_score, buy_date FROM sim_buy_signals ORDER BY id"
        ).fetchall()
    return [dict(r) for r in rows]


def build_arm_db(arm: str, signals: list[dict]) -> Path:
    """建一个干净的临时 sim 库：只含 sim_* 表 + 预置历史信号（status=pending）。"""
    db = OUT_DIR / f"replay_{arm}.db"
    if db.exists():
        db.unlink()
    init_sim_tables(str(db))
    with sqlite3.connect(db) as c:
        c.executemany(
            "INSERT INTO sim_buy_signals (symbol, strategy_from, llm_score, buy_date, status) "
            "VALUES (?, ?, ?, ?, 'pending')",
            [(s["symbol"], s["strategy_from"], s["llm_score"], s["buy_date"]) for s in signals],
        )
        c.commit()
    return db


def run_arm(arm: str, cfg: dict, days: list[str], signals: list[dict]) -> dict:
    """按日回放一个档位，返回账户曲线与交易统计。

    cfg 键：groups（规则组，None=全部）/ hard_stop_loss（覆盖硬止损，None=生产 -12%）
            / sell_rules_mode（"all"/"none"/"hard_stop_only"，None="all"）
    """
    from functools import partial

    db = build_arm_db(arm, signals)
    settings = get_settings()
    # 引擎：行情读主库（settings.db_path），模拟盘状态写临时库（db_path）
    sim = SimEngine(settings, db_path=str(db), push_tag="",
                    sell_rules_mode=cfg.get("sell_rules_mode") or "all")
    # 仅替换卖出规则集合 / 硬止损阈值（None=按默认，行为与生产一致）
    _eng.evaluate_exit = partial(_rules.evaluate_exit,
                                 rule_groups=cfg.get("groups"),
                                 hard_stop_loss=cfg.get("hard_stop_loss"))

    for i, d in enumerate(days, 1):
        _FakeDate._current = _REAL_DATE.fromisoformat(d)
        try:
            sim.run_daily(push_report=False)
        except Exception as e:  # 单日异常不中断回放，记录后继续（与生产"当日失败不影响次日"一致）
            logger.warning(f"[{arm}] {d} run_daily 异常: {e}")
        if i % 10 == 0 or i == len(days):
            v = _equity(db)
            print(f"  [{arm}] {i}/{len(days)} {d} 总资产 {v[-1][1]:,.0f}" if v else f"  [{arm}] {i}/{len(days)} {d}")

    return summarize(arm, db)


def _equity(db: Path) -> list[tuple[str, float]]:
    with sqlite3.connect(db) as c:
        return c.execute("SELECT date, total_value FROM sim_account_daily ORDER BY date").fetchall()


def summarize(arm: str, db: Path) -> dict:
    """期末总资产 / 最大回撤 / 夏普 / 交易统计。"""
    eq = _equity(db)
    if not eq:
        return {"arm": arm, "error": "无日结数据"}
    values = [v for _, v in eq]
    peak, mdd = values[0], 0.0
    for v in values:
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1.0)
    rets = [values[i] / values[i - 1] - 1 for i in range(1, len(values)) if values[i - 1] > 0]
    sharpe = (statistics.mean(rets) / statistics.stdev(rets) * (252 ** 0.5)) if len(rets) > 2 and statistics.stdev(rets) > 0 else float("nan")
    with sqlite3.connect(db) as c:
        n_trades, avg_pnl = c.execute(
            "SELECT COUNT(*), AVG(pnl_pct) FROM sim_closed_trades").fetchone()
        n_exec = c.execute("SELECT COUNT(*) FROM sim_buy_signals WHERE status='executed'").fetchone()[0]
        n_skip = c.execute("SELECT COUNT(*) FROM sim_buy_signals WHERE status!='executed'").fetchone()[0]
        by_cat = c.execute(
            "SELECT substr(exit_reason,1,6), COUNT(*) FROM sim_closed_trades GROUP BY 1 ORDER BY 2 DESC"
        ).fetchall()
    return {
        "arm": arm, "end_date": eq[-1][0], "final_value": values[-1],
        "total_return": values[-1] / values[0] - 1, "max_drawdown": mdd,
        "sharpe": sharpe, "n_trades": n_trades, "avg_pnl_pct": avg_pnl,
        "n_executed": n_exec, "n_skipped": n_skip, "exit_mix": by_cat,
    }


def actual_account() -> dict:
    """生产 LLM 账户实际结果（主库，只读）——用于 A 臂对拍。"""
    with sqlite3.connect(MAIN_DB) as c:
        eq = c.execute("SELECT date, total_value FROM sim_account_daily ORDER BY date").fetchall()
        n_trades, avg_pnl = c.execute("SELECT COUNT(*), AVG(pnl_pct) FROM sim_closed_trades").fetchone()
    return {"final_value": eq[-1][1], "end_date": eq[-1][0], "n_trades": n_trades, "avg_pnl_pct": avg_pnl}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A,B,C,D", help="要跑的臂，逗号分隔（键首字母）：A/B/C/D")
    args = ap.parse_args()

    days = trading_days()
    signals = load_signals()
    print(f"回放区间 {days[0]} ~ {days[-1]}（{len(days)} 个交易日）| 信号 {len(signals)} 条")
    print(f"临时库目录: {OUT_DIR}")

    wanted = {a.strip().upper() for a in args.arms.split(",") if a.strip()}
    key_map = {name[0]: name for name in ARMS}
    results = []
    for k in sorted(wanted):
        name = key_map.get(k)
        if name is None:
            print(f"未知臂: {k}")
            continue
        print(f"\n═══ 臂 {name} —— {ARMS[name]['note']} ═══")
        results.append(run_arm(name, ARMS[name], days, signals))

    # ── 汇总 ───────────────────────────────────────────────
    print("\n" + "=" * 96)
    print("回放结果汇总（判据：期末总资产更高 且 最大回撤劣化 ≤2pp）")
    print("=" * 96)
    print(f"{'臂':<14}{'期末总资产':>12}{'总收益':>9}{'最大回撤':>9}{'夏普':>7}"
          f"{'平仓笔数':>8}{'已买/跳过':>11}")
    for r in results:
        if "error" in r:
            print(f"{r['arm']:<14}{r['error']}")
            continue
        print(f"{r['arm']:<14}{r['final_value']:>12,.0f}{r['total_return']:>8.2%}"
              f"{r['max_drawdown']:>9.2%}{r['sharpe']:>7.2f}{r['n_trades']:>8}"
              f"{str(r['n_executed']) + '/' + str(r['n_skipped']):>11}")

    act = actual_account()
    print(f"\n对拍基准（生产实际）: 期末 {act['final_value']:,.2f}（{act['end_date']}）"
          f" | 平仓 {act['n_trades']} 笔 | 均值 {act['avg_pnl_pct']:.2%}")
    arm_a = next((r for r in results if r["arm"] == "A_all" and "error" not in r), None)
    if arm_a:
        diff = (arm_a["final_value"] - act["final_value"]) / act["final_value"]
        print(f"A 臂对拍偏差: {arm_a['final_value']:,.2f} vs {act['final_value']:,.2f} = {diff:+.2%}"
              f"  {'✅ 忠实（|偏差|<2%）' if abs(diff) < 0.02 else '⚠ 偏差偏大，结论需谨慎'}")

    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / "sell_policy_replay_summary.json", "w") as f:
        json.dump({"days": [days[0], days[-1]], "n_signals": len(signals),
                   "arms": results, "actual": act}, f, ensure_ascii=False, indent=2)
    print(f"\n汇总已写入 {OUT_DIR / 'sell_policy_replay_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
