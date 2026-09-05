"""⚠️ 已废弃（2026-09-05 后改用 rollback_sept_errsells_20260905.py 回溯重建）。
本"按 9/5 市价补回"路径已被 superseded：用户选定**回溯到 9/2 原始进场**而非 9/5 二次市价买入，
rollback_sept_errsells_20260905.py 已把误卖 9 只按原价重建满仓并清空本脚本所产生的 9/5 pending 信号。
请勿再运行本文件（否则会二次登记 9/5 信号造成重复）。

原功能（仅供历史参考）：9/5 一次性人工补回，把 9/4 被月内规则误卖的 9 只重新补回 V2 模拟盘。
  - 9/2 V4 月度换仓买满 TOP10；9/3 收盘规则(读买入前历史→死叉/负夏普即时触发)误把 9/10 标记待卖、
    9/4 开盘清出，账户只剩 603099 + ~89万现金闲置到月末。
  - 用户决策（见 memory min-hold-bug-and-month-end-hold / plan）：
      ① 9月现盘按"纯持有"人工续持 → 补回这 9 只回满 10 仓；
      ② V2 盘月内规则卖出冻结到 9/30（v2_simulation_daily.py 内已日期窗自动生效，仅留 -8% 硬止损）；
      ③ 保留 -8% 硬止损作为下行护栏。
  - 数据源：stock_daily 到 9/4。此处仅“登记买入信号(buy_date=今日9/5)”，
    实际以今日 9/5 OPEN 成交需等今晚 18:50 的 v2_simulation_daily（pipeline 先 18:10 sync 入 9/5 OHLC）。

用法（铁律六 py312）：
  /home/zhulei/anaconda3/envs/zhulei_py312/bin/python scripts/manual_sept_refill_20260905.py
  默认只打印计划(dry-run)；加 --commit 才真正写入 sim_buy_signals。

校验目标：commit 后今晚 run 成交 → sim_positions=10 只满仓、cash≈0(除100股整手尾差)。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import sqlite3
from sequoia_x.simulation.signals import submit_buy_signals

SIM_V2_DB = "data/sim_v2.db"
# 9/4 误卖、需补回的 9 只（第 10 只 603099 仍在仓）。Order 同 sim_buy_signals id 30-39 原序回退名单。
REFILL = ["601799", "603013", "603515", "603606", "603345",
          "601811", "600363", "600098", "603706"]
STRATEGY = "V2"


def main() -> None:
    ap = argparse.ArgumentParser(description="9月补回被误卖9只（默认 dry-run）")
    ap.add_argument("--commit", action="store_true", help="真正写信号（默认只打印计划）")
    args = ap.parse_args()

    c = sqlite3.connect(SIM_V2_DB)
    c.row_factory = sqlite3.Row
    held = {r["symbol"] for r in c.execute("SELECT symbol FROM sim_positions")}
    cash = c.execute("SELECT cash FROM sim_account_daily ORDER BY id DESC LIMIT 1").fetchone()["cash"]
    n_pending = c.execute("SELECT COUNT(*) FROM sim_buy_signals WHERE status='pending'").fetchone()[0]
    c.close()

    slots = 10 - len(held)
    print(f"账户现状: 持仓 {len(held)} 只 {sorted(held)} | 现金 {cash:,.0f} | 待执行信号 {n_pending}")
    print(f"可补空位: {slots} | 等权预算约 {cash / max(slots,1):,.0f}/只（>=满仓则按现金/槽均摊）")
    dup = [s for s in REFILL if s in held]
    if dup:
        print(f"[中止] 以下已在持仓，跳过以避免重复买入: {dup}")
    print(f"待补回 9 只: {REFILL}")

    if not args.commit:
        print("\n[dry-run] 未写库。确认无误后加 --commit 执行。")
        return

    n = submit_buy_signals(
        db_path=SIM_V2_DB, symbols=REFILL, strategy_name=STRATEGY, top_n=len(REFILL))
    print(f"[commit] 已登记买入信号 {n} 只（buy_date=今日）。")
    print("今晚 v2_simulation_daily 将以上证今日 OPEN 价成交；成交后应满仓 10 只。")


if __name__ == "__main__":
    main()
