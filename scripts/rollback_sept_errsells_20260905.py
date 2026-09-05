"""9/5 一次性回溯重建：把 9/4 月内规则误卖的同批 9 只"作废"，账本回到"9/2 满仓持有至今"。

（用户 2026-09-05 决议"回溯重建账户状态"，见 memory min-hold-bug-and-month-end-hold）
- 作废 9/4 误卖(closed id11-19, sell 09/02→09/04, 死叉/负夏普) → 该 9 笔 pnl 从 closed 抹去，
  使 closed.pnl 回到 8 月真实基(-7600.87)。
- 按 **9/2 原始进场价/股数/成本** 重建该 9 只持仓，与幸存 603099 并成满 10；
- 撤销 9/5 pending 补回信号(id40-48，为上一轮"9/5市价补回"方案留痕，现被本回溯取代)；
- 覆写 9/4 日结为"满仓10 纯持有"口径；9/2、9/3 已是纯持有行，不改。
结果即"自 9/2 满10持有，未发生过 9/4 误卖"，之后由 9 月 hard_stop_only(现已固化为 V4 长期)
持有到月末，不再被月内动量规则抛售。

只动 data/sim_v2.db(模拟账本)；不改主库、不发微信。幂等：前置校验没过即拒绝。
用法(铁律六 py312)：
  ... --dry-run   打印将执行的精确 SQL 与目标数字，不写库
  ... --yes       备份→单事务执行→终验
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
DB = _REPO / "data" / "sim_v2.db"
MAIN = _REPO / "data" / "sequoia_v2.db"

BUG_SYMS = ["601799", "603013", "603515", "603606", "603345",
            "601811", "600363", "600098", "603706"]      # 9/4 被误卖、需按 9/2 原价重建的 9 只

# 幸存 603099 从现持仓读(不动它除估值外)，故目标满仓 = 这 9 只 + 603099


def call(db, q, args=()):
    c = sqlite3.connect(db); c.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in c.execute(q, args)]
    finally:
        c.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()

    # ── 0. 前置校验(当前态必须匹配, 否则拒绝, 幂等) ──
    bug_closed = call(DB, "SELECT * FROM sim_closed_trades WHERE buy_date='2026-09-02' "
                          "AND sell_date='2026-09-04' AND exit_reason LIKE '%M(均线死叉)%'")
    assert len(bug_closed) == 9, f"[中止] 9/4 误卖 closed 应 9 条, 实际 {len(bug_closed)}"
    pend = call(DB, "SELECT * FROM sim_buy_signals WHERE status='pending' AND buy_date='2026-09-05'")
    assert len(pend) == 9, f"[中止] 9/5 pending 应 9 条, 实际 {len(pend)}"
    # 现持仓只应 603099
    pos = call(DB, "SELECT * FROM sim_positions")
    assert len(pos) == 1 and pos[0]["symbol"] == "603099", "[中止] 现持仓应为仅 603099"

    orig = {r["symbol"]: r for r in bug_closed}          # 9 只原始进场(buy_price/shares/total_cost)
    sv = call(DB, "SELECT buy_price,shares,total_cost FROM sim_positions WHERE symbol='603099'")[0]

    # 9/4 纯持有估值(满10), stock_value = Σ close*shares
    def closesum(date_):
        s = 0.0
        for sym in BUG_SYMS + ["603099"]:
            sh = orig[sym]["shares"] if sym in orig else sv["shares"]
            row = call(MAIN, "SELECT close FROM stock_daily WHERE symbol=? AND date=?", (sym, date_))
            assert row, f"缺 {sym} {date_} 收盘"
            s += row[0]["close"] * sh
        return round(s, 2)

    sv3_pure = closesum("2026-09-03")
    sv4_pure = closesum("2026-09-04")
    CASH_TARGET = 16538.67                                # 与 9/2/9/3 记账一致(满仓)
    total4 = round(CASH_TARGET + sv4_pure, 2)
    totalret4 = round(total4 / 1_000_000 - 1, 6)

    print("═══ 目标重建(校验) ═══")
    print(f"删 9/4 误卖 closed {len(bug_closed)} 条(ids={[r['id'] for r in bug_closed]}) → closed.pnl 回 8 月基")
    print(f"重建 9 只(9/2 原进场):")
    for sym in BUG_SYMS:
        o = orig[sym]
        print(f"  {sym}  @{o['buy_price']} × {o['shares']}  cost={o['total_cost']:.2f}")
    print(f"幸存 603099 cost={sv['total_cost']:.2f}")
    print(f"9/3 纯持有 sv={sv3_pure} | 9/4 纯持有 sv={sv4_pure} | total4={total4} totalret4={totalret4}")
    print(f"目标现金(导出)={CASH_TARGET} = 1,000,000 - Σtotalcost + closed.pnl")

    if a.dry_run:
        print("\n[dry-run] 未写库。--yes 执行(先备份)。")
        return
    if not a.yes:
        ap.error("需 --yes 执行(会改写生产账本, 先备份) 或 --dry-run")

    # ── 1. 备份 ──
    bak = DB.with_name(f"{DB.name}.bak_{time.strftime('%Y%m%d_%H%M%S')}_pre_rollback")
    shutil.copy2(DB, bak)
    print(f"\n[备份] {bak}")

    # ── 2. 单事务应用 ──
    sim = sqlite3.connect(DB)
    cur = sim.cursor()
    try:
        with sim:
            # 2a. 删 9/4 误卖 closed
            cur.execute("DELETE FROM sim_closed_trades WHERE buy_date='2026-09-02' "
                        "AND sell_date='2026-09-04' AND exit_reason LIKE '%M(均线死叉)%'")
            # 2b. 重建 9 只持仓(9/2 原进场): 与 insert_position 语义一致, today_opened=0(已持有)
            for sym in BUG_SYMS:
                o = orig[sym]
                cur.execute(
                    "INSERT INTO sim_positions(symbol,strategy_from,buy_date,buy_price,shares,"
                    "total_cost,highest_price,highest_value,current_price,current_value,"
                    "pnl,pnl_pct,hold_days,today_opened,signal_id,llm_override_count) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0)",
                    (sym, "V2", "2026-09-02", o["buy_price"], o["shares"], o["total_cost"],
                     o["buy_price"], o["buy_price"] * o["shares"],
                     o["buy_price"], o["buy_price"] * o["shares"], 0.0, 0.0, 3, 0, None))
            # 2c. 撤销 9/5 pending 补回信号
            cur.execute("DELETE FROM sim_buy_signals WHERE status='pending' AND buy_date='2026-09-05'")
            # 2d. 覆写 9/4 日结为"满仓10 纯持有"
            cur.execute(
                "UPDATE sim_account_daily SET cash=?, stock_value=?, total_value=?, "
                "daily_pnl=?, daily_pnl_pct=?, total_return=?, position_count=? WHERE date='2026-09-04'",
                (CASH_TARGET, sv4_pure, total4, round(sv4_pure - sv3_pure, 2),
                 round((sv4_pure - sv3_pure) / (CASH_TARGET + sv3_pure), 6), totalret4, 10))
    finally:
        sim.close()
    print("[已执行] 事务提交完成")

    # ── 3. 终验 ──
    npos = call(DB, "SELECT COUNT(*) n FROM sim_positions")[0]["n"]
    nclosed = call(DB, "SELECT COUNT(*) n FROM sim_closed_trades WHERE "
                       "buy_date='2026-09-02' AND sell_date='2026-09-04'")[0]["n"]
    npend = call(DB, "SELECT COUNT(*) n FROM sim_buy_signals WHERE status='pending'")[0]["n"]
    # 导出现金一致性
    pos_cost = call(DB, "SELECT COALESCE(SUM(total_cost),0) s FROM sim_positions")[0]["s"]
    closed_pnl = call(DB, "SELECT COALESCE(SUM(pnl),0) s FROM sim_closed_trades")[0]["s"]
    cash_derived = round(1_000_000 - pos_cost + closed_pnl, 2)
    print(f"[终验] positions={npos}(应10) | 残留 9/4 误卖 closed={nclosed}(应0) | pending={npend}(应0)")
    print(f"[终验] 导出现金={cash_derived} (应 16538.67) | closed.pnl={round(closed_pnl,2)} (应 -7600.87)")
    assert npos == 10 and nclosed == 0 and npend == 0, "终验未过"
    assert abs(cash_derived - 16538.67) < 0.01, f"现金导出不一致: {cash_derived}"
    print("[终验 ✅] 账本已回到 '9/2 满仓10', 现金/closeed 与 8 月基一致")


if __name__ == "__main__":
    main()
