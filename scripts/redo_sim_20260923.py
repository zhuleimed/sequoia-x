#!/usr/bin/env python
"""重做 2026-09-23 的 LLM 模拟盘 —— 替代 09-23 夜因数据错误产生的运行结果。

**背景（2026-09-24 事故）**
    09-23 18:10 的同步只写入了 1,643/5,222 只股票的当日行情（股票列表被误判退市所致，
    见 `_looks_like_mass_delisting`）。当晚模拟盘照常在残缺数据上跑完，产生 4 个错误动作：

        卖 002487 / 002980 / 002453  → **被取消**（"无今日行情，取消待卖出"）
        买 000756                    → **被取消**（"无今日行情数据"）
        09-23 日结                   → 17 只持仓中 9 只按**过期价**估值（总资产 946,711.68 失真）

    V4 模拟盘（sim_v2.db）当日 9 只持仓全有行情，结果正确，**不在本脚本处理范围内**。

**为什么要"补丁日期"**
    `SimEngine.run_daily()` 用 `date.today()`，无日期参数；今天跑会处理"今天"而非重做 09-23。
    本脚本在 import 应用模块**之前**把 `datetime.date` 换成一个 today() 恒返回 2026-09-23 的子类，
    使整条链路（engine/models/signals/strategy_summary + main.py 的函数内 `from datetime import date`）
    一致地认为"今天是 09-23"。（本路径无 `datetime.now()` 调用，故只需补 `date`。）

**回滚清单（只撤销"昨晚这一轮"造成的状态变更）**
    1. 恢复三笔被误取消的待卖标记（原因文本取自 logs/pipeline_20260922.log 的原始记录）
    2. 清掉 601330 的待卖标记 —— 它是**昨晚 09-23 这一步**才标记的（应 09-24 开盘执行），
       若不清，重跑会在 09-23 开盘就误卖；重跑时 Step 4 会重新推导
    3. `hold_days` 回滚 1 天 —— `_execute_pending_orders` 开头有
       `UPDATE sim_positions SET hold_days = hold_days + 1`，重跑会再加一次
    4. 恢复被误取消的买入信号 000756（id=116）
    5. 删除 09-23 由残缺数据产出的 pending 信号（600183 / 603217），交给选股重跑重建
    6. 删除 09-23 的账户日结行，由重跑 upsert 重建

**用法**
    python scripts/redo_sim_20260923.py --dry-run    # 只看将要改什么
    python scripts/redo_sim_20260923.py --rollback   # 执行回滚
    python scripts/redo_sim_20260923.py --rerun      # 补丁日期重跑 09-23 的模拟盘（含推送）
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import date as _REAL_DATE
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

DB = PROJ / "data/sequoia_v2.db"          # LLM 模拟盘（表与行情同库）
TARGET = "2026-09-23"
BACKUP = PROJ / "data/sequoia_v2.db.bak_20260924_pre_redo0923"

# 三笔被误取消的待卖：原因文本 = 09-22 收盘标记时的原文（logs/pipeline_20260922.log）
RESTORE_SELL = {
    "002487": "R(相对弱势)(60分); S(硬止损)(40分)",
    "002980": "硬止损(收盘触发): 成本128.39×0.92=118.12, 收盘113.10",
    "002453": "硬止损(收盘触发): 成本5.85×0.92=5.38, 收盘5.31",
}
CLEAR_SELL = ["601330"]                  # 昨晚本步标记的，须清掉由重跑重新推导
KEEP_HOLD_DAYS = {"600587"}              # 昨晚这一步新买入的，不在 +1 回滚范围
RESTORE_SIGNAL_IDS = [116]               # 000756 被误取消的买入
DELETE_SIGNAL_IDS = [118, 119]           # 09-23 残缺数据产出的 pending 信号


def _q(conn, sql, args=()):
    return conn.execute(sql, args).fetchall()


def show_state(conn, label: str) -> None:
    print(f"\n── {label} ──")
    n = _q(conn, "SELECT COUNT(*) FROM sim_positions")[0][0]
    cash = _q(conn, "SELECT cash FROM sim_account_daily WHERE date=?", (TARGET,))
    print(f"  持仓 {n} 只 | 09-23 日结行 {'存在' if cash else '不存在'}"
          + (f"（总资产 {_q(conn, 'SELECT total_value FROM sim_account_daily WHERE date=?', (TARGET,))[0][0]}）" if cash else ""))
    for s in list(RESTORE_SELL) + CLEAR_SELL:
        r = _q(conn, "SELECT hold_days, pending_sell_reason FROM sim_positions WHERE symbol=?", (s,))
        if r:
            print(f"  {s}: hold_days={r[0][0]} pending_sell={(r[0][1] or '无')[:40]}")
    for i in RESTORE_SIGNAL_IDS + DELETE_SIGNAL_IDS:
        r = _q(conn, "SELECT symbol, buy_date, status, cancel_reason FROM sim_buy_signals WHERE id=?", (i,))
        if r:
            print(f"  信号 id={i}: {r[0][0]} buy_date={r[0][1]} status={r[0][2]} cancel={r[0][3]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rollback", action="store_true")
    ap.add_argument("--rerun", action="store_true", help="重跑 09-23 的模拟盘（含日报推送）")
    ap.add_argument("--rerun-select", action="store_true",
                    help="重跑 09-23 的策略选股+LLM（含结果推送），重建 09-23 的买入信号")
    args = ap.parse_args()

    def _install_date_patch():
        """把 date.today() 钉在 2026-09-23。必须在 import 应用模块之前调用。"""
        import datetime as _dt

        class _PatchedDate(_REAL_DATE):
            @classmethod
            def today(cls):
                return cls(2026, 9, 23)

        _dt.date = _PatchedDate
        print(f"⏱️  日期补丁已装：date.today() → {_PatchedDate.today()}")

    if args.rerun_select:
        _install_date_patch()
        import runpy

        from dotenv import load_dotenv
        load_dotenv(PROJ / ".env")
        print("═══ 重跑 09-23 策略选股 + LLM（会推送结果）═══")
        sys.argv = [str(PROJ / "main.py")]      # 无附加参数 = 选股+LLM 模式
        runpy.run_path(str(PROJ / "main.py"), run_name="__main__")
        return 0

    if args.rerun:
        _install_date_patch()

        from dotenv import load_dotenv
        load_dotenv(PROJ / ".env")
        from sequoia_x.core.config import get_settings
        from sequoia_x.simulation.engine import SimEngine

        print("═══ 重跑 09-23 模拟盘（会推送日报）═══")
        sim = SimEngine(get_settings(), push_tag="LLM")
        result = sim.run_daily(push_report=True)
        print("结果:", result)
        return 0

    conn = sqlite3.connect(DB)
    show_state(conn, f"当前状态（目标日 {TARGET}）")

    if args.dry_run or not args.rollback:
        print("\n（干跑结束；加 --rollback 才真正改库）")
        conn.close()
        return 0

    shutil.copy2(DB, BACKUP)
    print(f"\n💾 已备份 → {BACKUP.name}")

    cur = conn.cursor()
    for sym, reason in RESTORE_SELL.items():
        cur.execute("UPDATE sim_positions SET pending_sell_reason=? WHERE symbol=?", (reason, sym))
        print(f"  ↺ 恢复待卖 {sym}（影响 {cur.rowcount} 行）")
    for sym in CLEAR_SELL:
        cur.execute("UPDATE sim_positions SET pending_sell_reason=NULL WHERE symbol=?", (sym,))
        print(f"  ✂️ 清除本步待卖 {sym}（影响 {cur.rowcount} 行）")
    placeholders = ",".join("?" for _ in KEEP_HOLD_DAYS)
    cur.execute(
        f"UPDATE sim_positions SET hold_days = hold_days - 1 "
        f"WHERE hold_days > 0 AND symbol NOT IN ({placeholders})", tuple(KEEP_HOLD_DAYS),
    )
    print(f"  ↺ hold_days 回滚 1 天（影响 {cur.rowcount} 行，跳过 {sorted(KEEP_HOLD_DAYS)}）")
    ids = ",".join(str(i) for i in RESTORE_SIGNAL_IDS)
    cur.execute(f"UPDATE sim_buy_signals SET status='pending', cancel_reason=NULL WHERE id IN ({ids})")
    print(f"  ↺ 恢复买入信号 id∈{RESTORE_SIGNAL_IDS}（影响 {cur.rowcount} 行）")
    ids = ",".join(str(i) for i in DELETE_SIGNAL_IDS)
    cur.execute(f"DELETE FROM sim_buy_signals WHERE id IN ({ids})")
    print(f"  🗑️ 删除 09-23 残缺信号 id∈{DELETE_SIGNAL_IDS}（影响 {cur.rowcount} 行）")
    cur.execute("DELETE FROM sim_account_daily WHERE date=?", (TARGET,))
    print(f"  🗑️ 删除 09-23 日结行（影响 {cur.rowcount} 行）")
    conn.commit()

    show_state(conn, "回滚后状态")
    conn.close()
    print("\n✅ 回滚完成。下一步：python scripts/redo_sim_20260923.py --rerun")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
