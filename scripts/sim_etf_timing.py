#!/usr/bin/env python3
"""ETF 择时模拟盘（2026-08-09, 独立策略, 100 万初始）

策略: Kronos 零样本 base 预测中证1000(000852) 未来 5 日收益,
信号 = pred_ret5 > -0.25%（阈值校正, V3 §9.9 验证: 6 年超额 +172pp）→ 持有 512100 / 空仓。

流程（pipeline 每日 18:10 调用, 复用 DataSync.is_trade_day 交易日开关, 自动跳过节假日）:
  周最后交易日（今天开盘 且 明天非交易日, 天然涵盖周五/节假日前一天）:
    ① 推理: Kronos 预测今日(ref) 未来 5 日收益（实测 ~1min, 1 进程 3 线程）
    ② 信号: pred_ret5 > -0.25% → 看多; 否则看空
    ③ 指令: 空仓+看多→BUY | 持仓+看空→SELL | 持仓+看多→HOLD（写 orders, pending）
    ④ 微信推送信号
  其它交易日:
    ① 执行 pending 指令（今日开盘价, 等价 T+1 开盘成交; 开盘价 18:10 已知）
    ② 估值（腾讯实时接口最新价; 回退 019 库今日/昨收）
    ③ 写 account_daily + 周报推送（每周五）

费率: 双边 0.05%（佣金 0.03% + 滑点 0.02%, ETF 无印花税）; 空仓期间现金 0 收益（保守）
数据库: data/sim_etf.db（orders / account_daily, 独立于 sim_v2）
用法: py312 python scripts/sim_etf_timing.py [--date 2026-08-07] [--no-push]
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

# ── 线程控制（铁律一）──
os.environ.pop("KMP_AFFINITY", None)
os.environ["OMP_NUM_THREADS"] = "3"

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "experiments/kronos"))

from sequoia_x.core.config import get_settings
from sequoia_x.data.sync import DataSync

DB_PATH = PROJECT_DIR / "data" / "sim_etf.db"
ETF_CODE = "512100"
ETF_NAME = "中证1000ETF南方"
INDEX_CODE = "sh.000852"
INIT_CASH = 1_000_000.0
COST = 0.0005          # 单边 0.05%（佣金+滑点）
SIGNAL_THRESHOLD = -0.25   # %: 预测 5 日收益 > -0.25% 看多
LOT = 100              # ETF 整手 100 份


# ─────────────────────────── 交易日历 ───────────────────────────

def is_trade_day(d: date) -> bool:
    """019 库数据驱动交易日判断（2026-08-09: 全量走 019 库原则）。

    ⚠️ 实测 DataSync.is_trade_day 的 baostock 路径卡死（30s+ 无响应, timeout 被杀）
    → 弃用。019 库 etf_daily 当日有 512100 数据 = 交易日（专门 ETF 库, 休市日无
    数据, 天然覆盖法定节假日; 数据由 019 pipeline 20:05 入库保证）。
    """
    try:
        conn = sqlite3.connect(ETF_DB_019)
        n = conn.execute("SELECT COUNT(*) FROM etf_daily WHERE symbol=? AND date=?",
                         (ETF_CODE, d.strftime("%Y-%m-%d"))).fetchone()[0]
        conn.close()
        return n > 0
    except Exception:
        return False


def next_n_trade_dates(after: date, n: int) -> list[str]:
    """复用 index_timing_check.next_n_trade_dates（chinese_calendar 本地日历）。

    2027+ 年份容错（节假日当工作日）——exec_date 仅是参考, 实际执行以
    019 库当日数据为准（非交易日自然跳过）。
    """
    import index_timing_check as ITC
    return ITC.next_n_trade_dates(after.strftime("%Y-%m-%d"), n)


# ─────────────────────────── 行情（019 库全量, 2026-08-09 用户确认）───────────────────────────

ETF_DB_019 = "/public/home/hpc/zhulei/superman/quant/code/019_etf_daily_sync_and_backtest/data/etf_daily.db"


def get_quote(d: date) -> dict:
    """行情全量走 019 库（专门 ETF 日线库, 当日数据 20:05 已入库, 模拟盘 20:30 运行）。

    open = 当日开盘价（T+1 成交价）, price = 当日收盘价（估值）。
    """
    try:
        conn = sqlite3.connect(ETF_DB_019)
        row = conn.execute(
            "SELECT open, close FROM etf_daily WHERE symbol=? AND date=?",
            (ETF_CODE, d.strftime("%Y-%m-%d"))).fetchone()
        conn.close()
        if row:
            return {"open": float(row[0]), "price": float(row[1])}
    except Exception as e:
        raise RuntimeError(f"{d} 019 库读取失败: {e}")
    raise RuntimeError(f"{d} 019 库无 {ETF_CODE} 当日数据（019 pipeline 未完成?）")


# ─────────────────────────── 信号推理 ───────────────────────────

def run_signal(d: date) -> dict:
    """Kronos 预测今日(ref) 未来 5 日收益 → 信号。

    复用 index_timing_check.predict_index（含 y_timestamp 修复经验）;
    y_timestamp 实盘无未来 DB 数据 → 用生成的交易日历（仅时间特征, 无价格泄漏）。
    2026-08-09: 指数数据全量走 019 库（KRONOS_INDEX_DB/SYMBOL 环境变量, 当日
    20:05 已入库; 004 库同步无 ETF 专属保障）。
    """
    os.environ["KRONOS_INDEX_DB"] = ETF_DB_019
    os.environ["KRONOS_INDEX_SYMBOL"] = "000852"
    import index_timing_check as ITC
    dates = ITC.ensure_index_data()
    if d.strftime("%Y-%m-%d") not in dates:
        raise RuntimeError(f"{d} 不在指数交易日历中（数据未同步?）")
    ITC._load_predictor()
    r = ITC.predict_index(d.strftime("%Y-%m-%d"))
    if r is None:
        raise RuntimeError(f"{d} 推理失败（历史不足或数据问题）")
    pred_ret5 = r["pred_ret5"]
    bullish = pred_ret5 > SIGNAL_THRESHOLD / 100.0
    return {"pred_ret5": pred_ret5, "bullish": bullish, "close": r["close"]}


# ─────────────────────────── 账务 ───────────────────────────

def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS account (
            id INTEGER PRIMARY KEY, date TEXT UNIQUE, cash REAL, shares INTEGER,
            etf_price REAL, total_value REAL, total_return REAL, created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY, create_date TEXT, action TEXT,
            signal_ret REAL, exec_date TEXT, status TEXT DEFAULT 'pending',
            exec_price REAL, shares INTEGER, cost REAL, created_at TEXT
        );
    """)
    # 初始账户（首条记录: 100 万现金）
    if conn.execute("SELECT COUNT(*) FROM account").fetchone()[0] == 0:
        conn.execute("INSERT INTO account (date, cash, shares, etf_price, total_value,"
                     " total_return, created_at) VALUES (?,?,?,?,?,?,?)",
                     (date.today().strftime("%Y-%m-%d"), INIT_CASH, 0, 0.0,
                      INIT_CASH, 0.0, datetime.now().isoformat(timespec="seconds")))
        conn.commit()
    conn.close()


def load_state(conn: sqlite3.Connection) -> dict:
    """最近一条账户状态（现金/持仓）。"""
    row = conn.execute(
        "SELECT date, cash, shares, etf_price, total_value FROM account "
        "ORDER BY id DESC LIMIT 1").fetchone()
    if row:
        return {"date": row[0], "cash": row[1], "shares": row[2],
                "price": row[3], "value": row[4]}
    return {"date": None, "cash": INIT_CASH, "shares": 0, "price": 0.0, "value": INIT_CASH}


def execute_orders(conn: sqlite3.Connection, d: date, quote: dict, state: dict,
                   push: bool) -> list[str]:
    """执行 pending 指令（今日开盘价, T+1 等价）。返回执行明细行。"""
    pend = conn.execute(
        "SELECT id, action, signal_ret, shares FROM orders "
        "WHERE status='pending' AND exec_date=? ORDER BY id",
        (d.strftime("%Y-%m-%d"),)).fetchall()
    msgs = []
    for oid, action, sig_ret, o_shares in pend:
        px = quote["open"]
        if action == "BUY" and state["shares"] == 0:
            budget = state["cash"] * (1 - COST)
            shares = int(budget / px / LOT) * LOT
            cost = shares * px * COST
            cash = state["cash"] - shares * px - cost
            conn.execute("UPDATE orders SET status='executed', exec_price=?, shares=?,"
                         " cost=? WHERE id=?", (px, shares, cost, oid))
            state.update(cash=cash, shares=shares, price=px)
            msgs.append(f"✅ 买入 {ETF_CODE}: {shares} 份 @{px:.3f}（成本 {cost:.0f}）")
        elif action == "SELL" and state["shares"] > 0:
            sold = state["shares"]                       # 实际卖出数（orders.shares 为空）
            revenue = sold * px * (1 - COST)
            cost = sold * px * COST
            conn.execute("UPDATE orders SET status='executed', exec_price=?, shares=?,"
                         " cost=? WHERE id=?", (px, sold, cost, oid))
            state["cash"] += revenue
            state["shares"] = 0
            msgs.append(f"✅ 卖出 {ETF_CODE}: {sold} 份 @{px:.3f}（净回笼 {revenue:.0f}）")
        else:
            conn.execute("UPDATE orders SET status='cancelled' WHERE id=?", (oid,))
            msgs.append(f"⏭ 指令作废（状态不符）: {action} 持仓={state['shares']}")
    return msgs


def daily_write(conn: sqlite3.Connection, d: date, state: dict, quote: dict) -> None:
    """写当日账户（估值）; 返回累计收益。"""
    value = state["cash"] + state["shares"] * quote["price"]
    conn.execute("INSERT OR REPLACE INTO account (date, cash, shares, etf_price,"
                 " total_value, total_return, created_at) VALUES (?,?,?,?,?,?,?)",
                 (d.strftime("%Y-%m-%d"), state["cash"], state["shares"], quote["price"],
                  value, value / INIT_CASH - 1, datetime.now().isoformat(timespec="seconds")))
    conn.commit()


# ─────────────────────────── 推送 ───────────────────────────

def notify(title: str, body: str) -> None:
    try:
        from wxpusher import WxPusher
        s = get_settings()
        WxPusher.send_message(content=f"{title}\n{body}", token=s.wxpusher_token,
                              topic_ids=s.wxpusher_topic_ids, content_type=1)
        print(f"[{datetime.now():%H:%M:%S}] 已推送: {title}", flush=True)
    except Exception as e:
        print(f"[{datetime.now():%H:%M:%S}] 推送失败: {e}", flush=True)


# ─────────────────────────── 主流程 ───────────────────────────

def main() -> None:
    global DB_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="指定日期（默认今天, 用于补跑/dry-run）")
    ap.add_argument("--no-push", action="store_true", help="不推送微信")
    ap.add_argument("--db", default=str(DB_PATH), help="数据库路径（dry-run 用临时库）")
    args = ap.parse_args()

    DB_PATH = Path(args.db)
    d = date.fromisoformat(args.date) if args.date else date.today()
    ds = d.strftime("%Y-%m-%d")
    print(f"[{datetime.now():%H:%M:%S}] ETF 择时模拟盘: {ds}", flush=True)

    # 1. 交易日开关（复用现有模拟盘逻辑, 自动跳过节假日）
    if not is_trade_day(d):
        print(f"  {ds} 非交易日, 跳过", flush=True)
        return

    # 2. 数据库 + 状态
    init_db()
    conn = sqlite3.connect(DB_PATH)
    state = load_state(conn)
    quote = get_quote(d)  # 今日行情（开盘/最新）

    # 3. 周最后交易日 → 推理信号; 否则 → 执行指令
    tomorrow = d + timedelta(days=1)
    is_week_end = is_trade_day(d) and not is_trade_day(tomorrow)
    msgs = []
    if is_week_end:
        print(f"  ⏳ 周最后交易日 → 推理信号...", flush=True)
        sig = run_signal(d)
        action = None
        if sig["bullish"] and state["shares"] == 0:
            action = "BUY"
        elif not sig["bullish"] and state["shares"] > 0:
            action = "SELL"
        else:
            action = "HOLD"
        exec_date = next_n_trade_dates(d, 1)[0]
        conn.execute("INSERT INTO orders (create_date, action, signal_ret, exec_date,"
                     " status, created_at) VALUES (?,?,?,?,?,?)",
                     (ds, action, sig["pred_ret5"], exec_date, "pending",
                      datetime.now().isoformat(timespec="seconds")))
        conn.commit()
        sig_txt = "🟢 看多" if sig["bullish"] else "🔴 看空"
        msgs.append(f"信号: {sig_txt}（预测 5 日 {sig['pred_ret5']*100:+.2f}%, 阈值 {SIGNAL_THRESHOLD}%）")
        msgs.append(f"指令: {action}（下一交易日 {exec_date} 开盘执行）")
        print("\n".join(msgs), flush=True)
        if not args.no_push:
            notify(f"📊 ETF 择时周信号（{ds}）", "\n".join(msgs))
    else:
        msgs = execute_orders(conn, d, quote, state, args.no_push)
        for m in msgs:
            print(f"  {m}", flush=True)
        if msgs and not args.no_push:
            notify(f"📊 ETF 择时成交（{ds}）", "\n".join(msgs))

    # 4. 估值 + 账户写入
    daily_write(conn, d, state, quote)
    value = state["cash"] + state["shares"] * quote["price"]
    ret = value / INIT_CASH - 1
    print(f"  📈 账户: 现金 {state['cash']:.0f} | 持仓 {state['shares']} 份 | "
          f"净值 {value:.0f}（累计 {ret*100:+.2f}%）", flush=True)

    # 5. 周报（周五推送）
    if is_week_end and not args.no_push:
        notify(f"📈 ETF 择时周报（{ds}）",
               f"净值 {value:,.0f}（累计 {ret*100:+.2f}%）\n"
               f"持仓: {'512100 × ' + str(state['shares']) + ' 份' if state['shares'] else '空仓（现金）'}")
    conn.close()


if __name__ == "__main__":
    main()
