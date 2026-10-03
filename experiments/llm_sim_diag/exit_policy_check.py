#!/usr/bin/env python
"""LLM 盘是否该切 sell_rules_mode="hard_stop_only"（V2 的 E1 档）—— 量化核对。

背景
    V2/V4 盘已切 E1（月内只留 -12% 硬止损，动量规则停用），证据来自 70 个月月度回测
    （experiments/compare_intra_exit.py）。LLM 盘（旧 main.py / 新 024）**未传**
    sell_rules_mode → 仍默认 "all"。本脚本用 LLM 盘自身的历史数据回答：
    "E1 是否可以直接照搬到 LLM 盘？"

两个核对角度
    ① 卖出规则贡献分解：93 笔平仓按类别（硬止损/移动止盈/动量规则）统计
       实现盈亏 + 卖出后 +5/+10/+20 交易日走势（"卖飞还是避开下跌"）
    ② 持仓槽占用模拟：若把非硬止损的卖出全部拦掉（hard_stop_only 的语义），
       按"累计买入 − 累计硬止损卖出"推算每日占用仓位 → 何时触顶 max_positions=20
       （触顶后 LLM 每日新信号将**无法买入** = 资金循环中断）
       注：这是上界估计——被拦下的持仓后续仍可能触发 -12%（未建模），
       且假设买入行为在触顶前不变。

用法（铁律六：必须 py312）
    /home/zhulei/anaconda3/envs/zhulei_py312/bin/python \
        experiments/llm_sim_diag/exit_policy_check.py
"""
from __future__ import annotations

import sqlite3
import statistics
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
DB = PROJ / "data" / "sequoia_v2.db"          # LLM 模拟盘（表与行情同库）
MAX_POSITIONS = 20                            # LLM 盘仓位上限（settings/config）
FWD_WINDOWS = (5, 10, 20)                     # 卖出后观察窗口（交易日）

# 卖出原因 → 类别（原因文本形如 "R(相对弱势)(60分); S(硬止损)(40分)" / "硬止损(开盘触发): ..."）
HARD_STOP_PREFIX = "硬止损"
TRAILING_PREFIXES = ("T(移动止盈)",)
MOMENTUM_PREFIXES = ("R(", "SH(", "M(", "D(", "LSTM")


def categorize(reason: str) -> str:
    """把 exit_reason 归类：hard_stop / trailing / momentum / other。"""
    r = reason or ""
    # 硬止损优先：多规则同时触发时原因串里可能含 "S(硬止损)(40分)"，但真正的穿透
    # 触发只有一种文本（"硬止损..."）。用开头判定，避免把动量单误归硬止损。
    if r.startswith(HARD_STOP_PREFIX):
        return "hard_stop"
    if any(p in r for p in TRAILING_PREFIXES):
        return "trailing"
    if any(p in r for p in MOMENTUM_PREFIXES):
        return "momentum"
    return "other"


def load_closed(conn) -> list[dict]:
    rows = conn.execute(
        "SELECT symbol, name, buy_date, sell_date, hold_days, pnl, pnl_pct, exit_reason "
        "FROM sim_closed_trades ORDER BY sell_date"
    ).fetchall()
    return [
        {"symbol": r[0], "name": r[1], "buy_date": r[2], "sell_date": r[3],
         "hold_days": r[4], "pnl": r[5] or 0.0, "pnl_pct": r[6] or 0.0,
         "reason": r[7] or "", "cat": categorize(r[7] or "")}
        for r in rows
    ]


def load_open(conn) -> list[dict]:
    rows = conn.execute(
        "SELECT symbol, buy_date, hold_days FROM sim_positions"
    ).fetchall()
    return [{"symbol": r[0], "buy_date": r[1], "hold_days": r[2]} for r in rows]


def forward_returns(conn, symbol: str, sell_date: str) -> dict[int, float | None]:
    """卖出日之后第 N 个交易日的收盘价相对卖出日收盘价的涨跌幅（不复权实际价）。

    用 close 序列（pctChg 按项目约定可能为 NULL）；取不到窗口的返回 None。
    """
    rows = conn.execute(
        "SELECT date, close FROM stock_daily WHERE symbol=? AND date>=? AND close>0 "
        "ORDER BY date LIMIT ?",
        (symbol, sell_date, max(FWD_WINDOWS) + 1),
    ).fetchall()
    if not rows or rows[0][0] != sell_date:
        return {w: None for w in FWD_WINDOWS}
    base = float(rows[0][1])
    out: dict[int, float | None] = {}
    for w in FWD_WINDOWS:
        if len(rows) > w:
            out[w] = float(rows[w][1]) / base - 1.0
        else:
            out[w] = None
    return out


def main() -> int:
    conn = sqlite3.connect(DB)
    closed = load_closed(conn)
    opened = load_open(conn)

    print(f"LLM 模拟盘平仓 {len(closed)} 笔 | 当前持仓 {len(opened)} 只")
    period = (closed[0]["buy_date"], closed[-1]["sell_date"])
    print(f"样本区间: {period[0]} ~ {period[1]}")

    # ── ① 卖出规则贡献分解 ───────────────────────────────
    print("\n" + "=" * 78)
    print("① 卖出规则贡献分解")
    print("=" * 78)
    cats: dict[str, list[dict]] = {}
    for t in closed:
        cats.setdefault(t["cat"], []).append(t)

    print(f"{'类别':<10}{'笔数':>5}{'合计盈亏¥':>12}{'均值pnl%':>10}{'中位%':>8}"
          f"{'胜率':>7}{'均持有日':>9}")
    for cat in ("hard_stop", "trailing", "momentum", "other"):
        ts = cats.get(cat) or []
        if not ts:
            continue
        pnl = [t["pnl"] for t in ts]
        pct = [t["pnl_pct"] * 100 for t in ts]
        win = sum(1 for x in pct if x > 0) / len(pct) * 100
        hd = [t["hold_days"] for t in ts]
        print(f"{cat:<10}{len(ts):>5}{sum(pnl):>12,.0f}{statistics.mean(pct):>10.2f}"
              f"{statistics.median(pct):>8.2f}{win:>6.0f}%{statistics.mean(hd):>9.1f}")

    # ── ② 卖出后走势（卖飞 or 避开下跌）─────────────────────
    print("\n" + "=" * 78)
    print("② 卖出后走势：动量/移动止盈的卖出，后来股价怎么走了？")
    print("   （正=卖出后继续涨=卖飞；负=卖出后继续跌=躲对了）")
    print("=" * 78)
    print(f"{'类别':<10}{'笔数':>5}" + "".join(f"{'+' + str(w) + 'd均值%':>12}" for w in FWD_WINDOWS)
          + f"{'+20d中位%':>11}")
    for cat in ("hard_stop", "trailing", "momentum"):
        ts = cats.get(cat) or []
        if not ts:
            continue
        fwd: dict[int, list[float]] = {w: [] for w in FWD_WINDOWS}
        for t in ts:
            fr = forward_returns(conn, t["symbol"], t["sell_date"])
            for w in FWD_WINDOWS:
                if fr.get(w) is not None:
                    fwd[w].append(fr[w] * 100)
        line = f"{cat:<10}{len(ts):>5}"
        for w in FWD_WINDOWS:
            v = fwd[w]
            line += f"{statistics.mean(v):>12.2f}" if v else f"{'n/a':>12}"
        med = statistics.median(fwd[20]) if fwd[20] else float("nan")
        line += f"{med:>11.2f}"
        print(line)

    # ── ③ 持仓槽占用模拟（hard_stop_only 的语义）─────────────
    print("\n" + "=" * 78)
    print("③ 持仓槽占用模拟：拦住非硬止损卖出后，何时占满 20 个仓位槽？")
    print("   口径: 非硬止损卖出全部拦下 → 槽位只在 -12% 硬止损时释放；满仓后新买入跳过")
    print("=" * 78)
    # 实际持仓数（来自日结表）对照
    actual = dict(conn.execute("SELECT date, position_count FROM sim_account_daily").fetchall())

    # 以"笔"为单位（同代码可能多笔；用序号区分，避免 set 去重低估占用）
    lots: list[dict] = []
    for i, t in enumerate(closed):
        lots.append({"lot": f"c{i}", "buy": t["buy_date"],
                     "exit_hs": t["cat"] == "hard_stop", "sell": t["sell_date"]})
    for i, t in enumerate(opened):
        lots.append({"lot": f"o{i}", "buy": t["buy_date"], "exit_hs": False, "sell": None})

    # 封顶模拟：按事件顺序推进，占用满 MAX_POSITIONS 后新买入被跳过（skip）。
    # 假设：被拦下的持仓在样本期内不再触发任何卖出（上界）。真实情况会有部分后续
    # 触发 -12% 释放槽位，但动量/止盈卖出合计占全部退出的 78%（见 ①），回收通道
    # 的主体被切断这一点不受影响。
    events: list[tuple[str, str, str]] = []
    for t in lots:
        events.append((t["buy"], "buy", t["lot"]))
        if t["exit_hs"] and t["sell"]:
            events.append((t["sell"], "hs_sell", t["lot"]))
    events.sort()

    held: set[str] = set()
    skipped = 0
    first_full = None
    rows_out: list[tuple[str, int, int]] = []   # (date, 模拟占用, 实际持仓)
    for d, kind, lot in events:
        if kind == "buy":
            if len(held) >= MAX_POSITIONS:
                skipped += 1
                continue
            held.add(lot)
        else:
            held.discard(lot)
        rows_out.append((d, len(held), actual.get(d, -1)))
        if len(held) >= MAX_POSITIONS and first_full is None:
            first_full = d

    # 打印每月抽样 + 首次触顶
    print(f"{'日期':<12}{'模拟占用':>8}{'实际持仓':>9}")
    seen_month = set()
    for d, n, a in rows_out:
        m = d[:7]
        if m not in seen_month:
            seen_month.add(m)
            print(f"{d:<12}{n:>8}{('-' if a < 0 else a):>9}")
    print(" ...")
    last = rows_out[-1] if rows_out else ("", 0, 0)
    print(f"{last[0]:<12}{last[1]:>8}{('-' if last[2] < 0 else last[2]):>9}")
    print()
    n_exit_confiscated = sum(1 for t in lots if not t["exit_hs"] and t["sell"])
    print(f"被拦下的卖出（非硬止损）: {n_exit_confiscated}/{len(closed)} 笔 "
          f"= {n_exit_confiscated / len(closed) * 100:.0f}% 的退出通道")
    if first_full:
        print(f"⚠ 模拟占用首次达到 {MAX_POSITIONS} 只: {first_full}"
              f"（样本起点后约 {first_full[5:]} 起满仓）")
        total_buys = len(closed) + len(opened)
        print(f"  样本期内被跳过的买入信号: {skipped} 笔"
              f"（占 {total_buys} 笔买入的 {skipped / total_buys * 100:.0f}%）")
    else:
        print(f"模拟占用从未达到 {MAX_POSITIONS}（资金循环不中断）")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
