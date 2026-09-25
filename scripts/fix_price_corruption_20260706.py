#!/usr/bin/env python3
"""修复 stock_daily 的价格错乱行（2026-09-25 发现，非单一日期）。

背景（LLM 模拟盘归因诊断时发现）
--------------------------------
`data/sequoia_v2.db` 的 stock_daily 表中，多个月份存在**整行价格错乱**：

| 日期 | 异常行数 | 说明 |
|------|----------|------|
| 2024-01-02 | 1,899 | |
| 2026-06-09 | **2,036** | 影响面最大 |
| 2026-07-06 | 626 | |
| 2026-07-07 | 631 | |

特征（已实测确认）：
  - **全部为深市代码**（000/001/002/003/300/301），沪市与北交所零命中
  - **off-by-one 行错位**：代码块内每只股票被写入了**上一只股票**当日的数据。
    实证：002181 于 07-06 的值 16.47 与 002180 的真实值 16.47 完全一致；
    以 07-07 真实开盘价（≈07-06 真实收盘）为基准，错位假设在异常块内 78.6% 命中，
    自身值假设 0% 命中。
  - 行内自洽（amount/volume ≈ close），是**整行被另一套数值覆盖**，非字段级错误
  - **不是除权所致**：坏股票与随机股票的「同期有除权事件」占比完全一致（均 32/400）
  - 当日 sync_log 状态 ok、coverage=1.0 —— **现有自检完全没发现**

影响面：
  - 等权均值类分析被严重污染（6/9 与 7/6 两日「全市场等权日均涨幅」被抬到 +20%）
  - **V4 重训（9/1）的训练数据含 2026-06-09 与 07-06/07-07**，这些股票的
    收益率标签是垃圾值；2024-01-02 则在 70 个月回测的训练集内
  - 排查过程见 `docs/2026-09_LLM模拟盘买入即跌归因诊断.md`

判定规则（可自证）
------------------
坏行 = 该股票自身价格序列里出现离谱跳变：

    |close / prev_close − 1| > 25%   或   |open / prev_close − 1| > 25%

A 股单日涨跌停 ±10%（创业板/科创板 ±20%），故 >25% 必为异常。

修复方式
--------
1. 按异常日期逐个处理：从 baostock 拉取受影响股票 **该日期前后 5 天**的日线
   （adjustflag=3 不复权，与项目铁律一致）
2. **交叉验证**：用区间内的**正常日期**比对 baostock 与库内值，只有 ≥2 天收盘价
   吻合（误差 <0.5%）才采信该股票的 baostock 数据——避免"用不可信的源覆盖不可信的源"
3. 备份原值 → 单事务 UPDATE → 复验

用法
----
    python scripts/fix_price_corruption_20260706.py                # 只体检（dry-run）
    python scripts/fix_price_corruption_20260706.py --apply        # 修复全部异常日期
    python scripts/fix_price_corruption_20260706.py --apply --dates 2026-06-09,2026-07-06
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DB = ROOT / "data" / "sequoia_v2.db"
BACKUP_DIR = ROOT / f"data/backup_price_fix_{datetime.now():%Y%m%d}"

JUMP = 0.25    # 单日 |涨跌| 超过 25% 视为坏行
TOL = 0.005    # 交叉验证的收盘价相对误差容差
PAD_DAYS = 7   # 拉取区间向两侧扩展的自然日数
# 扫描起点要早于首个目标日 —— 否则目标日会成为该股票的"首行"、拿不到前收而被漏掉
SCAN_SINCE = "2023-12-01"


def scan_anomalies(conn: sqlite3.Connection, since: str) -> pd.DataFrame:
    """扫描全表，返回所有 |单日涨跌| > JUMP 的行及其前收。"""
    df = pd.read_sql(
        "SELECT id, symbol, date, open, close FROM stock_daily "
        f"WHERE date >= '{since}' ORDER BY symbol, date",
        conn,
    )
    for c in ("open", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["prev_close"] = df.groupby("symbol")["close"].shift(1)
    df = df[df["prev_close"] > 0]
    jump = ((df["close"] / df["prev_close"] - 1).abs() > JUMP) | \
           ((df["open"] / df["prev_close"] - 1).abs() > JUMP)
    bad = df[jump].copy()
    bad["ratio"] = bad["close"] / bad["prev_close"]
    return bad.sort_values(["date", "symbol"]).reset_index(drop=True)


def fetch_baostock(symbols: list[str], start: str, end: str, to_bs_code) -> dict[str, pd.DataFrame]:
    """从 baostock 拉取区间日线（不复权）。"""
    import baostock as bs
    lg = bs.login()
    if lg.error_code != "0":
        print(f"  ❌ baostock 登录失败: {lg.error_msg}")
        return {}
    out: dict[str, pd.DataFrame] = {}
    fields = "date,open,high,low,close,volume,amount,turn,pctChg"
    try:
        for i, sym in enumerate(symbols, 1):
            try:
                rs = bs.query_history_k_data_plus(
                    to_bs_code(sym), fields, start_date=start, end_date=end,
                    frequency="d", adjustflag="3")
                if rs.error_code != "0":
                    continue
                rows = []
                while rs.next():
                    rows.append(rs.get_row_data())
                if not rows:
                    continue
                d = pd.DataFrame(rows, columns=rs.fields)
                for c in ("open", "high", "low", "close", "volume", "amount", "turn", "pctChg"):
                    d[c] = pd.to_numeric(d[c], errors="coerce")
                out[sym] = d
            except Exception as e:
                print(f"    ⚠️ {sym}: {e}")
            if i % 200 == 0:
                print(f"    ... {i}/{len(symbols)}")
    finally:
        bs.logout()
    return out


def cross_validate(conn: sqlite3.Connection, bs_data: dict[str, pd.DataFrame],
                   bad_syms: set[str], start: str, end: str,
                   bad_date: str) -> tuple[dict, list, list]:
    """用区间内的正常日交叉验证 baostock 是否可信。

    Returns:
        (trusted, rejected, insufficient) —— 只有 trusted 会被用于修复。
    """
    local = pd.read_sql(
        "SELECT symbol, date, close FROM stock_daily "
        f"WHERE date BETWEEN '{start}' AND '{end}' AND date <> '{bad_date}'", conn)
    local["close"] = pd.to_numeric(local["close"], errors="coerce")
    ref: dict[str, dict[str, float]] = {}
    for r in local.itertuples(index=False):
        ref.setdefault(r.symbol, {})[r.date] = r.close

    trusted, rejected, insufficient = {}, [], []
    for sym, df in bs_data.items():
        if sym not in bad_syms:
            continue
        known = ref.get(sym, {})
        if len(known) < 3:
            insufficient.append(sym)
            continue
        dmap = dict(zip(df["date"], df["close"]))
        pairs = [(dmap[d], px) for d, px in known.items() if d in dmap and px and dmap[d]]
        hits = sum(1 for a, b in pairs if abs(a / b - 1) <= TOL)
        if len(pairs) >= 3 and hits == len(pairs):
            trusted[sym] = df
        else:
            rejected.append(sym)
    return trusted, rejected, insufficient


def main() -> int:
    ap = argparse.ArgumentParser(description="修复 stock_daily 价格错乱行")
    ap.add_argument("--apply", action="store_true", help="执行修复（默认只体检）")
    ap.add_argument("--dates", type=str, default="",
                    help="逗号分隔的日期，默认修复全部扫描到的异常日期")
    args = ap.parse_args()

    if not DB.exists():
        print(f"❌ 数据库不存在: {DB}")
        return 1

    conn = sqlite3.connect(DB)
    print("=" * 92)
    print(f"【体检】扫描 {SCAN_SINCE} 起，|单日涨跌| > {JUMP:.0%} 的行")
    print("=" * 92)
    bad = scan_anomalies(conn, SCAN_SINCE)
    if bad.empty:
        print("✓ 未发现异常行")
        conn.close()
        return 0

    by_date = bad.groupby("date").size().sort_values(ascending=False)
    print(f"共 {len(bad)} 行异常，涉及 {bad['symbol'].nunique()} 只股票、{len(by_date)} 个交易日")
    print("\n异常行数 Top 12 的日期：")
    for d, n in by_date.head(12).items():
        sub = bad[bad["date"] == d]
        exch = "深市" if sub["symbol"].str.startswith(("0", "3")).all() else "混合"
        print(f"  {d}: {n:5d} 行  [{exch}]  比值中位 {sub['ratio'].median():.3f}")

    if not args.apply:
        print("\n（dry-run 模式，未修改数据库。加 --apply 执行修复）")
        conn.close()
        return 0

    # 待修复日期：显式指定，或取异常行数 >= 100 的日期（排除零星个例，那多为真实极端行情）
    targets = ([d.strip() for d in args.dates.split(",") if d.strip()]
               if args.dates else list(by_date[by_date >= 100].index))
    if not targets:
        print("\n无可修复日期（异常均为零星个例）")
        conn.close()
        return 0
    print(f"\n【修复】目标日期: {targets}")

    from sequoia_x.core.config import get_settings
    from sequoia_x.data.engine import DataEngine
    eng = DataEngine(get_settings())

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for d in targets:
        sub = bad[bad["date"] == d]
        syms = sorted(sub["symbol"].unique().tolist())
        start = (datetime.strptime(d, "%Y-%m-%d") - timedelta(days=PAD_DAYS)).strftime("%Y-%m-%d")
        end = (datetime.strptime(d, "%Y-%m-%d") + timedelta(days=PAD_DAYS)).strftime("%Y-%m-%d")
        print(f"\n── {d} ── 异常 {len(sub)} 行 / {len(syms)} 只，拉取 {start}~{end}")
        bs_data = fetch_baostock(syms, start, end, lambda s: eng._to_baostock_code(s))
        print(f"  baostock 返回 {len(bs_data)} 只")
        trusted, rejected, insufficient = cross_validate(
            conn, bs_data, set(syms), start, end, d)
        print(f"  交叉验证: ✓通过 {len(trusted)}  ✗不通过 {len(rejected)}  ⚠样本不足 {len(insufficient)}")

        if not trusted:
            print("  跳过（无股票通过交叉验证）")
            summary.append((d, 0, 0))
            continue

        # 备份
        bak = sub[sub["symbol"].isin(trusted)]
        bak_path = BACKUP_DIR / f"bad_rows_{d}.csv"
        bak.to_csv(bak_path, index=False, encoding="utf-8-sig")

        updated = 0
        with conn:  # 单事务
            for sym, df in trusted.items():
                row = df[df["date"] == d]
                if row.empty:
                    continue
                r = row.iloc[0]
                cur = conn.execute(
                    "UPDATE stock_daily SET open=?, high=?, low=?, close=?, "
                    "volume=?, amount=?, turnover=?, pctChg=? WHERE symbol=? AND date=?",
                    (float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]),
                     float(r["volume"]),
                     float(r["amount"]) if pd.notna(r["amount"]) else None,
                     float(r["turn"]) if pd.notna(r["turn"]) else None,
                     float(r["pctChg"]) if pd.notna(r["pctChg"]) else None, sym, d))
                updated += cur.rowcount
        print(f"  已更新 {updated} 行，备份 {bak_path.name}")
        summary.append((d, len(sub), updated))

    # ── 复验 ──
    print("\n" + "=" * 92)
    print("【复验】")
    after = scan_anomalies(conn, SCAN_SINCE)
    print(f"  修复后异常行: {len(after)}（修复前 {len(bad)}）")
    rem = after.groupby("date").size().sort_values(ascending=False).head(8)
    if len(rem):
        print("  残留 Top：")
        for dd, n in rem.items():
            print(f"    {dd}: {n} 行")

    print("\n【修复汇总】")
    for d, b, u in summary:
        print(f"  {d}: 异常 {b} 行 → 更新 {u} 行")

    json.dump({"before": len(bad), "after": len(after),
               "summary": [{"date": d, "bad": b, "updated": u} for d, b, u in summary]},
              open(BACKUP_DIR / "repair_summary.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    conn.close()
    print(f"\n完成。备份目录: {BACKUP_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
