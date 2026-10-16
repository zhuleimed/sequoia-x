"""修复 2026-10-16 科创板 volume ×100 事故。

## 事故

`tencent_source.py::_tencent_kline` 一直**一刀切** `volume *= 100`（假设腾讯返回"手"），
但**腾讯对科创板（688/689）返回的成交量单位是「股」** ⇒ 走 Tencent 路径的科创板被放大 100 倍。

证据（同花顺 dump 独立对拍，2026-10-16）：
  腾讯原始值 ÷ 同花顺真值：688001/688002/688005/688111/688036/688981/689009 全部 = 1.0000
                            600519/000001/300750/601398/002415/600036 全部 = 0.0100

为什么只有 82 只中招：`get_daily` 是**双源切换**的 —— 走 Sina 路径的 534 只 688 没被 ×100（Sina 返回"股"，
`_sina_kline` 不做转换）⇒ 侥幸正确；只有当天走了 Tencent 路径的那批被 ×100。

## 本脚本做什么

把**坏行的 volume ÷100**（只动 volume，**不碰 amount** —— 实测 amount 是对的）。

判定坏行：`symbol` 是科创板 **且** `amount/volume/close < 0.5`。
（非科创板的少量越界行 —— 如 601091/301551/301618 —— ratio 在 0.38~0.48，
 是**上市首日等真实极端行情**，不是单位错，**不动它们**。）

用法：
    python scripts/fix_688_volume_unit_20261016.py --dry-run   # 先看会改什么
    python scripts/fix_688_volume_unit_20261016.py             # 实际执行（自动备份）
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "sequoia_v2.db"
BACKUP_DIR = ROOT / "data" / "backup_688_volume_fix_20261016"


def log(m: str = "") -> None:
    print(m, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    con = sqlite3.connect(DB)
    rows = con.execute("""
        SELECT symbol, date, close, volume, amount
        FROM stock_daily
        WHERE (symbol LIKE '688%' OR symbol LIKE '689%')
          AND volume > 0 AND amount > 0 AND close > 0
          AND (amount / volume / close) < 0.5
        ORDER BY symbol, date
    """).fetchall()
    log(f"命中坏行: {len(rows)} 行")
    if not rows:
        log("无坏行，退出"); con.close(); return 0

    from collections import Counter
    log(f"  按日期: {dict(Counter(r[1] for r in rows))}")
    log(f"  代码范围: {min(r[0] for r in rows)} ~ {max(r[0] for r in rows)}")
    log(f"  样例:")
    for r in rows[:5]:
        log(f"    {r[0]} {r[1]}  close={r[2]}  volume={r[3]:,.0f}  "
            f"amount={r[4]:,.0f}  均价/收盘={r[4]/r[3]/r[2]:.4f}")

    # 备份
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    bak = BACKUP_DIR / "bad_rows_before.json"
    bak.write_text(json.dumps(
        [{"symbol": r[0], "date": r[1], "close": r[2], "volume": r[3], "amount": r[4]}
         for r in rows], ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"\n✅ 原值已备份: {bak}")

    if a.dry_run:
        log("\n[dry-run] 未写库。去掉 --dry-run 执行。")
        con.close(); return 0

    n = con.executemany(
        "UPDATE stock_daily SET volume = volume / 100.0 WHERE symbol = ? AND date = ?",
        [(r[0], r[1]) for r in rows]).rowcount
    con.commit()
    log(f"✅ 已更新 {n} 行（volume ÷100）")

    # ── 修后自证 ──
    left = con.execute("""
        SELECT COUNT(*) FROM stock_daily
        WHERE (symbol LIKE '688%' OR symbol LIKE '689%')
          AND volume > 0 AND amount > 0 AND close > 0
          AND (amount / volume / close) < 0.5
    """).fetchone()[0]
    log(f"\n自检：科创板越界行剩余 {left}  ← 必须是 0")
    if left:
        log("❌ 仍有残留，未通过"); con.close(); return 1

    # 抽 3 只看看修后的比价
    log("\n修后抽查（均价/收盘 应≈1）：")
    for sym in [r[0] for r in rows[:3]]:
        r = con.execute("""SELECT symbol,date,close,volume,amount FROM stock_daily
                           WHERE symbol=? AND date=? """, (sym, rows[0][1])).fetchone()
        if r:
            log(f"  {r[0]} {r[1]}  volume={r[3]:,.0f}  均价/收盘={r[4]/r[3]/r[2]:.4f}")
    con.close()
    log("\n✅ 修复完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
