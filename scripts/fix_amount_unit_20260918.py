#!/usr/bin/env python3
"""修复 stock_daily 的 amount 单位错误 + turnover 伪 0（2026-09-18）。

背景（022 项目侧发现，2026-09-18）：
  日线同步在**降级到腾讯实时行情**时，把接口的成交额（单位**万元**）直接写进了本库约定为
  **元**的 amount 列（小 10000 倍）；同一分支还把 turnover 写死 0.0（sina 分支把 amount 写 0）。
  实测 2026-08-14 起出现、08-19 起**全部约 5200 只股票**受影响，累计 11.8 万行。
  影响链：022 的 turnover_rate（=amount/float_cap×100）与所有 $amount/$adj_vwap 因子、
  021 因子库的换手类因子（翻译必败）全部用了这批脏数据。

判定与修复（可自证）：
  受影响行的「隐含价」= amount/volume ≈ close/10000 —— 乘 10000 后 ≈ close（实测中位比值 0.999871）。
  故：amount ← amount×10000；turnover ← NULL（0.0 是伪值，本库历史惯例用 NULL 表示未知）。

安全：改前把受影响行原值导出到 backup 目录；全程用单事务；改完逐项复验。

用法：
    python scripts/fix_amount_unit_20260918.py            # 只体检（dry-run，不改）
    python scripts/fix_amount_unit_20260918.py --apply    # 备份 + 修复 + 复验
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data/sequoia_v2.db"
BACKUP_DIR = ROOT / f"data/backup_amount_unit_fix_{datetime.now():%Y%m%d}"
SINCE = "2026-08-01"          # 体检起点（实际异常自 08-14 起）

# 受影响判定：amount > 0（排掉 sina 分支写的 0）且隐含价 < 收盘价/100（万元 vs 元差 10000 倍）
BAD_PRED = (
    "date >= ? AND volume > 0 AND close > 0 AND amount > 0 AND amount < close * volume / 100.0"
)


def main() -> int:
    apply = "--apply" in sys.argv
    con = sqlite3.connect(DB)
    con.execute("PRAGMA journal_mode=WAL")

    n_bad, d0, d1 = con.execute(
        f"SELECT COUNT(*), MIN(date), MAX(date) FROM stock_daily WHERE {BAD_PRED}", (SINCE,)
    ).fetchone()
    n_zero_turn = con.execute(
        "SELECT COUNT(*) FROM stock_daily WHERE date >= ? AND volume > 0 AND turnover = 0.0",
        (SINCE,),
    ).fetchone()[0]
    print(f"体检：amount 单位异常 {n_bad:,} 行（{d0} ~ {d1}）")
    print(f"      伪 turnover=0（有成交量却记 0）{n_zero_turn:,} 行")
    if not n_bad:
        print("✅ 无需修复")
        return 0
    if not apply:
        print("\n（dry-run，未改动。加 --apply 执行备份+修复）")
        return 0

    # ── 1) 备份受影响行原值 ──
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_sql(
        f"SELECT id, symbol, date, close, volume, amount, turnover FROM stock_daily WHERE {BAD_PRED}",
        con,
        params=(SINCE,),
    )
    fp = BACKUP_DIR / "affected_rows_before.parquet"
    df.to_parquet(fp, index=False)
    print(f"\n① 已备份 {len(df):,} 行原值 → {fp}")

    # ── 2) 单事务修复 ──
    try:
        con.execute("BEGIN")
        cur = con.execute(
            f"UPDATE stock_daily SET amount = amount * 10000.0 WHERE {BAD_PRED}", (SINCE,)
        )
        n_fix = cur.rowcount
        cur2 = con.execute(
            "UPDATE stock_daily SET turnover = NULL WHERE date >= ? AND volume > 0 AND turnover = 0.0",
            (SINCE,),
        )
        n_turn = cur2.rowcount
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    print(f"② 已修复 amount {n_fix:,} 行；turnover 置 NULL {n_turn:,} 行")

    # ── 3) 复验 ──
    chk = pd.read_sql(
        f"SELECT close, volume, amount, turnover FROM stock_daily WHERE {BAD_PRED}", con, params=(SINCE,)
    )
    # 上一步已把 amount 修好 → 用修复后的隐含量级判断"还有没有异常"
    still = con.execute(
        f"SELECT COUNT(*) FROM stock_daily WHERE {BAD_PRED}", (SINCE,)
    ).fetchone()[0]
    from_ = pd.read_sql(
        "SELECT date, close, volume, amount, turnover FROM stock_daily "
        "WHERE date >= '2026-08-19' AND volume > 0 LIMIT 200000",
        con,
    )
    r = (from_["amount"] / from_["volume"] / from_["close"])
    con.close()

    print(f"③ 复验：剩余单位异常 {still:,} 行（应为 0）")
    print(f"   08-19 之后隐含价/收盘价：中位 {r.median():.6f}（应 ≈1.0）"
          f"；落在 0.5~2 倍内 {((r - 1).abs() < 1).mean():.1%}")
    (BACKUP_DIR / "manifest.json").write_text(json.dumps({
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "reason": "腾讯实时行情成交额(万元)被写入 amount 列(元)；turnover 被写死 0.0",
        "affected_rows": int(n_bad),
        "date_range": [d0, d1],
        "fix": {"amount": "× 10000.0", "turnover": "0.0 → NULL（伪值）"},
        "rows_fixed_amount": int(n_fix),
        "rows_nulled_turnover": int(n_turn),
        "verify_median_implied_over_close": float(r.median()),
        "backup_file": fp.name,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"   manifest → {BACKUP_DIR/'manifest.json'}")
    return 0 if still == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
