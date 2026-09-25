#!/usr/bin/env python3
"""回填 stock_list.name 的历史空名（2026-09-26）。

问题
----
`stock_list` 共 5,931 行，其中 **5,327 行（89.8%）name 为 NULL** —— 包括
600000 浦发银行、600004 白云机场这种最主板的票。

根因（两条叠加）
----------------
1. `sync.py::sync_stock_list` 只在 **表为空**（全量）或 **有新股**（增量）时写 name，
   而写用的是 `INSERT OR IGNORE` —— 对**已存在**的行不会更新。
   于是历史上"带着空 name 插入"的行，之后无论同步多少次都补不上。
2. `stock_list` 的 `name` 列是**手工 ALTER** 加的，`CREATE TABLE` 语句里没有它，
   新建库会缺列直接报错（已在 sync.py 里补了幂等补列）。

影响
----
**不是正确性风险**：`get_base_stock_pool()` 的 ST/退市过滤用的是 baostock
`query_stock_basic` **实时返回**的名字，不读这张表。本表只服务展示层
（如日报/推送里的股票名）。所以这是"看不看得见名字"的问题，不影响选股。

做法
----
名字每次同步都能从**同花顺 tickers/list** 拿到（`DataSync.get_active_stocks()`），
本脚本就是把它一次性灌进空名行。写前备份、写后复验。

用法
----
    python scripts/backfill_stock_names.py            # 只体检（不联网、不改库）
    python scripts/backfill_stock_names.py --apply    # 联网取名字 + 备份 + 回填 + 复验
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DB = ROOT / "data" / "sequoia_v2.db"


def _missing(conn: sqlite3.Connection) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM stock_list WHERE name IS NULL OR name=''").fetchone()[0]


def main() -> int:
    ap = argparse.ArgumentParser(description="回填 stock_list.name 历史空名")
    ap.add_argument("--apply", action="store_true", help="执行回填（默认只体检）")
    args = ap.parse_args()

    conn = sqlite3.connect(DB)
    total = conn.execute("SELECT COUNT(*) FROM stock_list").fetchone()[0]
    miss = _missing(conn)
    print("=" * 84)
    print("【体检】stock_list.name 空名情况")
    print("=" * 84)
    print(f"  总行数 {total:,} | 空名 {miss:,}（{miss / total * 100:.1f}%）")
    if not miss:
        print("  ✓ 无需回填")
        conn.close()
        return 0
    if not args.apply:
        print("\n（dry-run，未联网、未改库。加 --apply 执行）")
        conn.close()
        return 0

    # ── 取名字（同花顺 tickers/list，与同步主源同一路径）──
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    from sequoia_x.core.config import get_settings
    from sequoia_x.data.sync import DataSync

    ds = DataSync(get_settings())
    active = ds.get_active_stocks()
    names: dict[str, str] = active.get("names", {}) or {}
    print(f"\n【取名字】同花顺返回 {len(names):,} 个 code→name")
    if not names:
        print("❌ 未取到名字（数据源不可用），中止。库未改动。")
        conn.close()
        return 1

    # ── 备份受影响行的旧值（均为 NULL/空，回滚=置回 NULL）──
    bak_dir = ROOT / f"data/backup_stock_names_{datetime.now():%Y%m%d}"
    bak_dir.mkdir(parents=True, exist_ok=True)
    keys = [r[0] for r in conn.execute(
        "SELECT symbol FROM stock_list WHERE name IS NULL OR name=''")]
    bak = bak_dir / "null_name_symbols.csv"
    with open(bak, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["symbol"])
        w.writerows((k,) for k in keys)
    print(f"【备份】{len(keys):,} 个空名 symbol → {bak}（原值均为 NULL，回滚=置回 NULL）")

    # ── 写入 ──
    filled = still_missing = 0
    with conn:
        for sym, nm in names.items():
            if not nm:
                continue
            cur = conn.execute(
                "UPDATE stock_list SET name=? WHERE symbol=? AND (name IS NULL OR name='')",
                (nm, sym))
            filled += cur.rowcount

    left = _missing(conn)
    print(f"\n【复验】写入 {filled:,} 行 | 剩余空名 {left:,}（回填前 {miss:,}）")
    # 未补上的多半是"本地有、同花顺已无"的退市股，属正常
    if left:
        sample = [r[0] for r in conn.execute(
            "SELECT symbol FROM stock_list WHERE name IS NULL OR name='' LIMIT 8")]
        print(f"  未补上的样例: {sample}")
    conn.close()
    print(f"\n完成。备份目录: {bak_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
