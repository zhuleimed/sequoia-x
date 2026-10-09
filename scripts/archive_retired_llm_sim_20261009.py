#!/usr/bin/env python
"""归档「老 LLM 模拟盘」残留数据（2026-10-09）。

**背景（为什么清）**
    老 LLM 模拟盘（"8 量化策略 → LLM 选股"，004 main.py）已于 2026-10-01 被
    独立子项目 024_llm_hs300（台账库 `024_llm_hs300/data/sim_hs300.db`）取代，
    004 管线里不再有它的步骤。但它的 4 张表还留在**主库** `data/sequoia_v2.db`：

        sim_positions      11 行（9/30 冻结的持仓）
        sim_closed_trades  93 行（8/3~9/30 平仓流水）
        sim_account_daily  62 行（日结，末日 2026-09-30 累计 -9.92%）
        sim_buy_signals   122 行（含 2 条 9/30 未执行的 pending）

    留着的问题：读者会误以为这是"当前在跑的盘"（2026-10-08 复盘即踩到：
    日报里的"已实现盈亏 +57,223.99"被误当成新产生的数字）。

**保留一份完整备份**：归档库 `data/archive/sim_llm_004_retired_20260930.db`
    含 4 张表的**全部行**（逐行 md5 校验与源库一致）。需要复现历史分析时，
    把 `experiments/llm_sim_diag/*.py`、`scripts/redo_sim_20260923.py` 等
    指向归档库即可（它们原先连的是主库）。

**删除前的前置审计（2026-10-09）**
    主库这 4 张表已无活引用：
      - V2/V4 盘台账 = `data/sim_v2.db`；LLM-HS300 盘台账 = 024 的 `sim_hs300.db`
      - 004 管线里没有 main.py（老 LLM 盘入口）这一步
      - 仅 `experiments/llm_sim_diag/`、`scripts/*_2026090X.py` 等**历史复盘脚本**读它们
        （都是"查历史"用途，指向归档库即可）

**用法**
    python scripts/archive_retired_llm_sim_20261009.py            # 只报告（dry-run，默认）
    python scripts/archive_retired_llm_sim_20261009.py --execute   # 归档 + 删除
"""
from __future__ import annotations

import argparse
import hashlib
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
MAIN_DB = PROJECT_DIR / "data" / "sequoia_v2.db"
ARCHIVE_DB = PROJECT_DIR / "data" / "archive" / "sim_llm_004_retired_20260930.db"
TABLES = ["sim_positions", "sim_closed_trades", "sim_account_daily", "sim_buy_signals"]


def table_rows_hash(conn: sqlite3.Connection, table: str) -> tuple[int, str]:
    """行数 + 全表逐行 md5（按 id 排序，保证可比）。"""
    rows = conn.execute(f"SELECT * FROM {table} ORDER BY id").fetchall()
    h = hashlib.md5()
    for r in rows:
        h.update(repr(r).encode("utf-8"))
    return len(rows), h.hexdigest()


def report(conn: sqlite3.Connection, label: str) -> dict[str, tuple[int, str]]:
    out = {}
    print(f"\n── {label} ──")
    for t in TABLES:
        n, dig = table_rows_hash(conn, t)
        out[t] = (n, dig)
        extra = ""
        if t == "sim_account_daily" and n:
            first, last = conn.execute(
                "SELECT MIN(date), MAX(date) FROM sim_account_daily").fetchone()
            extra = f"（{first} ~ {last}）"
        print(f"  {t:<20} {n:>5} 行  md5={dig[:12]}  {extra}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="归档老 LLM 盘残留数据")
    ap.add_argument("--execute", action="store_true", help="真正执行（默认只报告）")
    ap.add_argument("--main-db", default=str(MAIN_DB))
    ap.add_argument("--archive-db", default=str(ARCHIVE_DB))
    args = ap.parse_args()

    main_db = Path(args.main_db)
    archive_db = Path(args.archive_db)
    if not main_db.exists():
        print(f"❌ 主库不存在: {main_db}")
        return 1

    src = sqlite3.connect(str(main_db), timeout=60)
    src.execute("PRAGMA busy_timeout=60000")

    before = report(src, f"主库（源）: {main_db.name}")

    if not args.execute:
        print(f"\n[dry-run] 将把上述 4 张表的全部行归档到 {archive_db}，然后从主库删除。")
        print("          加 --execute 执行。")
        return 0

    if any(n == 0 for n, _ in before.values()):
        print("\n⚠️ 有表为空 —— 可能已经归档过。为避免重复操作，本次中止。")
        return 2

    # ── 1. 建归档库并逐表复制（先归档、校验通过才允许删除）──
    archive_db.parent.mkdir(parents=True, exist_ok=True)
    if archive_db.exists():
        print(f"❌ 归档库已存在，拒绝覆盖: {archive_db}")
        return 2
    dst = sqlite3.connect(str(archive_db))
    for t in TABLES:
        ddl = src.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (t,)
        ).fetchone()
        if not ddl:
            print(f"❌ 主库里没有表 {t}，中止。")
            return 2
        dst.execute(ddl[0])
        rows = src.execute(f"SELECT * FROM {t}").fetchall()
        cols = len(rows[0]) if rows else 0
        dst.executemany(
            f"INSERT INTO {t} VALUES ({','.join('?' * cols)})", rows)
    dst.execute(
        "CREATE TABLE archive_meta (key TEXT PRIMARY KEY, value TEXT)")
    dst.executemany("INSERT INTO archive_meta VALUES (?,?)", [
        ("archived_at", datetime.now().isoformat(timespec="seconds")),
        ("source_db", str(main_db)),
        ("reason", "老 LLM 模拟盘（main.py）已被 024_llm_hs300 取代；"
                   "台账迁出主库，避免与在跑的盘混淆"),
        ("note", "恢复：把 experiments/llm_sim_diag/*.py 等历史脚本指向本库即可"),
    ])
    dst.commit()

    after = report(dst, f"归档库（新）: {archive_db.name}")

    # ── 2. 逐表比对（行数 + 逐行 md5 必须完全一致）──
    bad = [t for t in TABLES if before[t] != after[t]]
    if bad:
        print(f"\n❌ 归档校验失败: {bad} —— 不删除任何数据。")
        return 2
    print("\n✅ 归档校验通过（4 张表行数与逐行 md5 全部一致）")

    # ── 3. 删除主库行（单事务）──
    with src:
        for t in TABLES:
            src.execute(f"DELETE FROM {t}")
    left = report(src, f"主库（删除后）: {main_db.name}")
    if any(n != 0 for n, _ in left.values()):
        print("❌ 删除后仍有残留，请人工检查。")
        return 2

    # ── 4. 数据库整体完好性（quick_check，比 integrity_check 快）──
    qc = src.execute("PRAGMA quick_check").fetchone()[0]
    print(f"\n主库 quick_check: {qc}")
    if qc != "ok":
        print("❌ 主库完整性检查未通过")
        return 2
    print(f"✅ 归档完成：{archive_db}（{archive_db.stat().st_size/1024:.0f} KB）")
    print("   主库 4 张 sim_* 表已清空（文件体积不变 —— 未 VACUUM，避免长时间锁库）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
