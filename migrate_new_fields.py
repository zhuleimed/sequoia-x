"""一次性迁移脚本：将 stock_daily 表中 2026-06-09 的数据补充新字段（pctChg/peTTM等）。"""
import baostock as bs
import pandas as pd
import sqlite3
import time

lg = bs.login()
if lg.error_code != "0":
    print(f"登录失败: {lg.error_msg}")
    exit(1)
print("baostock 登录成功")

db = "data/sequoia_v2.db"
conn = sqlite3.connect(db)
cur = conn.cursor()

# 获取所有 symbol
cur.execute("SELECT DISTINCT symbol FROM stock_daily")
symbols = [r[0] for r in cur.fetchall()]
print(f"共 {len(symbols)} 只股票")

# 只更新 2026-06-09 的数据
date_target = "2026-06-09"
new_fields = ["pctChg", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"]

updated = 0
failed = 0

for i, sym in enumerate(symbols):
    # 先检查是否已有 pctChg 数据
    cur.execute(
        f"SELECT pctChg FROM stock_daily WHERE symbol=? AND date=?",
        (sym, date_target)
    )
    existing = cur.fetchone()
    if existing and existing[0] is not None:
        continue  # 已有数据，跳过

    # 查询 baostock
    bs_code = f"sh.{sym}" if sym.startswith(("6","9")) else f"sz.{sym}"
    rs = bs.query_history_k_data_plus(
        bs_code,
        "date,open,high,low,close,volume,turn,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM",
        date_target, date_target, "d", "2"
    )
    if rs.error_code != "0":
        failed += 1
        continue

    # 手动逐行获取（兼容 pandas 3.0）
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        failed += 1
        continue

    data = pd.DataFrame(rows, columns=rs.fields)
    row = data.iloc[0]

    # UPDATE existing row with new fields
    try:
        cur.execute(
            """UPDATE stock_daily SET 
               pctChg=?, peTTM=?, pbMRQ=?, psTTM=?, pcfNcfTTM=?
               WHERE symbol=? AND date=?""",
            (
                float(row["pctChg"]) if row["pctChg"] else None,
                float(row["peTTM"]) if row["peTTM"] else None,
                float(row["pbMRQ"]) if row["pbMRQ"] else None,
                float(row["psTTM"]) if row["psTTM"] else None,
                float(row["pcfNcfTTM"]) if row["pcfNcfTTM"] else None,
                sym, date_target
            )
        )
        updated += 1
    except Exception as e:
        failed += 1

    if (i + 1) % 500 == 0:
        conn.commit()
        print(f"进度: {i+1}/{len(symbols)} | 已更新 {updated} | 失败 {failed}")

    time.sleep(0.05)

conn.commit()
bs.logout()

print(f"\n完成! 共 {len(symbols)} 只, 更新 {updated} 只, 失败 {failed} 只")

# 验证
cur.execute(f"SELECT COUNT(*) FROM stock_daily WHERE date=? AND pctChg IS NOT NULL", (date_target,))
total_filled = cur.fetchone()[0]
print(f"验证: {date_target} 日期有 pctChg 数据: {total_filled} 条")
conn.close()
