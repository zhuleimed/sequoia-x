import sqlite3
conn = sqlite3.connect("data/sequoia_v2.db")

# Check stock_daily
c = conn.execute("SELECT MAX(date), COUNT(DISTINCT symbol) FROM stock_daily")
r = c.fetchone()
print("===DB_STATUS===")
print("max_date:", r[0])
print("stock_count:", r[1])

# Check sync_log columns
c_cols = conn.execute("PRAGMA table_info(sync_log)")
cols = c_cols.fetchall()
print("sync_log_columns:", [col[1] for col in cols])

# Check sync_log
try:
    c2 = conn.execute("SELECT MAX(date), MAX(start_time) FROM sync_log")
    r2 = c2.fetchone()
    print("sync_max_date:", r2[0])
    print("sync_last_time:", r2[1])
except Exception as e:
    print("sync_log_error:", str(e))

# Check index_daily
c3 = conn.execute("SELECT MAX(date) FROM index_daily")
r3 = c3.fetchone()
print("index_max_date:", r3[0])

# Count records by date (last 5 dates)
c6 = conn.execute("SELECT date, COUNT(*) as cnt FROM stock_daily GROUP BY date ORDER BY date DESC LIMIT 5")
rows = c6.fetchall()
for row in rows:
    print("date_count:", row[0], row[1])

conn.close()
