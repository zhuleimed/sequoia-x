"""轻量脚本：仅补填 amount 字段，不重拉其他数据。"""
import baostock as bs
import sqlite3
import time
from datetime import datetime

DB = "data/sequoia_v2.db"
BATCH_SIZE = 500
SLEEP = 0.05  # baostock 请求间隔
MAX_RETRY = 50  # 连续错误重连阈值

WXPUSHER_TOKEN = "AT_hKGG0UfwrCP7bpcsO8cbQkrc4bZ9G3RX"
WXPUSHER_TOPIC_IDS = ["39277"]


def _push_notification(updated: int, filled: int, total: int, remaining: int, failed: int):
    """推送 amount 补填完成通知到微信。"""
    try:
        from wxpusher import WxPusher
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        pct = filled / total * 100 if total > 0 else 0
        msg_lines = [
            f"✅ amount 字段补填完成",
            f"时间: {now}",
            f"",
            f"本次补填: {updated:,} 行",
            f"失败: {failed:,} 行",
            f"最终 amount 填充: {filled:,}/{total:,} ({pct:.1f}%)",
            f"剩余缺失: {remaining:,}",
        ]
        if remaining == 0:
            msg_lines.append("🎉 全部完成，无缺失!")
        msg = "\n".join(msg_lines)

        result = WxPusher.send_message(
            content=msg,
            token=WXPUSHER_TOKEN,
            topic_ids=WXPUSHER_TOPIC_IDS,
            content_type=1,
        )
        if result.get("code") == 1000:
            print("WxPusher 推送成功", flush=True)
        else:
            print(f"WxPusher 推送失败: {result}", flush=True)
    except Exception as e:
        print(f"WxPusher 推送异常: {e}", flush=True)

conn = sqlite3.connect(DB)
cur = conn.cursor()

# ===== 获取缺失列表 =====
cur.execute("SELECT DISTINCT symbol FROM stock_daily WHERE amount IS NULL ORDER BY symbol")
symbols = [r[0] for r in cur.fetchall()]
print(f"需补 amount 的股票: {len(symbols)} 只", flush=True)

cur.execute("SELECT COUNT(*) FROM stock_daily WHERE amount IS NULL")
total_missing = cur.fetchone()[0]
print(f"总缺失 amount 行数: {total_missing:,}", flush=True)

# ===== baostock 登录 =====
lg = bs.login()
if lg.error_code != "0":
    print(f"baostock 登录失败: {lg.error_msg}")
    exit(1)
print("baostock 登录成功", flush=True)

updated = 0
failed = 0
batch = []
err_count = 0

# 注意：如果股票数太多、全 SSH 连接下长时间无输出可能导致断开，
# 先用前 100 只做测试，验证成功后改为全部
# 如要全部处理，将下面这行改为 symbols = symbols
symbols_to_process = symbols  # 全部处理

for i, sym in enumerate(symbols_to_process):
    # 断连恢复
    if err_count >= MAX_RETRY:
        print(f"[{i}] 连续 {err_count} 次错误，重连...", flush=True)
        bs.logout()
        time.sleep(1)
        if bs.login().error_code != "0":
            print("重连失败，终止", flush=True)
            break
        err_count = 0

    # 查询该股票的缺失日期范围
    cur.execute(
        "SELECT MIN(date), MAX(date) FROM stock_daily WHERE symbol=? AND amount IS NULL",
        (sym,),
    )
    row = cur.fetchone()
    if not row or row[0] is None:
        print(f"  {sym}: 无需补填（amount 已存在）", flush=True)
        continue
    start_date, end_date = row

    bs_code = f"sh.{sym}" if sym[0] in "69" else f"sz.{sym}"
    try:
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,amount",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="2",
        )
        if rs.error_code != "0":
            print(f"  {sym}: api error {rs.error_code}", flush=True)
            err_count += 1
            time.sleep(SLEEP)
            continue

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            print(f"  {sym}: 无数据返回", flush=True)
            err_count += 1
            time.sleep(SLEEP)
            continue

        for r in rows:
            d, amt = r[0], r[1]
            if d and amt and amt != "":
                batch.append((float(amt), sym, d))

        err_count = 0

    except Exception as e:
        print(f"  {sym}: 异常 {e}", flush=True)
        err_count += 1
        time.sleep(SLEEP)

    # 批量刷入
    if len(batch) >= BATCH_SIZE:
        try:
            cur.executemany(
                "UPDATE stock_daily SET amount=? WHERE symbol=? AND date=?",
                batch,
            )
            conn.commit()
            updated += len(batch)
            batch.clear()
        except Exception as e:
            print(f"  批量写入失败: {e}", flush=True)
            failed += len(batch)
            batch.clear()

    # 进度
    if (i + 1) % 20 == 0:
        print(f"[{i+1}/{len(symbols_to_process)}] updated={updated} failed={failed}", flush=True)

    time.sleep(SLEEP)

# 最后一刷
if batch:
    try:
        cur.executemany(
            "UPDATE stock_daily SET amount=? WHERE symbol=? AND date=?",
            batch,
        )
        conn.commit()
        updated += len(batch)
    except Exception:
        failed += len(batch)

bs.logout()

# 验证
cur.execute("SELECT COUNT(*) FROM stock_daily WHERE amount IS NULL")
remaining = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM stock_daily WHERE amount IS NOT NULL")
filled = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM stock_daily")
total = cur.fetchone()[0]
conn.close()

print(f"\n{'='*50}", flush=True)
print(f"完成!", flush=True)
print(f"补填行数: {updated}", flush=True)
print(f"失败: {failed}", flush=True)
print(f"最终 amount 填充: {filled}/{total} ({filled/total*100:.1f}%)", flush=True)
print(f"剩余缺失: {remaining}", flush=True)
print(f"{'='*50}", flush=True)

# WxPusher 推送完成结果
_push_notification(updated, filled, total, remaining, failed)

