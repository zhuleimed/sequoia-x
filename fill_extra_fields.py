#!/usr/bin/env python3
"""
补填 stock_daily 表中的扩展字段：amount, pctChg, peTTM, pbMRQ, psTTM, pcfNcfTTM
从 2024-01-02（数据库最早日期）开始，全部补全。
"""
import baostock as bs
import sqlite3
import time
from datetime import datetime

DB = "/public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x/data/sequoia_v2.db"
WDPUSHER_TOKEN = "AT_hKGG0UfwrCP7bpcsO8cbQkrc4bZ9G3RX"
WXPUSHER_TOPIC_IDS = ["39277"]

# 需要补填的字段列表
TARGET_FIELDS = ["amount", "pctChg", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"]
# baostock 查询字段（与 TARGET_FIELDS 对应）
BS_FIELDS = "date,amount,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM"

BATCH_SIZE = 500
SLEEP = 0.05
MAX_RETRY = 10


def push_notification(updated: int, total_stocks: int, failed: int):
    """推送补填完成通知到微信。"""
    try:
        from wxpusher import WxPusher
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [
            f"✅ 扩展字段补填完成",
            f"时间: {now}",
            f"",
            f"处理股票: {total_stocks} 只",
            f"补填行数: {updated:,}",
            f"失败: {failed:,}",
        ]
        msg = "\n".join(lines)
        result = WxPusher.send_message(
            content=msg, token=WDPUSHER_TOKEN,
            topic_ids=WXPUSHER_TOPIC_IDS, content_type=1,
        )
        if result.get("code") == 1000:
            print("WxPusher 推送成功", flush=True)
        else:
            print(f"WxPusher 推送失败: {result}", flush=True)
    except Exception as e:
        print(f"WxPusher 推送异常: {e}", flush=True)


def to_bs_code(symbol: str) -> str:
    prefix = "sh" if symbol.startswith(("6", "9")) else "sz"
    return f"{prefix}.{symbol}"


def main():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    # ── 1. 获取所有需要补填的股票 ──
    # 任意目标字段为 NULL 即纳入
    null_conditions = " OR ".join(f"{f} IS NULL" for f in TARGET_FIELDS)
    cur.execute(f"SELECT DISTINCT symbol FROM stock_daily WHERE {null_conditions} ORDER BY symbol")
    symbols = [r[0] for r in cur.fetchall()]
    print(f"需补填扩展字段的股票: {len(symbols)} 只", flush=True)

    # ── 2. baostock 登录 ──
    lg = bs.login()
    if lg.error_code != "0":
        print(f"baostock 登录失败: {lg.error_msg}", flush=True)
        exit(1)
    print("baostock 登录成功", flush=True)

    updated = 0
    failed = 0
    batch = []
    err_count = 0

    for i, sym in enumerate(symbols):
        # 断连恢复
        if err_count >= MAX_RETRY:
            print(f"[{i}] 连续 {err_count} 次错误，重连...", flush=True)
            bs.logout()
            time.sleep(1)
            if bs.login().error_code != "0":
                print("重连失败，终止", flush=True)
                break
            err_count = 0

        # 找该股票任意目标字段为 NULL 的日期范围（取并集：整段日期）
        cur.execute(
            f"SELECT MIN(date), MAX(date) FROM stock_daily WHERE symbol=? AND ({null_conditions})",
            (sym,),
        )
        row = cur.fetchone()
        if not row or row[0] is None:
            continue
        start_date, end_date = row

        bs_code = to_bs_code(sym)
        try:
            rs = bs.query_history_k_data_plus(
                bs_code, BS_FIELDS,
                start_date=start_date, end_date=end_date,
                frequency="d", adjustflag="2",
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
                err_count += 1
                time.sleep(SLEEP)
                continue

            for r in rows:
                d = r[0]
                if not d:
                    continue
                # 构建更新值：只有 baostock 返回了有效数据才更新
                vals = {}
                for idx, field in enumerate(["amount", "pctChg", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"]):
                    v = r[idx + 1]  # r[0]=date, r[1]=amount, r[2]=pctChg, ...
                    if v and v.strip() and v.strip() != "":
                        try:
                            vals[field] = float(v)
                        except ValueError:
                            pass

                if vals:
                    batch.append((vals.get("amount"), vals.get("pctChg"),
                                  vals.get("peTTM"), vals.get("pbMRQ"),
                                  vals.get("psTTM"), vals.get("pcfNcfTTM"),
                                  sym, d))

            err_count = 0

        except Exception as e:
            print(f"  {sym}: 异常 {e}", flush=True)
            err_count += 1
            time.sleep(SLEEP)

        # 批量刷入
        if len(batch) >= BATCH_SIZE:
            try:
                cur.executemany(
                    """UPDATE stock_daily SET
                       amount=?, pctChg=?, peTTM=?, pbMRQ=?, psTTM=?, pcfNcfTTM=?
                       WHERE symbol=? AND date=?""",
                    batch,
                )
                conn.commit()
                updated += len(batch)
                batch.clear()
            except Exception as e:
                print(f"  批量写入失败: {e}", flush=True)
                failed += len(batch)
                batch.clear()

        # 每 50 只打一次进度
        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{len(symbols)}] updated={updated:,} failed={failed}", flush=True)

        time.sleep(SLEEP)

    # 最后一刷
    if batch:
        try:
            cur.executemany(
                """UPDATE stock_daily SET
                   amount=?, pctChg=?, peTTM=?, pbMRQ=?, psTTM=?, pcfNcfTTM=?
                   WHERE symbol=? AND date=?""",
                batch,
            )
            conn.commit()
            updated += len(batch)
        except Exception:
            failed += len(batch)

    bs.logout()

    # ── 验证 ──
    print(f"\n{'='*50}", flush=True)
    print(f"补填完成!", flush=True)
    print(f"补填行数: {updated:,}", flush=True)
    print(f"失败: {failed:,}", flush=True)
    print()
    for field in TARGET_FIELDS:
        null_cnt = cur.execute(f"SELECT COUNT(*) FROM stock_daily WHERE {field} IS NULL").fetchone()[0]
        total = cur.execute("SELECT COUNT(*) FROM stock_daily").fetchone()[0]
        pct = null_cnt * 100.0 / total
        print(f"  {field}: 剩余 NULL={null_cnt:,}/{total:,} ({pct:.1f}%)", flush=True)
    print(f"{'='*50}", flush=True)

    conn.close()
    push_notification(updated, len(symbols), failed)


if __name__ == "__main__":
    main()
