#!/usr/bin/env python3
"""财联社快讯采集（news 备源, 独立风控面）— 全量流 → 按股票归档

2026-08-07 实测: 签名 = md5(sha1(排序后的 query_string)), 无盐（errno 10012 是盐错）。
接口: https://api3.cls.cn/v1/roll/get_roll_list
每条快讯自带 stock_list(关联股票: StockID/名称/涨跌幅) → 免代码匹配。

与 collector 的 news 主源(东财 search-api)互为备源:
  - 东财被封时, 用本脚本的快讯流补新闻面
  - 快讯为全市场流, 适合日频增量, 不适合按股回补

用法(cron 日频, 避开 18:10-18:45 同步窗口):
  30 20 * * 1-5 cd <project> && py312 python scripts/pull_cls_news.py >> logs/pull_cls_news_$(date +%Y%m).log 2>&1

落盘: data/extra_features/news_cls/{code}.parquet (code/ctime/title/content)
增量: data/extra_features/news_cls/.cursor (上次游标 unix 秒)
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_DIR / "data/extra_features/news_cls"
CURSOR_FILE = OUT_DIR / ".cursor"
API_URL = "https://api3.cls.cn/v1/roll/get_roll_list"
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.cls.cn/"}
PAGE_SIZE = 5  # 财联社 rn 上限=5(实测, 更大返回空)
# 增量模式: 拉取距上次游标之前的快讯; 首次运行拉最近 3 天
DEFAULT_BACK_DAYS = 3


def cls_sign(params: dict) -> str:
    """签名: md5(sha1(排序 query_string)), 无盐（2026-08-07 实测）"""
    qs = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    return hashlib.md5(hashlib.sha1(qs.encode()).hexdigest().encode()).hexdigest()


def get_roll(last_time: str, rn: int = PAGE_SIZE) -> dict:
    params = {"app": "CailianpressWeb", "category": "", "last_time": last_time,
              "last_time_1": "", "os": "web", "refresh_type": "1",
              "rn": str(rn), "sv": "7.7.5"}
    params["sign"] = cls_sign(params)
    r = requests.get(API_URL, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()


def norm_code(stock_id: str) -> str:
    """sz000426 → 000426"""
    return stock_id[2:] if stock_id and len(stock_id) > 2 else stock_id


def pull(last_time: str, cutoff: int) -> list:
    """游标分页拉取, 返回 [(code, ctime, title, content), ...]"""
    rows, seen = [], set()
    while True:
        d = get_roll(str(last_time))
        if d.get("errno") != 0:
            print(f"  接口错误: {d.get('msg')}")
            break
        items = d["data"]["roll_data"]
        if not items:
            break
        newest = items[0]["ctime"]
        oldest = items[-1]["ctime"]
        if oldest < cutoff:
            items = [it for it in items if it["ctime"] >= cutoff]
            if not items:
                break
        for it in items:
            ctime = it.get("ctime")
            for st in it.get("stock_list") or []:
                code = norm_code(st.get("StockID", ""))
                if not code or not code.isdigit():
                    continue
                key = (code, ctime)
                if key in seen:
                    continue
                seen.add(key)
                rows.append((code, ctime,
                             str(it.get("title") or "")[:200],
                             str(it.get("content") or "")[:1000]))
        last_time = items[-1]["ctime"]
        if len(items) < PAGE_SIZE or oldest < cutoff:
            break
        time.sleep(0.3)  # 温和限流
    return rows


def save_rows(rows: list) -> None:
    """按股票追加归档(去重)"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_code: dict[str, list] = {}
    for code, ctime, title, content in rows:
        by_code.setdefault(code, []).append((ctime, title, content))

    total = 0
    for code, items in by_code.items():
        fp = OUT_DIR / f"{code}.parquet"
        new = pd.DataFrame(items, columns=["ctime", "title", "content"])
        new = new.drop_duplicates(subset="ctime")
        if fp.exists():
            old = pd.read_parquet(fp)
            merged = pd.concat([old, new]).drop_duplicates(subset="ctime").sort_values("ctime")
        else:
            merged = new.sort_values("ctime")
        merged.to_parquet(fp, index=False)
        total += len(merged)
    print(f"  归档 {len(by_code)} 只股票, 总记录 {total}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 增量游标: 有 .cursor 从游标开始, 无则拉最近 N 天
    cutoff = int(time.time()) - DEFAULT_BACK_DAYS * 86400
    last_time = ""
    if CURSOR_FILE.exists():
        last_time = CURSOR_FILE.read_text().strip()
        print(f"增量模式: 游标={last_time}")
    else:
        print(f"首次运行: 拉最近 {DEFAULT_BACK_DAYS} 天")

    rows = pull(last_time, cutoff)
    print(f"拉取 {len(rows)} 条(股票,快讯)关联")
    if rows:
        save_rows(rows)
        # 更新游标 = 最新一条 ctime（下次从这之后拉）
        newest = max(r[1] for r in rows)
        CURSOR_FILE.write_text(str(newest))
        print(f"游标更新: {newest}")
    print("DONE")


if __name__ == "__main__":
    main()
