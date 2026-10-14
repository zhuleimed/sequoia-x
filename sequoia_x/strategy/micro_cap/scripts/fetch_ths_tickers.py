"""抓同花顺全量股票元数据（上市日期 / 退市日期）—— 修两个老毛病。

## 为什么要它

微盘研究有两个长期挂着的口径缺陷：

1. **幸存者偏差**：004 的 `stock_list` 5932 只里**只有 12 只有退市日**，而 A 股 30 年退市的远不止这些。
   退市股如果压根不在数据里，回测就会**系统性高估收益**（那些烂掉的股票被无声剔除了）。
2. **次新股识别**：`stock_list.listed_date` **全是 NULL**，之前只能用"库内首次出现日 + 180 天"近似。

同花顺的 `/api/meta/tickers/list` 同时给出：
  - `list_date`（上市日期）⇒ 修缺陷 2
  - `end_date` / `last_trade_date`（终止日 / 最后交易日）⇒ 识别退市股，修缺陷 1

## 分页

实测用 `offset` 翻页有效（`page` 无效）。每页上限 500。

用法：
    /home/zhulei/anaconda3/envs/zhulei_py312/bin/python fetch_ths_tickers.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://fuyao.aicubes.cn"
URL = f"{BASE}/api/meta/tickers/list"
KEY = os.environ.get("HITHINK_FINANCE_API_KEY", "")
OUT = Path(__file__).resolve().parents[1] / "data" / "ths_tickers.parquet"
PAGE = 500


def log(msg: str = "") -> None:
    print(msg, flush=True)


if not KEY:
    sys.exit("❌ 环境变量 HITHINK_FINANCE_API_KEY 未配置")

log("分页抓取全部 A 股元数据 ...")
items, offset = [], 0
while True:
    r = requests.get(URL, headers={"X-api-key": KEY},
                     params={"asset_type": "a-share", "limit": PAGE, "offset": offset},
                     timeout=60).json()
    if r.get("code") != 0:
        sys.exit(f"❌ 第 {offset} 条起失败: code={r.get('code')} {r.get('message')}")
    batch = r["data"]["item"]
    if not batch:
        break
    items.extend(batch)
    offset += PAGE
    log(f"  已取 {len(items):,} 条 ...")
    if len(batch) < PAGE:
        break
    time.sleep(0.2)

df = pd.DataFrame(items)
df["symbol"] = df["thscode"].astype(str).str.split(".").str[0].str.zfill(6)
df["list_date"] = pd.to_datetime(df["list_date"], errors="coerce")
df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
df["last_trade_date"] = pd.to_datetime(df["last_trade_date"], errors="coerce")

log("")
log("=" * 78)
log("自检")
log("=" * 78)
log(f"总条数: {len(df):,}")
log(f"股票数（去重）: {df['symbol'].nunique():,}")
log(f"上市日期有值: {df['list_date'].notna().sum():,}")
dead = df[df["end_date"].notna() | df["last_trade_date"].notna()]
log(f"★ 有退出日（退市/终止上市）: {len(dead):,} 只  ← 004 库里只有 12 只")
if len(dead):
    log("  退出年份分布:")
    log((dead["end_date"].fillna(dead["last_trade_date"]).dt.year
         .value_counts().sort_index()).to_string())
    log("  举例:")
    log(dead[["symbol", "name", "list_date", "end_date"]].head(10).to_string(index=False))

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT, index=False)
log("")
log(f"✅ 已写出: {OUT}  ({len(df):,} 行)")
