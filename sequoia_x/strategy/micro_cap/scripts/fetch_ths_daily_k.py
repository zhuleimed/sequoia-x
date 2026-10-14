"""下载同花顺「全市场十年日 K」dump —— 把微盘回测历史从 6.8 年拉长到 10 年。

## 为什么需要

004 的 `stock_daily` 只有 2020-01-02 起的数据（6.8 年），里面**只含 2 次崩盘**
（2024-01、2022-04）。**只有 2 个样本，任何风控结论都站不稳。**

同花顺官方开源接口提供一个预签名直链，可以**一次性下载全市场十年的日 K**
（约 173 MB，实测 20 秒左右），正好用来把回测历史往前推到 2016 年，
多覆盖 2016-01 熔断、2017-2018 小盘熊市等几轮周期。

## 接口（照抄 023_THS/scripts/daily_dump_verify.py 的用法）

    GET https://fuyao.aicubes.cn/api/dump/market-dumps/daily-k/download-url
    Header: X-api-key: $HITHINK_FINANCE_API_KEY
    → data.presigned_url（5 分钟有效）→ 直接 GET 下载 parquet

## 注意

dump 是**快照**，会过期。本脚本把文件按内容日期命名缓存，重复跑不会重复下载。

用法：
    /home/zhulei/anaconda3/envs/zhulei_py312/bin/python fetch_ths_daily_k.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import requests

BASE = "https://fuyao.aicubes.cn"
KEY = os.environ.get("HITHINK_FINANCE_API_KEY", "")
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "dump"


def log(msg: str) -> None:
    print(msg, flush=True)


if not KEY:
    sys.exit("❌ 环境变量 HITHINK_FINANCE_API_KEY 未配置")

OUT_DIR.mkdir(parents=True, exist_ok=True)

log("向同花顺申请下载直链 ...")
r = requests.get(
    f"{BASE}/api/dump/market-dumps/daily-k/download-url",
    headers={"X-api-key": KEY},
    timeout=30,
).json()
if r.get("code") != 0:
    sys.exit(f"❌ 获取直链失败: code={r.get('code')} {r.get('message')}")

url = r["data"]["presigned_url"]
fn = url.split("?")[0].split("/")[-1]
path = OUT_DIR / fn
log(f"文件名: {fn}")

if path.exists() and path.stat().st_size > 1_000_000:
    log(f"✅ 已缓存，跳过下载: {path}  ({path.stat().st_size / 1e6:.1f} MB)")
else:
    log("开始下载 ...")
    t0 = time.time()
    resp = requests.get(url, timeout=600)
    resp.raise_for_status()
    tmp = path.with_suffix(".parquet.tmp")
    tmp.write_bytes(resp.content)
    tmp.replace(path)
    log(f"✅ 已下载: {path}  {len(resp.content) / 1e6:.1f} MB  用时 {time.time() - t0:.1f}s")

# ── 自检：产物必须能读、且内容对得上（铁律一：不能只看 exit=0）──
import pandas as pd

df = pd.read_parquet(path)
log("")
log("=" * 70)
log("自检")
log("=" * 70)
log(f"行数: {len(df):,}")
log(f"列: {list(df.columns)}")

dcol = next((c for c in df.columns if "date" in c.lower() or c in ("日期",)), None)
scol = next((c for c in df.columns if "code" in c.lower() or c in ("代码", "symbol")), None)
log(f"日期列: {dcol} | 代码列: {scol}")

if dcol:
    d = pd.to_datetime(df[dcol])
    log(f"日期范围: {d.min().date()} ~ {d.max().date()}  （覆盖 {d.dt.year.nunique()} 个年份）")
    log("逐年行数（前 20 年）:")
    log(d.dt.year.value_counts().sort_index().head(20).to_string())
if scol:
    log(f"股票数: {df[scol].nunique():,}")

log("")
log("前 3 行样例:")
log(df.head(3).to_string())
