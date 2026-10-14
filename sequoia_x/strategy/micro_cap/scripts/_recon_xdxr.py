"""临时侦察脚本：看 xdxr 表能不能支撑"重建历史股本"。

不产出正式数据，只回答问题：
  1. 每只股票有没有"总股本"记录？覆盖多少只？
  2. 每只股票最早一条 总股本 记录是什么时候？是否早于 2020-01-02（我们回测起点）？
  3. 若某股票在 2020 之后没有新的股本变化，靠"向前填充"能不能拿到 2020 的值？
"""

from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[4]  # 004_sequoia-x/
XDXR_DIR = PROJ / "data" / "extra_features" / "xdxr"
START = pd.Timestamp("2020-01-02")

files = sorted(glob.glob(str(XDXR_DIR / "*.parquet")))
print(f"xdxr 文件数: {len(files)}")

rows = []
for f in files:
    df = pd.read_parquet(f)
    rows.append(df)

ev = pd.concat(rows, ignore_index=True)
print(f"合并后总行数: {len(ev):,}")
print(f"列: {list(ev.columns)}")
print(f"code 样例: {ev['code'].head(3).tolist()}  dtype={ev['code'].dtype}")
print()

# 只看带"总股本"信息的行
has_ts = ev[ev["houzongguben"].notna() | ev["qianzongguben"].notna()].copy()
print(f"含 总股本 信息的行: {len(has_ts):,}")
print(f"涉及股票数: {has_ts['code'].nunique()}")
print()

has_ts["date"] = pd.to_datetime(
    dict(year=has_ts["year"], month=has_ts["month"], day=has_ts["day"]), errors="coerce"
)
print("事件类型(name)分布:")
print(has_ts["name"].value_counts().head(10))
print()

# 每只股票最早/最晚的股本事件
g = has_ts.groupby("code")["date"].agg(["min", "max", "count"])
print("最早股本事件 的分布:")
print(f"  早于 {START.date()} 的股票数: {(g['min'] < START).sum()} / {len(g)}")
print(f"  晚于 {START.date()} 的股票数: {(g['min'] >= START).sum()} / {len(g)}")
print()
print("每只股票股本事件条数 分位:")
print(g["count"].describe().round(1))
print()

# stock_daily 里有多少只股票，与 xdxr 对得上吗
import sqlite3

con = sqlite3.connect(PROJ / "data" / "sequoia_v2.db")
daily_syms = pd.read_sql("select distinct symbol from stock_daily", con)["symbol"]
print(f"stock_daily 股票数: {len(daily_syms)}")
xdxr_syms = set(has_ts["code"].astype(str).str.zfill(6))
daily_syms = set(daily_syms.astype(str).str.zfill(6))
print(f"xdxr 有股本记录的股票数: {len(xdxr_syms)}")
print(f"日线有、xdxr 无股本记录: {len(daily_syms - xdxr_syms)}  ← 这些股票建不出市值")
missing = sorted(daily_syms - xdxr_syms)[:20]
print(f"  前 20 个: {missing}")
print()
print("样例: 平安银行(000001) 的股本事件(近 10 条, 按日期)")
s = has_ts[has_ts["code"].astype(str).str.zfill(6) == "000001"].sort_values("date")
print(s[["date", "name", "qianzongguben", "houzongguben", "panhouliutong"]].tail(10).to_string())
