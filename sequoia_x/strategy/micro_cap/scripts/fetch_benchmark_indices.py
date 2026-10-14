"""补齐微盘策略的对照基准指数（任务 3）。

## 为什么要补

004 的库里 `index_daily` 只有 7 个指数，**没有中证2000、没有微盘股指数**。
微盘策略跑出收益后，如果没有合适的尺子，就无法判断"是策略好还是小盘风格好"。

中证2000 是 A 股官方口径里最接近"微盘"的宽基（样本为剔除中证800、中证1000 后
市值靠后的 2000 只），是微盘策略最直接的对照基准。

## 为什么不直接写进生产库

`data/sequoia_v2.db` 由项目的每日流水线管理，且项目有严格的数据入库规范
（单位/复权口径/入库前自证）。新增指数属于**数据治理决策**，应先与项目流程对齐。
⇒ 本脚本把指数存到本目录的 `data/index_extra.parquet`，分析脚本自行合并。
（如需入库，另行确认后加一步插库即可。）

## 校验

用**中证1000**做对照：东财取的中证1000 应当与库里 `sh.000852` 高度一致
⇒ 证明东财这条数据线可信，那么同源取的中证2000 也可信。

用法：
    /home/zhulei/anaconda3/envs/zhulei_py312/bin/python fetch_benchmark_indices.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[4]
DB = PROJ / "data" / "sequoia_v2.db"
OUT_DIR = PROJ / "sequoia_x" / "strategy" / "micro_cap" / "data"
OUT = OUT_DIR / "index_extra.parquet"

# 起点用 2013：中证2000 虽然 2023-08 才发布，但官方回溯到 2013-12-31，
# 东财能取到全历史 ⇒ 拿到 13 年小盘历史（含 2015 股灾 / 2016 熔断 / 2017-18 熊市）。
START, END = "20130101", "20261013"

# 代码 → (中文名, 东财代码)。东财的宽基指数用 6 位代码，中证2000 = 932000。
WANTED = {
    "csi2000": ("中证2000", "932000"),
    "gn2000": ("国证2000", "399303"),
    "csi1000": ("中证1000", "000852"),   # 仅用于交叉校验，库里已有
}


def log(msg: str = "") -> None:
    print(msg, flush=True)


log("拉取指数日线（东财）...")
import akshare as ak

frames = []
for key, (cn, code) in WANTED.items():
    df = ak.index_zh_a_hist(symbol=code, period="daily", start_date=START, end_date=END)
    df = df.rename(columns={"日期": "date", "收盘": "close", "成交额": "amount"})
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = key
    frames.append(df[["symbol", "date", "close", "amount"]])
    log(f"  {cn}({code}): {len(df):,} 行  {df['date'].min().date()} ~ {df['date'].max().date()}")

extra = pd.concat(frames, ignore_index=True)

# ── 交叉校验：东财中证1000 vs 库里 sh.000852 ──
log("")
log("交叉校验：东财 中证1000 vs 库内 sh.000852")
con = sqlite3.connect(DB)
db = pd.read_sql(
    "select date, close from index_daily where symbol='sh.000852'", con
)
con.close()
db["date"] = pd.to_datetime(db["date"])

mine = extra[extra["symbol"] == "csi1000"][["date", "close"]].rename(columns={"close": "em"})
m = db.merge(mine, on="date", how="inner")
m["rel"] = (m["close"] - m["em"]).abs() / m["em"]
log(f"  可比交易日: {len(m):,}")
log(f"  相对误差中位数: {m['rel'].median():.6%}")
log(f"  相对误差 > 1% 的天数: {(m['rel'] > 0.01).sum()}")
log(f"  库内中证1000 最新日期: {db['date'].max().date()}")
log(f"  东财中证1000 最新日期: {mine['date'].max().date()}")

if m["rel"].median() > 0.005:
    log("")
    log("❌ 校验未通过：东财与库内中证1000 差异过大，中证2000 不可信")
    sys.exit(1)

# 顺带看看库里中证1000 缺了哪几天
gap = mine[~mine["date"].isin(db["date"])]
log("")
log(f"东财有、库内没有的交易日（库内中证1000 的缺口）: {len(gap)} 天")
if len(gap):
    log(f"  {gap['date'].min().date()} ~ {gap['date'].max().date()}")

OUT_DIR.mkdir(parents=True, exist_ok=True)
extra.to_parquet(OUT, index=False)
log("")
log(f"✅ 已写出: {OUT}  ({len(extra):,} 行)")
