"""校验同花顺十年日 K dump —— 能不能拿来把微盘回测拉长到 10 年？

## 为什么要校验

拉长历史的前提是：**同花顺这份数据，和 004 库里的数据对得上**。
如果两源在 2020-2026 的重叠期就不一致，那 2016-2019 那一段也不能信。

## 校什么

1. **结构**：日期范围 / 股票数 / 单位
2. **重叠期逐值比对**（2020-01-02 ~ 2026-10-13，与 `stock_daily` 比）：
   - 收盘价、成交量、成交额
3. **★ 退市股在不在**（这直接关系到幸存者偏差）：
   004 的 `stock_list` 里 5932 只只有 12 只有退市日，是**明显不正常**的
   （A 股 30 年退市/暂停上市的远不止 12 只）。
   如果 THS 这份 dump 里退市股齐全，那**顺手就修好了历史复盘的幸存者偏差**。

用法：
    /home/zhulei/anaconda3/envs/zhulei_py312/bin/python verify_ths_daily_k.py
"""

from __future__ import annotations

import glob
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[4]
DUMP_DIR = PROJ / "sequoia_x" / "strategy" / "micro_cap" / "data" / "dump"
DB = PROJ / "data" / "sequoia_v2.db"

pd.set_option("display.width", 200)


def log(msg: str = "") -> None:
    print(msg, flush=True)


files = sorted(glob.glob(str(DUMP_DIR / "*.parquet")))
if not files:
    raise SystemExit(f"❌ 没找到 dump，先跑 fetch_ths_daily_k.py")
path = Path(files[-1])
log(f"读取: {path.name}")
th = pd.read_parquet(path)
log(f"行数 {len(th):,}，列 {list(th.columns)}")

# ── 1. 结构 ──
# ⚠️ date_ms 是毫秒时间戳，有两个坑（都踩过）：
#   ① 必须指定 unit='ms'（不指定会被当成纳秒 ⇒ 解析成 1970-01-01）；
#   ② 时间戳是**北京时间的零点**（如 1476288000000 ms = 2016-10-13 00:00 北京
#      = 2016-10-12 16:00 UTC）。若按 UTC 归一化 ⇒ 整体**提前一天**，
#      与库比对会得到"收盘价差 1.4%、成交量差 25%"这种看似严重的假差异。
#      ⇒ 显式转到 Asia/Shanghai 再归一化。
th["date"] = (
    pd.to_datetime(th["date_ms"], unit="ms", utc=True)
    .dt.tz_convert("Asia/Shanghai")
    .dt.normalize()
    .dt.tz_localize(None)
)
th["symbol"] = th["thscode"].astype(str).str.split(".").str[0].str.zfill(6)

log("")
log("=" * 78)
log("第 1 步：结构")
log("=" * 78)
log(f"日期范围: {th['date'].min().date()} ~ {th['date'].max().date()}")
log(f"股票数: {th['symbol'].nunique():,}")
log(f"交易日数: {th['date'].nunique():,}")
log(f"adjusted 取值: {th['adjusted'].unique().tolist()}   （none = 不复权）")
log("逐年行数:")
log(th["date"].dt.year.value_counts().sort_index().to_string())
log("")
log("单位抽查（000001.SZ 最新一天）—— 应有 成交额/成交量 ≈ 收盘价:")
s = th[(th["symbol"] == "000001")].sort_values("date").iloc[-1]
log(f"  close={s['close_price']}  volume={s['volume']:,.0f}  turnover={s['turnover']:,.0f}")
log(f"  turnover/volume = {s['turnover'] / s['volume']:.4f}   ← 应≈close")

# ── 2. 与 004 库重叠期比对 ──
log("")
log("=" * 78)
log("第 2 步：与 004 库在重叠期逐值比对")
log("=" * 78)
con = sqlite3.connect(DB)
db = pd.read_sql("select symbol, date, close, volume, amount from stock_daily", con)
sl = pd.read_sql("select symbol, name, delisted_date from stock_list", con)
con.close()
db["date"] = pd.to_datetime(db["date"])

# 先统一改名再合并：避免 merge 的 suffixes 只在列名冲突时才生效这种坑
db_r = db.rename(columns={"close": "db_close", "volume": "db_volume", "amount": "db_amount"})
th_r = th[["symbol", "date", "close_price", "volume", "turnover"]].rename(
    columns={"close_price": "ths_close", "volume": "ths_volume", "turnover": "ths_turnover"}
)
m = db_r.merge(th_r, on=["symbol", "date"], how="inner")
log(f"可比行数: {len(m):,}   日期 {m['date'].min().date()} ~ {m['date'].max().date()}")

for col, tcol, name in [("db_close", "ths_close", "收盘价"),
                        ("db_volume", "ths_volume", "成交量"),
                        ("db_amount", "ths_turnover", "成交额")]:
    d = m[[col, tcol]].dropna()
    rel = (d[col] - d[tcol]).abs() / d[tcol].abs().replace(0, np.nan)
    log(f"  {name:<6} 相对误差 中位数={rel.median():.6%}  均值={rel.mean():.6%}  "
        f">1% 的行数={int((rel > 0.01).sum()):,}")

# ── 3. ★ 退市股 ──
log("")
log("=" * 78)
log("第 3 步：★ 退市股覆盖（幸存者偏差）")
log("=" * 78)
db_syms = set(db["symbol"].unique())
th_syms = set(th["symbol"].unique())
log(f"004 库股票数:  {len(db_syms):,}")
log(f"THS dump 股票数: {len(th_syms):,}")
log(f"THS 有、004 没有的: {len(th_syms - db_syms):,}   ← 这些很可能就是**退市股**")
log(f"004 有、THS 没有的: {len(db_syms - th_syms):,}")

only_ths = sorted(th_syms - db_syms)
if only_ths:
    # 看这些代码的"最后交易日"，若很早 ⇒ 确认是退市
    last = th[th["symbol"].isin(only_ths)].groupby("symbol")["date"].max()
    log("")
    log("这些股票的最后交易日分布:")
    log(f"  最后交易日 < 2025-01-01 的（基本可确认是退市/暂停上市）: "
        f"{int((last < pd.Timestamp('2025-01-01')).sum()):,} 只")
    log("  举例（代码 / 最后交易日 / 名称）:")
    nm = dict(zip(sl["symbol"], sl["name"]))
    for s in only_ths[:12]:
        log(f"    {s}  {last[s].date()}  {nm.get(s, '(004 无此股)')}")

log("")
log("=" * 78)
log("结论怎么用")
log("=" * 78)
log("  · 若第 2 步重叠期误差极小 ⇒ THS 数据可信，可用来把回测拉到 2016 年。")
log("  · 若第 3 步显示 THS 含大量 004 缺失的退市股 ⇒ 顺带修好了幸存者偏差，")
log("    历史复盘里「结果偏乐观」的警告（缺陷 3）可以部分解除。")
