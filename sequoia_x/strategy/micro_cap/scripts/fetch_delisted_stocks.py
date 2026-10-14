"""补齐「退市股」行情 —— 修幸存者偏差。

## 问题有多大（实测）

用 baostock 的「历史某一天的 A 股全集」对拍：2018-01-02 有 3470 只 A 股，
其中 **235 只**到 2026-10-13 已消失（「退市新亿」「*ST世茂」「退市罗顿」…）。
004 库和同花顺 dump **都不含**这些股票 ⇒ **回测系统性高估收益**。

## 两个坑（都踩过，所以这样写）

1. **必须带 `turn`（换手率）**：退市股**没有 xdxr 股本文件** ⇒ 算不出市值 ⇒ 进不了微盘池。
   解法：`流通市值 = 成交额 ÷ (换手率/100)`。
   已实测两个来源口径一致（平安银行：xdxr 流通股本 1,940,568.5 万股
   vs baostock 反推 1,940,583 万股，**差 0.0007%**）⇒ 可混用。
2. **必须逐日/逐股缓存**：baostock 每次查询约 40 秒，第一次写脚本时把缓存放在循环
   外面「全部查完再存」，结果第 6 天查询返回空值直接崩掉，**前 5 天白跑**。
   ⇒ 现在每完成一天/一只就落盘，**崩溃后重跑同一命令即可续跑**（项目铁律二）。

用法（崩溃/中断后**重跑同一条命令即可**）：
    /home/zhulei/anaconda3/envs/zhulei_py312/bin/python fetch_delisted_stocks.py
"""

from __future__ import annotations

import glob
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[4]
BASE = PROJ / "sequoia_x" / "strategy" / "micro_cap"
UDIR = BASE / "data" / "baostock_universe"      # 逐日缓存的股票全集
KDIR = BASE / "data" / "delisted_daily"         # 逐只缓存的日线
OUT = BASE / "data" / "delisted_daily.parquet"
XDXR_DIR = PROJ / "data" / "extra_features" / "xdxr"
DB = PROJ / "data" / "sequoia_v2.db"

START, END = "2016-10-01", "2026-10-13"
PROBE_DAYS = ["2016-10-13", "2017-06-01", "2018-06-01", "2019-06-03",
              "2020-06-01", "2021-06-01", "2022-06-01", "2023-06-01",
              "2024-06-03", "2025-06-03", "2026-10-13"]
FIELDS = "date,code,open,high,low,close,volume,amount,turn,pctChg,tradestatus"
A_SHARE = r"^(sh\.(60|68)|sz\.(00|30))\d{4}$"


def log(msg: str = "") -> None:
    print(msg, flush=True)


import baostock as bs  # noqa: E402

bs.login()


def all_stock(day: str, tries: int = 3) -> pd.DataFrame:
    """取某日的全部证券；返回 A 股子集。空结果/报错时重试，仍失败则返回空表。"""
    for k in range(tries):
        rs = bs.query_all_stock(day=day)
        if rs.error_code != "0":
            time.sleep(3)
            continue
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        if rows:
            d = pd.DataFrame(rows, columns=rs.fields)
            if "code" in d.columns:
                return d[d["code"].str.match(A_SHARE)].copy()
        time.sleep(3)
    return pd.DataFrame(columns=["code", "tradeStatus", "code_name"])


UDIR.mkdir(parents=True, exist_ok=True)
KDIR.mkdir(parents=True, exist_ok=True)

# ── 第 1 步：逐日取股票全集（每天单独缓存 ⇒ 可续跑）──
log("第 1 步：取窗口内各日的 A 股全集（每天单独缓存，崩溃可续跑）")
frames = []
for day in PROBE_DAYS:
    f = UDIR / f"{day}.parquet"
    if f.exists():
        d = pd.read_parquet(f)
    else:
        t0 = time.time()
        d = all_stock(day)
        d["day"] = pd.Timestamp(day)
        d.to_parquet(f, index=False)
        log(f"  {day}: A股 {len(d):,} 只  ({time.time() - t0:.0f}s)")
    frames.append(d)
uni = pd.concat(frames, ignore_index=True)

# ── 第 2 步：找出「窗口内曾存在、但我们没有行情」的股票 ──
th = pd.read_parquet(sorted(glob.glob(str(BASE / "data" / "dump" / "*.parquet")))[-1])
have = set(th["thscode"].astype(str).str.split(".").str[0].str.zfill(6))
con = sqlite3.connect(DB)
have |= set(pd.read_sql("select distinct symbol from stock_daily", con)["symbol"])
con.close()

uni["symbol"] = uni["code"].str.split(".").str[1]
ever = set(uni["symbol"])
missing = sorted(ever - have)
xd = {p.stem for p in Path(XDXR_DIR).glob("*.parquet")}

log("")
log("=" * 78)
log("对拍结果")
log("=" * 78)
log(f"窗口内曾存在的 A 股: {len(ever):,}")
log(f"已有行情的股票:      {len(have):,}")
log(f"★ 缺失的股票:        {len(missing):,}  ← 幸存者偏差的量")
log(f"  其中缺 xdxr 股本的: {len([s for s in missing if s not in xd])} 只 "
    f"⇒ 这批只能靠 `流通市值 = 成交额 ÷ 换手率` 反推")

if not missing:
    log("无需补数据")
    bs.logout()
    sys.exit(0)

# ── 第 3 步：逐只拉日线（每只单独缓存 ⇒ 可续跑）──
log("")
log(f"第 3 步：逐只拉 {len(missing)} 只的日线（含 turn 换手率，不复权）")
t0, done, fail = time.time(), 0, 0
for i, sym in enumerate(missing, 1):
    f = KDIR / f"{sym}.parquet"
    if f.exists():
        done += 1
        continue
    code = ("sh." if sym[0] == "6" else "sz.") + sym
    rs = bs.query_history_k_data_plus(code, FIELDS, start_date=START, end_date=END,
                                      frequency="d", adjustflag="3")
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    if rows:
        pd.DataFrame(rows, columns=rs.fields).to_parquet(f, index=False)
        done += 1
    else:
        fail += 1
        pd.DataFrame(columns=FIELDS.split(",")).to_parquet(f, index=False)   # 记空，避免反复重试
    if i % 50 == 0 or i == len(missing):
        log(f"  {i}/{len(missing)}  有数据 {done}  空 {fail}  用时 {time.time() - t0:.0f}s")
bs.logout()

# ── 第 4 步：合并 ──
parts = []
for f in sorted(KDIR.glob("*.parquet")):
    d = pd.read_parquet(f)
    if len(d):
        d["symbol"] = f.stem
        parts.append(d)
if not parts:
    sys.exit("❌ 一只都没取到")
out = pd.concat(parts, ignore_index=True)
for c in ["open", "high", "low", "close", "volume", "amount", "turn", "pctChg"]:
    out[c] = pd.to_numeric(out[c], errors="coerce")
out["date"] = pd.to_datetime(out["date"])
out = out[out["close"] > 0].sort_values(["symbol", "date"])
# 流通市值（元）= 成交额 ÷ (换手率/100)
# ⚠️ 换手率为 0 或缺失时会产生 inf ⇒ 必须置回 NaN（否则排名会被 inf 带偏）
_turn_frac = out["turn"] / 100
out["float_mv_bs"] = (out["amount"] / _turn_frac).where(_turn_frac > 0)
out["float_mv_bs"] = out["float_mv_bs"].replace([float("inf"), float("-inf")], pd.NA)

log("")
log("=" * 78)
log("自检")
log("=" * 78)
log(f"行数: {len(out):,}   股票: {out['symbol'].nunique()}")
log(f"日期: {out['date'].min().date()} ~ {out['date'].max().date()}")
log(f"流通市值可算出: {out['float_mv_bs'].notna().sum():,}/{len(out):,} = "
    f"{out['float_mv_bs'].notna().mean():.1%}")
bad = out["float_mv_bs"].notna() & (out["float_mv_bs"] <= 0)
log(f"流通市值 ≤ 0 的异常行: {int(bad.sum())}   ← 必须是 0")
q = (out["float_mv_bs"].dropna().quantile([0, .25, .5, .75, 1.0]) / 1e8).round(2)
log("流通市值分位（亿元）:"); log(q.to_string())
if out["float_mv_bs"].notna().mean() < 0.7:
    log("")
    log("❌ 流通市值覆盖率过低，检出自检不通过 ⇒ 不写缓存")
    sys.exit(1)

out.to_parquet(OUT, index=False)
log("")
log(f"✅ 已写出: {OUT}  ({len(out):,} 行, {out['symbol'].nunique()} 只)")
