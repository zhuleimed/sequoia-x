"""构建「每日市值」序列 —— 微盘策略的地基。

## 为什么要这个脚本

微盘策略的第一步是"按市值排序，买最小的那一批"。
但 004 的日线表 `stock_daily` **没有市值字段**，所以要先把它造出来。

## 市值是怎么算出来的

    市值 = 那一刻的真实股本 × 那一刻的真实股价

两个关键点：

1. **股本必须用"当时"的，不能用"今天"的。**
   因为股票会送转股（10 送 10 ⇒ 股本翻倍）。
   如果拿今天的股本去乘 5 年前的股价，会把小盘股误判成大盘股。
   → 历史股本从 `data/extra_features/xdxr/*.parquet`（除权除息表）里取，
     该表每只股票都记了历次「股本变化」事件的前/后总股本。

2. **股价必须用"不复权"的。**
   004 全库统一存不复权实际成交价（见项目 CLAUDE.md 复权口径铁律）。
   这正好配套：**送转股当天，股价除权下跌、股本同步翻倍 ⇒ 市值连续不变**。
   如果一边用不复权价、一边没有股本变化信息，市值就会出现假跳水。

## 输出的缓存

    micro_cap/data/market_cap.parquet
    列：symbol / date / total_mv（总市值，元）/ float_mv（流通市值，元）/ total_shares / float_shares

用法：
    /home/zhulei/anaconda3/envs/zhulei_py312/bin/python <本文件>
"""

from __future__ import annotations

import glob
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[4]  # 004_sequoia-x/
XDXR_DIR = PROJ / "data" / "extra_features" / "xdxr"
DB = PROJ / "data" / "sequoia_v2.db"
OUT_DIR = PROJ / "sequoia_x" / "strategy" / "micro_cap" / "data"
THS_DUMP_DIR = OUT_DIR / "dump"

# 价格来源：db = 004 生产库（2020-01 起，6.8 年）；ths = 同花顺十年 dump（2016-10 起，10 年）
# 用法：python build_market_cap.py [db|ths]
SOURCE = sys.argv[1] if len(sys.argv) > 1 else "db"
if SOURCE not in ("db", "ths"):
    sys.exit(f"❌ 用法: {Path(sys.argv[0]).name} [db|ths]")
OUT = OUT_DIR / (f"market_cap_{SOURCE}.parquet" if SOURCE == "ths" else "market_cap.parquet")


def log(msg: str) -> None:
    print(msg, flush=True)


# ── 第 1 步：把 5206 个 xdxr 文件读成一张事件表 ──
log("=" * 70)
log("第 1 步：读取 xdxr 除权除息表")
log("=" * 70)

files = sorted(glob.glob(str(XDXR_DIR / "*.parquet")))
log(f"文件数: {len(files)}")
if not files:
    sys.exit(f"❌ 找不到 xdxr 文件: {XDXR_DIR}")

ev = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
log(f"合并后 {len(ev):,} 行")

# 股票代码统一成 6 位字符串（'000001'）
# 末尾的 astype("object") 是必须的：str.zfill 在 pandas 2.x 会产出 StringDtype，
# 而 sqlite 读出来的 symbol 是 object，两者 merge_asof 时会报 dtype 不兼容。
if pd.api.types.is_numeric_dtype(ev["code"]):
    ev["symbol"] = ev["code"].astype("int64").astype(str).str.zfill(6).astype("object")
else:
    ev["symbol"] = (
        ev["code"].astype(str).str.split(".").str[0].str.zfill(6).astype("object")
    )

ev["date"] = pd.to_datetime(
    dict(year=ev["year"], month=ev["month"], day=ev["day"]), errors="coerce"
)
ev = ev[ev["date"].notna()]
log(f"有效日期行: {len(ev):,}")


def build_share_steps(col_before: str, col_after: str) -> pd.DataFrame:
    """把"前/后股本"事件表，转成"每个时点的股本"阶梯表。

    xdxr 一行记录的是「某个事件**发生时刻**的股本」：
    - 事件**之前**的股本 = 该行的"前股本"列
    - 事件**当天及之后**的股本 = 该行的"后股本"列

    所以对任意日期 t：股本(t) = 日期 ≤ t 的最近一条事件的"后股本"；
    若 t 早于所有事件，则用最早那条事件的"前股本"。

    做法：给每只股票**补一条 1900 年的虚拟事件**，其"后股本" = 最早事件的"前股本"
    ⇒ 之后统一用 merge_asof(backward) 即可，不用写特殊分支。
    """
    # ⚠️ xdxr 里有些记录的股本被填成 0.0（不是留空）。实测有两种害处：
    #   ① 有 156 只股票因此算出市值 0；
    #   ② 更隐蔽的：同一天会有两行（一行是真变动、一行是股本填 0 的垃圾行，name='15'），
    #      下面取"当天最后一条"时会正好选中垃圾行，导致当天股本变动**被静默丢弃**
    #      （实测 600136：真值 204082.6 万股被读成变动前的 58309.3 万股，市值差 71%）。
    # 股数永远不可能是 0 ⇒ 先统一把 0 当"缺失"，再据此剔除整行无效记录。
    # 依据：项目 CLAUDE.md「amount 单位铁律」——"0.0 读作真实值为 0，NULL 才是未知"。
    sub = ev.copy()
    for col in (col_before, col_after):
        sub[col] = sub[col].replace(0, np.nan)

    # 必须在"0→缺失"之后再筛，否则垃圾行会以 0.0 的形态混进来
    sub = sub[sub[col_after].notna() | sub[col_before].notna()].copy()

    sub["shares_after"] = sub[col_after]
    sub["shares_before"] = sub[col_before]

    # 同一天可能有多个事件：取最后一条（xdxr 文件按时间排序，最后一条是"最终结果"）
    sub = sub.sort_values(["symbol", "date"]).drop_duplicates(
        ["symbol", "date"], keep="last"
    )

    # 主阶梯：事件当天及之后 = 后股本
    main = sub[sub["shares_after"].notna()][["symbol", "date", "shares_after"]].copy()

    # 虚拟起点：每个股票最早事件的"前股本"，日期设成 1900-01-01
    first = (
        sub.sort_values("date")
        .drop_duplicates("symbol", keep="first")[["symbol", "shares_before", "shares_after"]]
        .copy()
    )
    # 前股本缺失或是 0（上市首条记录常见）时，退回用后股本
    first["shares_after"] = np.where(
        first["shares_before"].fillna(0) > 0, first["shares_before"], first["shares_after"]
    )
    first = first[first["shares_after"].notna()][["symbol", "shares_after"]]
    first["date"] = pd.Timestamp("1900-01-01")

    steps = pd.concat([first, main], ignore_index=True)
    return steps.sort_values(["symbol", "date"]).reset_index(drop=True)


log("构建 总股本 阶梯 ...")
total_steps = build_share_steps("qianzongguben", "houzongguben")
log(f"  阶梯点数: {len(total_steps):,}，覆盖股票 {total_steps['symbol'].nunique()} 只")

log("构建 流通股本 阶梯 ...")
float_steps = build_share_steps("panqianliutong", "panhouliutong")
log(f"  阶梯点数: {len(float_steps):,}，覆盖股票 {float_steps['symbol'].nunique()} 只")


# ── 第 2 步：读日线，把股本贴到每一天 ──
log("")
log("=" * 70)
log("第 2 步：与日线对齐（merge_asof 向后取最近一次股本）")
log("=" * 70)

def load_prices(source: str) -> pd.DataFrame:
    """读收盘价。两种来源（重叠期已实测逐值一致，收盘价/成交量误差中位数均为 0）：
       db  —— 004 生产库 stock_daily（2020-01 起）
       ths —— 同花顺「全市场十年日K」dump（2016-10 起）
    """
    if source == "db":
        con = sqlite3.connect(DB)
        d = pd.read_sql(
            "select symbol, date, close from stock_daily where close > 0", con
        )
        con.close()
    else:  # ths
        files = sorted(glob.glob(str(THS_DUMP_DIR / "*.parquet")))
        if not files:
            sys.exit("❌ 没找到 THS dump，请先跑 fetch_ths_daily_k.py")
        t = pd.read_parquet(files[-1])
        # ⚠️ date_ms 是**北京时间零点**的毫秒时间戳。必须转到 Asia/Shanghai 再归一化 ——
        #    按 UTC 解析会整体**提前一天**，实测会伪造出"收盘价差 1.4%、成交量差 25%"的假差异。
        t["date"] = (
            pd.to_datetime(t["date_ms"], unit="ms", utc=True)
            .dt.tz_convert("Asia/Shanghai")
            .dt.normalize()
            .dt.tz_localize(None)
        )
        t["symbol"] = t["thscode"].astype(str).str.split(".").str[0].str.zfill(6)
        d = t.rename(columns={"close_price": "close"})[["symbol", "date", "close"]]
        d = d[d["close"] > 0]
    # 统一到微秒精度：THS 的时间戳解析出来是 M8[ms]，而 xdxr 侧是 M8[us]，
    # 精度不一致会让 merge_asof 直接抛 dtype 不兼容（实测踩到）。
    d["date"] = pd.to_datetime(d["date"]).dt.as_unit("us")
    d["symbol"] = d["symbol"].astype("object")
    return d


daily = load_prices(SOURCE)
log(f"价格来源: {SOURCE} | 行数: {len(daily):,}，股票 {daily['symbol'].nunique()} 只 "
    f"| {daily['date'].min().date()} ~ {daily['date'].max().date()}")

# ⚠️ pandas 3.0 的 merge_asof(by=...) 要求**只按 on 键(date)全局排序**。
# 这跟 pandas 2.x 的习惯（按 [by, on] 即 [symbol, date] 排序）正好相反 ——
# 实测：按 [symbol, date] 排会报 "left keys must be sorted"。
daily = daily.sort_values("date").reset_index(drop=True)
total_steps = total_steps.sort_values("date").reset_index(drop=True)
float_steps = float_steps.sort_values("date").reset_index(drop=True)

mc = pd.merge_asof(
    daily,
    total_steps.rename(columns={"shares_after": "total_shares"}),
    on="date",
    by="symbol",
    direction="backward",
)
mc = pd.merge_asof(
    mc,
    float_steps.rename(columns={"shares_after": "float_shares"}),
    on="date",
    by="symbol",
    direction="backward",
)

# ── 第 3 步：算市值 ──
# 股本单位：xdxr 用「万股」⇒ ×1e4 换成「股」
mc["total_mv"] = mc["total_shares"] * 1e4 * mc["close"]
mc["float_mv"] = mc["float_shares"] * 1e4 * mc["close"]


# ── 第 4 步：运行时自检（铁律一：产物必须自检，不能只看 exit=0）──
log("")
log("=" * 70)
log("第 3 步：自检")
log("=" * 70)

n_all = len(mc)
ok_total = mc["total_mv"].notna().sum()
ok_float = mc["float_mv"].notna().sum()
log(f"总市值 可算出: {ok_total:,}/{n_all:,} = {ok_total / n_all:.2%}")
log(f"流通市值 可算出: {ok_float:,}/{n_all:,} = {ok_float / n_all:.2%}")

missing = mc[mc["total_mv"].isna()]["symbol"].value_counts()
log(f"总市值算不出的股票数: {len(missing)}")
if len(missing):
    log(f"  （这些多为新上市、xdxr 尚无股本记录）前 10: {list(missing.index[:10])}")

bad = mc[(mc["total_mv"].notna()) & (mc["total_mv"] <= 0)]
log(f"市值 ≤ 0 的异常行: {len(bad)}  ← 必须是 0")

mc_valid = mc[mc["total_mv"].notna()]
log("")
log("市值分布（最近一个交易日，单位：亿元）:")
last_day = mc_valid["date"].max()
snap = mc_valid[mc_valid["date"] == last_day]
q = snap["total_mv"].quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]) / 1e8
for p, v in q.items():
    log(f"  {p:>5.0%} 分位: {v:,.1f}")
log(f"  日期: {last_day.date()}，股票数: {len(snap)}")

log("")
log("抽查 3 只知名股票（最近交易日）:")
for sym in ["600519", "000001", "300750"]:
    r = snap[snap["symbol"] == sym]
    if len(r):
        r = r.iloc[0]
        log(
            f"  {sym}: 收盘 {r['close']:.2f} 元 × 总股本 {r['total_shares'] / 1e4:,.1f} 亿股"
            f" ⇒ 总市值 {r['total_mv'] / 1e8:,.1f} 亿元"
        )
    else:
        log(f"  {sym}: 无数据")

# 交叉校验：与 tdx_finance_snapshot 的"最新总股本"比
con = sqlite3.connect(DB)
snapdb = pd.read_sql("select symbol, total_shares from tdx_finance_snapshot", con)
con.close()
mine = snap[["symbol", "total_shares"]].copy()
mine["total_shares"] = mine["total_shares"] * 1e4
cmp = snapdb.merge(mine, on="symbol", suffixes=("_db", "_mine"))
cmp = cmp[cmp["total_shares_mine"].notna()]
cmp["rel_err"] = (cmp["total_shares_mine"] - cmp["total_shares_db"]).abs() / cmp[
    "total_shares_db"
].replace(0, np.nan)
log("")
log(f"交叉校验：重建股本 vs 数据库快照股本（{len(cmp):,} 只可比）")
log(f"  相对误差中位数: {cmp['rel_err'].median():.6%}")
log(f"  相对误差 > 1% 的股票数: {(cmp['rel_err'] > 0.01).sum()}  ← 越少越好")
if (cmp["rel_err"] > 0.01).sum():
    worst = cmp.nlargest(5, "rel_err")[["symbol", "total_shares_db", "total_shares_mine", "rel_err"]]
    log(worst.to_string(index=False))

if ok_total / n_all < 0.95 or len(bad) > 0:
    log("")
    log("❌ 自检未通过 ⇒ 不写缓存")
    sys.exit(1)

# ── 第 5 步：写缓存 ──
OUT_DIR.mkdir(parents=True, exist_ok=True)
out = mc[
    ["symbol", "date", "total_mv", "float_mv", "total_shares", "float_shares"]
].copy()
out.to_parquet(OUT, index=False)
log("")
log("=" * 70)
log(f"✅ 已写出: {OUT}  ({len(out):,} 行, {OUT.stat().st_size / 1e6:.1f} MB)")
log("=" * 70)
