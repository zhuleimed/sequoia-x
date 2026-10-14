"""微盘策略历史复盘 —— 回答"微盘到底哪一年、跌了多少、跌了多久"。

## 这个脚本要回答的 4 个问题

1. **微盘长期真的有超额吗？** 有多少？
2. **它崩过几次？每次跌多少、跌多久？**（用户记不清是 2024 还是 2025，用数据钉死）
3. **微盘 vs 中证1000 vs 沪深300** 的相对表现怎么变？
4. **如果买了不管，最坏情况有多惨？**（最大回撤 —— 这直接决定风控规则要定多严）

## 怎么做到"不偷看未来"（避免 look-ahead bias）

- 在**每月最后一个交易日收盘后**，才按当天市值排序选出池子；
- 池子在**下一个月**持有并计算收益。
- ⇒ 选股用的是"当时能看到的信息"，收益是"之后才发生的"，中间没有穿越。

## 已知的口径缺陷（先说清楚，别当成完美结果）

1. **ST 股用"今天的名字"过滤**：`stock_list.name` 只有一个最新值，
   所以 2023 年才被 ST 的股票，在 2020 年就已经被我剔除了 ⇒ 偏乐观一点点（用未来信息）。
2. **次新股按"库内首次出现日 + 180 天"剔除**：2020 年之前上市的老股在库里首日=2020-01-02，
   识别不出来是不是次新（但 2020 年初确有 2019 年底上市的新股混在里面）。
3. **库里有幸存者偏差嫌疑**：`stock_list` 5932 只里只有 12 只有退市日。

以上三条都会让结果偏乐观。**读结论时请把这个方向记在心里。**

用法：
    /home/zhulei/anaconda3/envs/zhulei_py312/bin/python analyze_microcap_history.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[4]
CACHE = PROJ / "sequoia_x" / "strategy" / "micro_cap" / "data" / "market_cap.parquet"
DB = PROJ / "data" / "sequoia_v2.db"
OUT_CSV = PROJ / "sequoia_x" / "strategy" / "micro_cap" / "data" / "microcap_monthly.csv"

pd.set_option("display.width", 220)


def log(msg: str = "") -> None:
    print(msg, flush=True)


# ── 读数据 ──
log("读取市值缓存 + 日线 ...")
mc = pd.read_parquet(CACHE)
con = sqlite3.connect(DB)
d = pd.read_sql("select symbol, date, close, pctChg, volume from stock_daily", con)
names = pd.read_sql("select symbol, name from stock_list", con)
con.close()

d["date"] = pd.to_datetime(d["date"])
mc["date"] = pd.to_datetime(mc["date"])
log(f"日线 {len(d):,} 行；市值 {len(mc):,} 行")

df = d.merge(mc[["symbol", "date", "total_mv", "float_mv"]], on=["symbol", "date"], how="left")
df = df.merge(names, on="symbol", how="left")

# ── 过滤（都写成"当天可知"的规则，避免穿越）──
n0 = len(df)
df = df[df["volume"] > 0]                                  # 停牌日不参与
df = df[df["pctChg"].notna()]
df = df[df["total_mv"].notna() & (df["total_mv"] > 0)]     # 市值算得出来
df = df[~df["name"].astype(str).str.contains("ST", na=False)]  # ⚠️ 用今天的名字，见注释缺陷1
first_date = df.groupby("symbol")["date"].transform("min")
age = (df["date"] - first_date).dt.days
df = df[~((first_date > pd.Timestamp("2020-01-02")) & (age < 180))]  # ⚠️ 次新，见注释缺陷2
log(f"过滤: {n0:,} → {len(df):,} 行（剔停牌/无市值/ST/次新）")

df["ym"] = df["date"].dt.to_period("M")

# ── 每只股票每月收益（用 log 相加再 exp，比 apply lambda 快几十倍）──
df["logret"] = np.log1p(df["pctChg"] / 100.0)
mret = df.groupby(["symbol", "ym"])["logret"].sum().rename("ret").reset_index()
mret["ret"] = np.expm1(mret["ret"])
log(f"月度收益表: {len(mret):,} 条（股票×月）")

# ── 月末快照：选池子 ──
is_me = df["date"] == df.groupby("ym")["date"].transform("max")
me = df[is_me][["ym", "symbol", "total_mv", "float_mv"]].copy()
me["rank_tot"] = me.groupby("ym")["total_mv"].rank(pct=True)
me["rank_flt"] = me.groupby("ym")["float_mv"].rank(pct=True)
me["hold_ym"] = me["ym"] + 1   # 本月末选出的池子，下个月持有
log(f"月末快照: {len(me):,} 条，覆盖 {me['ym'].nunique()} 个月 "
    f"({me['ym'].min()} ~ {me['ym'].max()})")


def portfolio(rank_col: str, q: float) -> pd.Series:
    """在月末按 rank_col 取最小的 q 比例，等权持有下一个月，返回月度收益。"""
    m = me[me[rank_col] <= q][["hold_ym", "symbol"]].rename(columns={"hold_ym": "ym"})
    sub = mret.merge(m, on=["ym", "symbol"], how="inner")
    return sub.groupby("ym")["ret"].mean()


# ── 对照基准 ──
con = sqlite3.connect(DB)
idx = pd.read_sql(
    "select symbol, date, close from index_daily where symbol in ('sh.000300','sh.000852','sh.000905')",
    con,
)
con.close()
idx["date"] = pd.to_datetime(idx["date"])
idx["ym"] = idx["date"].dt.to_period("M")
im = idx.groupby(["ym", "symbol"])["close"].last().unstack()
idxret = im.pct_change()

# ── 汇总 ──
out = pd.DataFrame(
    {
        "微盘10%(总市值)": portfolio("rank_tot", 0.10),
        "微盘20%(总市值)": portfolio("rank_tot", 0.20),
        "微盘30%(总市值)": portfolio("rank_tot", 0.30),
        "微盘20%(流通市值)": portfolio("rank_flt", 0.20),
        "中证1000": idxret["sh.000852"],
        "中证500": idxret["sh.000905"],
        "沪深300": idxret["sh.000300"],
    }
)
out = out.dropna(how="all")
out["微盘20-中证1000"] = out["微盘20%(总市值)"] - out["中证1000"]
out["微盘20-沪深300"] = out["微盘20%(总市值)"] - out["沪深300"]

log("")
log("=" * 100)
log("【问题 1】年度收益（%）—— 微盘长期到底有没有超额")
log("=" * 100)
ann = out.groupby(out.index.year).apply(lambda x: (1 + x).prod() - 1) * 100
log(ann.round(1).to_string())
log("")
log("怎么读：最后一列『微盘20-沪深300』为正，表示「微盘那年比沪深300多涨了这么多」。")
log("        注意看有几年是负的、负多少 —— 那就是「微盘跑输」的年份。")

log("")
log("=" * 100)
log("【问题 2】最惨的 12 个月（按微盘20%）—— 崩盘到底发生在什么时候")
log("=" * 100)
worst = out.sort_values("微盘20%(总市值)").head(12)
show = (worst[["微盘10%(总市值)", "微盘20%(总市值)", "中证1000", "沪深300", "微盘20-沪深300"]] * 100).round(1)
log(show.to_string())
log("")
log("怎么读：第一列是「市值最小的 10% 那批股票，那个月平均跌了多少」。")
log("        -20 就是「平均跌两成」。连着一字跌停卖不掉的话，实际会更惨。")

log("")
log("=" * 100)
log("【问题 3】最大回撤（从最高点跌下来最深的一次，以及花了多久爬起来）")
log("=" * 100)


def max_dd(series: pd.Series) -> tuple[float, pd.Period, pd.Period, int]:
    """返回 (最大回撤, 峰值月, 谷底月, 从峰值到谷底的月数)。"""
    nav = (1 + series.fillna(0)).cumprod()
    peak = nav.cummax()
    dd = nav / peak - 1
    trough = dd.idxmin()
    at_peak = nav[:trough].idxmax()
    n_months = len(nav[at_peak:trough])
    return dd.min(), at_peak, trough, n_months


for col in ["微盘10%(总市值)", "微盘20%(总市值)", "中证1000", "沪深300"]:
    dd, pk, tr, n = max_dd(out[col])
    log(f"  {col:<20} 最大回撤 {dd:7.1%}   从 {pk} 跌到 {tr}（历时 {n} 个月）")

log("")
log("=" * 100)
log("【问题 4】逐月明细（微盘20% 为主）")
log("=" * 100)
detail = (out[["微盘10%(总市值)", "微盘20%(总市值)", "中证1000", "沪深300", "微盘20-沪深300"]] * 100).round(1)
log(detail.to_string())

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT_CSV, float_format="%.6f")
log("")
log(f"✅ 月度明细已存: {OUT_CSV}")
