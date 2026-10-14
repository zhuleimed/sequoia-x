"""微盘研究共用的数据层 —— 一个地方定义「微盘池怎么选、收益怎么算」。

## 为什么要有这个模块

微盘研究已经有 3 个脚本（历史复盘 / 风控回测 / 稳健性检查），都要做同样几件事：
读日线 → 过滤 → 月末按市值选池 → 算池内等权日收益。
而且现在有**两种数据源**（004 库 6.8 年 / 同花顺 dump 10 年）。

如果每个脚本各写一遍，口径迟早会漂。所以把这段共用逻辑收在这里。

## 两种数据源

| source | 价格/收益 | 成交额 | 市值 | 覆盖 |
|---|---|---|---|---|
| `db`  | `stock_daily` 的 `pctChg` | `amount` | `market_cap.parquet` | 2020-01 ~ 至今（6.8 年） |
| `ths` | THS dump 的 `close_price` 算 | `turnover` | `market_cap_ths.parquet` | 2016-10 ~ 至今（10 年） |

两源在重叠期（2020-01 ~ 2026-10）已实测逐值一致：收盘价、成交量的相对误差中位数都是 0。

## 口径过滤（三个脚本统一）

1. `volume > 0` —— 停牌日不参与
2. 收益可得（`pctChg` / 前收盘价）
3. 市值算得出来且 > 0
4. **剔除 ST** —— ⚠️ 用的是**今天的名字**（库/源里只有最新名称）⇒ 用到了未来信息，**偏乐观**
5. **剔除次新**（库内首次出现 + 180 天）—— ⚠️ 2020 年前上市的老股识别不出来

## 已知缺陷（各脚本报告里都要照抄这一段，别只写在自己的日志里）

- 缺陷 4/5 都会让结果**偏乐观**。
- 库里只有 12 只有退市日 ⇒ **幸存者偏差仍在**（THS dump 也没解决：它多出的 349 只全是北交所，不是退市股）。

用法：
    from microcap_data import load, build_pool, daily_portfolio
    df = load("ths")
    members = build_pool(df, q=0.20)
    port = daily_portfolio(df, members)
"""

from __future__ import annotations

import glob
import sqlite3
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[4]
BASE = PROJ / "sequoia_x" / "strategy" / "micro_cap"
DATA = BASE / "data"
DB = PROJ / "data" / "sequoia_v2.db"

NEW_STOCK_DAYS = 180   # 次新剔除窗口（自然日）


def _norm_symbol(s: pd.Series) -> pd.Series:
    return s.astype(str).str.split(".").str[0].str.zfill(6).astype("object")


def _load_names() -> pd.DataFrame:
    con = sqlite3.connect(DB)
    n = pd.read_sql("select symbol, name from stock_list", con)
    con.close()
    n["symbol"] = _norm_symbol(n["symbol"])
    return n


# ── 「疑似 ST」的价格反推规则（无未来函数）──
# ST/*ST 的日涨跌幅被限制在 ±5%（普通主板 ±10%）。据此反推：
#   过去 ST_WIN 个交易日里 |涨跌幅| 从未超过 ST_CAP，且至少有 ST_HITS 天达到 ST_HIT 以上
# ⇒ 判为「疑似 ST」。只用 t 及之前的信息，**无未来函数**。
# 参数偏保守（窗口长、要求多次触及上限），是为了压低误杀 —— 实测召回/误报见 st_lookahead_check.py。
ST_WIN, ST_MINP = 120, 60
ST_CAP, ST_HIT, ST_HITS = 0.054, 0.046, 3


def st_proxy_flag(df: pd.DataFrame) -> pd.Series:
    """返回「疑似 ST」的布尔列（与 df 同索引，按 symbol 分组滚动计算）。"""
    d = df[["symbol", "ret"]].copy()
    g = d.groupby("symbol")["ret"]
    r_max = g.transform(lambda s: s.abs().rolling(ST_WIN, min_periods=ST_MINP).max())
    hits = g.transform(
        lambda s: s.abs().between(ST_HIT, ST_CAP).astype(float).rolling(ST_WIN, min_periods=ST_MINP).sum()
    )
    return (r_max <= ST_CAP) & (hits >= ST_HITS)


DELISTED = DATA / "delisted_daily.parquet"


def load(source: str = "db", st_mode: str = "today", with_delisted: bool = True) -> pd.DataFrame:
    """读并清洗日线。返回列：symbol / date / close / ret / amount / total_mv / name

    st_mode —— ST 股怎么处理（这是本项目唯一的**未来函数**所在，所以做成显式开关便于对照）：
      "today"  用**今天的名字**剔除（⚠️ 用到了未来信息，会让结果偏乐观）—— 默认，与早期报告一致
      "none"   完全不过滤（无未来函数，但把 ST 股也买进去了，不现实）
      "proxy"  **只用当时可见的价格信息**推断"疑似 ST"：
               ST/*ST 的日涨跌幅被限制在 ±5%，所以「过去 20 日 |涨跌幅| 从未超过 5.4%，
               且至少有 1 天达到 4.6% 以上」的股票，判为疑似 ST ⇒ 无未来函数。
               （准确率见 `st_lookahead_check.py` 的实测）
    """
    if source == "db":
        con = sqlite3.connect(DB)
        d = pd.read_sql(
            "select symbol, date, close, high, low, pctChg, volume, amount from stock_daily", con
        )
        con.close()
        d["date"] = pd.to_datetime(d["date"])
        d["symbol"] = _norm_symbol(d["symbol"])
        d["ret"] = d["pctChg"] / 100.0
        mv = pd.read_parquet(DATA / "market_cap.parquet")

    elif source == "ths":
        files = sorted(glob.glob(str(DATA / "dump" / "*.parquet")))
        if not files:
            raise SystemExit("❌ 没找到 THS dump，请先跑 fetch_ths_daily_k.py")
        t = pd.read_parquet(files[-1])
        # 北京时间零点的时间戳：必须转 Asia/Shanghai 再归一化，否则整体提前一天
        t["date"] = (
            pd.to_datetime(t["date_ms"], unit="ms", utc=True)
            .dt.tz_convert("Asia/Shanghai")
            .dt.normalize()
            .dt.tz_localize(None)
        )
        t["symbol"] = _norm_symbol(t["thscode"])
        t = t.rename(columns={"close_price": "close", "turnover": "amount",
                              "high_price": "high", "low_price": "low"})
        t = t.sort_values(["symbol", "date"])
        # THS 只有不复权价 ⇒ 日收益由相邻收盘价算。
        # 注意：不复权价在除权日有跳空，这点与 004 库的 pctChg 口径一致（实测重叠期一致）。
        t["ret"] = t.groupby("symbol")["close"].pct_change()
        d = t[["symbol", "date", "close", "high", "low", "ret", "volume", "amount"]]
        mv = pd.read_parquet(DATA / "market_cap_ths.parquet")
    else:
        raise SystemExit(f"❌ 未知 source: {source}（可选 db / ths）")

    mv["date"] = pd.to_datetime(mv["date"]).dt.as_unit("us")
    mv["symbol"] = _norm_symbol(mv["symbol"])
    d["date"] = d["date"].dt.as_unit("us")

    # ── 并入退市股（修幸存者偏差）──
    # 退市股没有 xdxr 股本 ⇒ 算不出**总**市值，但 baostock 的换手率可反推**流通**市值。
    # ⇒ 含退市股的分析必须按 float_mv 排序（两只来源口径已实测一致，差 0.0007%）。
    # 若按 total_mv 排序，退市股会因 NaN 被 rank 排除 ⇒ 结果与不含时一致（向后兼容）。
    if with_delisted and DELISTED.exists():
        dl = pd.read_parquet(DELISTED)
        dl["date"] = pd.to_datetime(dl["date"]).dt.as_unit("us")
        dl["symbol"] = _norm_symbol(dl["symbol"])
        dl = dl.sort_values(["symbol", "date"])
        dl["ret"] = dl.groupby("symbol")["close"].pct_change()
        dl["float_mv"] = dl["float_mv_bs"]
        # 退市股**没有总股本** ⇒ 用「总市值/流通市值」的**月度中位数**把流通市值折成总市值。
        # 依据：实测该比值中位数 0.91（75 分位 = 1.0），即多数股票已全流通，1:1 折算误差不大；
        # 而**不能用流通市值直接排序** —— 那会把限售比例高的次新股拉进微盘池，
        # 实测毛年化会从 13.0% 掉到 4.2%，等于偷偷换了策略。
        _tmp = mv[mv["total_mv"].notna() & (mv["float_mv"] > 0)].copy()
        _tmp["_r"] = _tmp["total_mv"] / _tmp["float_mv"]
        _ratio = _tmp.groupby(_tmp["date"].dt.to_period("M"))["_r"].median()
        dl["total_mv"] = dl["float_mv"] * dl["date"].dt.to_period("M").map(_ratio).astype(float)
        d = pd.concat(
            [d, dl[["symbol", "date", "close", "high", "low", "ret", "volume", "amount"]]],
            ignore_index=True,
        )
        mv = pd.concat(
            [mv, dl[["symbol", "date", "total_mv", "float_mv"]]], ignore_index=True
        )
        print(f"[microcap_data] 并入退市股 {dl['symbol'].nunique()} 只 / {len(dl):,} 行", flush=True)

    df = d.merge(mv[["symbol", "date", "total_mv", "float_mv"]], on=["symbol", "date"], how="left")
    df = df.merge(_load_names(), on="symbol", how="left")

    n0 = len(df)
    df = df[df["volume"] > 0]
    df = df[df["ret"].notna()]
    # 只要有**任一**市值可用且为正就保留（退市股只有 float_mv；幸存股两者都有）
    # 排序时另一列是 NaN，`rank()` 会给 NaN ⇒ 自然被排除，不需要额外分支。
    df = df[(df["total_mv"] > 0) | (df["float_mv"] > 0)]
    df = df.sort_values(["symbol", "date"])

    if st_mode == "today":
        df = df[~df["name"].astype(str).str.contains("ST", na=False)]   # ⚠️ 缺陷 4：用了未来信息
    elif st_mode == "proxy":
        df = df[~st_proxy_flag(df)]
    elif st_mode != "none":
        raise SystemExit(f"❌ 未知 st_mode: {st_mode}（可选 today / none / proxy）")

    fd = df.groupby("symbol")["date"].transform("min")
    df = df[~((fd > df["date"].min()) & ((df["date"] - fd).dt.days < NEW_STOCK_DAYS))]  # ⚠️ 缺陷 5
    df["ym"] = df["date"].dt.to_period("M")
    print(f"[microcap_data] source={source}  {n0:,} → {len(df):,} 行  "
          f"({df['date'].min().date()} ~ {df['date'].max().date()})", flush=True)
    return df


def build_pool(df: pd.DataFrame, q: float = 0.20, by: str = "total_mv") -> pd.DataFrame:
    """月末按 `by` 升序取最小的 q 比例。返回 symbol / hold_ym（= 持有月）。"""
    is_me = df["date"] == df.groupby("ym")["date"].transform("max")
    me = df[is_me][["ym", "symbol", by]].copy()
    me["rank"] = me.groupby("ym")[by].rank(pct=True)
    m = me[me["rank"] <= q][["ym", "symbol"]].copy()
    m["hold_ym"] = m["ym"] + 1        # 月末选出 → 下月持有（避免用到未来信息）
    return m


def daily_portfolio(df: pd.DataFrame, members: pd.DataFrame) -> pd.Series:
    """池内等权的日度收益序列。

    等权 ⇒ 组合当日收益 = 池内个股当日收益的**算术平均**（这正是等权的定义，且天然向量化）。
    """
    d = df.copy()
    d["hold_ym"] = d["ym"]
    held = d.merge(members[["hold_ym", "symbol"]], on=["hold_ym", "symbol"], how="inner")
    s = held.groupby("date")["ret"].mean()
    s.index = pd.to_datetime(s.index)
    return s.rename("port_ret")


def pool_amount_share(df: pd.DataFrame, members: pd.DataFrame) -> pd.Series:
    """微盘池成交额占全市场成交额的比例（拥挤度指标的原料）。"""
    d = df.copy()
    d["hold_ym"] = d["ym"]
    held = d.merge(members[["hold_ym", "symbol"]], on=["hold_ym", "symbol"], how="inner")
    s = held.groupby("date")["amount"].sum() / d.groupby("date")["amount"].sum()
    s.index = pd.to_datetime(s.index)
    return s.rename("share")


# ── 用户口径的「初选股池」──
# 非北交所（8xxxxx/920xxx）、非科创板（688）、非创业板（300/301）⇒ 只剩沪深主板
MAIN_BOARD_PREFIX = ("60", "00")     # 沪主板 600/601/603/605 + 深主板 000/001/002/003
MIN_LISTED_DAYS = 250                # 非次新：上市满 250 个自然日
MIN_PRICE = 2.0                      # 股价 > 2 元


def apply_initial_pool(df: pd.DataFrame,
                       min_price: float = MIN_PRICE,
                       min_days: int = MIN_LISTED_DAYS) -> pd.DataFrame:
    """按用户口径过滤出「初选股池」。

    规则：非北交所 + 非科创板 + 非创业板 + 非ST + 非退市 + 上市>250日 + 股价>2元。

    说明：
    - **板块**按代码前两位判断（沪深主板 = 60/00）⇒ 天然排除北交所(8x/920)、科创(688)、创业(300/301)。
    - **非ST**：`stock_list.name` 只有最新名称 ⇒ 这里仍带未来函数，但实测影响落在噪声内
      （见 `st_lookahead_check.py`），且本池子**本来就排除 ST**，所以不再单独处理。
    - **非退市**：池子按"当时是否有行情"自然判定，退市后就没有行情了。
    - **上市日**：用「库内首次出现日」近似。数据从 2016-10-13 起，
      所以 2016-10-13 之后上市的日期是**准确的**；更早上市的会记成 2016-10-13（只会少算几天，无影响）。
    """
    d = df[df["symbol"].str[:2].isin(MAIN_BOARD_PREFIX)].copy()
    d = d[d["close"] > min_price]
    fd = d.groupby("symbol")["date"].transform("min")
    d = d[d["date"] - fd >= pd.Timedelta(days=min_days)]
    return d


def pick_top_n(df: pd.DataFrame, n: int = 20, by: str = "total_mv") -> pd.DataFrame:
    """月末在初选池里按 `by` 升序取**固定 N 支**（不是取比例）。

    返回列：symbol / hold_ym（= 持有月）。与 `build_pool` 的区别：
    `build_pool` 取「最小的 q 比例」（约 800 只，等于一个指数）；
    本函数取**固定只数**（10 或 20 支），对应真实的集中持仓。
    """
    is_me = df["date"] == df.groupby("ym")["date"].transform("max")
    me = df[is_me].dropna(subset=[by]).copy()
    me = me.sort_values(["ym", by])
    me["_rk"] = me.groupby("ym").cumcount()
    m = me[me["_rk"] < n][["ym", "symbol"]].copy()
    m["hold_ym"] = m["ym"] + 1        # 月末选出 → 下月持有（不穿越）
    return m


def apply_stop_loss(nav: pd.DataFrame, threshold: float = -0.12) -> tuple[pd.DataFrame, int]:
    """月内**硬止损**（项目 `#18` 定的 E1 规则：单票浮亏 ≤ −12% 即卖出）。

    nav: index=date, 列=symbol, 值=(1+日收益).cumprod()（相对月初的净值，起点 1.0）
    返回 (处理后的 nav, 触发次数)。触发后该股**冻结在触发日净值**（= 卖出转现金，不再涨跌）。

    口径说明：按**收盘价**判定与执行（与项目回测一致），未建模"跌停卖不掉"。
    """
    out = nav.copy()
    hits = 0
    for s in nav.columns:
        col = nav[s]
        breach = (col - 1.0) <= threshold
        if breach.any():
            d0 = breach.idxmax()                  # 首次触发日
            mask = out.index >= d0
            out.loc[mask, s] = col.loc[d0]        # 冻结在卖出价
            hits += 1
    return out, hits


SIGNAL_COLS = ["S1_趋势", "S2_相对趋势", "S3_拥挤度", "S4_波动率", "组合_S2且S4", "组合_四个都要"]


def build_signals(port: pd.Series, share: pd.Series, bench_ret: pd.Series,
                  ma: int = 60, q_win: int = 250, q_level: float = 0.80) -> pd.DataFrame:
    """构造风控信号（**定义唯一来源**，风控回测与成本回测共用，避免口径漂移）。

    输出的是「当天收盘时看得到的状态」；**调用方必须自己 `.shift(1)` 再乘收益**，
    否则就是用当天信息交易当天，属于穿越。

    信号含义：
      S1_趋势      净值 > N 日均线才持有（经典趋势跟踪）
      S2_相对趋势  微盘/中证2000 的相对净值 > 其均线才持有
      S3_拥挤度    微盘成交额占比的 N 日均值 < 过去 M 日的 q 分位才持有
      S4_波动率    微盘近 20 日波动 < 过去 M 日的 q 分位才持有
    """
    nav = (1 + port).cumprod()
    bench_nav = (1 + bench_ret.fillna(0)).cumprod()      # 不预先 reindex：否则会丢掉指数独有的交易日
    rel = nav / bench_nav
    crowd = share.rolling(ma).mean()
    vol = port.rolling(20).std()

    sig = pd.DataFrame(index=port.index)
    sig["S1_趋势"] = nav > nav.rolling(ma).mean()
    sig["S2_相对趋势"] = rel > rel.rolling(ma).mean()
    sig["S3_拥挤度"] = crowd < crowd.rolling(q_win).quantile(q_level)
    sig["S4_波动率"] = vol < vol.rolling(q_win).quantile(q_level)
    sig["组合_S2且S4"] = sig["S2_相对趋势"] & sig["S4_波动率"]
    sig["组合_四个都要"] = sig[["S1_趋势", "S2_相对趋势", "S3_拥挤度", "S4_波动率"]].all(axis=1)
    return sig[SIGNAL_COLS]
