#!/usr/bin/env python3
"""build_extra_features.py — 扩展维度特征工程（8 类原始数据 → 35-44 维）

与 model_selection_v2/features.py 的 88 维特征对齐:
  输入: 交易日序列 + close 价格 + data/extra_features/ 的 parquet
  输出: (n_days, N_extra) 稠密数值数组(fillna(0)) + 特征名 + 覆盖率

频率与对齐(防 look-ahead):
  - 日频:   fund_flow / news(发布时间) / news_cls → 直接对齐
  - 季频:   finance(报告期→法定披露日 Q1:4/30 H1:8/31 Q3:10/31 年报:次年4/30)
            holders(公告日 HOLD_NOTICE_DATE) → asof 前向填充
  - 事件:   xdxr(事件日) → asof 前向填充  (forecast 2026-08-20 移除)
  - 快照:   consensus(无日期, 可用日=文件 mtime 采集日) → 仅采集日后有效
            (回测早期样本无此特征, 树模型自动忽略; 3-6月积累后形成历史序列)

覆盖率: 每特征返回"非零比例"; 特征层剔除逻辑由调用方按覆盖率阈值处理。
"""
from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
EXTRA_DIR = PROJECT_DIR / "data/extra_features"


# ════════════════════════════════════════════════════════════
#  读取辅助
# ════════════════════════════════════════════════════════════

@lru_cache(maxsize=2048)
def _load(subset: str, code: str) -> pd.DataFrame | None:
    """读取单只股票的原始数据 parquet（lru_cache: 训练缓存构建时同股票被反复调用）。

    缓存的是原始 DataFrame——调用方（各特征函数）必须 .copy() 后再修改，
    防止污染缓存（fund_flow/news 等函数会改列/索引）。
    """
    fp = EXTRA_DIR / subset / f"{code}.parquet"
    if not fp.exists():
        return None
    return pd.read_parquet(fp)


def _disclose_date(report_date: str) -> date:
    """财务报告期 → 法定披露日(保守):
    Q1(03-31)→04-30 | H1(06-30)→08-31 | Q3(09-30)→10-31 | 年报(12-31)→次年04-30"""
    d = pd.to_datetime(report_date)
    if d.month == 3:
        return date(d.year, 4, 30)
    if d.month == 6:
        return date(d.year, 8, 31)
    if d.month == 9:
        return date(d.year, 10, 31)
    return date(d.year + 1, 4, 30)  # 年报


def _asof_align(dates: pd.DatetimeIndex, events: pd.DataFrame,
                avail_col: str, value_cols: list[str]) -> pd.DataFrame:
    """事件/低频数据按可用日(披露日/发布日) asof 前向填充到交易日序列。

    Args:
        dates: 交易日序列
        events: 事件表, 含 avail_col(可用日期列) 和 value_cols(值列)
        avail_col: 可用日期列名（披露日/发布日/事件日）
    Returns:
        DataFrame(index=dates, columns=value_cols), 可用日前为 NaN
    """
    if events is None or len(events) == 0:
        return pd.DataFrame(index=dates, columns=value_cols, dtype=float)
    ev = events.copy()
    ev[avail_col] = pd.to_datetime(ev[avail_col], errors="coerce")
    ev = ev.dropna(subset=[avail_col]).sort_values(avail_col)
    # 同日多条(如年报与Q1同披露日) → 保留最新一条, 避免 reindex 重复索引报错
    ev = ev.drop_duplicates(subset=[avail_col], keep="last")
    if len(ev) == 0:
        return pd.DataFrame(index=dates, columns=value_cols, dtype=float)
    ev = ev.set_index(avail_col)[value_cols].apply(pd.to_numeric, errors="coerce")
    # asof 前向填充: 每个交易日取"最近一次已披露"的值
    return ev.reindex(dates, method="ffill")


# ════════════════════════════════════════════════════════════
#  各数据面特征
# ════════════════════════════════════════════════════════════

def _fund_flow_features(code: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """资金流向(日频): 主力占比/5日均值/净流入天数/20日累计/动量/超大单占比"""
    df = _load("fund_flow", code)
    cols = ["主力净流入-净占比", "超大单净流入-净占比"]
    if df is None or len(df) == 0:
        return pd.DataFrame(index=dates, columns=[f"ff_{c}" for c in
                             ["main_ratio", "main_ratio_5d", "main_inflow_days_5",
                              "main_amt_20d", "main_momentum", "xbig_ratio_5d"]], dtype=float)
    df = df.copy()  # lru_cache: 防污染缓存
    df["日期"] = pd.to_datetime(df["日期"])
    df = df.set_index("日期").sort_index()
    s = df["主力净流入-净占比"].astype(float) / 100.0  # % → 小数
    xb = df["超大单净流入-净占比"].astype(float) / 100.0
    amt = df["主力净流入-净额"].astype(float)
    out = pd.DataFrame(index=dates)
    out["ff_main_ratio"] = s.reindex(dates).fillna(0.0)
    out["ff_main_ratio_5d"] = s.rolling(5).mean().reindex(dates).fillna(0.0)
    out["ff_main_inflow_days_5"] = (s > 0).astype(float).rolling(5).sum().reindex(dates).fillna(0.0)
    out["ff_main_amt_20d"] = amt.rolling(20).sum().reindex(dates).fillna(0.0) / 1e8  # 亿元
    out["ff_main_momentum"] = (s.rolling(5).mean() - s.rolling(20).mean()).reindex(dates).fillna(0.0)
    out["ff_xbig_ratio_5d"] = xb.rolling(5).mean().reindex(dates).fillna(0.0)
    return out


def _finance_get(df: pd.DataFrame, col: str) -> pd.Series:
    """财务列容错读取: 列缺失 → 0（金融股无毛利率/净利率等列, 同花顺按行业返回不同列集）"""
    if col in df.columns:
        return df[col].astype(str).str.replace("%", "", regex=False)
    return pd.Series("0", index=df.index)


def _finance_features(code: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """财务摘要(季频, 披露日对齐): 16 维财务基本面特征。

    2026-08-07 修复: 金融股(银行/保险/券商)无毛利率等列 → 逐列容错(_finance_get),
    缺失列填 0, 保证输出恒定列数（固定列契约, 拼接维度一致）。
    2026-08-20 扩展(10→16): 新增 摊薄ROE/营收增速变化/流动比率/速动比率/存货周转天数/应收周转天数
    """
    df = _load("finance", code)
    if df is None or len(df) == 0:
        return pd.DataFrame(index=dates, columns=_EMPTY_COLS["finance"], dtype=float)
    # 百分比口径列(数值为 %, 存为 0.xx): 折算 /100
    pct_cols = ["roe", "gp_margin", "np_margin", "debt_ratio", "rev_yoy",
                "profit_yoy", "roe_tb"]
    pct_conf = {
        "roe": "净资产收益率", "gp_margin": "销售毛利率", "np_margin": "销售净利率",
        "debt_ratio": "资产负债率", "rev_yoy": "营业总收入同比增长率",
        "profit_yoy": "净利润同比增长率", "roe_tb": "净资产收益率-摊薄",
    }
    # 比率/天数口径列(数值即倍数/天, 不折算)
    raw_cols = {
        "eps": "基本每股收益",       # 元/股
        "bps": "每股净资产",         # 元/股
        "ocf": "每股经营现金流",     # 元/股
        "undist_rps": "每股未分配利润",  # 元/股
        "cap_rps": "每股资本公积金",     # 元/股
        "current_ratio": "流动比率",
        "quick_ratio": "速动比率",
        "inv_turn_days": "存货周转天数",
        "ar_turn_days": "应收账款周转天数",
    }
    ev = pd.DataFrame({"avail": [str(_disclose_date(r)) for r in df["报告期"]]})
    for k, col in pct_conf.items():
        ev[k] = pd.to_numeric(_finance_get(df, col), errors="coerce") / 100.0
    for k, col in raw_cols.items():
        ev[k] = pd.to_numeric(_finance_get(df, col), errors="coerce")
    aligned = _asof_align(dates, ev, "avail", [c for c in ev.columns if c != "avail"])
    out = pd.DataFrame(index=dates)
    out["fin_roe"] = aligned["roe"].fillna(0.0)
    out["fin_gp_margin"] = aligned["gp_margin"].fillna(0.0)
    out["fin_np_margin"] = aligned["np_margin"].fillna(0.0)
    out["fin_debt_ratio"] = aligned["debt_ratio"].fillna(0.0)
    out["fin_rev_yoy"] = aligned["rev_yoy"].fillna(0.0)
    out["fin_profit_yoy"] = aligned["profit_yoy"].fillna(0.0)
    out["fin_profit_yoy_chg"] = aligned["profit_yoy"].diff().fillna(0.0)  # 净利增速加速/减速
    out["fin_cf_quality"] = (aligned["ocf"] / aligned["eps"].replace(0, np.nan)).fillna(0.0)
    out["fin_eps"] = aligned["eps"].fillna(0.0)
    out["fin_bps"] = aligned["bps"].fillna(0.0)
    # ── 2026-08-20 新增 6 维 ──
    out["fin_roe_tb"] = aligned["roe_tb"].fillna(0.0)                    # 摊薄ROE
    out["fin_rev_yoy_chg"] = aligned["rev_yoy"].diff().fillna(0.0)      # 营收增速变化
    out["fin_current_ratio"] = aligned["current_ratio"].fillna(0.0)     # 流动比率
    out["fin_quick_ratio"] = aligned["quick_ratio"].fillna(0.0)         # 速动比率
    out["fin_inv_turn_days"] = aligned["inv_turn_days"].fillna(0.0)     # 存货周转天数
    out["fin_ar_turn_days"] = aligned["ar_turn_days"].fillna(0.0)       # 应收周转天数
    return out


def _holders_features(code: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """股东结构(季频, 公告日对齐): 户数环比/户均市值/筹码集中×股价"""
    df = _load("holders", code)
    if df is None or len(df) == 0:
        return pd.DataFrame(index=dates, columns=["hd_num_chg", "hd_avg_mcap"], dtype=float)
    ev = pd.DataFrame({
        "avail": df["股东户数公告日期"],
        "num_chg": df["股东户数-增减比例"].astype(float) / 100.0,
        "avg_mcap": df["户均持股市值"].astype(float),
    })
    aligned = _asof_align(dates, ev, "avail", ["num_chg", "avg_mcap"])
    out = pd.DataFrame(index=dates)
    out["hd_num_chg"] = aligned["num_chg"].fillna(0.0)      # 筹码集中度(负=集中)
    out["hd_avg_mcap"] = np.log1p(aligned["avg_mcap"].fillna(0.0))  # 对数化
    return out


def _consensus_features(code: str, dates: pd.DatetimeIndex, close: pd.Series) -> pd.DataFrame:
    """一致预期(快照, 采集日对齐): 买入占比/覆盖数/预测PE/目标价偏离/分歧度"""
    df = _load("consensus", code)
    cols = ["cs_buy_ratio", "cs_org_num", "cs_pred_pe", "cs_aim_dev", "cs_aim_spread"]
    if df is None or len(df) == 0:
        return pd.DataFrame(index=dates, columns=cols, dtype=float)
    r = df.iloc[0]  # 快照(单行)
    org = float(r["机构数"] or 0)
    buy = float(r["买入数"] or 0)
    eps2 = float(r["Y2预测EPS"] or 0)
    aim_hi = float(r["目标价上限"] or 0)
    aim_lo = float(r["目标价下限"] or 0)
    # 可用日（2026-09-25 修复）：
    #   先明确一点——**把"快照采集日之前"的样本掩掉是正确行为**：
    #   该快照当时还不存在，拿它当历史特征就是前视偏差。所以
    #   "老样本上 consensus 为 0"不是 bug，是应有之义。
    #
    #   真正的问题是**判断"采集日"的依据错了**：原实现读文件 **mtime**，
    #   而 mtime 是文件系统元数据，会被复制 / rsync / 重写 parquet / 重新下载
    #   任意改写 —— 一次不经意的文件操作就能把"快照生效日"整体挪位，
    #   而且**不会报错**（实测：目录里 3,334 个文件 mtime 全被推到 2026-08-31，
    #   而训练采样日最大只到 2026-07-21 → 掩码命中 0 天，5 列全 0；
    #   更早一次构建时 mtime 是 2026-08-07，同一份数据就"有值"）。
    #
    #   改为**优先读 parquet 里的 `snapshot_date` 列**（采集时写入的业务日期，
    #   属数据本身、随文件走，且刷新不改写），缺失时才退回 mtime（老数据兼容）。
    snap = r.get("snapshot_date") if hasattr(r, "get") else None
    if snap is not None and pd.notna(snap):
        avail = pd.Timestamp(snap)
    else:
        fp = EXTRA_DIR / "consensus" / f"{code}.parquet"
        avail = (pd.Timestamp(datetime.fromtimestamp(os.path.getmtime(fp)))
                 if fp.exists() else pd.Timestamp(date.today()))
    mask = dates >= avail  # 快照采集日前 = NaN(回测早期无此特征)
    out = pd.DataFrame(index=dates, columns=cols, dtype=float)
    out.loc[mask, "cs_buy_ratio"] = buy / org if org else 0.0
    out.loc[mask, "cs_org_num"] = np.log1p(org)
    if eps2 > 0:
        out.loc[mask, "cs_pred_pe"] = close[mask] / eps2
    if aim_hi > 0 and aim_lo > 0:
        mid = (aim_hi + aim_lo) / 2
        out.loc[mask, "cs_aim_dev"] = close[mask] / mid - 1.0
        out.loc[mask, "cs_aim_spread"] = (aim_hi - aim_lo) / mid
    return out.fillna(0.0)


def _news_features(code: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """新闻(日频): 近5/20日条数/来源多样性(数量信号第一档)"""
    df = _load("news", code)
    cols = ["nw_cnt_5d", "nw_cnt_20d", "nw_src_div"]
    if df is None or len(df) == 0:
        return pd.DataFrame(index=dates, columns=cols, dtype=float)
    df = df.copy()  # lru_cache: 防污染缓存
    df["发布时间"] = pd.to_datetime(df["发布时间"], errors="coerce")
    df = df.dropna(subset=["发布时间"])
    if len(df) == 0:
        return pd.DataFrame(index=dates, columns=cols, dtype=float)
    count = df.groupby(df["发布时间"].dt.normalize()).size()
    src = df.groupby(df["发布时间"].dt.normalize())["文章来源"].nunique()
    cnt_series = count.reindex(dates).fillna(0)
    src_series = src.reindex(dates).fillna(0)
    out = pd.DataFrame(index=dates)
    out["nw_cnt_5d"] = cnt_series.rolling(5, min_periods=1).sum().fillna(0.0)
    out["nw_cnt_20d"] = cnt_series.rolling(20, min_periods=1).sum().fillna(0.0)
    out["nw_src_div"] = src_series.rolling(5, min_periods=1).mean().fillna(0.0)
    return out


def _xdxr_features(code: str, dates: pd.DatetimeIndex, close: pd.Series) -> pd.DataFrame:
    """分红(低频): 股息率(近12月分红/价)/近3年分红·送转次数"""
    df = _load("xdxr", code)
    cols = ["xd_yield", "xd_div_cnt_3y", "xd_song_cnt_3y"]
    if df is None or len(df) == 0:
        return pd.DataFrame(index=dates, columns=cols, dtype=float)
    ev = pd.DataFrame({
        "avail": pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str) + "-" + df["day"].astype(str)),
        "fenhong": pd.to_numeric(df["fenhong"], errors="coerce").fillna(0.0),   # 每股分红
        "songzhuangu": pd.to_numeric(df["songzhuangu"], errors="coerce").fillna(0.0),
    })
    ev = ev.dropna(subset=["avail"]).drop_duplicates(subset="avail", keep="last")
    if len(ev) == 0:
        return pd.DataFrame(index=dates, columns=cols, dtype=float)
    # ── 逐日 as-of 口径（2026-09-25 修复）──
    # 原实现有两处结构性问题，导致这两维**永远恒 0、零信息**（实测生产缓存里
    # `xd_div_cnt_3y`/`xd_song_cnt_3y` 零方差）：
    #   ① L264-268 原把"近3年事件次数"算成**一个标量**贴满整列 → 常数列会被
    #      `model_selection_v2/features.py` 的零方差保护强制清 0；
    #      且 `song_3y` 用的是**全历史**次数（不是近 3 年）——语义也不对。
    #   ② `xd_yield` 用 `div_cum.reindex(dates)`，而 div_cum 的索引是**事件日**，
    #      对齐到交易日只在除权当天有值（实测 1,733 个交易日里仅 14 天非零）。
    # 现在三列都改成"对每个交易日 d，取其及之前已知的信息"。
    out = pd.DataFrame(index=dates)
    div_cum = ev.set_index("avail")["fenhong"].sort_index().rolling("365D", min_periods=1).sum()
    # ① 近 12 月累计分红按日 as-of 对齐（取 d 之前最近一次事件的滚动值，而非只在事件日有值）
    out["xd_yield"] = (
        div_cum.reindex(dates, method="ffill").fillna(0.0) / close.replace(0, np.nan)
    ).fillna(0.0)

    # ② 近 3 年事件次数：按日滚动计数（avail ∈ [d-1095天, d]），随日变化
    def _count_within_3y(ev_dates: pd.Series) -> np.ndarray:
        arr = np.sort(pd.to_datetime(ev_dates).values)
        if len(arr) == 0:
            return np.zeros(len(dates))
        d_arr = dates.values
        hi = np.searchsorted(arr, d_arr, side="right")
        lo = np.searchsorted(arr, d_arr - np.timedelta64(1095, "D"), side="left")
        return (hi - lo).astype(float)

    out["xd_div_cnt_3y"] = _count_within_3y(ev.loc[ev["fenhong"] > 0, "avail"])
    out["xd_song_cnt_3y"] = _count_within_3y(ev.loc[ev["songzhuangu"] > 0, "avail"])
    return out


# ════════════════════════════════════════════════════════════
#  2026-08-20 新增：同花顺短线情绪维度（事件日 asof）
#  龙虎榜 / 涨停池——采集见 scripts/fetch_hithink_sentiment.py（近一年日榜→每股parquet）
# ════════════════════════════════════════════════════════════


def _latest_event_state(df, dates, value_cols, next_day: bool = False):
    """事件 asof 前向填充 value_cols + 返回 avail 序列（用于算滚动事件次数）。

    Args:
        next_day: True=事件在"下一交易日"才生效（消除前视）。
            ⚠️ 2026-08-21 修复: 龙虎榜/涨停是"收盘后/当日"才知的事件, 若当日生效则
            回测在月末采样(ref_date=上榜日)用到"当天还没公布的龙虎榜/涨停"→ 前视,
            系统性抢跑虚增收益(920156 4-30上榜被当天4-30使用, 实测确认)。必须次日生效。
            fund_flow 等当日资金流不属此类, 保持当日(False)。

    Returns (aligned_df[value_cols], avail: DatetimeIndex)
    """
    sub = df.dropna(subset=value_cols, how="all")
    if sub.empty:
        return None, None
    ev = pd.DataFrame({"avail": pd.to_datetime(sub["date"])})
    for c in value_cols:
        ev[c] = pd.to_numeric(sub[c], errors="coerce")
    ev = ev.dropna(subset=["avail"]).drop_duplicates(subset="avail", keep="last")
    if ev.empty:
        return None, None
    if next_day:
        # 事件移到下一交易日生效: avail = 涨停/上榜日的下一交易日（pandas 原生, 无 numpy 类型陷阱）
        # dates 是当前股票的交易日序列(DatetimeIndex)。对每个上榜日 d:
        #   searchsorted(左边界) 给 >=d 的首位置; 若该位就是 d(上榜日也是交易日) → 取下一位。
        # 保证 event 最早从"次日"起可用, 消除"当天上榜被当天选股使用"的前视。
        @np.vectorize
        def _next_td(d):
            d_ts = pd.Timestamp(d)
            pos = dates.searchsorted(d_ts, side="left")
            if pos < len(dates) and dates[pos] == d_ts:
                pos += 1
            return dates[pos] if pos < len(dates) else pd.NaT
        ev["avail"] = [pd.Timestamp(x) for x in _next_td(ev["avail"].values)]
        ev = ev.dropna(subset=["avail"]).drop_duplicates(subset="avail", keep="last")
    aligned = _asof_align(dates, ev, "avail", value_cols)
    return aligned, ev["avail"].drop_duplicates()


def _event_count_30d(avail, dates):
    """近30日事件次数（无事件→0）。avail: 事件日 Series(DatetimeIndex 值)。"""
    if avail is None or len(avail) == 0:
        return pd.Series(0.0, index=dates)
    # 转 DatetimeIndex 事件序列，再滚动 30D 计数
    ev = pd.Series(1.0, index=pd.DatetimeIndex(avail)).sort_index()
    cnt = ev.rolling("30D", min_periods=1).count()
    return cnt.reindex(dates).fillna(0.0).astype(float)


def _dragon_tiger_features(code: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """龙虎榜(事件, 近一年): 净买入(亿)/净买入率/近30日上榜次数"""
    df = _load("dragon_tiger", code)
    cols = ["dt_net_buy", "dt_net_rate", "dt_cnt_30d"]
    if df is None or len(df) == 0:
        return pd.DataFrame(index=dates, columns=cols, dtype=float)
    aligned, avail = _latest_event_state(df, dates, ["dt_net_buy", "dt_net_rate"], next_day=True)  # 龙虎榜盘后才公布, 次日生效防前视
    if aligned is None:
        return pd.DataFrame(index=dates, columns=cols, dtype=float)
    res = pd.DataFrame(index=dates)
    res["dt_net_buy"] = (aligned["dt_net_buy"] / 1e8).fillna(0.0)   # 亿
    res["dt_net_rate"] = aligned["dt_net_rate"].fillna(0.0)         # 净买入率(小数)
    res["dt_cnt_30d"] = _event_count_30d(avail, dates)
    return res


def _limit_up_features(code: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """涨停(事件, 近一年): 连板数/封单额(亿)/近30日涨停次数"""
    df = _load("limit_up", code)
    cols = ["lu_lianban", "lu_seal", "lu_cnt_30d"]
    if df is None or len(df) == 0:
        return pd.DataFrame(index=dates, columns=cols, dtype=float)
    aligned, avail = _latest_event_state(df, dates, ["lu_lianban", "lu_seal"], next_day=True)  # 涨停当日才最终确认, 次日生效防前视
    if aligned is None:
        return pd.DataFrame(index=dates, columns=cols, dtype=float)
    res = pd.DataFrame(index=dates)
    res["lu_lianban"] = aligned["lu_lianban"].fillna(0.0)     # 连板数
    res["lu_seal"] = (aligned["lu_seal"] / 1e8).fillna(0.0)   # 封单额(亿)
    res["lu_cnt_30d"] = _event_count_30d(avail, dates)
    return res


# ════════════════════════════════════════════════════════════
#  同花顺财务质量维度（V6-A 实验，2026-10-06）
# ════════════════════════════════════════════════════════════
#
#  背景：004 原有 16 维财务特征偏「盈利/偿债/营运」的**规模与水平**；
#        本组补的是「**质量**」——赚的是不是真钱、收入的现金转化、研发投入强度。
#
#  ★ 门控（重要）：本组受环境变量 FEATURE_V6_THS_FINANCE=1 控制才启用。
#     原因：`groups = groups or list(FEATURE_GROUPS.keys())` 会**默认启用全部组**，
#     若不加门控，仅"注册"就会静默改变生产行为（见 _default_groups()）。
#
#  ★ asof 口径：**绝不能用同花顺的 report_date_ms**（实测系统性晚一年，
#     茅台 2023 年报真实披露 2024-04-03、接口返回 2025-04-03）。
#     沿用 _disclose_date() 的保守法定披露日兜底。
#
#  数据源：data/extra_features/finance_ths_v2/{indicators,income}/{code}.parquet
#         （023_THS 于 2026-10-06 采集，覆盖 2012 起、2020 后非空率 100%）

THS_FIN_DIR = EXTRA_DIR / "finance_ths_v2"

# 同花顺指标报告期编码 → 报告期末日期（"2025-3" → "2025-09-30"）
_THS_Q_END = {"1": "03-31", "2": "06-30", "3": "09-30", "4": "12-31"}


def _ths_report_end(rep: str) -> str:
    """'2025-3' → '2025-09-30'"""
    y, q = str(rep).split("-")
    return f"{y}-{_THS_Q_END[q]}"


@lru_cache(maxsize=2048)
def _load_ths_fin(sub: str, code: str) -> pd.DataFrame | None:
    """读取 finance_ths_v2 的子目录（indicators / income）。

    注意：该目录**嵌套两层**，与 _load() 的 {subset}/{code}.parquet 结构不同，
    故单独实现。lru_cache 与 _load 同理（同股票被反复调用）。
    """
    fp = THS_FIN_DIR / sub / f"{code}.parquet"
    if not fp.exists():
        return None
    try:
        return pd.read_parquet(fp)
    except Exception:
        return None


def _ths_finance_features(code: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """同花顺财务质量 5 维：现金流质量 4 项 + 研发强度 1 项。

    全部按【法定披露日】asof 前向填充，缺失 fillna(0)（与 004 惯例一致，
    保证固定列数契约）。
    """
    cols = _EMPTY_COLS["ths_finance"]
    empty = pd.DataFrame(index=dates, columns=cols, dtype=float)

    parts: list[pd.DataFrame] = []

    # ── ① 现金流质量 4 项（来自 indicators，接口给的是百分数 → /100）──
    ind = _load_ths_fin("indicators", code)
    if ind is not None and len(ind) and "report" in ind.columns:
        ev = pd.DataFrame({
            "avail": [_disclose_date(_ths_report_end(r)) for r in ind["report"]],
            "fin_ths_cash_quality": pd.to_numeric(
                ind.get("net_profit_cash_content"), errors="coerce") / 100.0,
            "fin_ths_cash_sales": pd.to_numeric(
                ind.get("operating_cash_flow_net_divide_income"), errors="coerce") / 100.0,
            "fin_ths_cash_index": pd.to_numeric(
                ind.get("cash_operating_index"), errors="coerce") / 100.0,
            "fin_ths_cash_invest": pd.to_numeric(
                ind.get("cash_meet_invest_ratio"), errors="coerce") / 100.0,
        })
        parts.append(_asof_align(dates, ev, "avail", cols[:4]))

    # ── ② 研发强度 = 研发费用 / 营业收入（来自利润表）──
    inc = _load_ths_fin("income", code)
    if inc is not None and len(inc) and "period_end_ms" in inc.columns:
        rd = pd.to_numeric(inc.get("research_and_development_expenses"), errors="coerce")
        oi = pd.to_numeric(inc.get("operating_income"), errors="coerce")
        ratio = rd / oi.replace(0, np.nan)
        ends = pd.to_datetime(inc["period_end_ms"], unit="ms", utc=True,
                              errors="coerce").dt.tz_convert("Asia/Shanghai")
        ev2 = pd.DataFrame({
            "avail": [_disclose_date(d.strftime("%Y-%m-%d")) if pd.notna(d) else None
                      for d in ends],
            "fin_ths_rd_intensity": ratio,
        })
        parts.append(_asof_align(dates, ev2, "avail", ["fin_ths_rd_intensity"]))

    # ★ V6-C：按开关裁列。不设 V6C_DROP_DIMS 时 keep == cols ⇒ 与 V6-A 逐位一致。
    keep = _group_cols("ths_finance")
    if not parts:
        return empty.reindex(columns=keep)
    out = pd.concat(parts, axis=1).reindex(columns=cols)
    return out.fillna(0.0).reindex(columns=keep)


# ════════════════════════════════════════════════════════════
#  主入口
# ════════════════════════════════════════════════════════════

FEATURE_GROUPS = {
    "fund_flow": _fund_flow_features,
    "finance": _finance_features,
    "holders": _holders_features,
    "consensus": _consensus_features,
    "news": _news_features,
    "xdxr": _xdxr_features,
    # 2026-08-20: forecast(业绩预告) 移除——ths/mootdx 均无此接口，信息量可由
    # finance 16维 + 短线情绪维度替代（feature_version→4）
    # 2026-08-20: 新增同花顺短线情绪维度
    # hot_rank 移除(2026-08-20): 热榜仅 30 只稳定股票、全市场覆盖 0.6%，
    #   特征对 99%+ 股票恒 0 → 无信息量（数据源本身是稳定 Top30 榜）
    "dragon_tiger": _dragon_tiger_features,
    "limit_up": _limit_up_features,
    # 2026-10-06 V6-A 实验组：同花顺财务质量 5 维
    #   ★ 默认【不启用】—— 必须设 FEATURE_V6_THS_FINANCE=1 才生效，
    #     门控见下方 _default_groups()（否则注册即生效，会静默改生产行为）
    "ths_finance": _ths_finance_features,
    # 2026-10-10 V6-D 实验组「财务深度 14 维」在**函数定义之后**注册（见文件下方
    #   `FEATURE_GROUPS["ths_deep"] = _ths_deep_features`）—— 因为该函数定义在本字典之后，
    #   在这里直接引用会 NameError。注册时机仍在 import 期，行为与写在字典里等价。
}

# 各组固定列名模板（异常兜底时全 0 填充, 保证输出恒 33 列 → 拼接维度一致）
_EMPTY_COLS = {
    "fund_flow": ["ff_main_ratio", "ff_main_ratio_5d", "ff_main_inflow_days_5",
                  "ff_main_amt_20d", "ff_main_momentum", "ff_xbig_ratio_5d"],
    "finance": ["fin_roe", "fin_gp_margin", "fin_np_margin", "fin_debt_ratio",
                "fin_rev_yoy", "fin_profit_yoy", "fin_profit_yoy_chg",
                "fin_cf_quality", "fin_eps", "fin_bps",
                "fin_roe_tb", "fin_rev_yoy_chg", "fin_current_ratio",
                "fin_quick_ratio", "fin_inv_turn_days", "fin_ar_turn_days"],
    "holders": ["hd_num_chg", "hd_avg_mcap"],
    "consensus": ["cs_buy_ratio", "cs_org_num", "cs_pred_pe", "cs_aim_dev", "cs_aim_spread"],
    "news": ["nw_cnt_5d", "nw_cnt_20d", "nw_src_div"],
    "xdxr": ["xd_yield", "xd_div_cnt_3y", "xd_song_cnt_3y"],
    "dragon_tiger": ["dt_net_buy", "dt_net_rate", "dt_cnt_30d"],
    "limit_up": ["lu_lianban", "lu_seal", "lu_cnt_30d"],
    # V6-A 实验组（默认不启用，见 _default_groups）
    "ths_finance": ["fin_ths_cash_quality", "fin_ths_cash_sales",
                    "fin_ths_cash_index", "fin_ths_cash_invest",
                    "fin_ths_rd_intensity"],
    # V6-D 实验组（2026-10-10，默认不启用）：**财务深度 14 维**
    #   来源：同花顺 4 张表里"早已采集、但从未接入"的字段（筛选后剩下的真·新信息）
    "ths_deep": [
        "fin_dp_invest_cf_to_rev",          # 投资活动现金流净额 ÷ 营收
        "fin_dp_capex_to_rev",              # 购建固定资产等支出 ÷ 营收
        "fin_dp_finance_cf_to_rev",         # 筹资活动现金流净额 ÷ 营收
        "fin_dp_dividend_to_rev",           # 分红付息支出 ÷ 营收
        "fin_dp_manage_fee_to_rev",         # 管理费用 ÷ 营收
        "fin_dp_sales_fee_to_rev",          # 销售费用 ÷ 营收
        "fin_dp_cash_to_assets",            # 货币资金 ÷ 总资产
        "fin_dp_current_assets_to_assets",  # 流动资产 ÷ 总资产
        "fin_dp_ar_to_assets",              # 应收账款 ÷ 总资产
        "fin_dp_assets_turnover",           # 总资产周转率（已是比率）
        "fin_dp_current_assets_turnover",   # 流动资产周转率（已是比率）
        "fin_dp_assets_growth",             # 总资产增长率（接口给百分数 → ÷100）
        "fin_dp_fixed_asset_expansion",     # 固定资产扩张率（接口给百分数 → ÷100）
        "fin_dp_lt_debt_equity",            # 长期负债权益比（接口给百分数 → ÷100）
    ],
    # V6-E 实验组（2026-10-10，默认不启用）：**mootdx 股东/机构 16 维**
    #   来源：`data/extra_features/mootdx_finance/`（119 期，1990~2026）——**早已采集、从未接入**
    #   核心是"聪明钱在买什么"：QFII/券商/保险/基金/社保/私募/银行/年金/特殊法人/一般法人/国家队
    #   —— 现有 134 维里几乎完全没有（只有 holders 2 列）
    "mootdx_holder": [
        "fin_mh_qfii",          # QFII 持股 ÷ 流通A股
        "fin_mh_broker",        # 券商持股 ÷ 流通A股
        "fin_mh_insurance",     # 保险持股 ÷ 流通A股
        "fin_mh_fund",          # 基金持股 ÷ 流通A股
        "fin_mh_socialsec",     # 社保持股 ÷ 流通A股
        "fin_mh_private",       # 私募持股 ÷ 流通A股
        "fin_mh_bank",          # 银行持股 ÷ 流通A股
        "fin_mh_annuity",       # 年金持股 ÷ 流通A股
        "fin_mh_special_legal", # 特殊法人持股 ÷ 流通A股
        "fin_mh_general_legal", # 一般法人持股 ÷ 流通A股
        "fin_mh_national",      # 国家队持股 ÷ 流通A股（源字段是万股，需 ×10000）
        "fin_mh_top10_float",   # 十大流通股东合计 ÷ 流通A股（筹码集中度）
        "fin_mh_top1",          # 第一大股东持股 ÷ 流通A股
        "fin_mh_restricted",    # 受限流通A股 ÷ 流通A股（解禁压力）
        "fin_mh_nholders",      # 股东人数（户）——原始计数
        "fin_mh_freefloat",     # 自由流通股 ÷ 流通A股
    ],
}

# ★ V6-A 实验开关：不设/≠1 时，ths_finance 组完全不参与，行为与 V5 逐位一致
THS_FINANCE_ENV = "FEATURE_V6_THS_FINANCE"

# ★ V6-B 实验开关（2026-10-09）：**整组删掉**（半截/死列组）。
#   值 = 逗号分隔的**组名**，如 `V6B_DROP_GROUPS=consensus,news,dragon_tiger,limit_up`。
#   与 V6-C 的 `V6C_DROP_DIMS`（在 ths_finance 组内砍列）是**两种不同形态**，二者不能同时设
#   （见 labels.py 的强制校验）。
#   ★ 默认不设 ⇒ 行为与以前**逐位一致**。
#   ⚠️ 本开关**不依赖** `FEATURE_V6_THS_FINANCE`（删的是别的组），
#      但 labels.py 要求同时开 V6 特征 + 版本号才升到 9 —— 目的是让版本号与特征集**一一对应**，
#      避免"特征变了、版本号没变"⇒ 新列数写进旧缓存目录。
V6B_DROP_ENV = "V6B_DROP_GROUPS"


def v6b_dropped_groups() -> list[str]:
    """当前要整组删掉的特征组（默认空 = 一组都不删）。

    组名写错**直接抛错**，不静默忽略 —— 静默忽略会得到"以为删了、其实没删"的假实验。
    """
    raw = os.environ.get(V6B_DROP_ENV, "").strip()
    if not raw:
        return []
    want = [s.strip() for s in raw.split(",") if s.strip()]
    unknown = [w for w in want if w not in FEATURE_GROUPS]
    if unknown:
        raise ValueError(
            f"{V6B_DROP_ENV} 里有未知组名 {unknown}；"
            f"可选：{sorted(FEATURE_GROUPS)}")
    return want


# ★ V6-D 实验开关（2026-10-10）：**财务深度 14 维 + 删 consensus**（= "新129维"+14）
#   一个开关代表**一整个目标配置**（而不是多个正交开关叠加）——目的是让
#   「版本号 ↔ 特征集」一一对应，避免"两个不同特征集共用一个版本号"（SOP §3）。
#   · 开启时：**加** `ths_deep` 14 维；**删** `consensus` 整组（5 列、只一行数据、永远不起作用）
#   · 目标维度：88 + (46 − 5 + 14) = **143 维**
#   · ★ 默认不设 ⇒ 行为与以前**逐位一致**
V6D_DEEP_ENV = "FEATURE_V6_DEEP_FINANCE"

# 同花顺 4 张表里的**元数据列**（不是数值特征），合成表时要排除
_THS_META_COLS = {"code", "thscode", "ticker", "period", "fiscal_year", "fiscal_period",
                  "report", "report_date_ms", "report_end_ms", "period_end",
                  "snapshot_date", "currency", "avail"}


def v6d_deep_enabled() -> bool:
    """V6-D 实验配置是否启用（默认 False = 生产行为不变）"""
    return os.environ.get(V6D_DEEP_ENV) == "1"


# ★ V6-E 实验开关（2026-10-10）：**mootdx 股东/机构 16 维**（"聪明钱在买什么"）
#   一个开关 = **一整个目标配置**：加 mootdx 16 维 + 加 ths_deep 14 维（V6-D 那批）+ 删 consensus
#   ⇒ 目标 159 维 = 88 + (46 − 5 + 14 + 16)
#   ★ 默认不设 ⇒ 行为与以前**逐位一致**
V6E_MOOTDX_ENV = "FEATURE_V6_MOOTDX_HOLDER"


def v6e_mootdx_enabled() -> bool:
    """V6-E 实验配置是否启用（默认 False = 生产行为不变）"""
    return os.environ.get(V6E_MOOTDX_ENV) == "1"


# 实验组 → 开关名。**新增实验组必须在这里加一行**，否则会被 _default_groups() 当成"默认启用"。
# ★ 2026-10-10 教训：原实现是按 `ths_` 前缀过滤，新增 `ths_deep` 时它同时被"前缀分支"和
#   "V6-D 分支"各加一次 ⇒ 维度从 143 变 157（**静默错**，不报错）。现改为**显式集合**。
EXP_GROUP_SWITCH = {
    "ths_finance": THS_FINANCE_ENV,      # V6-A
    "ths_deep": V6D_DEEP_ENV,            # V6-D
    "mootdx_holder": V6E_MOOTDX_ENV,     # V6-E
}
_EXP_GROUPS = set(EXP_GROUP_SWITCH)


def _exp_group_enabled(group: str) -> bool:
    """实验组是否启用。V6-E 是"完整配置"⇒ 它同时启用 ths_deep。"""
    if group == "ths_deep":
        return v6d_deep_enabled() or v6e_mootdx_enabled()
    env = EXP_GROUP_SWITCH.get(group)
    return bool(env) and os.environ.get(env) == "1"


def _ths_deep_features(code: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """**财务深度 14 维**（2026-10-10）——同花顺 4 张表里"已采集、未接入"的字段。

    设计与筛选依据见 `023_THS/reports/实验设计_V6D_财务深度_20261010.md`：
    这 14 个字段经"与现有 134 维的截面相关性"筛选，全部 ≤ 0.62（真·新信息）。

    ★ 对齐规则（**必须与 `_ths_finance_features` 完全一致**）：
      按【法定披露日】asof 前向填充；缺失 fillna(0)（固定列数契约）。
    ★★ **绝不能用 `report_date_ms`** —— 实测它**系统性晚一年**
      （000001 的 2009Q4 返回 2011-02-25，真实披露日 2010-03）⇒ 用了等于白丢一年信息。
      本函数从「财年 + 季度」推报告期末，再走 `_disclose_date` 法定披露日逻辑。
    """
    cols = _EMPTY_COLS["ths_deep"]
    empty = pd.DataFrame(index=dates, columns=cols, dtype=float)

    # ── ① 逐表取数，建「披露日 → 值」事件表 ──
    ev_rows: list[pd.DataFrame] = []
    for sub in ("income", "balance", "cashflow", "indicators"):
        d = _load_ths_fin(sub, code)
        if d is None or len(d) == 0:
            continue
        d = d.copy()
        if sub == "indicators":
            if "report" not in d.columns:
                continue
            d["avail"] = [_disclose_date(_ths_report_end(r)) for r in d["report"]]
        else:
            if not {"fiscal_year", "fiscal_period"} <= set(d.columns):
                continue
            av = []
            for fy, fq in zip(d["fiscal_year"], d["fiscal_period"]):
                q = str(fq).upper().replace("Q", "").strip()
                av.append(_disclose_date(f"{int(fy)}-{_THS_Q_END[q]}") if q in _THS_Q_END else None)
            d["avail"] = av
        d = d.dropna(subset=["avail"])
        if len(d) == 0:
            continue
        d = d.set_index("avail")
        # ★ 同一披露日可能多行（重述/更正）⇒ 每张表先各自去重，否则下面建表会报
        #   "cannot reindex on an axis with duplicate labels"
        d = d[~d.index.duplicated(keep="last")]
        ev_rows.append(d)
    if not ev_rows:
        return empty

    # ── ② 合成一张表（同名不同表的列互不冲突）并按披露日去重（保留最新）──
    allcols = {}
    for d in ev_rows:
        for c in d.columns:
            if c not in _THS_META_COLS and c not in allcols:
                allcols[c] = d[c]
    E = pd.DataFrame(allcols).sort_index()
    E = E[~E.index.duplicated(keep="last")]
    if len(E) == 0:
        return empty

    def num(name: str) -> pd.Series:
        return pd.to_numeric(E.get(name), errors="coerce") if name in E.columns else pd.Series(np.nan, index=E.index)

    rev = num("operating_income").replace(0, np.nan)     # 营收（分母，现金流量表/利润表共用）
    ast = num("assets_total").replace(0, np.nan)         # 总资产（分母，资产负债表）
    spec = {
        "fin_dp_invest_cf_to_rev": num("invest_cash_flow_net") / rev,
        "fin_dp_capex_to_rev": num("pay_fixed_assets_etc_cash") / rev,
        "fin_dp_finance_cf_to_rev": num("financing_cash_flow_net") / rev,
        "fin_dp_dividend_to_rev": num("pay_dividends_profits_interest_cash") / rev,
        "fin_dp_manage_fee_to_rev": num("manage_fee") / rev,
        "fin_dp_sales_fee_to_rev": num("sales_fee") / rev,
        "fin_dp_cash_to_assets": num("cash") / ast,
        "fin_dp_current_assets_to_assets": num("total_current_assets") / ast,
        "fin_dp_ar_to_assets": num("accounts_receivable") / ast,
        "fin_dp_assets_turnover": num("total_assets_turnover_ratio"),
        "fin_dp_current_assets_turnover": num("current_assets_turnover_ratio"),
        # ↓ 接口给的是**百分数**（实测：总资产增长率中位 4.0、长期负债权益比 6.0）⇒ ÷100
        "fin_dp_assets_growth": num("total_assets_growth_ratio") / 100.0,
        "fin_dp_fixed_asset_expansion": num("fixed_asset_invest_expansion_ratio") / 100.0,
        "fin_dp_lt_debt_equity": num("long_term_debt_equity_ratio") / 100.0,
    }
    ev = pd.DataFrame({"avail": E.index, **{k: np.asarray(v) for k, v in spec.items()}})
    out = _asof_align(dates, ev, "avail", cols)
    return out.reindex(columns=cols).fillna(0.0)


# 在函数定义之后注册（原因见 FEATURE_GROUPS 里的注释）
FEATURE_GROUPS["ths_deep"] = _ths_deep_features


# ══════════════════════════════════════════════════════════════════════
#  V6-E：mootdx 股东/机构 16 维（2026-10-10）
#  ★ 数据是**按期存储**的（`mootdx_finance/gpcwYYYYMMDD.parquet`，119 期）
#    ⇒ 必须**全局建一次索引** {股票 → [(披露日, 值)]}，不能像别的组那样按股票读文件。
#  ★ 单位（已实测核验）：源字段除"国家队持股数量（万股)"外都是**股**；
#    占比类与分母 `已上市流通A股` 同单位 ⇒ 直接相除即可（国家队需 ×10000）。
# ══════════════════════════════════════════════════════════════════════
_MOOTDX_SRC = {
    "fin_mh_qfii":          ("QFII持股量", 1.0),
    "fin_mh_broker":        ("券商持股量", 1.0),
    "fin_mh_insurance":     ("保险持股量", 1.0),
    "fin_mh_fund":          ("基金持股量", 1.0),
    "fin_mh_socialsec":     ("社保持股量", 1.0),
    "fin_mh_private":       ("私募持股量", 1.0),
    "fin_mh_bank":          ("银行持股量(股)(机构持股)", 1.0),
    "fin_mh_annuity":       ("年金持股量", 1.0),
    "fin_mh_special_legal": ("特殊法人持股量(股)(机构持股)", 1.0),
    "fin_mh_general_legal": ("一般法人持股量(股)(机构持股)", 1.0),
    "fin_mh_national":      ("国家队持股数量（万股)", 10000.0),   # ★ 源字段单位是万股
    "fin_mh_top10_float":   ("十大流通股东持股数量合计(股)", 1.0),
    "fin_mh_top1":          ("第一大股东的持股数量", 1.0),
    "fin_mh_restricted":    ("受限流通A股(股)", 1.0),
    "fin_mh_freefloat":     ("自由流通股(股)", 1.0),
}
_MOOTDX_DENOM = "已上市流通A股"      # 占比的分母（流通股本）
_MOOTDX_NHOLDERS = "股东人数(户)"    # 原始计数，不做比例
_MOOTDX_INDEX: dict | None = None   # {code: [(avail, {col: val}), ...]}


def _mootdx_holder_index() -> dict:
    """建 {股票代码: [(披露日, {列: 值})]} 索引 —— **只在首次调用时**读一次全部期（119 个文件）。"""
    global _MOOTDX_INDEX
    if _MOOTDX_INDEX is not None:
        return _MOOTDX_INDEX
    import pyarrow.parquet as _pq
    d_moot = EXTRA_DIR / "mootdx_finance"
    want = {_MOOTDX_DENOM, "report_date", _MOOTDX_NHOLDERS} | {v[0] for v in _MOOTDX_SRC.values()}
    cache: dict = {}
    files = sorted(d_moot.glob("*.parquet")) if d_moot.exists() else []
    print(f"[mootdx] 建股东/机构索引：读取 {len(files)} 期 …", flush=True)
    for f in files:
        try:
            names = set(_pq.read_schema(f).names)
            cols = [c for c in want if c in names]
            if "report_date" not in cols or _MOOTDX_DENOM not in cols:
                continue
            d = pd.read_parquet(f, columns=cols)
            rd = int(d["report_date"].iloc[0])
            avail = _disclose_date(f"{rd // 10000}-{(rd // 100) % 100:02d}-{rd % 100:02d}")
            den = pd.to_numeric(d[_MOOTDX_DENOM], errors="coerce").replace(0, np.nan).values
            vals = {}
            for name, (src, mult) in _MOOTDX_SRC.items():
                if src in d.columns:
                    vals[name] = pd.to_numeric(d[src], errors="coerce").values * mult / den
                else:
                    vals[name] = np.full(len(d), np.nan)
            nhold = (pd.to_numeric(d[_MOOTDX_NHOLDERS], errors="coerce").values
                     if _MOOTDX_NHOLDERS in d.columns else np.full(len(d), np.nan))
            for i, sym in enumerate(d.index.values):
                rec = {k: float(v[i]) for k, v in vals.items()}
                rec["fin_mh_nholders"] = float(nhold[i])
                cache.setdefault(str(sym), []).append((avail, rec))
        except Exception:
            continue
    _MOOTDX_INDEX = cache
    print(f"[mootdx] 索引完成：{len(cache)} 只股票", flush=True)
    return cache


def _mootdx_holder_features(code: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """**mootdx 股东/机构 16 维** —— "聪明钱在买什么"。

    对齐规则与其他财务组一致：按【法定披露日】asof 前向填充、缺失 fillna(0)。
    ★ `report_date` 是**报告期末**（且是整数 20260630），必须走 `_disclose_date` 法定披露日逻辑。
    ★ 占比类**裁剪到 [0,1]**：分母 `已上市流通A股` 对部分股票**过时**（未随解禁更新），
      实测"十大流通股东合计÷流通A股"最大 25、"第一大股东"33、"受限流通"49
      ⇒ 不裁剪会造出畸变值。
    """
    cols = _EMPTY_COLS["mootdx_holder"]
    empty = pd.DataFrame(index=dates, columns=cols, dtype=float)
    rows = _mootdx_holder_index().get(code)
    if not rows:
        return empty
    ev = pd.DataFrame([{"avail": a, **v} for a, v in sorted(rows, key=lambda x: x[0])])
    out = _asof_align(dates, ev, "avail", cols)
    ratio_cols = [c for c in cols if c != "fin_mh_nholders"]
    out[ratio_cols] = out[ratio_cols].clip(0.0, 1.0)
    return out.reindex(columns=cols).fillna(0.0)


FEATURE_GROUPS["mootdx_holder"] = _mootdx_holder_features


def ths_finance_enabled() -> bool:
    """V6-A 实验组是否启用（默认 False = 生产行为不变）"""
    return os.environ.get(THS_FINANCE_ENV) == "1"


# ★ V6-C 实验开关（2026-10-09）：**在 V6-A 的 5 维里再砍掉若干列**（维度缩减）。
#   值 = 逗号分隔的列名，如 `V6C_DROP_DIMS=fin_ths_cash_quality,fin_ths_cash_sales`。
#   ★ 默认不设 ⇒ **行为与 V6-A 逐位一致**（生产与既有实验不受影响）。
#   必须同时设 FEATURE_V6_THS_FINANCE=1（否则该组根本不参与，本开关空转）。
#   ⚠️ 它**不直接改 FEATURE_VERSION** —— 版本号由 labels.py 用同一套判断统一决定
#      （设本开关且非安慰剂 → feature_version 8）。**两处必须一致，改一处要改两处**。
V6C_DROP_ENV = "V6C_DROP_DIMS"


def v6c_dropped_dims() -> list[str]:
    """当前要砍掉的 ths_finance 列（默认空 = 一列都不砍）。

    列名写错**直接抛错**，不静默忽略 —— 静默忽略会得到"以为砍了、其实没砍"的假实验，
    而"静默失效"正是本项目最常见的失败类型。
    """
    raw = os.environ.get(V6C_DROP_ENV, "").strip()
    if not raw:
        return []
    want = [s.strip() for s in raw.split(",") if s.strip()]
    unknown = [w for w in want if w not in _EMPTY_COLS["ths_finance"]]
    if unknown:
        raise ValueError(
            f"{V6C_DROP_ENV} 里有未知列名 {unknown}；"
            f"可选：{_EMPTY_COLS['ths_finance']}")
    return want


def ths_finance_cols() -> list[str]:
    """ths_finance 组**实际输出**的列（受 V6C_DROP_DIMS 影响）。

    ★ 这是"固定列数契约"的唯一入口：**正常路径与全 0 兜底都必须走它**，
      否则异常兜底输出 5 列、正常路径输出 3 列 ⇒ 拼接维度错位（本项目有过这类事故）。
    """
    drop = set(v6c_dropped_dims())
    return [c for c in _EMPTY_COLS["ths_finance"] if c not in drop]


def _group_cols(group: str) -> list[str]:
    """某组的输出列清单（目前只有 ths_finance 会随实验开关变）"""
    return ths_finance_cols() if group == "ths_finance" else _EMPTY_COLS[group]


def _default_groups() -> list[str]:
    """默认启用的特征组。

    ★ 为什么需要这个函数：原实现是 `groups = groups or list(FEATURE_GROUPS.keys())`，
      意味着**只要往 FEATURE_GROUPS 里注册新组，它就会立刻被所有调用方用上**，
      无需任何显式开启 —— 等于"加个组"就静默改了生产行为。
      本函数把实验组从默认集中排除，只在其环境变量显式开启时才加入。
    """
    # ★ 实验组**逐个按自己的开关**加入（显式集合，不靠命名前缀 —— 见 EXP_GROUP_SWITCH 的教训注释）
    gs = [g for g in FEATURE_GROUPS if g not in _EXP_GROUPS]
    for _g in FEATURE_GROUPS:
        if _g in _EXP_GROUPS and _exp_group_enabled(_g):
            gs.append(_g)
    # ★ V6-B（2026-10-09）：按开关删掉整组（如 consensus/news/dragon_tiger/limit_up）
    drop = set(v6b_dropped_groups())
    if drop:
        gs = [g for g in gs if g not in drop]
    # ★ V6-D / V6-E（2026-10-10）：一个开关 = 一个完整配置 ⇒ 顺带**删掉 consensus**（5 列死列）
    if v6d_deep_enabled() or v6e_mootdx_enabled():
        gs = [g for g in gs if g != "consensus"]
    return gs


def _zero_features(group: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """单数据面异常/缺失 → 全 0 列（保持固定列数契约, 覆盖率 0 → 关键面缺失由 incomplete 判定）"""
    return pd.DataFrame(0.0, index=dates, columns=_group_cols(group))

# ════════════════════════════════════════════════════════════
#  缺口处置（数据完整性标记）
#
#  关键数据面(缺失=数据问题, 应剔除):
#    fund_flow/finance/holders — 每只股票理应都有(成交/财报/股东)
#  语义可缺失面(缺失=合法信息, 不剔除):
#    consensus(46% A股无研报覆盖) / news(小盘股无新闻) / xdxr(次新股无分红记录)
# ════════════════════════════════════════════════════════════
KEY_GROUPS = ("fund_flow", "finance", "holders")
# 阈值 5%: 区分"窗口限制"与"数据缺失"
#   fund_flow 仅 120 天窗口 → 250 天 dates 覆盖 48% (合法, 非缺失)
#   数据面完全未采到 → 覆盖 0% (缺失 → incomplete)
EXTRA_COVERAGE_THRESHOLD = 0.05
_PREFIX_TO_GROUP = {"ff": "fund_flow", "fin": "finance", "hd": "holders",
                    "cs": "consensus", "nw": "news", "xd": "xdxr",
                    "dt": "dragon_tiger", "lu": "limit_up"}


def build_extra_with_flag(dates: pd.DatetimeIndex, close: pd.Series,
                          code: str, groups: list[str] | None = None
                          ) -> tuple[pd.DataFrame, bool, dict]:
    """构建特征 + 数据完整性标记（缺口股票处置入口）。

    Returns:
        (features_df, incomplete, coverage)
        incomplete=True → 关键数据面覆盖率 < 阈值, 训练/预测端应剔除该股票
    """
    groups = groups or _default_groups()   # ★ V6-A 门控：实验组需显式开启
    feats, cov = build_extra_features(dates, close, code, groups)
    # 逐个关键面检查"有值天数占比", 任一关键面缺失(<阈值) → incomplete
    key_covs = {}
    for g in KEY_GROUPS:
        if g not in groups:
            continue
        cols = [c for c in feats.columns if _PREFIX_TO_GROUP.get(c.split("_", 1)[0]) == g]
        if cols:
            key_covs[g] = float((feats[cols] != 0).any(axis=1).mean())
    incomplete = bool(key_covs) and any(c < EXTRA_COVERAGE_THRESHOLD for c in key_covs.values())
    # 2026-09-01: 精准放宽 — 东财 RPT_HOLDERNUM_DET 对部分大盘蓝筹(如 600027/601899)不披露股东户数,
    # 若仅 holders 面缺失、而 fund_flow 与 finance 均达标 → 视为数据源覆盖缺失, 不整只剔除。
    # 避免误剔正常标的(历史月份均在池、且有真实预测)。如多个关键面缺失, 维持剔除。
    if incomplete and key_covs.get("holders", 1.0) < EXTRA_COVERAGE_THRESHOLD:
        other_ok = all(
            key_covs.get(g, 1.0) >= EXTRA_COVERAGE_THRESHOLD
            for g in ("fund_flow", "finance")
        )
        if other_ok:
            incomplete = False
    return feats, incomplete, cov


def scan_incomplete(codes: list[str], get_close,
                    groups: list[str] | None = None) -> dict[str, bool]:
    """批量扫描数据完整性（训练/回测前调用）。

    Args:
        codes: 股票代码列表
        get_close: 回调 code -> (dates, close_series)（调用方提供行情）
    Returns:
        {code: incomplete}
    """
    out = {}
    for code in codes:
        try:
            dates, close = get_close(code)
            if dates is None or len(dates) == 0:
                out[code] = True  # 无行情 → 视为不完整
                continue
            _, incomplete, _ = build_extra_with_flag(dates, close, code, groups)
            out[code] = incomplete
        except Exception:
            out[code] = True
    return out


def build_extra_features(dates: pd.DatetimeIndex, close: pd.Series,
                         code: str, groups: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """构建单只股票的扩展维度特征。

    Args:
        dates: 交易日序列(DatetimeIndex)
        close: 收盘价序列(index=dates)
        code: 股票代码
        groups: 需要的数据面列表(None=全部 7 类)
    Returns:
        (features_df, coverage): features_df(index=dates, columns=特征名, 稠密fillna0);
                                  coverage = {特征名: 非零比例}
    """
    groups = groups or _default_groups()   # ★ V6-A 门控：实验组需显式开启
    parts = []
    for g in groups:
        try:
            fn = FEATURE_GROUPS[g]
            if g in ("consensus", "xdxr"):
                part = fn(code, dates, close)
            else:
                part = fn(code, dates)
            parts.append(part)
        except Exception:
            # 单数据面失败 → 全 0 列（2026-08-07: 保持固定 33 列契约,
            # 不跳过——跳过会导致拼接维度错位; 缺失由 coverage/incomplete 机制处置）
            parts.append(_zero_features(g, dates))
    if parts:
        features = pd.concat(parts, axis=1)
    else:
        features = pd.DataFrame(index=dates)
    features = features.fillna(0.0).astype(np.float32)
    coverage = {c: float((features[c] != 0).mean()) for c in features.columns}
    return features, coverage


# ════════════════════════════════════════════════════════════
#  自测: 单只股票验证 + 覆盖率统计
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    code = sys.argv[1] if len(sys.argv) > 1 else "600519"
    # 交易日序列(近 250 个交易日, 用 funds_flow 日期或生成)
    ff = _load("fund_flow", code)
    if ff is not None:
        dates = pd.to_datetime(ff["日期"])
    else:
        dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=250)
    rng = np.random.default_rng(0)
    close = pd.Series(rng.uniform(10, 50, len(dates)), index=dates, name="close")

    feats, cov = build_extra_features(dates, close, code)
    print(f"\n=== {code} 扩展维度特征 ===")
    print(f"交易日: {len(dates)} | 特征数: {len(feats.columns)}")
    print(f"\n特征名与覆盖率(近250日):")
    for c, cv in cov.items():
        print(f"  {c:24s} 覆盖率 {cv*100:5.1f}%  均值 {feats[c].mean():+.4f}")
