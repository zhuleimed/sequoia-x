#!/usr/bin/env python3
"""build_extra_features.py — 扩展维度特征工程（8 类原始数据 → 35-44 维）

与 model_selection_v2/features.py 的 88 维特征对齐:
  输入: 交易日序列 + close 价格 + data/extra_features/ 的 parquet
  输出: (n_days, N_extra) 稠密数值数组(fillna(0)) + 特征名 + 覆盖率

频率与对齐(防 look-ahead):
  - 日频:   fund_flow / news(发布时间) / news_cls → 直接对齐
  - 季频:   finance(报告期→法定披露日 Q1:4/30 H1:8/31 Q3:10/31 年报:次年4/30)
            holders(公告日 HOLD_NOTICE_DATE) → asof 前向填充
  - 事件:   forecast(发布日) / xdxr(事件日) → asof 前向填充
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
    """财务摘要(季频, 披露日对齐): ROE/毛利率/净利率/负债率/增速/现金流质量

    2026-08-07 修复: 金融股(银行/保险/券商)无毛利率等列 → 逐列容错(_finance_get),
    缺失列填 0, 保证输出恒 10 列（固定列契约, 拼接维度一致）。
    """
    df = _load("finance", code)
    if df is None or len(df) == 0:
        return pd.DataFrame(index=dates, columns=[f"fin_{c}" for c in
                             ["roe", "gp_margin", "np_margin", "debt_ratio", "rev_yoy",
                              "profit_yoy", "profit_yoy_chg", "cf_quality", "eps", "bps"]],
                            dtype=float)
    ev = pd.DataFrame({
        "avail": [str(_disclose_date(r)) for r in df["报告期"]],
        "roe": _finance_get(df, "净资产收益率"),
        "gp_margin": _finance_get(df, "销售毛利率"),
        "np_margin": _finance_get(df, "销售净利率"),
        "debt_ratio": _finance_get(df, "资产负债率"),
        "rev_yoy": _finance_get(df, "营业总收入同比增长率"),
        "profit_yoy": _finance_get(df, "净利润同比增长率"),
        "eps": _finance_get(df, "基本每股收益"),
        "bps": _finance_get(df, "每股净资产"),
        "ocf": _finance_get(df, "每股经营现金流"),
    })
    for c in ["roe", "gp_margin", "np_margin", "debt_ratio", "rev_yoy", "profit_yoy", "eps", "bps", "ocf"]:
        ev[c] = pd.to_numeric(ev[c], errors="coerce") / (100.0 if "ratio" in c or "margin" in c or c == "debt_ratio" else 1.0)
    aligned = _asof_align(dates, ev, "avail", [c for c in ev.columns if c != "avail"])
    out = pd.DataFrame(index=dates)
    out["fin_roe"] = aligned["roe"].fillna(0.0)
    out["fin_gp_margin"] = aligned["gp_margin"].fillna(0.0)
    out["fin_np_margin"] = aligned["np_margin"].fillna(0.0)
    out["fin_debt_ratio"] = aligned["debt_ratio"].fillna(0.0)
    out["fin_rev_yoy"] = aligned["rev_yoy"].fillna(0.0)
    out["fin_profit_yoy"] = aligned["profit_yoy"].fillna(0.0)
    out["fin_profit_yoy_chg"] = aligned["profit_yoy"].diff().fillna(0.0)  # 加速/减速
    out["fin_cf_quality"] = (aligned["ocf"] / aligned["eps"].replace(0, np.nan)).fillna(0.0)
    out["fin_eps"] = aligned["eps"].fillna(0.0)
    out["fin_bps"] = aligned["bps"].fillna(0.0)
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
    # 可用日 = 文件采集日(快照之后才有效)
    fp = EXTRA_DIR / "consensus" / f"{code}.parquet"
    avail = datetime.fromtimestamp(os.path.getmtime(fp)).date() if fp.exists() else date.today()
    mask = dates >= pd.Timestamp(avail)  # 采集日前 = NaN(回测早期无此特征)
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
    # 滚动 12 月累计分红/送转(按事件日对齐)
    out = pd.DataFrame(index=dates)
    div_cum = ev.set_index("avail")["fenhong"].sort_index().rolling("365D", min_periods=1).sum()
    song_cum = ev.set_index("avail")["songzhuangu"].sort_index().rolling("365D", min_periods=1).sum()
    div_3y = ev[ev["avail"] >= dates[-1] - pd.Timedelta(days=1095)].shape[0]  # 近3年事件次数
    song_3y = (ev["songzhuangu"] > 0).sum()
    out["xd_yield"] = (div_cum.reindex(dates).fillna(0.0) / close.replace(0, np.nan)).fillna(0.0)
    out["xd_div_cnt_3y"] = float(div_3y)
    out["xd_song_cnt_3y"] = float(song_3y)
    return out


def _forecast_features(code: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """业绩预告(事件, 发布日对齐): 近12月类型分/最大增幅/次数/新鲜度"""
    df = _load("forecast", code)
    cols = ["fc_type_12m", "fc_max_chg_12m", "fc_cnt_12m", "fc_freshness"]
    if df is None or len(df) == 0:
        return pd.DataFrame(index=dates, columns=cols, dtype=float)
    TYPE_SCORE = {"预增": 1.0, "略增": 0.5, "扭亏": 0.8, "续盈": 0.3, "预盈": 0.6,
                  "预减": -1.0, "略减": -0.5, "首亏": -0.8, "续亏": -1.0, "增亏": -0.6}
    ev = pd.DataFrame({
        "avail": pd.to_datetime(df["profitForcastExpPubDate"], errors="coerce"),
        "score": df["profitForcastType"].map(TYPE_SCORE).fillna(0.0),
        "chg": pd.to_numeric(df["profitForcastChgPctUp"], errors="coerce").fillna(0.0) / 100.0,
    })
    ev = ev.dropna(subset=["avail"]).sort_values("avail")
    if len(ev) == 0:
        return pd.DataFrame(index=dates, columns=cols, dtype=float)
    out = pd.DataFrame(index=dates, dtype=float)
    for i, d in enumerate(dates):
        w = ev[ev["avail"] <= d]
        if len(w) == 0:
            continue
        w12 = w[w["avail"] >= d - pd.Timedelta(days=365)]
        out.loc[d, "fc_type_12m"] = w12["score"].sum() / max(len(w12), 1)
        out.loc[d, "fc_max_chg_12m"] = w12["chg"].max()
        out.loc[d, "fc_cnt_12m"] = len(w12)
        out.loc[d, "fc_freshness"] = 1.0 / (1.0 + (d - w["avail"].iloc[-1]).days)  # 距最近预告
    return out.fillna(0.0)


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
    "forecast": _forecast_features,
}

# 各组固定列名模板（异常兜底时全 0 填充, 保证输出恒 33 列 → 拼接维度一致）
_EMPTY_COLS = {
    "fund_flow": ["ff_main_ratio", "ff_main_ratio_5d", "ff_main_inflow_days_5",
                  "ff_main_amt_20d", "ff_main_momentum", "ff_xbig_ratio_5d"],
    "finance": ["fin_roe", "fin_gp_margin", "fin_np_margin", "fin_debt_ratio",
                "fin_rev_yoy", "fin_profit_yoy", "fin_profit_yoy_chg",
                "fin_cf_quality", "fin_eps", "fin_bps"],
    "holders": ["hd_num_chg", "hd_avg_mcap"],
    "consensus": ["cs_buy_ratio", "cs_org_num", "cs_pred_pe", "cs_aim_dev", "cs_aim_spread"],
    "news": ["nw_cnt_5d", "nw_cnt_20d", "nw_src_div"],
    "xdxr": ["xd_yield", "xd_div_cnt_3y", "xd_song_cnt_3y"],
    "forecast": ["fc_type_12m", "fc_max_chg_12m", "fc_cnt_12m", "fc_freshness"],
}


def _zero_features(group: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """单数据面异常/缺失 → 全 0 列（保持固定列数契约, 覆盖率 0 → 关键面缺失由 incomplete 判定）"""
    return pd.DataFrame(0.0, index=dates, columns=_EMPTY_COLS[group])

# ════════════════════════════════════════════════════════════
#  缺口处置（数据完整性标记）
#
#  关键数据面(缺失=数据问题, 应剔除):
#    fund_flow/finance/holders — 每只股票理应都有(成交/财报/股东)
#  语义可缺失面(缺失=合法信息, 不剔除):
#    consensus(46% A股无研报覆盖) / forecast(业绩稳定无预告)
#    news(小盘股无新闻) / xdxr(次新股无分红记录)
# ════════════════════════════════════════════════════════════
KEY_GROUPS = ("fund_flow", "finance", "holders")
# 阈值 5%: 区分"窗口限制"与"数据缺失"
#   fund_flow 仅 120 天窗口 → 250 天 dates 覆盖 48% (合法, 非缺失)
#   数据面完全未采到 → 覆盖 0% (缺失 → incomplete)
EXTRA_COVERAGE_THRESHOLD = 0.05
_PREFIX_TO_GROUP = {"ff": "fund_flow", "fin": "finance", "hd": "holders",
                    "cs": "consensus", "nw": "news", "xd": "xdxr", "fc": "forecast"}


def build_extra_with_flag(dates: pd.DatetimeIndex, close: pd.Series,
                          code: str, groups: list[str] | None = None
                          ) -> tuple[pd.DataFrame, bool, dict]:
    """构建特征 + 数据完整性标记（缺口股票处置入口）。

    Returns:
        (features_df, incomplete, coverage)
        incomplete=True → 关键数据面覆盖率 < 阈值, 训练/预测端应剔除该股票
    """
    groups = groups or list(FEATURE_GROUPS.keys())
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
    groups = groups or list(FEATURE_GROUPS.keys())
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
