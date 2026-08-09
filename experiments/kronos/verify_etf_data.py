#!/usr/bin/env python3
"""ETF/指数数据多途径核验（2026-08-09, 铁律: 独立源交叉 + 关键点对照）

核验对象: 中证1000 指数 000852 + ETF 512100
途径:
  指数 000852: ①019 etf_daily.db  ②本项目 sequoia_v2.db(腾讯源)  ③baostock
              ④腾讯直连(web.ifzq.gtimg.cn 指数 day)  ⑤新浪(hq.sinajs.cn)
  ETF 512100: ①019 etf_daily.db  ②腾讯直连 qfq（2023-12 起）
核验内容:
  A. 全序列对齐: close 最大相对偏差 + 相关
  B. 关键时点 5 点（2020-08-03/2021-01-04/2022-01-04/2024-01-02/2026-07-28）逐值
  C. 关键事件: 2024-10-08 +16%（真实暴涨开盘?）; 2022-09-05（指数应无跳变）
  D. 重叠段(2023-12 起) ETF 019库 vs 腾讯qfq

用法: py312 python experiments/kronos/verify_etf_data.py
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
ETF_DB = "/public/home/hpc/zhulei/superman/quant/code/019_etf_daily_sync_and_backtest/data/etf_daily.db"
SEQ_DB = str(PROJECT_ROOT / "data/sequoia_v2.db")
KEY_DATES = ["2020-08-03", "2021-01-04", "2022-01-04", "2024-01-02", "2026-07-28"]


def from_019(table: str, symbol: str) -> pd.DataFrame:
    conn = sqlite3.connect(ETF_DB)
    df = pd.read_sql(f"SELECT date, open, close FROM {table} WHERE symbol=? ORDER BY date",
                     conn, params=[symbol])
    conn.close()
    return df


def from_seqdb(symbol: str) -> pd.DataFrame:
    conn = sqlite3.connect(SEQ_DB)
    df = pd.read_sql("SELECT date, open, close FROM index_daily WHERE symbol=? ORDER BY date",
                     conn, params=[symbol])
    conn.close()
    return df


def from_tencent_index(code: str, days: int = 2600) -> pd.DataFrame | None:
    """腾讯直连指数日线（day 无复权问题; 指数无拆分）。"""
    import requests
    url = f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,,,{days},"
    try:
        r = requests.get(url, timeout=15)
        data = r.json().get("data", {}).get(code, {})
        rows = data.get("day") or data.get("qfqday") or []
        df = pd.DataFrame(rows, columns=["date", "open", "close", "high", "low", "volume"])
        df[["open", "close"]] = df[["open", "close"]].apply(pd.to_numeric, errors="coerce")
        return df[["date", "open", "close"]]
    except Exception as e:
        print(f"  ⚠️ 腾讯指数 {code} 拉取失败: {e}")
        return None


def from_baostock_index(code: str) -> pd.DataFrame | None:
    """baostock 指数日线（主力源, 应全量）。"""
    import baostock as bs
    bs.login()
    rs = bs.query_history_k_data_plus(
        code, "date,open,high,low,close,volume",
        start_date="2019-01-01", end_date="2026-08-07", frequency="d")
    rows = []
    while (rs.error_code == "0") and rs.next():
        rows.append(rs.get_row_data())
    bs.logout()
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=rs.fields)
    df[["open", "close"]] = df[["open", "close"]].apply(pd.to_numeric, errors="coerce")
    return df[["date", "open", "close"]]


def from_sina_index(code: str) -> pd.DataFrame | None:
    """新浪指数日线。"""
    import requests
    url = ("https://quotes.sina.cn/cn/api/json_v2.php/"
           f"CN_MarketData.getKLineData?symbol={code}&scale=240&ma=no&datalen=2600")
    try:
        r = requests.get(url, timeout=15)
        data = r.json()
        if not isinstance(data, list) or not data:
            return None
        df = pd.DataFrame(data)[["day", "open", "close"]]
        df = df.rename(columns={"day": "date"})
        df[["open", "close"]] = df[["open", "close"]].apply(pd.to_numeric, errors="coerce")
        return df
    except Exception as e:
        print(f"  ⚠️ 新浪 {code} 拉取失败: {e}")
        return None


def align_compare(name: str, base: pd.DataFrame, other: pd.DataFrame | None) -> None:
    """全序列对齐对比（close 相对偏差 + 关键点）。"""
    if other is None or other.empty:
        print(f"  {name}: 无数据")
        return
    m = base.merge(other, on="date", suffixes=("_a", "_b"))
    if len(m) < 10:
        print(f"  {name}: 对齐仅 {len(m)} 行")
        return
    rel = (m["close_a"] - m["close_b"]).abs() / m["close_a"]
    print(f"  {name}: 对齐 {len(m)} 行 | close 最大偏差 {rel.max()*100:.4f}% "
          f"| 中位偏差 {rel.median()*100:.4f}% | 相关 {np.corrcoef(m['close_a'], m['close_b'])[0,1]:.6f}")


def main() -> None:
    print("═══ A. 指数 000852 多途径核验 ═══")
    d_019 = from_019("index_daily", "000852")
    d_seq = from_seqdb("sh.000852")
    d_bs = from_baostock_index("sh.000852")
    d_tx = from_tencent_index("sh000852")
    d_sina = from_sina_index("sh000852")

    print(f"数据量: 019={len(d_019)} | 项目库={len(d_seq)} | baostock={len(d_bs) if d_bs is not None else 0} "
          f"| 腾讯={len(d_tx) if d_tx is not None else 0} | 新浪={len(d_sina) if d_sina is not None else 0}")
    align_compare("baostock vs 019", d_019, d_bs)
    align_compare("腾讯直连 vs 019", d_019, d_tx)
    align_compare("新浪 vs 019", d_019, d_sina)
    align_compare("项目库 vs 019", d_019, d_seq)

    print("\n═══ B. 关键时点逐值对照（open/close）═══")
    srcs = [("019", d_019), ("项目库", d_seq), ("baostock", d_bs),
            ("腾讯", d_tx), ("新浪", d_sina)]
    for d in KEY_DATES:
        print(f"  {d}:")
        for nm, df in srcs:
            if df is None or df.empty:
                continue
            row = df[df["date"] == d]
            if len(row):
                print(f"    {nm:<6} open {row['open'].iloc[0]:>10.2f}  close {row['close'].iloc[0]:>10.2f}")
            else:
                print(f"    {nm:<6} （无此日）")

    print("\n═══ C. 关键事件确认 ═══")
    # 2024-10-08 跳变（国庆后暴涨）: 所有源是否一致
    for nm, df in srcs:
        if df is None or df.empty:
            continue
        i = df.index[df["date"] == "2024-10-08"]
        if len(i):
            i = i[0]
            prev = df["close"].iloc[i - 1] if i > 0 else np.nan
            chg = df["open"].iloc[i] / prev - 1 if prev > 0 else np.nan
            print(f"  2024-10-08 {nm:<6}: open 跳变 {chg*100:+.1f}%")
    # 2022-09-05 指数应无跳变（ETF 拆分仅影响 ETF 自身）
    for nm, df in srcs:
        if df is None or df.empty:
            continue
        i = df.index[df["date"] == "2022-09-05"]
        if len(i):
            i = i[0]
            prev = df["close"].iloc[i - 1] if i > 0 else np.nan
            chg = df["open"].iloc[i] / prev - 1 if prev > 0 else np.nan
            print(f"  2022-09-05 {nm:<6}: open 跳变 {chg*100:+.2f}% （应≈0）")

    print("\n═══ D. ETF 512100 重叠段核验（2023-12 起: 019库 vs 腾讯qfq）═══")
    etf_019 = from_019("etf_daily", "512100")
    etf_qfq = from_tencent_index("sh512100", 2000)
    if etf_qfq is not None:
        etf_qfq = etf_qfq.rename(columns={"open": "open_q", "close": "close_q"})
        m = etf_019.merge(etf_qfq, on="date")
        if len(m):
            rel = (m["close"] - m["close_q"]).abs() / m["close_q"]
            print(f"  重叠 {len(m)} 行 ({m['date'].iloc[0]}~{m['date'].iloc[-1]})")
            print(f"  close 最大偏差 {rel.max()*100:.3f}% | 中位 {rel.median()*100:.3f}%")
            # 比例恒定性检查（qfq 整体缩放 vs 019）
            ratio = m["close"] / m["close_q"]
            print(f"  close 比值: 均 {ratio.mean():.4f} 波动 {ratio.std():.6f}（≈0 则缩放恒定）")
            print(f"  腾讯 qfq 2023-12 前数据: {etf_qfq['date'].iloc[0]} 起 {len(etf_qfq)} 行")
        else:
            print("  无重叠")
    else:
        print("  ⚠️ 腾讯 qfq 拉取失败")


if __name__ == "__main__":
    main()
