#!/usr/bin/env python3
"""20 只对拍（修正版）：同花顺财务重建历史估值 vs baostock 库存值。

关键修正（2026-08-20 发现）：
  - THS income 的 report_date_ms 近期期数据**不可信**（Y2025 Q3 误标 2025-10-30），
    不能用于 asof！改用**法定披露日**（同项目 _disclose_date 惯例）：
      Q1→04-30 | H1→08-31 | Q3→10-31 | 年报→次年04-30
  - basic_eps 是**财年内累计 YTD**：季度值 = 当年累计 − 上年同季度累计；Q1=本身。
  - TTM EPS = 最近 4 个已披露报告季的季度值之和（asof 时点为最近披露期）。

判定：diff ≤ 25% 视为口径一致 → 同花顺可重建历史估值，走全量迁移；
      否则 → 历史库存保留 baostock（B1 过渡），同花顺只补新增日。
只读，不写库。
"""
import os
import sys
import sqlite3
import datetime
import requests

B = "https://fuyao.aicubes.cn"
KEY = os.environ.get("HITHINK_FINANCE_API_KEY", "")
H = {"X-api-key": KEY}
DB = "data/sequoia_v2.db"
CANDIDATES = ['600519', '000858', '000002', '601398', '600036', '300750', '002594',
              '688981', '601318', '000001', '600900', '601899', '002415', '600276',
              '000333', '601012', '002714', '300142', '601988', '600031']
ASOF = "2026-08-19"


def ms(s): return int(datetime.datetime.strptime(s, "%Y%m%d").timestamp() * 1000)
def thscode(c): return f"{c}.SH" if c[0] in "69" else f"{c}.SZ"


def disclose_date(fy, fp):
    """法定披露日（保守，与项目 _disclose_date 一致）。"""
    if fp == "Q1":  return datetime.date(fy, 4, 30)
    if fp == "Q2":  return datetime.date(fy, 8, 31)
    if fp == "Q3":  return datetime.date(fy, 10, 31)
    if fp == "Q4":  return datetime.date(fy + 1, 4, 30)
    return datetime.date(fy + 1, 4, 30)


def get_income(code):
    j = requests.get(B + "/api/a-share/financials/income-statements",
                     params={"thscode": thscode(code), "period": "quarterly",
                             "start": ms("20220101"), "end": ms("20270101")},
                     headers=H, timeout=20).json()
    if not j.get("code") == 0:
        return None
    items = []
    for it in j["data"]["item"]:
        try:
            items.append({"fy": int(it["fiscal_year"]),
                          "fp": it["fiscal_period"],
                          "eps": float(it.get("basic_eps") or 0.0)})
        except (KeyError, TypeError, ValueError):
            continue
    # 去重：同 (fy,fp) 保留 eps 绝对值最大（防重复/脏行）
    best = {}
    for it in items:
        k = (it["fy"], it["fp"])
        if k not in best or abs(it["eps"]) > abs(best[k]["eps"]):
            best[k] = it
    out = []
    for (fy, fp), it in best.items():
        it["avail"] = disclose_date(fy, fp)
        out.append(it)
    out.sort(key=lambda x: (x["avail"], x["fy"], _qnum(x["fp"])))
    return out


def _qnum(fp):
    return {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}.get(fp, 0)


def ttm_eps(items, asof):
    """截至 asof 已披露期的 TTM EPS = 最近 4 个披露季的季度值之和。"""
    avail = [it for it in items if it["avail"] <= asof]
    if len(avail) < 4:
        return 0.0, len(avail)
    last4 = avail[-4:]
    # 季度值：Q1=累计本身；其余=当年累计−上月季累计（跨年同句柄）
    quarters = []
    # 用 (fy,q) 定位，计算季度值需当年内前一季的累计
    bykey = {(it["fy"], it["fp"]): it["eps"] for it in items}
    for it in last4:
        fy, fp = it["fy"], it["fp"]
        if fp == "Q1":
            q = it["eps"]
        else:
            prev_fp = {"Q2": "Q1", "Q3": "Q2", "Q4": "Q3"}[fp]
            prev = bykey.get((fy, prev_fp))
            q = it["eps"] - (prev if prev is not None else 0.0)
        quarters.append(q)
    return sum(quarters), len(last4)


def main():
    if len(KEY) < 8:
        print("缺 KEY"); sys.exit(1)
    conn = sqlite3.connect(DB)
    print(f"{'股票':<7}{'隐含EPS(bs)':>12}{'THS TTM':>12}{'相符%':>8}  判断")
    print("-" * 58)
    ok = fail = 0
    for code in CANDIDATES:
        row = conn.execute("SELECT peTTM, close FROM stock_daily WHERE symbol=? AND date=?",
                           (code, ASOF)).fetchone()
        if not row or not row[0]:
            print(f"{code:<7} 无 baostock {ASOF} 值，跳过"); continue
        bs_pe, close = row[0], row[1]
        implied = close / bs_pe if bs_pe else 0.0
        items = get_income(code)
        if not items:
            print(f"{code:<7} ths 拉取失败"); continue
        ttm, n = ttm_eps(items, datetime.date(*map(int, ASOF.split("-"))))
        if abs(implied) > 1e-6:
            diff = abs(ttm - implied) / abs(implied) * 100
        else:
            diff = float('nan')
        flag = "✅一致" if diff <= 25 else "⚠️偏差"
        ok += flag == "✅一致"; fail += flag != "✅一致"
        print(f"{code:<7}{implied:>12.3f}{ttm:>12.3f}{diff:>7.1f}%  {flag}")
    print("-" * 58)
    print(f"对拍(asof{ASOF}): 一致 {ok} / 偏差 {fail} / 总数 {ok+fail}")
    conn.close()


if __name__ == "__main__":
    main()
