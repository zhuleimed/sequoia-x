#!/usr/bin/env python3
"""同花顺短线情绪维度采集：龙虎榜 / 涨停池 / 热榜（近一年，跨截面日榜→每股 parquet）。

2026-08-20 实现（Task #6）。数据特性：
  - 三接口均**仅支持近一年**（dragon-tiger/limit-up/hot-stock-list-history codes=1003 超期）
  - 均为**按交易日返回日榜**（跨截面），非每股时间序列 → 采集后按每股转置成事件 parquet
  - 由此构建的事件特征用"事件日 asof 前向填充"，近一年外的时间全 0（语义合理：短线情绪重近期）

输出（每股 parquet，`data/extra_features/<subset>/<code>.parquet`）：
  dragon_tiger: 龙虎榜上榜事件 (date, dt_net_buy, dt_net_rate, dt_hot_rank)
  limit_up:     涨停事件 (date, lu_lianban, lu_seal, lu_is_st)
  hot_rank:     热榜事件 (date, hr_rank, hr_heat)

断点续跑（铁律二）：按交易日处理完即存每日汇总，启动重跑可跳过已处理日。

用法：
  HITHINK_FINANCE_API_KEY=<key> py312 python scripts/fetch_hithink_sentiment.py
    [--days 365]     只采最近 N 天
    [--limit-days N]  限交易日数（调试）
"""
import argparse
import datetime
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests

PROJ = Path(__file__).resolve().parent.parent
OUT = PROJ / "data" / "extra_features"
SUBSETS = ("dragon_tiger", "limit_up", "hot_rank")
PROGRESS = PROJ / "scripts" / "tmp" / "hithink_sentiment_progress.json"
B = "https://fuyao.aicubes.cn"
KEY = os.environ.get("HITHINK_FINANCE_API_KEY", "")
H = {"X-api-key": KEY}

# 每股事件累积: code -> list[dict(date, 字段...)]
_EV: dict[str, list[dict]] = defaultdict(list)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def trade_days_since(cutoff: datetime.date) -> list[datetime.date]:
    """用同花顺交易日历（近一年窗口）取 cutoff 之后的交易日。"""
    r = requests.get(B + "/api/a-share/calendar/trading-days", headers=H, timeout=20).json()
    items = r.get("data", {}).get("item", [])
    days = sorted(it["date"] for it in items)  # yyyyMMdd
    out = []
    for s in days:
        d = datetime.datetime.strptime(s, "%Y%m%d").date()
        if d >= cutoff:
            out.append(d)
    return out


def _to_code(thscode: str) -> str:
    return thscode.split(".")[0] if "." in thscode else thscode


def pull_dragon_tiger(d: datetime.date):
    j = requests.get(B + "/api/a-share/special-data/dragon-tiger-list",
                     params={"board_type": "all", "date": d.isoformat()}, headers=H, timeout=20).json()
    if j.get("code") != 0:
        return
    for it in j.get("data", {}).get("stock_items", []):
        code = _to_code(it.get("thscode", ""))
        if not code:
            continue
        _EV[code].append({
            "date": d.isoformat(),
            "dt_net_buy": it.get("net_value"),
            "dt_net_rate": it.get("net_rate"),
            "dt_hot_rank": it.get("hot_rank"),
        })


def pull_limit_up(d: datetime.date):
    # 分页（每页 200，连板涨停总量 <200/日页足够，保守翻页）
    for page in range(1, 4):
        j = requests.get(B + "/api/a-share/special-data/limit-up-pool",
                         params={"date_ms": int(datetime.datetime.combine(d, datetime.time()).timestamp() * 1000),
                                 "page": page, "size": 200}, headers=H, timeout=20).json()
        if j.get("code") != 0:
            break
        data = j.get("data", {})
        items = data.get("item", [])
        for it in items:
            code = _to_code(it.get("thscode", ""))
            if not code:
                continue
            _EV[code].append({
                "date": d.isoformat(),
                "lu_lianban": it.get("continue_day_cnt"),
                "lu_seal": it.get("seal_money"),
                "lu_is_st": 1 if it.get("is_st") else 0,
            })
        if len(items) < 200:
            break


def pull_hot_rank(d: datetime.date):
    j = requests.get(B + "/api/a-share/special-data/hot-stock-list-history",
                     params={"date": d.isoformat()}, headers=H, timeout=20).json()
    if j.get("code") != 0:
        return
    for it in j.get("data", {}).get("item", []):
        code = _to_code(it.get("thscode", ""))
        if not code:
            continue
        _EV[code].append({
            "date": d.isoformat(),
            "hr_rank": it.get("rank"),
            "hr_heat": it.get("heat"),
        })


def flush(code: str):
    """把某股在全部三个数据面的累积事件写各自的 parquet + 清空（省内存）。"""
    recs = _EV.pop(code, None)
    if not recs:
        return
    df = pd.DataFrame(recs)
    df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    # 按数据面分列：每只股票只需要它出现的那些面
    by_sub = {"dragon_tiger": [c for c in df.columns if c.startswith("dt_")],
              "limit_up": [c for c in df.columns if c.startswith("lu_")],
              "hot_rank": [c for c in df.columns if c.startswith("hr_")]}
    for subset, cols in by_sub.items():
        if not cols:
            continue
        sub_df = df[["date"] + cols].drop_duplicates(subset=["date"], keep="last")
        # 只保留该数据面有真实事件的行（该股在别的面出现产生的 NaN 行丢弃）
        sub_df = sub_df.dropna(subset=cols, how="all")
        if sub_df.empty:
            continue
        od = OUT / subset
        od.mkdir(parents=True, exist_ok=True)
        sub_df.to_parquet(od / f"{code}.parquet", index=False)


def main(days_cutoff: datetime.date, done: set, debug_days: int | None = None):
    t0 = time.time()
    pull_funcs = [pull_dragon_tiger, pull_limit_up, pull_hot_rank]
    day_list = trade_days_since(days_cutoff)
    day_list = [d for d in day_list if d.isoformat() not in done]
    if debug_days:
        day_list = day_list[:debug_days]
        PROGRESS.unlink(missing_ok=True)  # 调试时不写进度，避免污染断点续跑
    log(f"待处理交易日 {len(day_list)} 个（自 {days_cutoff}），已完成 {len(done)}")

    for i, d in enumerate(day_list):
        for fn in pull_funcs:
            try:
                fn(d)
            except Exception as e:
                log(f"  {d} {fn.__name__} 异常: {e}")
        if (i + 1) % 20 == 0 or i == len(day_list) - 1:
            log(f"  {d} 处理 {i+1}/{len(day_list)} 天, 累积 {len(_EV)} 股, "
                f"耗时 {(time.time()-t0)/60:.1f}min")
        # 每 20 天 flush 一次 + 存进度（断点续跑）
        if (i + 1) % 20 == 0:
            for code in list(_EV.keys()):
                flush(code)
            done.add(d.isoformat())
            PROGRESS.parent.mkdir(exist_ok=True)
            json.dump({"done": sorted(done)}, open(PROGRESS, "w"))

    # 收尾 flush
    for code in list(_EV.keys()):
        flush(code)
    done.update(d.isoformat() for d in day_list)
    PROGRESS.parent.mkdir(exist_ok=True)
    json.dump({"done": sorted(done)}, open(PROGRESS, "w"))
    log(f"完成: 写 {len(list(OUT.glob('dragon_tiger/*.parquet')))} 只龙虎榜, "
        f"{len(list(OUT.glob('limit_up/*.parquet')))} 只涨停, "
        f"{len(list(OUT.glob('hot_rank/*.parquet')))} 只热榜")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--debug-days", type=int, default=None, help="只处理前 N 个交易日(调试)")
    a = ap.parse_args()
    if len(KEY) < 8:
        sys.exit("缺 HITHINK_FINANCE_API_KEY")
    out_base = datetime.date.today() - datetime.timedelta(days=a.days)
    # 断点续跑
    done = set()
    if PROGRESS.exists():
        done = set(json.load(open(PROGRESS)).get("done", []))
    main(out_base, done, debug_days=a.debug_days)
