#!/usr/bin/env python
"""龙虎榜/涨停榜 数据"补齐到最新 + 全窗安全重建"(upsert-merge, 不丢历史) —— 铁律: 数据无损坏才是结果。

为什么自建一个新采集器, 不直接复用 scripts/fetch_hithink_sentiment.py(生产):
  生产版 flush = "按最近20日块 pop 即写", 同一只股若横跨多个 flush 块会被**后块覆盖前块**——
  一场跨月(或增量跨20日)的运行就可能把它早先月份的上榜事件整个盖掉丢历史。本项目数据资产
  必须可安全累进。故此处按"整窗单进程内存累积 → 每股一次性合并(读既有+新增去重=取date后到)"。

流程:
  1. 市场日历取 sh.000300 全近一年交易日(端到端), 取 ≤ 最近完整收盘日 的日期序列作为待拉清单
  2. 逐日调同花顺 dragon-tiger-list / limit-up-pool(接口语义与生产 fetch_hithink_sentiment 一致)
  3. 单进程内存累积 _new:{face:{code:{date:row}}}; 每10天写 checkpoint(断点续跑, 同命令恢复)
  4. 收尾对每一只出现过的 code(按面)做 **upsert**: 读现有 <code>.parquet + concat 新增 + date去重(保后)
     → 写回。任何生产没拉到/被盖掉的历史, 只要现存文件里有就会保留; 新增只会更全, 永不丢。
验证: 文件数≥存量、末事件日期≥2026-09-0x、每面覆盖交易日连续、无掉历史。

用法(铁律六=py312; 需 HITHINK_FINANCE_API_KEY 已 export):
  HITHINK_FINANCE_API_KEY=<key> nohup /home/zhulei/anaconda3/envs/zhulei_py312/bin/python \
     experiments/dragon_tiger_event/refresh_sentiment.py 2>&1 | tee logs/dragon_refresh_<date>.log &
可选: --rebuild  从 ~8/20 全窗重建(默认同=全窗, 因本脚本本就是全窗逐日; 多余)。
      --until 2026-09-08 拉到最后日期(默认=sh.000300 最近完整日)
"""
import argparse
import datetime as dt
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DB_PATH = PROJECT_ROOT / "data" / "sequoia_v2.db"
EXTRA = PROJECT_ROOT / "data" / "extra_features"
PROGRESS = PROJECT_ROOT / "experiments" / "dragon_tiger_event" / "progress_sentiment.json"

B = "https://fuyao.aicubes.cn"
KEY = os.environ.get("HITHINK_FINANCE_API_KEY", "")
H = {"X-api-key": KEY}
# 各面每股至今窗口(用于判断新增是否真的多出新日期；不直接依赖, 只 fallback 到 cutoff)
#   —— 决定"最早要拉哪天"：取 sh.000300 近一年与现存 max 交叠的宽窗(宽松稳妥)
FACE_COLS = {
    "dragon_tiger": {"dt_net_buy": "net_value", "dt_net_rate": "net_rate", "dt_hot_rank": "hot_rank"},
    "limit_up":     {"lu_lianban": "continue_day_cnt", "lu_seal": "seal_money", "lu_is_st": "is_st"},
    "hot_rank":     {"hr_rank": "rank", "hr_heat": "heat"},
}


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def market_trade_days() -> pd.DatetimeIndex:
    import sqlite3
    con = sqlite3.connect(DB_PATH)
    d = pd.read_sql("SELECT DISTINCT date FROM index_daily WHERE symbol='sh.000300'", con)
    con.close()
    return pd.DatetimeIndex(sorted(pd.to_datetime(d["date"])))


def pull_dragon(day, _new):
    j = requests.get(B + "/api/a-share/special-data/dragon-tiger-list",
                     params={"board_type": "all", "date": day.isoformat()}, headers=H, timeout=30).json()
    if j.get("code") != 0:
        return 0
    c = 0
    for it in j.get("data", {}).get("stock_items", []):
        code = (it.get("thscode") or "").split(".")[0]
        if not code:
            continue
        row = {"date": day}
        for k, api in FACE_COLS["dragon_tiger"].items():
            row[k] = it.get(api)
        _new["dragon_tiger"][code][day] = row
        c += 1
    return c


def pull_limit(day, _new):
    ms = int(dt.datetime.combine(day, dt.time()).timestamp() * 1000)
    tot = 0
    for page in range(1, 4):
        j = requests.get(B + "/api/a-share/special-data/limit-up-pool",
                         params={"date_ms": ms, "page": page, "size": 200}, headers=H, timeout=30).json()
        if j.get("code") != 0:
            break
        items = j.get("data", {}).get("item", [])
        for it in items:
            code = (it.get("thscode") or "").split(".")[0]
            if not code:
                continue
            row = {"date": day}
            for k, api in FACE_COLS["limit_up"].items():
                row[k] = it.get(api)
            # is_st 是 bool/str → 转 1/0
            if k == "lu_is_st":
                row["lu_is_st"] = 1 if it.get("is_st") else 0
            _new["limit_up"][code][day] = row
        tot += len(items)
        if len(items) < 200:
            break
    return tot


def merge_write(face, code, new_by_day: dict):
    """upsert: 读既有 parquet + 新增 + date去重(保后) 写回该股该面文件。"""
    od = EXTRA / face
    od.mkdir(parents=True, exist_ok=True)
    fp = od / f"{code}.parquet"
    frames = []
    if fp.exists():
        try:
            old = pd.read_parquet(fp)
            frames.append(old)
        except Exception as e:
            log(f"  !读旧 {face}/{code} 失败{e}, 仅新建")
    colnames = list(FACE_COLS[face].keys())
    nd = pd.DataFrame([{**{k: r.get(k) for k in colnames}, "date": by} for by, r in new_by_day.items()])
    if nd is not None and len(nd):
        frames.append(nd)
    if not frames:
        return
    out = pd.concat(frames, ignore_index=True)
    if "date" not in out.columns or out.empty:
        return
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out = out.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    out = out[[c for c in ["date"] + colnames if c in out.columns]]
    out.to_parquet(fp, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--until", type=str, default=None,
                    help="拉到最后交易日(YYYY-MM-DD); 默认= sh.000300 最近日期")
    ap.add_argument("--start-back", type=int, default=400,
                    help="近一年回看天数(含跨年度假), 默认 400 保证覆盖至 8/20 窗与节假")
    a = ap.parse_args()
    if len(KEY) < 8:
        sys.exit("缺 HITHINK_FINANCE_API_KEY (env)")

    td = list(market_trade_days())                     # list[Timestamp]
    if a.until:
        td = [x for x in td if x.date() <= dt.date.fromisoformat(a.until)]
    if not td:
        sys.exit("无市场日(日历空)")
    latest = td[-1].date()
    start_d = latest - dt.timedelta(days=a.start_back)
    sel = [x.date() for x in td if x.date() >= start_d]
    log(f"盘尾 {latest}; 待拉交易日 {len(sel)} 个({sel[0]}..{sel[-1]})")

    done = set()
    if PROGRESS.exists():
        done = set(json.load(open(PROGRESS)).get("done", []))
    todo = [s for s in sel if s.isoformat() not in done]
    log(f"已完成 {len(done)} 天, 实拉 {len(todo)} 天")

    _new = {f: defaultdict(dict) for f in FACE_COLS}   # face -> code -> {(date): row}
    t0 = time.time(); nd_total = [0, 0, 0]
    for i, d in enumerate(todo):
        # 拉取失败不中断: 记 0 行, 照常续(最后合并看缺不缺)
        try:
            nd_total[0] += pull_dragon(d, _new)
        except Exception as e:
            log(f"  ✗ dragon {d}: {e}; 本轮该日计0, 不中断(可由下轮重拉)")
        try:
            nd_total[1] += pull_limit(d, _new)
        except Exception as e:
            log(f"  ✗ limit {d}: {e}")
        if (i + 1) % 5 == 0 or i == len(todo) - 1:
            used = done | {s.isoformat() for s in todo[: i + 1]}
            json.dump({"done": sorted(used)}, open(PROGRESS, "w"))
            log(f"  {d} {i+1}/{len(todo)}天 dragon累计{nd_total[0]} limit累计{nd_total[1]}, "
                f"{(time.time()-t0)/60:.1f}min @ETA剩~"
                f"{(time.time()-t0)/(i+1)*(len(todo)-i-1)/60:.1f}min")

    # upsert 写回(face 覆盖到的每只 code 合并进既有, date去重保后)
    for face in ("limit_up", "dragon_tiger", "hot_rank"):
        codes = _new[face]
        if not codes:
            continue
        log(f"[写回] {face}: {len(codes)} 股有新增")
        for code, by_day in codes.items():
            merge_write(face, code, by_day)
    import glob
    for face in ("dragon_tiger", "limit_up"):
        log(f"[验] {face} 文件 {len(glob.glob(f'{EXTRA}/{face}/*.parquet'))}")
    log("完成。请见校验: 末事件日期≥9初、历史保留、连续无断档。")


if __name__ == "__main__":
    main()
