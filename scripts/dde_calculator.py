#!/usr/bin/env python3
"""mootdx DDE 资金流自算器（路 A）— 不依赖任何第三方资金流接口

原理:
    A股"主力资金流"本质 = 逐笔成交按金额分档 + 按买卖方向累计。
    mootdx 的 transactions() 可拉历史任意日期的逐笔(含买卖方向),
    按东财口径分档(超大>100万 / 大20-100万 / 中5-20万 / 小<5万)自算。

实测验证(2026-08-07, 600519 2026-08-06):
    DDE 主力净额 -1.03亿 vs 东财基准 -1.35亿 → 方向一致、量级同档。
    差异来源: 尾盘集合竞价盘(buyorsell=8, 152笔)未归入买卖 + 东财阈值细节。

buyorsell 编码(通达信逐笔协议原始值, 实测分布 0/1/2/5/8):
    0 = 主动买   1 = 主动卖   2/5/8 = 中性盘(集合竞价等, 不归买卖)

用法:
    python3 dde_calculator.py --codes codes.txt --start 20260701 --end 20260807
    输出: <OUT>/dde/{code}.parquet  (每日: 主力/超大/大/中/小单净额+占比)
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mootdx_client import get_client

# ── 东财口径分档阈值(单笔成交金额) ──
TIER_BOUNDS = {"超大": 100e4, "大": 20e4, "中": 5e4}  # 元; 其余=小单

# 买卖方向映射(假设 A, 经东财对照验证)
BUY_SET = {0}       # 主动买
SELL_SET = {1}      # 主动卖
# 2/5/8 = 中性盘(集合竞价/尾盘撮合), 不归买卖

# 服务器(复用已验证的行情服务器)
SERVER = ("180.153.18.170", 7709)


def _tier(amount: float) -> str:
    if amount > TIER_BOUNDS["超大"]:
        return "超大"
    if amount > TIER_BOUNDS["大"]:
        return "大"
    if amount > TIER_BOUNDS["中"]:
        return "中"
    return "小"


def fetch_day_ticks(client, code: str, date: str) -> pd.DataFrame:
    """拉取单日全部逐笔(分页拉全)。返回 DataFrame 或 None。"""
    frames, start = [], 0
    while True:
        df = client.transactions(symbol=code, date=date, start=start, offset=2000)
        if df is None or len(df) == 0:
            break
        frames.append(df)
        start += len(df)
        if len(df) < 2000:
            break
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def compute_dde(code: str, date: str, client) -> dict | None:
    """单股单日 DDE 资金流。返回 dict 或 None(无成交/停牌)。"""
    ticks = fetch_day_ticks(client, code, date)
    if ticks is None or len(ticks) == 0:
        return None
    # 金额 = 价格 × 量(手) × 100
    ticks["amount"] = ticks["price"] * ticks["vol"] * 100
    ticks["tier"] = ticks["amount"].apply(_tier)

    buy = ticks[ticks["buyorsell"].isin(BUY_SET)]["amount"].sum()
    sell = ticks[ticks["buyorsell"].isin(SELL_SET)]["amount"].sum()
    total = buy + sell  # 中性盘不计入流向, 但计入成交额口径
    main_amt = 0.0
    row = {"code": code, "date": date, "total_amount": float(ticks["amount"].sum())}
    for tier in ("超大", "大", "中", "小"):
        sub = ticks[ticks["tier"] == tier]
        b = sub[sub["buyorsell"].isin(BUY_SET)]["amount"].sum()
        s = sub[sub["buyorsell"].isin(SELL_SET)]["amount"].sum()
        net = b - s
        row[f"{tier}净额"] = float(net)
        row[f"{tier}占比"] = float(net / total) if total else 0.0
        if tier in ("超大", "大"):
            main_amt += net
    row["主力净额"] = float(main_amt)
    row["主力占比"] = float(main_amt / total) if total else 0.0
    row["买额"] = float(buy)
    row["卖额"] = float(sell)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", required=True, help="股票代码文件(txt 每行一个)")
    ap.add_argument("--start", required=True, help="起始日期 YYYYMMDD")
    ap.add_argument("--end", required=True, help="结束日期 YYYYMMDD")
    ap.add_argument("--out", default="/public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x/data/extra_features/dde")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    codes = [l.strip() for l in open(args.codes) if l.strip()]
    dates = pd.date_range(args.start, args.end, freq="B").strftime("%Y%m%d").tolist()  # 工作日
    print(f"股票 {len(codes)} 只 × 日期 {len(dates)} 天 = {len(codes)*len(dates)} 个任务")

    os.makedirs(args.out, exist_ok=True)
    client = get_client()
    if client is None:
        sys.exit("mootdx 连接失败")

    done = {f[:-8] for f in os.listdir(args.out) if f.endswith(".parquet")}  # 断点续跑
    todo = [c for c in codes if c not in done]
    print(f"待计算 {len(todo)} 只(已跳过 {len(done)})")

    ok = fail = 0
    t0 = time.time()

    def work(code):
        rows = []
        for d in dates:
            try:
                r = compute_dde(code, d, client)
                if r:
                    rows.append(r)
            except Exception:
                pass
        if rows:
            df = pd.DataFrame(rows)
            df.to_parquet(os.path.join(args.out, f"{code}.parquet"), index=False)
            return True
        return False

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, c): c for c in todo}
        for i, f in enumerate(as_completed(futs), 1):
            if f.result():
                ok += 1
            else:
                fail += 1
            if i % 10 == 0 or i == len(todo):
                el = time.time() - t0
                eta = el / i * (len(todo) - i)
                print(f"  {i}/{len(todo)} ({100*i/len(todo):.0f}%) 成功{ok} 失败{fail} "
                      f"耗时{el/60:.1f}min ETA {eta/60:.1f}min", flush=True)

    print(f"\n完成: 成功 {ok} 只, 无成交/失败 {fail} 只 → {args.out}/")


if __name__ == "__main__":
    main()
