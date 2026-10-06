#!/usr/bin/env python3
"""★每日数据校验：用同花顺全市场 dump 当独立标尺，盯住 004 的当日入库数据。

━━ 为什么需要它 ━━
2026-10-06 的跨源对拍，一次查出 3 类历史事故（07-06 污染 / 复权口径 / 单位错乱），
但它们**都是几个月后才被发现的**。这三类问题本可以在**发生当天**就抓到。

本脚本每交易日跑一次，成本极低（1MB 流量 / ~2 秒），却能当场报警。

━━ 校验逻辑（全部来自实测经验）━━
  1. 下载 daily-k-10d（最近 10 个交易日全市场，约 1MB）
  2. 与 004 的 stock_daily 逐值比对：OHLC / 成交量 / 成交额
  3. **分类报警**（不只是"有差异"，而是指出像哪类事故）：
       · 单位疑似错误   —— 量或额比值 ≈ 0.01 / 100（手↔股、万元↔元）
       · 复权口径疑似漂移 —— 同一股票多日价格比值恒定 ≠ 1
       · 数值不一致     —— 散点式差异（疑似同步故障，如 2026-07-06）
  4. 已知的**正常**情况会被排除，避免误报狼来了：
       · 补缺路径的 amount 是 `close × volume` 估算值（非真实成交额）
       · 004 尚未同步当日数据（正常时序差，不算错）

用法：
  HITHINK_FINANCE_API_KEY=<key> python scripts/daily_dump_verify.py
  # 不要推送：加 --no-push；  只查某天：--days 5
退出码：0=一切正常  1=发现问题（供 cron 判断）
"""
import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ★API Key 必须从 .env 读，不能依赖 ~/.bashrc ——
#   cron 不加载 bashrc（.env 里已有 2026-09-24 的教训注释：
#   「同花顺主源自 08-20 上线起每夜静默失效、掉回 baostock」）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE = "https://fuyao.aicubes.cn"
KEY = os.environ.get("HITHINK_FINANCE_API_KEY", "")
# 自定位：本脚本位于 <004>/scripts/，向上两级即项目根
SX = Path(__file__).resolve().parent.parent
DB = str(SX / "data" / "sequoia_v2.db")
CACHE = str(SX / "data" / "verify")

# ── 容差设计（2026-10-06 实测调参，避免误报）──
# 价格：两源逐值完全一致（实测 0 差异）→ 极小容差
TOL_PRICE = 1e-6
# 成交量：004 从「手」精度源换算而来，最大差半手=50 股（实测 max=50）
#         → 用「相对 1e-4 或 绝对 100 股」双条件，两者取宽
TOL_VOL_REL = 1e-4
TOL_VOL_ABS = 100.0
# 成交额：★不做绝对值比对。
#   原因：补缺路径写入的是 close×volume 估算值，而真实值=volume×VWAP，
#   两者天然差 (close−vwap)/vwap（实测 p50=0.5%、p90=1.6%）——无法区分
#   "估算值" 与 "真实值"，硬比必然大面积误报。
#   改用【物理约束】：成交额÷成交量 = 均价，必须落在当日 [low, high] 内。
#   该约束对 100 倍 / 10000 倍的单位错误极其敏感（历史三次事故都能抓到）。
VWAP_LO = 0.95        # 均价 / 最低价 的下界
VWAP_HI = 1.15        # 均价 / 最高价 的上界
RATIO_UNIT_HI = 50    # 比值 > 50 或 < 1/50 → 疑似 100 倍单位错
RATIO_UNIT_LO = 0.02


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def download_dump():
    """获取 daily-k-10d 并缓存到本地（按内容日期命名，当天重复跑不重复下载）。"""
    os.makedirs(CACHE, exist_ok=True)
    r = requests.get(f"{BASE}/api/dump/market-dumps/daily-k-10d/download-url",
                     headers={"X-api-key": KEY}, timeout=30).json()
    if r.get("code") != 0:
        raise RuntimeError(f"获取下载链接失败: code={r.get('code')} {r.get('message')}")
    url = r["data"]["presigned_url"]
    fn = url.split("?")[0].split("/")[-1]           # 如 ..._10d_20261005.parquet
    path = os.path.join(CACHE, fn)
    if os.path.exists(path) and os.path.getsize(path) > 100_000:
        log(f"dump 已缓存，跳过下载: {fn}")
        return path
    t0 = time.time()
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(resp.content)
    os.replace(tmp, path)
    log(f"dump 已下载: {fn}  {len(resp.content)/1024**2:.2f} MB  {time.time()-t0:.1f}s")
    return path


def classify(df):
    """把差异行归类，返回 {类别: DataFrame} + 统计。

    判据设计（2026-10-06 实测调参，目标=零误报）：
      · 价格  —— 逐值比对（两源实测完全一致）
      · 成交量 —— 逐值比对，容忍「手」精度（相对 1e-4 或 绝对 100 股）
      · 成交额 —— **不比绝对值**，改用物理约束 vwap∈[low,high]
                  （补缺估算值天然偏差 0.5~1.6%，硬比必误报）
    """
    out = {}
    n = len(df)

    # ── 价格 ──
    price_bad = np.zeros(n, bool)
    for c in ["open", "high", "low", "close"]:
        rel = ((df[c].astype(float) - df[f"{c}_t"].astype(float)).abs()
               / np.maximum(df[f"{c}_t"].astype(float).abs(), 1e-12))
        price_bad |= (rel > TOL_PRICE).values

    # ── 成交量 ──
    vol_ratio = df["volume"].astype(float) / df["volume_t"].astype(float).replace(0, np.nan)
    vol_abs = (df["volume"].astype(float) - df["volume_t"].astype(float)).abs()
    vol_bad = ((vol_ratio - 1).abs() > TOL_VOL_REL) & (vol_abs > TOL_VOL_ABS)

    # ── 成交额：物理约束 ──
    vwap = df["amount"].astype(float) / df["volume"].astype(float).replace(0, np.nan)
    lo = df["low"].astype(float)
    hi = df["high"].astype(float)
    amt_bad = ((vwap < lo * VWAP_LO) | (vwap > hi * VWAP_HI)) & vwap.notna()

    # ① 单位疑似错误：比值极端
    amt_ratio = df["amount"].astype(float) / df["turnover"].astype(float).replace(0, np.nan)
    unit = df[(vol_ratio > RATIO_UNIT_HI) | (vol_ratio < RATIO_UNIT_LO) |
              (amt_ratio > RATIO_UNIT_HI) | (amt_ratio < RATIO_UNIT_LO)]
    if len(unit):
        out["①单位疑似错误(手↔股 / 万元↔元)"] = unit

    # ② 复权口径疑似漂移：同股多日价格比值恒定 ≠ 1
    g = df.assign(pr=df["close"].astype(float) / df["close_t"].astype(float)) \
          .groupby("symbol")["pr"].agg(["median", "std", "size"])
    g = g[(g["size"] >= 5) & ((g["median"] - 1).abs() > 1e-3) & (g["std"] < 1e-3)]
    if len(g):
        out["②复权口径疑似漂移"] = df[df["symbol"].isin(g.index)]

    # ③ 物理约束被违反（均价跑到当日价格区间外）
    if amt_bad.any():
        out["③均价越界(单位/数值异常)"] = df[amt_bad]

    # ④ 价格/成交量逐值不符（散点）
    rest = df[(price_bad | vol_bad)]
    if len(out):
        # 显式 astype(str)：symbol/date 可能是 Arrow string dtype，
        # 与 object dtype 直接用 + 拼接会抛 TypeError
        key = lambda x: x["symbol"].astype(str) + "|" + x["date"].astype(str)
        known = set(key(pd.concat(out.values())))
        rest = rest[~key(rest).isin(known)]
    if len(rest):
        out["④价格或成交量不符"] = rest

    stats = {"n_rows": n,
             "price_diff": int(price_bad.sum()),
             "vol_diff": int(vol_bad.sum()),
             "vwap_out": int(amt_bad.sum())}
    return out, stats


def selftest(m):
    """★自测：把三类历史事故人为注入数据，验证分类器真能抓到。

    只报"正常"的检查器等于没用——必须证明它在出事时会响。
    """
    log("=" * 70)
    log("自测模式：注入三类历史事故，验证能否捕获")
    log("=" * 70)
    # ★按「股票」取样而非按行：判据②要求同一股票 ≥5 天数据，
    #   随机抽行会让每只股票不足 5 天，导致漏检（自测假失败）
    syms = sorted(m["symbol"].unique())[:600]
    base = m[m["symbol"].isin(syms)].copy()
    log(f"  自测样本：{len(base):,} 行 / {base['symbol'].nunique()} 只"
        f"（按股票取样，保证每只有多日数据）")
    cases = [
        ("①单位错-v"
         "olume手", lambda d: d.assign(volume=d["volume"] / 100.0),
         "①单位疑似错误(手↔股 / 万元↔元)"),
        ("①单位错-amount万元", lambda d: d.assign(amount=d["amount"] / 10000.0),
         "①单位疑似错误(手↔股 / 万元↔元)"),
        ("②复权口径漂移", lambda d: d.assign(
            **{c: d[c] * 2.889 for c in ["open", "high", "low", "close"]}),
         "②复权口径疑似漂移"),
        ("③均价越界(额×100)", lambda d: d.assign(amount=d["amount"] * 100.0),
         "③均价越界(单位/数值异常)"),
        ("④散点价格错", lambda d: d.assign(
            close=d["close"] * 1.5, high=d["high"] * 1.5), "④价格或成交量不符"),
    ]
    ok_all = True
    for name, mutate, expect in cases:
        d = mutate(base).copy()
        buckets, _ = classify(d)
        hit = any(expect in k for k in buckets)
        n = sum(len(v) for k, v in buckets.items() if expect in k)
        ok_all &= hit
        log(f"  {'✅' if hit else '❌'} {name:22s} → 期望『{expect}』"
            f"  命中 {n:,} 行")
    # 干净数据不应报警
    buckets, _ = classify(base)
    clean_ok = len(buckets) == 0
    ok_all &= clean_ok
    log(f"  {'✅' if clean_ok else '❌'} {'干净数据应无告警':22s} → "
        f"{'无告警' if clean_ok else '误报: ' + str(list(buckets))}")
    log("")
    log("自测结果：" + ("✅ 全部通过（能抓事故、不误报）" if ok_all else "❌ 有失败项"))
    return 0 if ok_all else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--selftest", action="store_true",
                    help="注入三类历史事故，验证分类器能否捕获（不碰生产库）")
    ap.add_argument("--days", type=int, default=10, help="只检查最近 N 个交易日")
    args = ap.parse_args()
    if len(KEY) < 8:
        sys.exit("❌ 缺 HITHINK_FINANCE_API_KEY")

    log("=" * 70)
    log("每日数据校验：同花顺 dump  vs  004 stock_daily")
    log("=" * 70)

    try:
        path = download_dump()
    except Exception as e:
        log(f"❌ 下载失败: {e}")
        sys.exit(1)

    t = pd.read_parquet(path, columns=["thscode", "date_ms", "open_price", "high_price",
                                       "low_price", "close_price", "volume", "turnover"])
    t["symbol"] = t["thscode"].str.split(".").str[0]
    t["date"] = (pd.to_datetime(t["date_ms"], unit="ms", utc=True)
                 .dt.tz_convert("Asia/Shanghai").dt.strftime("%Y-%m-%d"))
    t = t.drop(columns=["thscode", "date_ms"]).rename(
        columns={"open_price": "open_t", "high_price": "high_t", "low_price": "low_t",
                 "close_price": "close_t", "volume": "volume_t", "turnover": "turnover"})
    dump_dates = sorted(t["date"].unique())
    log(f"dump 覆盖 {len(dump_dates)} 个交易日: {dump_dates[0]} ~ {dump_dates[-1]}"
        f"  股票 {t['symbol'].nunique():,} 只")

    conn = sqlite3.connect(DB)
    d = pd.read_sql("SELECT symbol,date,open,high,low,close,volume,amount FROM stock_daily "
                    "WHERE date>=?", conn, params=(dump_dates[0],))
    conn.close()
    log(f"004 同期 {len(d):,} 行 / {d['symbol'].nunique():,} 只")
    if not len(d):
        log("⚠️ 004 该区间无数据（可能尚未同步）—— 跳过，非错误")
        sys.exit(0)

    m = t.merge(d, on=["symbol", "date"], how="inner")
    m = m[(m["close_t"] > 0)].copy()
    # 只查最近 N 个交易日
    keep = sorted(m["date"].unique())[-args.days:]
    m = m[m["date"].isin(keep)]
    log(f"比对 {len(m):,} 行（最近 {len(keep)} 个交易日）\n")

    # 覆盖缺口提示（不算错误，但值得知道）
    only_dump = len(t[t["date"].isin(keep)]) - len(m)
    if only_dump:
        log(f"提示：同花顺有而 004 无 {only_dump:,} 行（新股/未同步/北交所，属正常范围）\n")

    if args.selftest:
        sys.exit(selftest(m))

    buckets, stats = classify(m)

    log("=" * 70)
    log(f"逐值统计：价格不符 {stats['price_diff']:,} 行 ｜ 成交量不符 {stats['vol_diff']:,} 行 "
        f"｜ 均价越界 {stats['vwap_out']:,} 行（共 {stats['n_rows']:,} 行）")
    if not buckets:
        log("✅ 全部正常：价格 / 成交量一致，成交额通过物理约束（均价落在当日区间内）")
        log("   注：成交额未做绝对值比对——补缺路径的估算值天然偏差 0.5~1.6%，硬比必误报")
        log("       故改用物理约束，对 100倍/10000倍 单位错误同样敏感")
        sys.exit(0)

    log("🔴 发现问题：")
    lines = []
    for k in sorted(buckets):
        sub = buckets[k]
        msg = (f"  {k}: {len(sub):,} 行 / {sub['symbol'].nunique()} 只股票 "
               f"（{sub['date'].min()} ~ {sub['date'].max()}）")
        log(msg)
        lines.append(msg)
        cols = ["symbol", "date", "close", "close_t", "volume", "volume_t"]
        sample = sub.head(5)[[c for c in cols if c in sub.columns]]
        for r in sample.itertuples():
            s = (f"      {r.symbol} {r.date}  收盘 {r.close} vs {r.close_t}"
                 f"   量 {r.volume:,.0f} vs {r.volume_t:,.0f}")
            log(s)
            lines.append(s)
        if len(sub) > 5:
            lines.append(f"      … 其余 {len(sub)-5:,} 行见日志")

    # ── 推送 ──
    if not args.no_push:
        try:
            sys.path.insert(0, str(SX))
            from sequoia_x.core.config import get_settings
            from wxpusher import WxPusher
            st = get_settings()
            text = ("【004 数据校验告警】\n"
                    f"校验区间 {keep[0]} ~ {keep[-1]}\n\n" + "\n".join(lines[:40]))
            WxPusher.send_message(content=text, token=st.wxpusher_token,
                                  topic_ids=st.wxpusher_topic_ids, content_type=1)
            log("\n已推送微信告警")
        except Exception as e:
            log(f"\n推送失败（不影响判定）: {str(e)[:80]}")

    # 落盘记录
    os.makedirs(CACHE, exist_ok=True)
    rec = {"time": str(datetime.now()), "range": [keep[0], keep[-1]],
           "buckets": {k: {"rows": len(v), "stocks": int(v["symbol"].nunique())}
                       for k, v in buckets.items()}}
    with open(os.path.join(CACHE, "verify_history.jsonl"), "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    sys.exit(1)


if __name__ == "__main__":
    main()
