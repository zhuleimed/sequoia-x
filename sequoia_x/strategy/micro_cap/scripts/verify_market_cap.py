"""校验「每日市值」缓存 —— 拿独立第三方数据当裁判。

## 为什么要单独一个脚本

构建脚本里的交叉校验用的是 `tdx_finance_snapshot.total_shares`，
但实测发现**这个快照本身可能是旧的**（例：600960 在 2026-08-21 有一次股本变化，
快照还停在变化前的 95051.6 万股）。

所以需要一个**独立来源**当裁判：用东方财富的实时行情快照（akshare
`stock_zh_a_spot_em`）里的「总市值」列，和本缓存**最新交易日**的市值逐一比对。
东财是第三方口径，跟本项目的数据链完全无关，出错方向也独立。

判据：
  - 相对误差中位数 < 0.1%   ⇒ 通过
  - 相对误差 > 5% 的股票数  ⇒ 打印出来人工看（可能是快照时点差异）

用法：
    /home/zhulei/anaconda3/envs/zhulei_py312/bin/python verify_market_cap.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[4]
CACHE = PROJ / "sequoia_x" / "strategy" / "micro_cap" / "data" / "market_cap.parquet"


def log(msg: str) -> None:
    print(msg, flush=True)


log(f"读取缓存: {CACHE}")
mc = pd.read_parquet(CACHE)
last_day = mc["date"].max()
mine = mc[(mc["date"] == last_day) & mc["total_mv"].notna()][["symbol", "total_mv"]].copy()
log(f"缓存最新交易日: {last_day.date()}，{len(mine):,} 只股票")

log("拉取东方财富实时快照（约 60~90 秒，请稍候）...")
import akshare as ak

spot = ak.stock_zh_a_spot_em()
log(f"东财返回 {len(spot):,} 行，列含: {[c for c in spot.columns if '市值' in c]}")

em = spot[["代码", "名称", "总市值"]].rename(
    columns={"代码": "symbol", "名称": "name", "总市值": "em_mv"}
)
em["symbol"] = em["symbol"].astype(str).str.zfill(6).astype("object")
em["em_mv"] = pd.to_numeric(em["em_mv"], errors="coerce")

cmp = mine.merge(em, on="symbol", how="inner")
cmp = cmp[(cmp["em_mv"] > 0) & (cmp["total_mv"] > 0)]
cmp["rel_err"] = (cmp["total_mv"] - cmp["em_mv"]).abs() / cmp["em_mv"]

log("")
log("=" * 70)
log(f"可比股票数: {len(cmp):,}")
log(f"相对误差中位数: {cmp['rel_err'].median():.4%}")
log(f"相对误差均值:   {cmp['rel_err'].mean():.4%}")
log(f"误差 < 0.1% 的占比: {(cmp['rel_err'] < 0.001).mean():.2%}")
log(f"误差 < 1%   的占比: {(cmp['rel_err'] < 0.01).mean():.2%}")
log(f"误差 > 5%   的股票数: {(cmp['rel_err'] > 0.05).sum()}")
log("=" * 70)

bad = cmp[cmp["rel_err"] > 0.05].copy()
if len(bad):
    bad["我的市值(亿)"] = (bad["total_mv"] / 1e8).round(1)
    bad["东财市值(亿)"] = (bad["em_mv"] / 1e8).round(1)
    log("")
    log(f"误差最大的 25 只（按我的市值升序，微盘优先看）:")
    show = bad.nsmallest(25, "total_mv")[
        ["symbol", "name", "我的市值(亿)", "东财市值(亿)", "rel_err"]
    ]
    log(show.to_string(index=False))
    log("")
    log(f"这 {len(bad)} 只里，我的市值 < 50 亿的（会进微盘池）: {(bad['total_mv'] < 50e8).sum()}")

    # 微盘池受影响程度：按我的市值排最小的 20%
    mine["rank"] = mine["total_mv"].rank(pct=True)
    pool = set(mine[mine["rank"] <= 0.20]["symbol"])
    badset = set(bad["symbol"])
    log(f"我的微盘池(最小20%={len(pool)}只) 中，与东财对不上的: {len(pool & badset)} 只")

if cmp["rel_err"].median() < 0.001:
    log("")
    log("✅ 校验通过")
else:
    log("")
    log("❌ 校验未通过：中位数误差偏大，需要排查")
    sys.exit(1)
