"""抓「候选叠加资产」的日线 —— 用于研究"微盘为主、叠加其它策略"。

## 为什么要它

用户提议「以微盘为主，叠加其它优化策略」。**叠加有没有用，取决于一个前提**：
其它资产要能在**微盘崩盘的时候不崩**（否则叠了等于没叠，还摊薄收益）。

所以先要把候选资产的历史拉下来，算**相关性**（尤其是崩盘月份的相关性）。

候选（按"能否降 beta / 与微盘是否互补"选）：

| 代号 | 名称 | 为什么选它 |
|---|---|---|
| `div_sh` | 上证红利 | **高股息代表**（用户点名）—— 通常低 beta、防御性强 |
| `sz50` | 上证50 | 大盘价值，与微盘风格对立 |
| `hs300` | 沪深300 | 市场基准 |
| `csi500` | 中证500 | 中盘 |
| `gem` | 创业板指 | 成长风格 |
| `bond` | 上证国债 | **唯一真正的"避险"资产**（股债跷跷板） |

（中证2000 / 国证2000 / 中证1000 已在 `index_extra.parquet` 里，不重复拉。）

用法：
    /home/zhulei/anaconda3/envs/zhulei_py312/bin/python fetch_extra_assets.py
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "data" / "assets_extra.parquet"

WANTED = {
    "div_sh": ("上证红利", "000015"),
    "sz50": ("上证50", "000016"),
    "hs300": ("沪深300", "000300"),
    "csi500": ("中证500", "000905"),
    "gem": ("创业板指", "399006"),
    "bond": ("上证国债", "000012"),
}
START, END = "20160101", "20261013"


def log(msg: str = "") -> None:
    print(msg, flush=True)


import akshare as ak  # noqa: E402

frames = []
for key, (cn, code) in WANTED.items():
    t0 = time.time()
    df = ak.index_zh_a_hist(symbol=code, period="daily", start_date=START, end_date=END)
    df = df.rename(columns={"日期": "date", "收盘": "close"})
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = key
    frames.append(df[["symbol", "date", "close"]])
    log(f"  {cn}({code}): {len(df):,} 行  {df['date'].min().date()} ~ {df['date'].max().date()}  ({time.time() - t0:.0f}s)")

out = pd.concat(frames, ignore_index=True)

log("")
log("=" * 70)
log("自检")
log("=" * 70)
for key, (cn, _) in WANTED.items():
    d = out[out["symbol"] == key]
    r = d["close"].pct_change().dropna()
    log(f"  {cn:<8} 行数 {len(d):>5}  年化 {((1 + r).prod() ** (244 / len(r)) - 1):>7.1%}  "
        f"最大回撤 {((d['close'] / d['close'].cummax() - 1).min()):>7.1%}")
bad = out[out["close"] <= 0]
log(f"\n价格 ≤ 0 的异常行: {len(bad)}   ← 必须是 0")

OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_parquet(OUT, index=False)
log(f"\n✅ 已写出: {OUT}  ({len(out):,} 行)")
