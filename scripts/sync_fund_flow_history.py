"""新浪资金流全历史同步（2019-09 起，全市场）——2026-08-11 本项目生产版。

背景:
- 东财 push2his 个股资金流仅返回近 ~120 天（实测），无法补历史 → 121 维扩展特征
  采样日被锁死在 120 天窗口内。
- 新浪 MoneyFlow.ssl_qsfx_lscjfb 历史可翻到 2010 年（实测 2019-09 至今 ~1700 交易日）,
  2026-08-10 021 项目已验证全市场可同步。

输出（本项目格式，与东财版列兼容，_fund_flow_features 只读 3 列）:
  data/extra_features/fund_flow/{symbol}.parquet
  列: 日期 | 主力净流入-净额(元) | 主力净流入-净占比(%) | 超大单净流入-净占比(%)
换算（新浪原始 14 字段 → 本项目列）:
  主力净占比(%) = ratioamount × 100            （ratioamount 为小数）
  超大单净占比(%) = ratioamount × r0_net/netamount × 100
    （东财语义: 超大单净占比 = 超大单净额/成交额 = 主力净占比 × 超大单净额/主力净额,
      新浪无成交额字段, 用 r0_net/netamount 还原; netamount==0 → 0）

断点续跑: parquet 已存在即跳过该股票 → 崩溃后重跑同一命令即可。
用法: py312 python scripts/sync_fund_flow_history.py [--test N] [--workers W] [--start-date 2019-09-01]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv()

from sequoia_x.core.config import get_settings
from sequoia_x.core.logger import get_logger

logger = get_logger("sync_fund_flow_history")

SINA_URL = ("https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
            "MoneyFlow.ssl_qsfx_lscjfb?page={page}&num=1000&sort=opendate&asc=0&daima={daima}")
FIELDS = ["opendate", "trade", "changeratio", "turnover", "netamount", "ratioamount",
          "r0", "r1", "r2", "r3", "r0_net", "r1_net", "r2_net", "r3_net"]

# 目标输出目录（本项目）
OUT_DIR = ROOT / "data" / "extra_features" / "fund_flow"


def to_daima(symbol: str) -> str | None:
    """6 位代码 → 新浪 daima（sz/sh 前缀）；北交所/老三板无数据返回 None。"""
    if symbol.startswith(("60", "68")):
        return f"sh{symbol}"
    if symbol.startswith(("00", "30")):
        return f"sz{symbol}"
    return None


def fetch_page(daima: str, page: int) -> list[dict]:
    """请求一页（num=1000，降序：最新在前）。失败抛异常由调用方重试。"""
    req = urllib.request.Request(SINA_URL.format(page=page, daima=daima),
                                 headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        txt = r.read().decode("utf-8", errors="replace")
    if not txt.strip() or txt.strip() == "null":
        return []
    return json.loads(txt)


def fetch_sina_fund_flow(symbol: str, start_date: str = "2019-09-01") -> pd.DataFrame | None:
    """拉取单只股票新浪资金流全历史并转换为本项目格式（不落盘）。

    Returns:
        本项目格式 DataFrame（日期/收盘价/涨跌幅/五档净额/五档占比）或 None。
    """
    daima = to_daima(symbol)
    if daima is None:
        return None

    rows_all: list[dict] = []
    for page in range(1, 5):
        try:
            rows = fetch_page(daima, page)
        except Exception as e:
            logger.warning(f"{symbol} 第{page}页失败: {e}")
            time.sleep(1.0)
            try:
                rows = fetch_page(daima, page)
            except Exception as e2:
                logger.error(f"{symbol} 第{page}页重试仍失败: {e2}")
                break
        if not rows:
            break
        rows_all.extend(rows)
        oldest = rows[-1]["opendate"]
        if str(oldest) < start_date:
            break
        time.sleep(0.2)  # 翻页限速

    if not rows_all:
        return None

    df = pd.DataFrame(rows_all)
    for c in FIELDS[1:]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ── 本项目格式转换 ──
    # 关键: 新浪 netamount = 四档净额之和(全市场口径), 而本项目/东财"主力" = 超大单+大单。
    # 还原东财口径: 主力净额 = r0_net + r1_net; 各档占比 = 全口径占比 × 档净额/全净额
    # （恒等式: 档占比 = 档净额/成交额 = 全净额/成交额 × 档净额/全净额; 全净额≈0 → 0）
    netamount = df["netamount"].astype(float)
    ratio = df["ratioamount"].astype(float) * 100.0  # 全口径占比 小数 → %
    r0_net = df["r0_net"].astype(float)
    r1_net = df["r1_net"].astype(float)
    r2_net = df["r2_net"].astype(float)
    r3_net = df["r3_net"].astype(float)
    main_net = r0_net + r1_net  # 主力 = 超大单+大单（东财口径等价）

    def pct_of(net_col: pd.Series) -> np.ndarray:
        return np.where(np.abs(netamount) > 1e-8, ratio * net_col / netamount, 0.0)

    out = pd.DataFrame({
        "日期": pd.to_datetime(df["opendate"]),
        "收盘价": df["trade"].astype(float),
        "涨跌幅": df["changeratio"].astype(float),
        "主力净流入-净额": main_net,
        "小单净流入-净额": r3_net,
        "中单净流入-净额": r2_net,
        "大单净流入-净额": r1_net,
        "超大单净流入-净额": r0_net,
        "主力净流入-净占比": pct_of(main_net),
        "小单净流入-净占比": pct_of(r3_net),
        "中单净流入-净占比": pct_of(r2_net),
        "大单净流入-净占比": pct_of(r1_net),
        "超大单净流入-净占比": pct_of(r0_net),
    })
    out = out.sort_values("日期").drop_duplicates(subset="日期", keep="last")
    return out


def sync_one(symbol: str, start_date: str) -> tuple[int, str, str]:
    """同步单只股票（fetch_sina_fund_flow + 落盘）。返回 (行数, 最早日期, 最新日期)。"""
    out = fetch_sina_fund_flow(symbol, start_date)
    if out is None or len(out) == 0:
        return 0, "-", "-"
    out.to_parquet(OUT_DIR / f"{symbol}.parquet", index=False)
    return len(out), str(out["日期"].min())[:10], str(out["日期"].max())[:10]


def _worker(args: tuple[str, str]) -> tuple[str, int, str, str]:
    """多进程 worker（必须模块级才能 pickle）。"""
    symbol, start_date = args
    if (OUT_DIR / f"{symbol}.parquet").exists():
        return symbol, 0, "-", "-"  # 断点续跑：跳过已有
    n, d0, d1 = sync_one(symbol, start_date)
    return symbol, n, d0, d1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=int, default=0, help="只同步前 N 只（单进程测速）")
    ap.add_argument("--workers", type=int, default=16, help="并行进程数")
    ap.add_argument("--start-date", default="2019-09-01", help="同步起点（留 20 日动量预热）")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    settings = get_settings()

    # 股票清单（sequoia_v2.db 只读）
    conn = sqlite3.connect(f"file:{settings.db_path}?mode=ro", uri=True)
    symbols = sorted(r[0] for r in conn.execute("SELECT DISTINCT symbol FROM stock_daily"))
    conn.close()
    logger.info(f"股票清单: {len(symbols)} 只 | 输出: {OUT_DIR} | workers={args.workers}")

    todo = [s for s in symbols if not (OUT_DIR / f"{s}.parquet").exists()]
    logger.info(f"待同步: {len(todo)} 只（已有 {len(symbols) - len(todo)} 只跳过）")

    if args.test > 0:
        for s in todo[: args.test]:
            n, d0, d1 = sync_one(s, args.start_date)
            logger.info(f"[test] {s}: {n} 行 ({d0} ~ {d1})")
        return

    t0 = time.time()
    from multiprocessing import Pool

    ok = fail = 0
    with Pool(processes=args.workers) as pool:
        for i, (symbol, n, d0, d1) in enumerate(
            pool.imap_unordered(_worker, [(s, args.start_date) for s in todo], chunksize=4)
        ):
            ok += 1 if n > 0 else 0
            fail += 1 if n == 0 else 0
            if (i + 1) % 200 == 0 or i == len(todo) - 1:
                el = time.time() - t0
                eta = el / (i + 1) * (len(todo) - i - 1) if i + 1 > 0 else 0
                logger.info(
                    f"进度: {i + 1}/{len(todo)} (ok={ok} fail={fail}) "
                    f"耗时{el / 60:.1f}min 速率{el / (i + 1):.1f}s/只 "
                    f"ETA {eta / 60:.0f}min"
                )
    logger.info(f"同步完成: ok={ok} fail={fail} 总耗时 {(time.time() - t0) / 60:.1f}min")


if __name__ == "__main__":
    main()
