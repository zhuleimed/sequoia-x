#!/usr/bin/env python3
"""重建历史 peTTM（方案 A：PE 彻底摆脱 baostock，PB 保留库内 baostock 历史值）。

2026-08-20 由「20 只对拍」验证（见 scripts/compare_ths_vs_baostock_valuation.py）：
  - 用 finance parquet 的「基本每股收益」(累计YTD) 计算 rolling TTM（标准滚动口径）
    TTM = 当前YTD累计 + (去年全年 − 去年YTD至同相对季)
  - 20 只全符合 (asof 2026-08-19 最大差 9.2% 多数<5%)，亏损股(万科/隆基)也一致
  - 关键: 不得用 report_date_ms，必须用法定披露日 asof（防 look-ahead，已修）

方案 A 决策：
  - 重写 stock_daily.peTTM 为 finance 重建值（全历史，统一口径，无 baostock 依赖）
  - pbMRQ / psTTM / pcfNcfTTM 不动（保留 baostock 历史真值）
  - 仅 pbMRQ=None 或 =0 时用「每股净资产」补 (close/bps)

用法：
  # 全量（无参数）
  py312 python scripts/rebuild_valuation_history.py
  # 只重写 N 只（测试）
  py312 python scripts/rebuild_valuation_history.py --limit 5
  # 只写 peTTM，不写 pb（纯 PE 试点）
  py312 python scripts/rebuild_valuation_history.py --pe-only
断点续跑（铁律二）：每只完成写 progress json，启动时跳过已完成股票，同命令恢复。
"""
import argparse
import datetime
import json
import os
import sqlite3
import time
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
DB = Path(os.environ.get("VALUATION_DB", PROJ / "data/sequoia_v2.db"))
FIN = PROJ / "data/extra_features" / "finance"
PROGRESS = PROJ / "scripts" / "tmp" / "valuation_rebuild_progress.json"

LOGGER_PREFIX = "rebuild_valuation_history"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── 口径函数（2026-08-20 20只验证） ─────────────────────────
def _disclose(date):
    d = pd.to_datetime(date)
    if d.month == 3: return datetime.date(d.year, 4, 30)
    if d.month == 6: return datetime.date(d.year, 8, 31)
    if d.month == 9: return datetime.date(d.year, 10, 31)
    return datetime.date(d.year + 1, 4, 30)  # 12-31 年报，次年 4/30


def _build_records(code):
    """读 finance parquet → {(year,quarter): (披露日, 累计eps)}"""
    fp = FIN / f"{code}.parquet"
    if not fp.exists():
        return None
    df = pd.read_parquet(fp)
    if "基本每股收益" not in df.columns or "报告期" not in df.columns:
        return None
    eps = pd.to_numeric(df["基本每股收益"].astype(str).str.replace("%", ""),
                        errors="coerce")
    rec = {}
    for (_, r), e in zip(df.iterrows(), eps):
        d = pd.to_datetime(r["报告期"])
        q = {3: 1, 6: 2, 9: 3, 12: 4}.get(d.month)
        if q is None:
            continue
        rec[(d.year, q)] = (_disclose(r["报告期"]), e)
    return rec


def _rolling_ttm(rec, on_date):
    """截至 on_date 的 rolling TTM EPS（最近已披露季）。返回 float 或 None。"""
    cand = [(av, y, q) for (y, q), (av, _) in rec.items() if av <= on_date]
    if not cand:
        return None
    av, y, q = max(cand)
    cur = rec[(y, q)][1]
    prev = rec.get((y - 1, q), (None, None))[1]
    lastf = rec.get((y - 1, 4), (None, None))[1]
    if lastf is None:
        lastf = prev
    if prev is None or cur is None or lastf is None:
        return None
    return cur + (lastf - prev)


# ── 主流程 ────────────────────────────────────────────


def rebuild(limit=None, pe_only=False):
    conn = sqlite3.connect(DB)
    # 股票清单：从 stock_list 或 stock_daily 取有 finance parquet 的
    syms = [r[0] for r in conn.execute(
        "SELECT DISTINCT symbol FROM stock_daily ORDER BY symbol").fetchall()]
    if limit:
        syms = syms[:limit]

    # 断点续跑
    done = set()
    if PROGRESS.exists():
        done = set(json.load(open(PROGRESS)).get("done", []))
    todo = [s for s in syms if s not in done]
    log(f"共 {len(syms)} 只，待处理 {len(todo)}，已完成 {len(done)}")

    t0 = time.time()
    n_rebuilt = 0
    for i, code in enumerate(todo):
        rec = _build_records(code)
        if not rec:
            done.add(code)
            continue
        # 拉该股全部 peTTM 数据（按日重建）
        rows = conn.execute(
            "SELECT date, close, peTTM, pbMRQ FROM stock_daily WHERE symbol=? "
            "AND close IS NOT NULL AND close>0 ORDER BY date", (code,)).fetchall()
        updates = 0
        for dstr, close, pe_old, pb_old in rows:
            d = datetime.date.fromisoformat(dstr)
            ttm = _rolling_ttm(rec, d)
            new_pe = None
            if ttm is not None and abs(ttm) > 1e-9:
                new_pe = round(close / ttm, 6)
            if new_pe is None:
                continue
            if pe_only:
                if pe_old is None or abs(new_pe - pe_old) > 1e-9:
                    conn.execute("UPDATE stock_daily SET peTTM=? WHERE symbol=? AND date=?",
                                 (new_pe, code, dstr)); updates += 1
            else:
                # 只在"要变"时更新
                if pe_old is None or abs(new_pe - pe_old) > 1e-9:
                    conn.execute("UPDATE stock_daily SET peTTM=? WHERE symbol=? AND date=?",
                                 (new_pe, code, dstr)); updates += 1
            # PB: 方案 A 保留库值; 仅缺(NULL/0)时尝试补(需要 bps, 本次不补, 留给 Phase3b)
        conn.commit()
        n_rebuilt += updates
        done.add(code)
        # 即时持久化进度（断点续跑）
        PROGRESS.parent.mkdir(exist_ok=True)
        json.dump({"done": sorted(done)}, open(PROGRESS, "w"))
        if (i + 1) % 50 == 0 or i == len(todo) - 1:
            log(f"  进度 {i+1}/{len(todo)} 只, 累计重写 {n_rebuilt} 行, "
                f"耗时 {(time.time()-t0)/60:.1f}min")
    conn.close()
    log(f"完成: 重写 {n_rebuilt} 行 peTTM（{len(done)} 只）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pe-only", action="store_true", help="只重建 peTTM，不动 pb")
    a = ap.parse_args()
    rebuild(limit=a.limit, pe_only=a.pe_only)
