#!/usr/bin/env python3
"""V3 修订二: 合成占比试调分析（5% vs 10% vs 基线, 2026-08-09）

对比 2026-05/06 月: 基线(无合成) / sr5(≈5% 注入) / sr10(≈10% 注入) 的 T2 Rank IC,
判断小占比是否"减稀释保保护"。结果微信推送。

用法: py312 python scripts/analyze_synth_ratio.py [--no-push]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.stats as st

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
DB = str(PROJECT_DIR / "data/sequoia_v2.db")
MONTHS = ["2026-05", "2026-06"]
VARIANTS = [("基线", "prediction_cache_base"),
            ("6%(r25)", "prediction_cache_sr25"),     # ratio 0.25 → 300 样本 ≈ 6%
            ("12%(r5)", "prediction_cache_sr5"),      # ratio 0.5 → 600 样本 ≈ 12%
            ("96%(r2)", "prediction_cache_sr50")]     # ratio 2.0 → 全量 4800 ≈ 96%


def calc_ic(path: Path, month: str) -> tuple[float, float, int] | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    if month not in d:
        return None
    m = d[month]
    syms, pred = m["symbols"], np.array(m["t2"])
    conn = sqlite3.connect(DB)
    ref = conn.execute(
        "SELECT MAX(date) FROM stock_daily WHERE date>=? AND date<?",
        (month + "-01", f"{int(month[:4])}-{int(month[5:7])+1:02d}-01")).fetchone()[0]
    actual = []
    for s in syms:
        rows = conn.execute(
            "SELECT close FROM stock_daily WHERE symbol=? AND date>? ORDER BY date LIMIT 20",
            (s, ref)).fetchall()
        c0 = conn.execute(
            "SELECT close FROM stock_daily WHERE symbol=? AND date=?", (s, ref)).fetchone()
        actual.append(rows[-1][0] / c0[0] - 1 if len(rows) == 20 and c0 else np.nan)
    conn.close()
    a = np.array(actual, dtype=float)
    mask = np.isfinite(a) & np.isfinite(pred)
    if mask.sum() < 100:
        return None
    ic, p = st.spearmanr(pred[mask], a[mask])
    return float(ic), float(p), int(mask.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    rows = {m: {} for m in MONTHS}
    complete = True
    for mth in MONTHS:
        for label, stem in VARIANTS:
            ic = calc_ic(PROJECT_DIR / f"output/backtest_v2/{stem}_{mth}.json", mth)
            if ic is None:
                complete = False
            else:
                rows[mth][label] = ic

    if not complete:
        print(f"[{datetime.now():%H:%M:%S}] ⏳ 部分任务未完成, 稍后再查", flush=True)
        return

    lines = ["📊 V3修订二 合成占比试调 II（T2 Rank IC, 800 只, 6%/12%/96% vs 基线）",
             "月份    基线     6%       12%      96%"]
    for mth in MONTHS:
        r = rows[mth]
        lines.append(f"{mth}  {r['基线'][0]:+.4f}  {r['6%(r25)'][0]:+.4f}  "
                     f"{r['12%(r5)'][0]:+.4f}  {r['96%(r2)'][0]:+.4f}")
    # 均值
    mean_b = np.mean([rows[m]['基线'][0] for m in MONTHS])
    mean_6 = np.mean([rows[m]['6%(r25)'][0] for m in MONTHS])
    mean_12 = np.mean([rows[m]['12%(r5)'][0] for m in MONTHS])
    mean_96 = np.mean([rows[m]['96%(r2)'][0] for m in MONTHS])
    lines.append(f"均值   {mean_b:+.4f}  {mean_6:+.4f}  {mean_12:+.4f}  {mean_96:+.4f}")

    # 判定: 96% vs 基线/6%/12%（24% 全量 3 个月均值 +0.0081 为参照）
    d96 = mean_96 - mean_b
    if d96 > 0:
        verdict = "✅ 96% 有效（保护强度随占比单调 → 占比越大越好）"
    elif d96 < 0:
        verdict = "❌ 96% 无效（占比过大有害 → 24% 为最优）"
    else:
        verdict = "⚠️ 无差异"
    lines.append(f"\nΔ96%={d96:+.4f} → {verdict}")
    lines.append("下一步: 若单调 → 走合成完整序列（无限扩量）; 若 24% 最优 → 定格保险配置")

    report = "\n".join(lines)
    print(report, flush=True)
    if not args.no_push:
        try:
            from wxpusher import WxPusher
            from sequoia_x.core.config import get_settings
            s = get_settings()
            WxPusher.send_message(content=report, token=s.wxpusher_token,
                                  topic_ids=s.wxpusher_topic_ids, content_type=1)
            print(f"[{datetime.now():%H:%M:%S}] ✅ 已微信推送", flush=True)
        except Exception as e:
            print(f"[{datetime.now():%H:%M:%S}] 推送失败: {e}", flush=True)
    (PROJECT_DIR / "experiments/kronos/output/synth_ratio_report.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
