#!/usr/bin/env python3
"""V3 修订二: 合成完整序列实验分析 + 微信推送（2026-08-09）

对比 2026-05/06 月: 基线 / 完整序列增强（真·数据增强）的 T2 Rank IC,
并参照标签替换 24%/96% 结果。结果微信推送。

用法: py312 python scripts/analyze_synth_series.py [--no-push]
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

    rows = {}
    complete = True
    for mth in MONTHS:
        base = calc_ic(PROJECT_DIR / f"output/backtest_v2/prediction_cache_base_{mth}.json", mth)
        ser = calc_ic(PROJECT_DIR / f"output/backtest_v2/prediction_cache_series_{mth}.json", mth)
        if base is None or ser is None:
            complete = False
        else:
            rows[mth] = {"base": base[0], "series": ser[0]}

    if not complete:
        print(f"[{datetime.now():%H:%M:%S}] ⏳ 部分任务未完成, 稍后再查", flush=True)
        return

    lines = ["📊 V3修订二 合成完整序列（真·数据增强, T2 Rank IC, 800 只）",
             "月份    基线     完整序列   ΔIC"]
    for mth in MONTHS:
        r = rows[mth]
        d = r["series"] - r["base"]
        lines.append(f"{mth}  {r['base']:+.4f}  {r['series']:+.4f}  {d:+.4f}")
    mb = np.mean([rows[m]["base"] for m in MONTHS])
    ms = np.mean([rows[m]["series"] for m in MONTHS])
    lines.append(f"均值   {mb:+.4f}  {ms:+.4f}  {ms-mb:+.4f}")

    d = ms - mb
    if d > 0.02:
        verdict = "🚀 完整序列增强显著有效（真·数据增强成立, 样本量扩充 → IC 提升）"
        nxt = "下一步: 扩大合成规模（48/96 只）→ 纳入 V3 正式训练管线"
    elif d > 0:
        verdict = "✅ 完整序列有效（轻微提升）"
        nxt = "下一步: 扩大合成规模验证"
    else:
        verdict = "❌ 完整序列无效（特征/标签自洽但无增益, 机制需重审）"
        nxt = "下一步: 检查合成序列质量/特征分布, 或定格标签替换 96% 为保险配置"
    lines.append(f"\nΔ均值={d:+.4f} → {verdict}")
    lines.append(nxt)

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
    (PROJECT_DIR / "experiments/kronos/output/synth_series_report.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
