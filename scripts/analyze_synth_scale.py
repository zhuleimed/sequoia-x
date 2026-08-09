#!/usr/bin/env python3
"""V3 修订二: 合成规模效应分析（24 vs 96 只 vs 基线, 2026-08-10）

对比 2026-04/05/06 三个月: 基线 / 24 只序列 / 96 只序列的 T2 Rank IC,
验证规模效应 + 3 个月一致性。结果微信推送。

用法: py312 python scripts/analyze_synth_scale.py [--no-push]
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
MONTHS = ["2026-04", "2026-05", "2026-06"]
VARIANTS = [("基线", "prediction_cache_base"), ("24只", "prediction_cache_series"),
            ("96只", "prediction_cache_series96")]


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
                rows[mth][label] = ic[0]

    if not complete:
        print(f"[{datetime.now():%H:%M:%S}] ⏳ 部分任务未完成, 稍后再查", flush=True)
        return

    lines = ["📊 V3修订二 合成规模效应（T2 Rank IC, 800 只, 3 个月）",
             "月份    基线     24只     96只"]
    for mth in MONTHS:
        r = rows[mth]
        lines.append(f"{mth}  {r['基线']:+.4f}  {r['24只']:+.4f}  {r['96只']:+.4f}")
    mb = np.mean([rows[m]['基线'] for m in MONTHS])
    m24 = np.mean([rows[m]['24只'] for m in MONTHS])
    m96 = np.mean([rows[m]['96只'] for m in MONTHS])
    lines.append(f"均值   {mb:+.4f}  {m24:+.4f}  {m96:+.4f}")

    d24, d96 = m24 - mb, m96 - mb
    if d96 > d24 > 0:
        verdict = "🚀 规模效应确认: 96 只 > 24 只 > 基线（样本量越大增强越强）"
        nxt = "下一步: 纳入 V3 正式训练管线（88 维模式 + 自动降级告警）"
    elif d96 > 0 and d96 <= d24:
        verdict = "⚠️ 96 只无额外增益（24 只已达饱和）"
        nxt = "下一步: 定格 24 只配置, 纳入正式管线"
    elif d96 <= 0:
        verdict = "❌ 96 只反而有害（过拟合合成数据）"
        nxt = "下一步: 回退 24 只, 检查生成质量"
    else:
        verdict = "⚠️ 混合信号, 需细看各月"
        nxt = "下一步: 分月诊断"
    lines.append(f"\nΔ24只={d24:+.4f} Δ96只={d96:+.4f} → {verdict}")
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
    (PROJECT_DIR / "experiments/kronos/output/synth_scale_report.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
