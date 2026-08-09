#!/usr/bin/env python3
"""V3 修订二: 3 个月合成增强对比分析 + 微信推送（2026-08-09）

分析 2026-04/05/06 基线 vs 增强（+1200 Kronos 合成标签）的 T2 Rank IC,
验证"向中性收缩"模式（失效月保护/有效月稀释）是否成立, 结果微信推送。

用法: py312 python scripts/analyze_synth_exp.py [--no-push]
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


def calc_ic(path: Path, month: str) -> tuple[float, float, int] | None:
    """T2 预测 vs 实际 20 日收益的 Rank IC。"""
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


def analyze() -> dict:
    rows = []
    for mth in MONTHS:
        base = calc_ic(PROJECT_DIR / f"output/backtest_v2/prediction_cache_base_{mth}.json", mth)
        syn = calc_ic(PROJECT_DIR / f"output/backtest_v2/prediction_cache_synth_{mth}.json", mth)
        if base is None or syn is None:
            return {"complete": False, "rows": rows, "missing": mth}
        rows.append({"month": mth, "base_ic": base[0], "base_p": base[1],
                     "synth_ic": syn[0], "synth_p": syn[1], "n": base[2]})
    return {"complete": True, "rows": rows}


def fmt_report(res: dict) -> str:
    lines = ["📊 V3修订二 合成增强 3 个月对比（T2 Rank IC, 800 只）",
             "月份    基线     增强     ΔIC    基线p",
             "─" * 42]
    for r in res["rows"]:
        d = r["synth_ic"] - r["base_ic"]
        lines.append(f"{r['month']}  {r['base_ic']:+.4f}  {r['synth_ic']:+.4f}  "
                     f"{d:+.4f}  {r['base_p']:.2f}")
    ics_b = [r["base_ic"] for r in res["rows"]]
    ics_s = [r["synth_ic"] for r in res["rows"]]
    mean_b, mean_s = np.mean(ics_b), np.mean(ics_s)
    d_mean = mean_s - mean_b
    lines.append(f"均值   {mean_b:+.4f}  {mean_s:+.4f}  {d_mean:+.4f}")
    # 模式判断: 基线正月增强降 / 基线下月增强升 → 向中性收缩
    shrink = all((r["base_ic"] > 0 and r["synth_ic"] < r["base_ic"]) or
                 (r["base_ic"] < 0 and r["synth_ic"] > r["base_ic"])
                 for r in res["rows"])
    if shrink:
        lines.append("\n→ 模式确认: 向中性收缩（失效月保护 / 有效月稀释）= 保险非增强")
        lines.append("→ 下一步建议: ①合成占比 24%→5-10% 试调 ②合成完整序列真·数据增强")
    else:
        lines.append("\n→ 模式未完全确认, 需结合各月方向细看")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()
    res = analyze()
    if not res["complete"]:
        print(f"[{datetime.now():%H:%M:%S}] ⏳ {res.get('missing')} 未完成, 稍后再查",
              flush=True)
        return
    report = fmt_report(res)
    print(report, flush=True)
    if not args.no_push:
        try:
            from wxpusher import WxPusher
            from sequoia_x.core.config import get_settings
            s = get_settings()
            WxPusher.send_message(content=f"{report}", token=s.wxpusher_token,
                                  topic_ids=s.wxpusher_topic_ids, content_type=1)
            print(f"[{datetime.now():%H:%M:%S}] ✅ 已微信推送", flush=True)
        except Exception as e:
            print(f"[{datetime.now():%H:%M:%S}] 推送失败: {e}", flush=True)
    # 写结果文件供后续引用
    (PROJECT_DIR / "experiments/kronos/output/synth_exp_3m_report.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
