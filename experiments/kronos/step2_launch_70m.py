#!/usr/bin/env python3
"""3a 70 个月全量启动器（2026-08-08, nohup 运行）

逐月调用 step2_launch.py（36 独立进程并行推理）→ 每月完成自动 step3 IC 分析 →
追加 output/70m_monthly_ic.csv。全部完成后汇总由 monitor_70m.py 触发 step5_summary.py。

断点续跑（铁律二）: month_<YM>.jsonl 行数 ≥ 池子数×0.99 → 该月视为完成, 跳过。
2026-06 已完成 → 自动跳过。崩溃后重跑同命令即可。

分批策略（用户决策 2026-08-08）: --max-months N 只跑前 N 个待跑月份
（6 个月先行验证 → 判定后再决定是否续跑全量; 续跑 = 重跑同命令不带参数,
已完成的月份自动跳过）。

用法（nohup, 铁律二）:
  env -u KMP_AFFINITY nohup /home/zhulei/anaconda3/envs/zhulei_py312/bin/python -u \
      experiments/kronos/step2_launch_70m.py [--max-months N] > logs/kronos_launch_70m.log 2>&1 &
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJECT_ROOT / "experiments/kronos/output"
POOLS_DIR = PROJECT_ROOT / "output/backtest_v2/kronos_pools"
IC_CSV = OUT_DIR / "70m_monthly_ic.csv"
PYTHON = "/home/zhulei/anaconda3/envs/zhulei_py312/bin/python"
DONE_FRAC = 0.99  # 行数 ≥ 池子数×0.99 视为该月完整（推理偶发失败留 1% 余量）


def month_list() -> list[str]:
    """按月动态池文件名排序（2020-09 → 2026-06, 70 个月）。"""
    return sorted(p.stem for p in POOLS_DIR.glob("*.json"))


def pool_size(month: str) -> int:
    return len(json.loads((POOLS_DIR / f"{month}.json").read_text()))


def month_done(month: str) -> bool:
    """断点续跑: merged 文件行数达到池子数×0.99 → 完成。"""
    f = OUT_DIR / f"month_{month}.jsonl"
    if not f.exists():
        return False
    n = sum(1 for _ in open(f, encoding="utf-8") if _.strip())
    return n >= pool_size(month) * DONE_FRAC


def parse_ic_from_step3(text: str) -> dict:
    """解析 step3_analyze 输出 → {ic, p, top10, bot10, spread, samples}"""
    r: dict = {}
    for line in text.splitlines():
        if "有效样本" in line:
            r["samples"] = int(line.split(":")[1].split("只")[0].strip())
        elif "Rank IC" in line:
            parts = line.split("=")[1].split("(")[0].strip()
            r["ic"] = float(parts)
            r["p"] = line.split("p=")[1].rstrip(")")
        elif "TOP10%" in line:
            r["top10"] = line.split(":")[1].strip()
        elif "BOT10%" in line:
            r["bot10"] = line.split(":")[1].strip()
    return r


def append_ic_row(month: str, r: dict) -> None:
    """每月 IC 追加 CSV（增量保存, 铁律二）。"""
    new = not IC_CSV.exists()
    with open(IC_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["month", "ic", "p", "samples",
                                          "top10", "bot10", "spread", "ts"])
        if new:
            w.writeheader()
        w.writerow({"month": month, "ts": datetime.now().strftime("%m-%d %H:%M"),
                    **r})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-months", type=int, default=0,
                    help="只跑前 N 个待跑月份（0=全部; 6 个月先行验证用）")
    args = ap.parse_args()

    months = month_list()
    print(f"[{datetime.now():%H:%M:%S}] 70 个月全量启动: 共 {len(months)} 个月, "
          f"36 进程/月 × OMP=1, 预计 ~1.6h/月 ≈ 4.7 天", flush=True)

    # 启动自检（铁律五）: 池子/缓存/已完成
    todo = [m for m in months if not month_done(m)]
    done = [m for m in months if month_done(m)]
    if args.max_months > 0:
        todo = todo[:args.max_months]
        print(f"[分批] --max-months {args.max_months}: 本批只跑 {len(todo)} 个月 "
              f"(剩余自动跳过, 续跑时补)", flush=True)
    print(f"[自检] 已完成 {len(done)} 个月（自动跳过）: {done[:3]}... "
          f"本批待跑 {len(todo)} 个月: {todo[:3]}...", flush=True)
    if not todo:
        print("✅ 全部完成，无待跑月份", flush=True)
        return

    t0 = time.time()
    for i, month in enumerate(todo):
        if month_done(month):
            print(f"[{datetime.now():%H:%M:%S}] [{i+1}/{len(todo)}] {month} 已完成, 跳过",
                  flush=True)
            continue
        print(f"[{datetime.now():%H:%M:%S}] [{i+1}/{len(todo)}] ⏳ 启动 {month} "
              f"(池子 {pool_size(month)} 只, 已过 {time.time()-t0:.0f}s)",
              flush=True)
        m0 = time.time()
        r = subprocess.run(
            [PYTHON, "-u", str(PROJECT_ROOT / "experiments/kronos/step2_launch.py"),
             "--month", month],
            env={**os.environ, "OMP_NUM_THREADS": "1"},
            capture_output=True, text=True, timeout=12 * 3600)
        out = r.stdout + r.stderr
        # 尾部日志（失败排查用）
        print(out[-1500:], flush=True)
        if r.returncode != 0:
            print(f"⚠️ {month} 启动器退出码 {r.returncode}, 继续下一月（日志见 "
                  f"logs/kronos_batch_{month}_*.log）", flush=True)
        # 完成校验（铁律三: 验证产出有效性）
        n = sum(1 for _ in open(OUT_DIR / f"month_{month}.jsonl",
                                encoding="utf-8") if _.strip()) if month_done(month) else 0
        if month_done(month):
            print(f"[{datetime.now():%H:%M:%S}] ✅ {month} 完成: {n} 只, "
                  f"耗时 {(time.time()-m0)/60:.0f}min, 运行 {(time.time()-t0)/3600:.1f}h, "
                  f"ETA 剩余 {(time.time()-m0) * (len(todo)-i-1)/3600:.1f}h",
                  flush=True)
            # 自动逐月 IC 分析（快, ~1min）→ 追加 CSV
            r2 = subprocess.run(
                [PYTHON, "-u", str(PROJECT_ROOT / "experiments/kronos/step3_analyze.py"),
                 "--month", month],
                capture_output=True, text=True, timeout=600)
            txt = r2.stdout + r2.stderr
            ic = parse_ic_from_step3(txt)
            if "ic" in ic:
                ic["spread"] = (ic.get("top10", "").replace("+", "").replace("%", "") or "?")
                append_ic_row(month, ic)
                print(f"  📊 {month} Rank IC = {ic['ic']:+.4f} (样本 {ic.get('samples','?')})",
                      flush=True)
            else:
                print(f"  ⚠️ {month} IC 解析失败:\n{txt[-500:]}", flush=True)
        else:
            print(f"⚠️ {month} 完整性校验未过（{n}/{pool_size(month)} 只）, 记录待重跑",
                  flush=True)

    print(f"\n[{datetime.now():%H:%M:%S}] 🎉 本批主循环结束（{len(todo)} 个月）, 总耗时 "
          f"{(time.time()-t0)/3600:.1f}h（剩余月份重跑同命令 --max-months 0 即续跑）",
          flush=True)


if __name__ == "__main__":
    main()
