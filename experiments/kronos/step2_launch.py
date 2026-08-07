#!/usr/bin/env python3
"""3a 单月全量启动器: 36 独立进程 × OMP=1 并行推理（2026-08-07）

背景（实测结论）:
  - multiprocessing Pool(fork) + torch 并发推理死锁（多 worker 同时推理时 0 产出）
  - 独立 python 进程（非 fork）单线程稳定: sample=10 ~40s/股（36 线程满配时）
  - 36 进程 × 1 线程 = 36 核满配 → 单月 2900 只 ≈ 50-60min

用法:
  py312 python step2_launch.py --month 2026-06 [--procs 36] [--samples 10] [--limit N]

输出: experiments/kronos/output/month_<YM>_<i>.jsonl 分片 + 完成汇总
断点续跑: 分片文件已有 code 跳过; 重跑同命令只补缺失。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
OUT_DIR = PROJECT_ROOT / "experiments/kronos/output"
LOG_DIR = PROJECT_ROOT / "logs"
POOLS_DIR = PROJECT_ROOT / "output/backtest_v2/kronos_pools"
PYTHON = "/home/zhulei/anaconda3/envs/zhulei_py312/bin/python"


def load_pool(month: str, limit: int = 0) -> list[str]:
    """按月动态池（2026-08-08: 回测必须用该月时点池, 避免幸存者偏差）。
    优先 kronos_pools/{month}.json（build_pools.py 生成）; 缺失回退当前快照。"""
    pool_f = POOLS_DIR / f"{month}.json"
    if pool_f.exists():
        symbols = json.loads(pool_f.read_text())
        print(f"股票池（按月动态）: {len(symbols)} 只 ({pool_f.name})")
    else:
        symbols = json.loads((PROJECT_ROOT / "output/backtest_v2/.stock_pool.json").read_text())
        print(f"股票池（当前快照）: {len(symbols)} 只")
    return symbols[:limit] if limit > 0 else symbols


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default="2026-06")
    ap.add_argument("--procs", type=int, default=36)
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 只（调试）")
    args = ap.parse_args()

    sys.path.insert(0, str(PROJECT_ROOT / "experiments/kronos"))
    import step2_monthly as S
    ref_date = S.get_month_last_trade_date(args.month)
    symbols = load_pool(args.month, args.limit)
    print(f"单月全量: month={args.month} ref={ref_date} pool={len(symbols)} 只 "
          f"procs={args.procs} samples={args.samples}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ── 分片（按已有分片断点续跑: 完成的分片跳过）──
    n_procs = min(args.procs, len(symbols))
    chunk = (len(symbols) + n_procs - 1) // n_procs
    procs = []
    t0 = time.time()
    for i in range(n_procs):
        batch = symbols[i * chunk:(i + 1) * chunk]
        if not batch:
            continue
        out_f = OUT_DIR / f"month_{args.month}_{i}.jsonl"
        # 分片级断点: 分片内全部完成才跳过（批内断点由 step2_batch 处理）
        log_f = LOG_DIR / f"kronos_batch_{args.month}_{i}.log"
        env = dict(os.environ)
        env.pop("KMP_AFFINITY", None)
        env["OMP_NUM_THREADS"] = "1"
        cmd = [PYTHON, "-u", str(PROJECT_ROOT / "experiments/kronos/step2_batch.py"),
               "--codes", ",".join(batch), "--ref", ref_date,
               "--out", str(out_f), "--samples", str(args.samples)]
        with open(log_f, "w") as lf:
            p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        procs.append((i, p, out_f))
        print(f"  启动分片 {i}: {len(batch)} 只 (PID {p.pid})", flush=True)

    # ── 监控进度（铁律一: 跑到哪/速率/ETA）──
    done_counts = [0] * n_procs
    while True:
        time.sleep(60)
        alive = 0
        total = 0
        for i, p, out_f in procs:
            if p.poll() is None:
                alive += 1
            if out_f.exists():
                n = sum(1 for _ in open(out_f, encoding="utf-8"))
                done_counts[i] = n
                total += n
            else:
                done_counts[i] = 0
        el = time.time() - t0
        rate = total / el if el > 0 else 0
        eta = (len(symbols) - total) / rate / 60 if rate > 0 else 0
        print(f"  [{datetime.now():%H:%M:%S}] 完成 {total}/{len(symbols)} "
              f"存活进程={alive} 速率={rate:.2f}股/s ETA={eta:.0f}min", flush=True)
        if alive == 0:
            break

    # ── 汇总: 合并分片 → month_<YM>.jsonl ──
    merged = OUT_DIR / f"month_{args.month}.jsonl"
    seen = set()
    with open(merged, "w", encoding="utf-8") as mf:
        for i, p, out_f in procs:
            if out_f.exists():
                for line in open(out_f, encoding="utf-8"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if r["code"] not in seen:
                        seen.add(r["code"])
                        mf.write(json.dumps(r) + "\n")
    el = time.time() - t0
    print(f"\n✅ 单月完成: {len(seen)}/{len(symbols)} 只, 总耗时 {el/60:.1f}min, "
          f"合并输出 {merged}")
    if len(seen) < len(symbols):
        print(f"⚠️ 缺失 {len(symbols) - len(seen)} 只（分片日志见 logs/kronos_batch_{args.month}_*.log）")


if __name__ == "__main__":
    main()
