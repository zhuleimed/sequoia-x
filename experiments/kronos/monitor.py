#!/usr/bin/env python3
"""3a 单月推理独立监测器（2026-08-07, nohup 运行, 用户退出会话后继续）

职责:
  1. 等待 step2_launch 完成（轮询分片文件, 每 2min）
  2. 完成后自动跑 step3_analyze（单月 Rank IC）
  3. wxpusher 微信推送结果（完成 + IC 初值）
  4. 超时（6h）→ 告警不退出, 等完成→分析→推送

用法: nohup py312 python -u experiments/kronos/monitor.py --month 2026-06 \
        > logs/kronos_monitor.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
OUT_DIR = PROJECT_ROOT / "experiments/kronos/output"
POOLS_DIR = PROJECT_ROOT / "output/backtest_v2/kronos_pools"
PYTHON = "/home/zhulei/anaconda3/envs/zhulei_py312/bin/python"
TIMEOUT_H = 6
POOL_SIZE = 0  # 动态: 由 month 池子文件决定（2026-08-08 修复, 原硬编码 2978）


def notify(title: str, body: str) -> None:
    try:
        from wxpusher import WxPusher
        from sequoia_x.core.config import get_settings
        s = get_settings()
        WxPusher.send_message(content=f"{title}\n{body}", token=s.wxpusher_token,
                              topic_ids=s.wxpusher_topic_ids, content_type=1)
        print(f"[{datetime.now():%H:%M:%S}] 已推送: {title}", flush=True)
    except Exception as e:
        print(f"[{datetime.now():%H:%M:%S}] 推送失败: {e}", flush=True)


def count_done(month: str) -> int:
    total = 0
    for f in OUT_DIR.glob(f"month_{month}_*.jsonl"):
        try:
            total += sum(1 for _ in open(f, encoding="utf-8") if _.strip())
        except Exception:
            pass
    return total


def main() -> None:
    global POOL_SIZE
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default="2026-06")
    args = ap.parse_args()
    month = args.month
    merged = OUT_DIR / f"month_{month}.jsonl"
    pool_f = POOLS_DIR / f"{month}.json"
    POOL_SIZE = len(json.loads(pool_f.read_text())) if pool_f.exists() else 0
    print(f"  动态池: {POOL_SIZE} 只", flush=True)

    print(f"[{datetime.now():%H:%M:%S}] 监测器启动: month={month} 目标={POOL_SIZE} 只 "
          f"超时={TIMEOUT_H}h", flush=True)
    t0 = time.time()
    last_count = -1
    _timeout_notified = False
    stall_since = None

    while True:
        time.sleep(120)
        n = count_done(month)
        elapsed = time.time() - t0

        # 完成判定: 合并文件存在且达到目标数
        if merged.exists():
            try:
                n_merged = sum(1 for _ in open(merged, encoding="utf-8") if _.strip())
            except Exception:
                n_merged = 0
            if n_merged >= POOL_SIZE:
                break

        # 停滞检测: 3 个轮询周期（6min）无进展 → 告警（进程可能挂了）
        if n == last_count:
            if stall_since is None:
                stall_since = time.time()
            elif time.time() - stall_since > 360:
                notify("⚠️ Kronos 单月推理停滞", f"{elapsed/60:.0f}min 完成 {n}/{POOL_SIZE}, "
                       f"6min 无进展, 请检查 logs/kronos_launch_*.log")
                stall_since = time.time()  # 避免重复推送
        else:
            stall_since = None
        last_count = n

        # 超时判定: 告警一次但不退出（用户不在时仍要等完成→分析→推送）
        if elapsed > TIMEOUT_H * 3600 and not _timeout_notified:
            _timeout_notified = True
            notify("⚠️ Kronos 单月推理超时", f"{TIMEOUT_H}h 完成 {n}/{POOL_SIZE} 只, "
                   f"仍在继续等待完成（不退出）")

        # 周期进度日志
        if int(elapsed / 120) % 5 == 0:
            print(f"[{datetime.now():%H:%M:%S}] {n}/{POOL_SIZE} "
                  f"({elapsed/60:.0f}min)", flush=True)

    # ── 完成 → 自动分析 + 推送 ──
    print(f"[{datetime.now():%H:%M:%S}] ✅ 推理完成, 开始单月 IC 分析...", flush=True)
    r = subprocess.run(
        [PYTHON, "-u", str(PROJECT_ROOT / "experiments/kronos/step3_analyze.py"),
         "--month", month],
        capture_output=True, text=True, timeout=600)
    out = r.stdout + r.stderr
    print(out, flush=True)

    ic_line = next((l for l in out.splitlines() if "Rank IC" in l), "Rank IC 未解析")
    top_line = next((l for l in out.splitlines() if "TOP10%" in l), "")
    notify("✅ Kronos 零样本单月完成（2026-06）",
           f"完成 {POOL_SIZE} 只\n{ic_line}\n{top_line}\n详见 logs/kronos_launch_2026-06.log")
    print(f"[{datetime.now():%H:%M:%S}] 监测器结束", flush=True)


if __name__ == "__main__":
    main()
