#!/usr/bin/env python3
"""3a 70 个月全量独立监测器（2026-08-08, nohup 运行, 用户退出会话后继续）

职责:
  1. 轮询（每 3min）: 已完成月数 / 目标 + 当前月分片进度
  2. 停滞检测（2h 无任何进展）→ 微信告警（重复推送间隔 ≥2h）
  3. 超时（7 天）→ 微信告警一次（不退出）
  4. 目标月全部完成 → 自动跑 step5_summary.py（逐月 IC + corr 汇总）→ 微信推送判定

分批策略（用户决策 2026-08-08）: --max-months N 与主循环同参数——只等前 N 个月
完成即退出（6 个月先行验证; 全量续跑时 monitor 不带参数等全部）。

用法: env -u KMP_AFFINITY nohup /home/zhulei/anaconda3/envs/zhulei_py312/bin/python -u \
          experiments/kronos/monitor_70m.py [--max-months N] > logs/kronos_monitor_70m.log 2>&1 &
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
sys.path.insert(0, str(PROJECT_ROOT))  # 供 wxpusher/config 导入
OUT_DIR = PROJECT_ROOT / "experiments/kronos/output"
POOLS_DIR = PROJECT_ROOT / "output/backtest_v2/kronos_pools"
PYTHON = "/home/zhulei/anaconda3/envs/zhulei_py312/bin/python"
STALL_H = 2          # 停滞告警阈值
TIMEOUT_H = 7 * 24   # 超时告警阈值（70×1.6h≈4.7 天, 留余量）
DONE_FRAC = 0.99


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


def months() -> list[str]:
    return sorted(p.stem for p in POOLS_DIR.glob("*.json"))


def pool_size(month: str) -> int:
    return len(json.loads((POOLS_DIR / f"{month}.json").read_text()))


def done_months() -> list[str]:
    out = []
    for m in months():
        f = OUT_DIR / f"month_{m}.jsonl"
        if f.exists():
            n = sum(1 for _ in open(f, encoding="utf-8") if _.strip())
            if n >= pool_size(m) * DONE_FRAC:
                out.append(m)
    return out


def current_month_progress() -> tuple[str | None, int, int]:
    """当前进行中月份 + 分片完成数/目标。"""
    done = set(done_months())
    for m in months():
        if m in done:
            continue
        n = 0
        for f in OUT_DIR.glob(f"month_{m}_*.jsonl"):
            try:
                n += sum(1 for _ in open(f, encoding="utf-8") if _.strip())
            except Exception:
                pass
        return m, n, pool_size(m)
    return None, 0, 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-months", type=int, default=0,
                    help="只等前 N 个月完成（0=全部 70 个月）")
    args = ap.parse_args()

    total = len(months())
    if args.max_months > 0:
        total = min(total, args.max_months)
    print(f"[{datetime.now():%H:%M:%S}] 70 个月监测器启动: 目标 {total} 个月, "
          f"停滞 {STALL_H}h 告警, 超时 {TIMEOUT_H/24:.0f} 天告警", flush=True)
    t0 = time.time()
    last_progress = (-1, -1)   # (已完月数, 当月分片数)
    stall_since = None
    timeout_notified = False

    while True:
        time.sleep(180)
        done = done_months()
        cm, cdone, ctarget = current_month_progress()
        elapsed = time.time() - t0
        prog = (min(len(done), total), cdone)

        # 完成判定: 前 total 个目标月全部 done（done 按时间序覆盖前 N 月）
        if done[:total] == months()[:total]:
            break

        # 停滞: 2h 无任何进展（月数或当月分片均不变）→ 告警（间隔 2h）
        if prog == last_progress:
            if stall_since is None:
                stall_since = time.time()
            elif time.time() - stall_since > STALL_H * 3600:
                notify("⚠️ Kronos 70 个月推理停滞",
                       f"{elapsed/3600:.1f}h: 完成 {len(done)}/{total} 月, "
                       f"当前 {cm} {cdone}/{ctarget}, {STALL_H}h 无进展 → 请检查 "
                       f"logs/kronos_launch_70m.log 与 batch 日志")
                stall_since = time.time()
        else:
            stall_since = None
        last_progress = prog

        # 超时告警一次
        if elapsed > TIMEOUT_H * 3600 and not timeout_notified:
            timeout_notified = True
            notify("⚠️ Kronos 70 个月推理超时",
                   f"{TIMEOUT_H/24:.0f} 天完成 {len(done)}/{total} 月, 仍在继续（不退出）")

        # 周期日志（铁律一: 跑到哪/ETA）
        if int(elapsed / 180) % 10 == 0:
            rate = (len(done)) / elapsed * 3600 if elapsed > 0 else 0
            eta = (total - len(done)) / rate if rate > 0 else 0
            print(f"[{datetime.now():%H:%M:%S}] 完成 {len(done)}/{total} 月 "
                  f"({elapsed/3600:.1f}h, 速率 {rate:.2f}月/h, ETA {eta:.0f}h) | "
                  f"当前 {cm} {cdone}/{ctarget}", flush=True)

    # ── 全部完成 → 汇总分析 + 推送 ──
    print(f"[{datetime.now():%H:%M:%S}] 🎉 70 个月全部完成, 开始汇总分析...", flush=True)
    r = subprocess.run(
        [PYTHON, "-u", str(PROJECT_ROOT / "experiments/kronos/step5_summary.py")],
        capture_output=True, text=True, timeout=1800)
    out = r.stdout + r.stderr
    print(out, flush=True)
    notify("🎉 Kronos 3a 70 个月全量完成",
           f"逐月 IC + corr 汇总见 output/70m_summary.csv\n{out[-600:]}")
    print(f"[{datetime.now():%H:%M:%S}] 监测器结束", flush=True)


if __name__ == "__main__":
    main()
