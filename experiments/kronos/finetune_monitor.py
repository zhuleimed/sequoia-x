#!/usr/bin/env python3
"""3b 微调完成监测器（2026-08-08, nohup 运行, 用户退出会话后继续）

官方 train_sequential.py 无推送 → 本监测器补上:
  1. 每 2min 轮询 basemodel/best_model 落盘（训练完成标志）+ 日志尾部
  2. 停滞检测: 日志 30min 无更新 → 微信告警（间隔 ≥30min）
  3. 完成后微信推送（含 tokenizer/predictor 两阶段耗时与日志尾部）

用法: nohup py312 python -u experiments/kronos/finetune_monitor.py \
        > logs/finetune_monitor.log 2>&1 &
"""
from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
FINETUNED_DIR = (PROJECT_ROOT / "experiments/kronos/finetune_csv/finetuned/a_share_50_small")
LOG_F = PROJECT_ROOT / "logs/finetune_a_share_50.log"
STALL_MIN = 30
EXP = "a_share_50_small"


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


def log_tail(n: int = 8) -> str:
    try:
        lines = LOG_F.read_text(encoding="utf-8").strip().splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return "(日志不可读)"


def main() -> None:
    best_model = FINETUNED_DIR / "basemodel" / "best_model"
    print(f"[{datetime.now():%H:%M:%S}] 微调监测器启动: 等 {best_model}", flush=True)
    t0 = time.time()
    last_mtime = LOG_F.stat().st_mtime if LOG_F.exists() else time.time()
    stall_notified = False

    while True:
        time.sleep(120)
        # 完成判定: basemodel/best_model 落盘
        if (best_model / "model.safetensors").exists() or \
           (best_model / "pytorch_model.bin").exists():
            el = time.time() - t0
            notify("✅ Kronos A 股微调完成（small, 50 只）",
                   f"耗时 {el/60:.0f}min, 模型 {best_model}\n"
                   f"日志尾部:\n{log_tail()}")
            print(f"[{datetime.now():%H:%M:%S}] 完成推送, 监测器结束", flush=True)
            return

        # 停滞检测: 日志 30min 无更新
        if LOG_F.exists():
            mtime = LOG_F.stat().st_mtime
            if mtime != last_mtime:
                last_mtime = mtime
                stall_notified = False
            elif time.time() - mtime > STALL_MIN * 60 and not stall_notified:
                stall_notified = True
                notify("⚠️ Kronos 微调疑似停滞",
                       f"{STALL_MIN}min 日志无更新, 请检查 logs/finetune_a_share_50.log:\n{log_tail(5)}")
        else:
            time.sleep(10)

        # 周期日志
        if int((time.time() - t0) / 120) % 15 == 0:
            print(f"[{datetime.now():%H:%M:%S}] 训练中 {elapsed_min:.0f}min"
                  .replace("elapsed_min", str((time.time() - t0) / 60)), flush=True)


if __name__ == "__main__":
    main()
