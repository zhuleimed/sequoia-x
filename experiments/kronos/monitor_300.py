#!/usr/bin/env python3
"""3b 300 只微调完成监测 + 自动评估（2026-08-08, nohup 运行）

300 只（stride=5）训练完成 → 自动评估（2 月推理 + 增量分析）→
微信推送完整评估理由（各月明细 + 判定）。
**不自动启动 800 只**（用户决定: 7h 大投入需人工确认）。

由 auto_300.py 在 300 只启动后调用本脚本（或手动启动）。

用法: env -u KMP_AFFINITY nohup py312 python -u experiments/kronos/monitor_300.py \
        > logs/monitor_300.log 2>&1 &
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
FINETUNED = PROJECT_ROOT / "experiments/kronos/finetune_csv/finetuned/a_share_300_small"
FT_TOK = FINETUNED / "tokenizer" / "best_model"
FT_PRED = FINETUNED / "basemodel" / "best_model"
FINETUNE_LOG = PROJECT_ROOT / "logs/finetune_a_share_300.log"
PYTHON = "/home/zhulei/anaconda3/envs/zhulei_py312/bin/python"
STALL_MIN = 40


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


def fmt_reason(res: dict) -> str:
    lines = ["📊 300 只各月明细（增量口径）:",
             "月份 | 微调后IC | 反转IC | 均值回归IC | corr(K,mR) | 增量"]
    for r in res.get("rows", []):
        lines.append(f"{r['month']} | {r['ic_ft']:+.4f} | {r['ic_rev']:+.4f} | "
                     f"{r['ic_mr']:+.4f} | {r['corr_mr']:+.3f} | {r['inc']:+.4f}")
    lines.append(f"\n增量均值: {res['inc_mean']:+.4f}（门槛 ≥ {res['threshold_inc']}）"
                 f"{'✅' if res['main_pass'] else '❌'}")
    if res.get("ic_zero_mean") is not None:
        lines.append(f"微调后 IC 均值: {res['ic_ft_mean']:+.4f} vs 零样本 base: "
                     f"{res['ic_zero_mean']:+.4f} {'✅' if res['aux_pass'] else '❌'}")
    return "\n".join(lines)


def main() -> None:
    print(f"[{datetime.now():%H:%M:%S}] 300 只监测器启动: 等 best_model ({FT_PRED})",
          flush=True)
    t0 = time.time()
    last_mtime = FINETUNE_LOG.stat().st_mtime if FINETUNE_LOG.exists() else time.time()
    stall_notified = False

    while True:
        time.sleep(180)
        if (FT_PRED / "model.safetensors").exists() or (FT_PRED / "pytorch_model.bin").exists():
            break
        if FINETUNE_LOG.exists():
            mtime = FINETUNE_LOG.stat().st_mtime
            if mtime != last_mtime:
                last_mtime = mtime
                stall_notified = False
            elif time.time() - mtime > STALL_MIN * 60 and not stall_notified:
                stall_notified = True
                notify("⚠️ Kronos 300 只微调疑似停滞",
                       f"{STALL_MIN}min 日志无更新: {FINETUNE_LOG}")
        if int((time.time() - t0) / 180) % 20 == 0:
            print(f"[{datetime.now():%H:%M:%S}] 等待 300 只微调完成 "
                  f"{(time.time()-t0)/60:.0f}min", flush=True)

    notify("✅ Kronos 300 只微调完成", "开始自动评估（2 月推理 + 增量判定）")
    print(f"[{datetime.now():%H:%M:%S}] 300 只微调完成, 开始评估...", flush=True)
    env = {**os.environ,
           "KRONOS_TOKENIZER_DIR": str(FT_TOK),
           "KRONOS_PREDICTOR_DIR": str(FT_PRED)}
    r = subprocess.run(
        [PYTHON, "-u", str(PROJECT_ROOT / "experiments/kronos/eval_ft.py"),
         "--months", "2026-06", "2026-03"],
        env=env, capture_output=True, text=True, timeout=12 * 3600)
    out = r.stdout + r.stderr
    print(out, flush=True)

    try:
        res = json.loads((PROJECT_ROOT / "experiments/kronos/output/ft_eval_result.json")
                         .read_text(encoding="utf-8"))
    except Exception as e:
        notify("⚠️ Kronos 300 只评估异常", f"判定文件解析失败:\n{out[-800:]}")
        return

    reason = fmt_reason(res)
    inc_mean = res.get("inc_mean", -99)
    if res.get("verdict"):
        notify("📊 Kronos 300 只评估达标",
               f"判定理由:\n{reason}\n\n→ 增量 {inc_mean:+.4f} 达标; "
               f"是否启动 800 只（~7h）由你确认后再执行")
    else:
        notify("📊 Kronos 300 只评估未达标",
               f"判定理由:\n{reason}\n\n→ 扩量无增益迹象, 建议方向四收尾")
    print(f"[{datetime.now():%H:%M:%S}] 300 只监测器结束", flush=True)


if __name__ == "__main__":
    main()
