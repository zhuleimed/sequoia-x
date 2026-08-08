#!/usr/bin/env python3
"""3b 自动判定链（2026-08-08, nohup 运行, 用户退出会话后继续）

50 只微调完成 → 自动评估（2 月推理 + 增量分析）→ 阈值判定 →
达标: 自动启动 300 只微调（stride=5, ~2.5h）+ 推送
未达标: 推送结论, 方向四收尾

阈值（用户确认放宽, 2026-08-08）:
  主判据: 两月增量 IC 均值 ≥ +0.005
  辅助: 微调后 IC 均值 > 零样本 base IC 均值
300 只完成后不自动启动 800 只（用户决定: 需人工确认）——本脚本 300 只
训练启动后即结束, 300 只的完成/评估由新监测器接手。

用法: env -u KMP_AFFINITY nohup py312 python -u experiments/kronos/auto_300.py \
        > logs/auto_300.log 2>&1 &
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
FINETUNED = PROJECT_ROOT / "experiments/kronos/finetune_csv/finetuned/a_share_50_small"
FT_TOK = FINETUNED / "tokenizer" / "best_model"
FT_PRED = FINETUNED / "basemodel" / "best_model"
FINETUNE_LOG = PROJECT_ROOT / "logs/finetune_a_share_50.log"
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
    """评估理由格式化（微信推送用, 含各月明细 + 判定依据）。"""
    lines = ["📊 各月明细（增量口径）:",
             "月份 | 微调后IC | 反转IC | 均值回归IC | corr(K,mR) | 增量"]
    for r in res.get("rows", []):
        lines.append(f"{r['month']} | {r['ic_ft']:+.4f} | {r['ic_rev']:+.4f} | "
                     f"{r['ic_mr']:+.4f} | {r['corr_mr']:+.3f} | {r['inc']:+.4f}")
    lines.append(f"\n增量均值: {res['inc_mean']:+.4f}（门槛 ≥ {res['threshold_inc']}）"
                 f"{'✅' if res['main_pass'] else '❌'}")
    if res.get("ic_zero_mean") is not None:
        lines.append(f"微调后 IC 均值: {res['ic_ft_mean']:+.4f} vs 零样本 base: "
                     f"{res['ic_zero_mean']:+.4f} "
                     f"{'✅' if res['aux_pass'] else '❌'}")
    return "\n".join(lines)


def launch_300() -> None:
    """启动 300 只微调（nohup + 独立日志 + 完成监测器 monitor_300）。"""
    cd = PROJECT_ROOT / "experiments/kronos/finetune_csv"
    cmd = ("env -u KMP_AFFINITY nohup " + PYTHON + " -u train_sequential.py "
           "--config configs/config_a_share_300.yaml "
           f"> {PROJECT_ROOT}/logs/finetune_a_share_300.log 2>&1 & echo $!")
    r = subprocess.run(cmd, shell=True, cwd=str(cd), capture_output=True, text=True)
    pid = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "?"
    print(f"[{datetime.now():%H:%M:%S}] 300 只微调已启动 PID={pid}", flush=True)
    # 挂 300 只完成监测器（完成 → 自动评估 → 推送理由; 不自动 800）
    cmd2 = ("env -u KMP_AFFINITY nohup " + PYTHON + " -u "
            f"{PROJECT_ROOT}/experiments/kronos/monitor_300.py "
            f"> {PROJECT_ROOT}/logs/monitor_300.log 2>&1 & echo $!")
    r2 = subprocess.run(cmd2, shell=True, capture_output=True, text=True)
    pid2 = r2.stdout.strip().splitlines()[-1] if r2.stdout.strip() else "?"
    print(f"[{datetime.now():%H:%M:%S}] monitor_300 已启动 PID={pid2}", flush=True)


def main() -> None:
    print(f"[{datetime.now():%H:%M:%S}] 自动判定链启动: 等 50 只微调完成 "
          f"(best_model: {FT_PRED})", flush=True)
    t0 = time.time()
    last_mtime = FINETUNE_LOG.stat().st_mtime if FINETUNE_LOG.exists() else time.time()
    stall_notified = False

    while True:
        time.sleep(180)
        # 完成判定
        if (FT_PRED / "model.safetensors").exists() or (FT_PRED / "pytorch_model.bin").exists():
            break
        # 停滞告警
        if FINETUNE_LOG.exists():
            mtime = FINETUNE_LOG.stat().st_mtime
            if mtime != last_mtime:
                last_mtime = mtime
                stall_notified = False
            elif time.time() - mtime > STALL_MIN * 60 and not stall_notified:
                stall_notified = True
                notify("⚠️ Kronos 微调疑似停滞",
                       f"{STALL_MIN}min 日志无更新: {FINETUNE_LOG}")
        if int((time.time() - t0) / 180) % 20 == 0:
            print(f"[{datetime.now():%H:%M:%S}] 等待微调完成 "
                  f"{(time.time()-t0)/60:.0f}min", flush=True)

    # ── 微调完成 → 评估 ──
    notify("✅ Kronos 50 只微调完成", "开始自动评估（2 月推理 + 增量判定）")
    print(f"[{datetime.now():%H:%M:%S}] 微调完成, 开始评估...", flush=True)
    env = {**os.environ,
           "KRONOS_TOKENIZER_DIR": str(FT_TOK),
           "KRONOS_PREDICTOR_DIR": str(FT_PRED)}
    r = subprocess.run(
        [PYTHON, "-u", str(PROJECT_ROOT / "experiments/kronos/eval_ft.py"),
         "--months", "2026-06", "2026-03"],
        env=env, capture_output=True, text=True, timeout=12 * 3600)
    out = r.stdout + r.stderr
    print(out, flush=True)

    # ── 判定 → 决策 ──
    try:
        res = json.loads((PROJECT_ROOT / "experiments/kronos/output/ft_eval_result.json")
                         .read_text(encoding="utf-8"))
        verdict = res.get("verdict", False)
        inc_mean = res.get("inc_mean", -99)
    except Exception as e:
        print(f"⚠️ 判定结果解析失败: {e}", flush=True)
        notify("⚠️ Kronos 50 只评估异常", f"判定文件解析失败:\n{out[-800:]}")
        return

    reason = fmt_reason(res)
    if verdict:
        notify("🚀 Kronos 50 只评估达标 → 自动启动 300 只",
               f"判定理由:\n{reason}\n\n→ 两门槛均过, 启动 300 只微调（stride=5, ~2.5h）")
        launch_300()
        notify("300 只微调已启动", "完成后自动评估并推送理由; 800 只等你看结果后确认")
    else:
        notify("❌ Kronos 50 只评估未达标, 300 只不启动",
               f"判定理由:\n{reason}\n\n→ 主/辅助门槛未过, 方向四 3b 收尾（详见 §9.7）")
    print(f"[{datetime.now():%H:%M:%S}] 自动判定链结束", flush=True)


if __name__ == "__main__":
    main()
