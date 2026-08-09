#!/usr/bin/env python3
"""方案 B 指数微调监测链（2026-08-09, nohup 运行, 用户退出会话后继续）

中证1000 指数微调（Kronos-small, config_index_small.yaml）完成 →
自动评估（index_timing_check 2 年窗口, 微调模型）→ 对比零样本 base →
微信推送结果 + 下一步建议。

完成判定（auto_300 教训 2026-08-08）: 训练日志含 "Training completed successfully!"
标志 + 硬超时——不能只看 best_model 文件存在（训练中途每个 epoch 就保存
checkpoint, 会提前触发用中途模型评估）。
停滞检测: 日志 30min 无更新 → 微信告警（间隔 ≥30min）。

⚠️ 必须项目根 cwd 启动（wxpusher env_file 相对 cwd 教训, auto_300 推送失败根因）:
  env -u KMP_AFFINITY nohup py312 python -u experiments/kronos/monitor_index_ft.py \
    > logs/monitor_index_ft.log 2>&1 &
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

FINETUNED = PROJECT_ROOT / "experiments/kronos/finetune_csv/finetuned/index_000852_small"
FT_TOK = FINETUNED / "tokenizer" / "best_model"
FT_PRED = FINETUNED / "basemodel" / "best_model"
TRAIN_LOG = PROJECT_ROOT / "logs/finetune_index_000852.log"
EVAL_OUT = PROJECT_ROOT / "experiments/kronos/output/index_timing_check_2y_ft_index.json"
ZERO_JSON = PROJECT_ROOT / "experiments/kronos/output/index_timing_check_2y.json"  # 零样本 base
PYTHON = "/home/zhulei/anaconda3/envs/zhulei_py312/bin/python"
STALL_MIN = 30
TRAIN_DONE_MARK = "Training completed successfully!"
TRAIN_TIMEOUT_H = 6  # 单序列 small 预计 <1h, 6h 硬超时保护
EVAL_TIMEOUT_H = 3


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


def fmt_compare(zero: dict, ft: dict) -> str:
    """零样本 base vs 指数微调 对比表（微信推送）。"""
    def row(tag, key, fmt="+.4f"):
        return (f"{tag:<9} 零样本 {zero[key]:{fmt}}  |  指数微调 {ft[key]:{fmt}}"
                if key in zero and key in ft else "")
    lines = [
        "📊 中证1000 5日择时（2年145点, 同口径）",
        "指标          零样本base        指数微调small",
        f"Spearman(收益)  {zero['spearman_ret']:+.4f}          {ft['spearman_ret']:+.4f}",
        f"方向准确率      {zero['accuracy']*100:.1f}%           {ft['accuracy']*100:.1f}%",
        f"择时累计        {zero['strat_nav']:+.1%}         {ft['strat_nav']:+.1%}",
        f"买入持有        {zero['buyhold_nav']:+.1%}         {ft['buyhold_nav']:+.1%}",
        f"超额            {zero['excess']:+.1%}         {ft['excess']:+.1%}",
    ]
    return "\n".join(lines)


def main() -> None:
    print(f"[{datetime.now():%H:%M:%S}] 指数微调监测链启动: 等训练完成 "
          f"(best_model: {FT_PRED})", flush=True)
    t0 = time.time()
    last_mtime = TRAIN_LOG.stat().st_mtime if TRAIN_LOG.exists() else time.time()
    stall_notified = False

    while True:
        time.sleep(180)
        if TRAIN_LOG.exists():
            log_text = TRAIN_LOG.read_text(encoding="utf-8", errors="ignore")
            if TRAIN_DONE_MARK in log_text:
                break
        # 硬超时保护
        if time.time() - t0 > TRAIN_TIMEOUT_H * 3600:
            notify("⚠️ 指数微调监测超时",
                   f"{TRAIN_TIMEOUT_H}h 未见训练完成标志（{TRAIN_DONE_MARK}）, 监测链退出")
            print(f"[{datetime.now():%H:%M:%S}] ⚠️ 超时退出, 未执行评估", flush=True)
            return
        # 停滞告警
        if TRAIN_LOG.exists():
            mtime = TRAIN_LOG.stat().st_mtime
            if mtime != last_mtime:
                last_mtime = mtime
                stall_notified = False
            elif time.time() - mtime > STALL_MIN * 60 and not stall_notified:
                stall_notified = True
                notify("⚠️ 指数微调疑似停滞", f"{STALL_MIN}min 日志无更新: {TRAIN_LOG}")
        if int((time.time() - t0) / 180) % 20 == 0:
            print(f"[{datetime.now():%H:%M:%S}] 等待微调完成 "
                  f"{(time.time()-t0)/60:.0f}min", flush=True)

    # ── 训练完成 → 评估 ──
    notify("✅ 中证1000 指数微调完成", "开始自动评估（2 年 145 点择时, 微调模型）")
    print(f"[{datetime.now():%H:%M:%S}] 微调完成, 开始评估...", flush=True)
    env = {**os.environ,
           "KRONOS_TOKENIZER_DIR": str(FT_TOK),
           "KRONOS_PREDICTOR_DIR": str(FT_PRED)}
    r = subprocess.run(
        [PYTHON, "-u", str(PROJECT_ROOT / "experiments/kronos/index_timing_check.py"),
         "--days", "730", "--workers", "12",
         "--out", "index_timing_check_2y_ft_index.json"],
        env=env, capture_output=True, text=True, timeout=EVAL_TIMEOUT_H * 3600)
    out = (r.stdout + r.stderr).replace("Loading weights from local directory\n", "")
    print(out, flush=True)

    # ── 判定: 对比零样本 base ──
    try:
        ft = json.loads(EVAL_OUT.read_text(encoding="utf-8"))
        zero = json.loads(ZERO_JSON.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"⚠️ 评估结果解析失败: {e}", flush=True)
        notify("⚠️ 指数微调评估异常", f"判定文件解析失败:\n{out[-800:]}")
        return

    d_spear = ft["spearman_ret"] - zero["spearman_ret"]
    d_acc = ft["accuracy"] - zero["accuracy"]
    d_excess = ft["excess"] - zero["excess"]
    improved = d_spear > 0.02 and d_acc > 0.01  # Spearman 至少 +0.02 且方向准确率不降
    verdict = "✅ 改善, 方向对!" if improved else ("⚠️ 部分改善" if d_excess > 0 or d_spear > 0 else "❌ 未改善")
    reason = [
        fmt_compare(zero, ft),
        f"\nΔSpearman {d_spear:+.4f} | Δ方向 {d_acc*100:+.1f}pp | Δ超额 {d_excess:+.1f}pp",
        f"→ 判定: {verdict}",
    ]
    if improved:
        reason.append("下一步: 滚动微调（每 2 个月重训一轮）+ 滚动推理参数调整（中金核心 trick）→ 需你确认后启动")
    else:
        reason.append("下一步建议: ①调训练参数(epochs/学习率)重试 ②换 Kronos-base 基座夜跑 ③或收尾方案 B")
    body = "\n".join(reason)
    notify(f"📊 方案B 指数微调评估: {verdict}", body)
    print(body, flush=True)
    print(f"[{datetime.now():%H:%M:%S}] 监测链结束", flush=True)


if __name__ == "__main__":
    main()
