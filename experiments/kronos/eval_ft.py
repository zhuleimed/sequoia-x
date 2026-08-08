#!/usr/bin/env python3
"""3b 微调后评估编排（2026-08-08）: 单月推理 + 基线增量分析 + 判定

流程:
  1. 用微调后模型（KRONOS_TOKENIZER_DIR/KRONOS_PREDICTOR_DIR 环境变量指定）对
     --months 各月推理（step2_launch 36 进程, 输出 month_<YM>_ft.jsonl）
  2. analyze_baseline.py --suffix _ft（增量 IC 口径: Kronos IC - max(反转, 均值回归)）
  3. 判定（放宽阈值, 用户确认 2026-08-08）:
     主判据: 各月增量均值 ≥ +0.005
     辅助: 微调后 IC 均值 > 零样本 base IC 均值（确保微调有改善）
  4. 输出判定 JSON: output/ft_eval_result.json + 打印

用法:
  KRONOS_TOKENIZER_DIR=<ft tokenizer> KRONOS_PREDICTOR_DIR=<ft predictor> \\
    py312 python eval_ft.py --months 2026-06 2026-03 [--zero-ic 0.4844 0.0130]
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJECT_ROOT / "experiments/kronos/output"
PYTHON = "/home/zhulei/anaconda3/envs/zhulei_py312/bin/python"
THRESHOLD_INC = 0.005  # 放宽后主判据（用户确认）
MONTHS_DEFAULT = ["2026-06", "2026-03"]
ZERO_DEFAULT = {"2026-06": 0.4844, "2026-03": 0.0130}  # 零样本 base IC（3a 实测）


def run_inference(month: str) -> bool:
    """单月推理（微调模型, 输出 month_<YM>_ft.jsonl）。"""
    print(f"[eval] ⏳ {month} 单月推理（微调模型）...", flush=True)
    r = subprocess.run(
        [PYTHON, "-u", str(PROJECT_ROOT / "experiments/kronos/step2_launch.py"),
         "--month", month, "--suffix", "_ft"],
        env={**os.environ, "OMP_NUM_THREADS": "1"},
        capture_output=True, text=True, timeout=12 * 3600)
    out = r.stdout + r.stderr
    print(out[-800:], flush=True)
    # 完成校验: _ft.jsonl 行数 ≈ 池子数
    fp = OUT_DIR / f"month_{month}_ft.jsonl"
    n = sum(1 for _ in open(fp, encoding="utf-8") if _.strip()) if fp.exists() else 0
    if n < 3000:
        print(f"⚠️ {month} 推理产出 {n} 行（<3000）, 视为失败", flush=True)
        return False
    print(f"✅ {month} 推理完成: {n} 只", flush=True)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="+", default=MONTHS_DEFAULT)
    ap.add_argument("--zero-ic", type=float, nargs="+", default=None,
                    help="零样本 base IC（按 months 顺序）")
    args = ap.parse_args()
    zero_ics = args.zero_ic or [ZERO_DEFAULT.get(m, None) for m in args.months]

    # 1. 推理（两月串行, 36 核预算）
    ok = all(run_inference(m) for m in args.months)
    if not ok:
        print("❌ 推理失败, 中止判定", flush=True)
        return

    # 2. 基线增量分析
    print("\n[eval] 基线增量分析（_ft 口径）...", flush=True)
    r = subprocess.run(
        [PYTHON, "-u", str(PROJECT_ROOT / "experiments/kronos/analyze_baseline.py")]
        + args.months + ["--suffix", "_ft"],
        capture_output=True, text=True, timeout=1800)
    out = r.stdout + r.stderr
    print(out, flush=True)

    # 3. 解析增量 IC（完整列: month idx ic_ft ic_rev ic_mr corr inc）
    #    analyze_baseline.py 实际输出 7 字段（2026-08-09 修复: 之前误以为 8 字段,
    #    导致永远解析 0 行, inc_mean=-99 强制判负, 误判方向四收尾）
    rows = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 7 and parts[0].startswith("20"):
            rows.append({"month": parts[0], "idx_ret": float(parts[1].replace("%", "")) / 100,
                         "ic_ft": float(parts[2]), "ic_rev": float(parts[3]),
                         "ic_mr": float(parts[4]), "corr_mr": float(parts[5]),
                         "inc": float(parts[6])})
    if len(rows) != len(args.months):
        print(f"⚠️ 解析到 {len(rows)} 个月（期望 {len(args.months)}）", flush=True)
    incs = [r["inc"] for r in rows]
    inc_mean = sum(incs) / len(incs) if incs else -99
    ics_ft = [r["ic_ft"] for r in rows]
    ic_ft_mean = sum(ics_ft) / len(ics_ft) if ics_ft else -99
    ic_zero_avail = [z for z in zero_ics if z is not None]
    ic_zero_mean = sum(ic_zero_avail) / len(ic_zero_avail) if ic_zero_avail else None

    # 4. 判定（放宽阈值, 用户确认）
    main_pass = inc_mean >= THRESHOLD_INC
    aux = (ic_zero_mean is None) or (ic_ft_mean > ic_zero_mean)
    verdict = main_pass and aux
    result = {
        "months": args.months, "inc_mean": inc_mean, "inc_list": incs,
        "ic_ft_mean": ic_ft_mean, "ic_zero_mean": ic_zero_mean,
        "threshold_inc": THRESHOLD_INC, "main_pass": main_pass, "aux_pass": aux,
        "verdict": verdict, "rows": rows,
    }
    (OUT_DIR / "ft_eval_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2))
    print("\n" + "=" * 60, flush=True)
    print(f"增量均值 = {inc_mean:+.4f}（门槛 ≥ {THRESHOLD_INC}）{'✅' if main_pass else '❌'}")
    print(f"微调后 IC 均值 = {ic_ft_mean:+.4f} vs 零样本均值 = "
          f"{ic_zero_mean if ic_zero_mean is not None else 'N/A'}"
          f"{' ✅' if aux else ' ❌'}")
    print(f"→ 判定: {'✅ 达标 → 启动 300 只扩量' if verdict else '❌ 未达标 → 300 只不启动, 方向四收尾'}")
    print(f"结果已写 {OUT_DIR / 'ft_eval_result.json'}")


if __name__ == "__main__":
    main()
