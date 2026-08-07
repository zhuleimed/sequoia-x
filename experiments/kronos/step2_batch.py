#!/usr/bin/env python3
"""3a 批次推理进程（step2_monthly 的子进程 worker, 每进程独立加载模型）

用法: py312 python step2_batch.py --codes 600519,600000 --ref 2026-06-30 --out <jsonl>
独立 python 进程（非 fork）→ 无 torch 线程池继承问题; 每进程单线程（OMP=1）,
由调度方以 N 进程并行凑满 36 核。
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.pop("KMP_AFFINITY", None)
os.environ["OMP_NUM_THREADS"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "experiments/kronos"))
sys.path.insert(0, str(PROJECT_ROOT))

from model import Kronos, KronosTokenizer, KronosPredictor
import step2_monthly as S  # 复用 predict_one


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", required=True, help="逗号分隔股票代码")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True, help="jsonl 输出文件（追加）")
    ap.add_argument("--samples", default="30", help="Kronos 采样数(30/10)")
    args = ap.parse_args()

    tokenizer = KronosTokenizer.from_pretrained(
        str(S.MODELS_DIR / "Kronos-Tokenizer-base"), local_files_only=True)
    model = Kronos.from_pretrained(str(S.MODELS_DIR / "Kronos-base"), local_files_only=True)
    S._PREDICTOR = KronosPredictor(model, tokenizer, device="cpu", max_context=512)
    SAMPLES = int(args.samples)

    codes = [c for c in args.codes.split(",") if c]
    # 批内断点续跑: 跳过 out 文件已有 code
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["code"])
                except Exception:
                    pass
    codes = [c for c in codes if c not in done]
    n_ok = 0
    for code in codes:
        try:
            r = S.predict_one(code, args.ref)
            if r is not None:
                with open(args.out, "a", encoding="utf-8") as f:
                    f.write(json.dumps(r) + "\n")
                n_ok += 1
        except Exception:
            pass
    print(f"[batch {os.getpid()}] {n_ok}/{len(codes)} 完成 → {args.out}", flush=True)


if __name__ == "__main__":
    main()

# 批内断点续跑: 启动时跳过 out 文件已有 code（进程崩溃后重跑同命令只补缺失）
