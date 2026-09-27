#!/usr/bin/env python3
"""D0 扩充验证驱动：10 个训练切点 × 3 档训练量。

为什么不只用一个大切点
----------------------
单个切点（或少数几个评估日）的结论**会被 regime 绑架**——首轮实验里
"全量档 ICIR 1.86"就是用 4 个连续上涨日的评估日换来的假象，换两个切点后消失。
故此处取 10 个切点、每个用其后 13 个采样日做样本外评估：
· 切点间隔 6 个月，评估窗口 ~6 个月 → **窗口基本不重叠**，观测近似独立
· 汇总时同时给两种口径：
  ① 逐日配对（n=130，但跨切点重叠 → 独立性偏弱，p 值偏乐观）
  ② **按切点先取均值再跨切点配对（n=10，保守，推荐以此为准）**

用法
----
    python experiments/d0_train_size/driver.py            # 断点续跑（已有 json 会跳过）
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data/cache/v2_dataset/3ccb10244903"
HERE = Path(__file__).resolve().parent
PY = "/home/zhulei/anaconda3/envs/zhulei_py312/bin/python"

CUTS = ["2021-08-06", "2022-02-11", "2022-08-05", "2023-02-07", "2023-08-07",
        "2024-02-07", "2024-08-07", "2025-02-11", "2025-08-07", "2026-02-06"]
N_EVAL = 13


def main() -> int:
    ds = sorted(set(json.load(open(CACHE / "dates.json"))))
    jobs = []
    for cut in CUTS:
        if cut not in ds:
            print(f"⚠️ 切点 {cut} 不在采样日里，跳过")
            continue
        after = [d for d in ds if d > cut][:N_EVAL]
        if len(after) < N_EVAL:
            print(f"⚠️ 切点 {cut} 之后只有 {len(after)} 个采样日，跳过")
            continue
        jobs.append((cut, after))

    print(f"共 {len(jobs)} 个切点，每点 {N_EVAL} 个样本外评估日；切点间隔半年 → 窗口基本不重叠\n")
    for i, (cut, ev) in enumerate(jobs, 1):
        tag = f"_c{cut.replace('-', '')}"
        done = (HERE / "out" / f"summary{tag}.json").exists()
        print(f"[{i}/{len(jobs)}] 切点 {cut}  评估 {ev[0]} ~ {ev[-1]}  "
              f"{'（已完成，跳过）' if done else ''}", flush=True)
        if done:
            continue
        cmd = [PY, "-u", str(HERE / "run.py"), "--train-end", cut,
               "--eval-dates", ",".join(ev), "--tag", tag]
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        log = HERE / f"driver_{cut}.log"
        log.write_text(r.stdout + "\n---STDERR---\n" + r.stderr)
        if r.returncode != 0:
            print(f"  ❌ exit={r.returncode}，见 {log.name}")
        else:
            s = json.loads((HERE / "out" / f"summary{tag}.json").read_text())
            line = "  ".join(f"{x['n_used']:,}→IC{x['ic_mean']:+.4f}/IR{x['icir']}" for x in s)
            print(f"  ✅ {line}", flush=True)
    print("\n全部完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
