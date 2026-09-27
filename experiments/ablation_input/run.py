#!/usr/bin/env python3
"""消融实验：模型输入表示的两种改法，是否比现状更好。

背景（今天已确立的事实）
------------------------
1. 现状：1 个样本 = **120 天 × 129 维展平 = 15,480 个特征**，训练样本 5,000 条
   （`MAX_TRAIN_SAMPLES`，按日期排序取尾部 → 实际只有 ~2 个采样日）
2. D0 实验已证明：**样本加到 77,000 条（14 倍）IC 没有改善**（p=0.49，胜率 4/10）
   ⇒ 瓶颈不在样本量，那就在**特征表示**
3. V5 的 T1 模型最重要的 10 个特征里 **8 个是"几号"**（§7d 时间日历 col 66/67）
   ⇒ 训练集只有 2 个采样日 → "几号"只有 2 个取值 → 退化成"我是哪一批"的编号

两个候选改法
------------
A) **降维**：120 天里相邻天高度相关，展平等于把同一信号重复 120 遍。改用
   · `lastday`      = 只用窗口最后一天（129 维）
   · `lastday_agg`  = 最后一天 + 窗口均值 + 窗口标准差（129×3 = 387 维）
B) **日历消融**：把 §7d 那 4 列（col 65..68）置零，训练与推理同步置零
   · `nocal`         = 现状表示 + 日历列置零
   · `lastday_nocal` = 降维 + 日历列置零

基线
----
直接复用今天 D0 已跑出的生产配置结果 `experiments/d0_train_size/out/cap_5000_c<cut>.json`
（同切点、同评估日、同超参、同 cap=5000）→ 严格配对可比，且省掉 10 次训练。

用法
----
    python experiments/ablation_input/run.py            # 断点续跑
    python experiments/ablation_input/run.py --variants lastday,nocal
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CACHE = ROOT / "data/cache/v2_dataset/3ccb10244903"
OUT = Path(__file__).resolve().parent / "out"
D0_OUT = ROOT / "experiments/d0_train_size/out"
CAP = 5000                     # 与生产一致
N_EVAL = 13
CUTS = ["2021-08-06", "2022-02-11", "2022-08-05", "2023-02-07", "2023-08-07",
        "2024-02-07", "2024-08-07", "2025-02-11", "2025-08-07", "2026-02-06"]
CAL_COLS = slice(65, 69)       # §7d 时间日历（sin/cos weekday、day_of_month、is_quarter_end）
VARIANTS = ["lastday", "lastday_agg", "nocal", "lastday_nocal"]


def _train_start(end: str, months: int = 12) -> str:
    y, m = int(end[:4]), int(end[5:7])
    m -= months
    while m <= 0:
        m += 12
        y -= 1
    return f"{y}-{m:02d}-01"


def build_X(raw: np.ndarray, variant: str) -> np.ndarray:
    """把 (n, 120, 129) 的原始块按变体转成 2D 训练/预测矩阵。"""
    if variant == "lastday":
        return raw[:, -1, :]
    if variant == "lastday_agg":
        return np.concatenate([raw[:, -1, :], raw.mean(axis=1), raw.std(axis=1)], axis=1)
    if variant == "nocal":
        r = raw.copy()
        r[:, :, CAL_COLS] = 0.0
        return r.reshape(len(r), -1)
    if variant == "lastday_nocal":
        r = raw.copy()
        r[:, :, CAL_COLS] = 0.0
        return r[:, -1, :]
    raise ValueError(variant)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=str, default="", help="逗号分隔，默认全部")
    args = ap.parse_args()
    variants = [v.strip() for v in args.variants.split(",")] if args.variants else VARIANTS

    OUT.mkdir(parents=True, exist_ok=True)
    X = np.load(str(CACHE / "X.npy"), mmap_mode="r")
    y2 = np.load(str(CACHE / "y2.npy"))
    dates = np.array(json.load(open(CACHE / "dates.json")))
    print("=" * 96)
    print(f"消融实验  cap={CAP}（与生产一致）  变体={variants}")
    print(f"基线 = experiments/d0_train_size/out/cap_5000_c<cut>.json（今天已跑，严格配对可比）")
    print("=" * 96)

    from sequoia_x.model_selection_v2.config import get_config
    from sequoia_x.model_selection_v2.models.tree_reg import train_reg, predict_reg
    cfg = get_config()

    summary = []       # [{cut, variant, ic_mean, icir, per_date:[...], sec, n_feat}]
    for cut in CUTS:
        if cut not in dates:
            print(f"⚠️ 切点 {cut} 不存在，跳过"); continue
        tr_idx = np.where((dates >= _train_start(cut)) & (dates <= cut))[0]
        if len(tr_idx) < 100:
            print(f"⚠️ 切点 {cut} 样本不足"); continue
        tr_idx = tr_idx[-CAP:]
        eval_dates = [d for d in sorted(set(dates)) if d > cut][:N_EVAL]
        ev_idx = {d: np.where(dates == d)[0] for d in eval_dates}
        ev_y = {d: y2[i].astype(float) for d, i in ev_idx.items()}

        # 只读一次原始块，变体在其上变换
        raw_tr = np.asarray(X[tr_idx])
        raw_ev = {d: np.asarray(X[i]) for d, i in ev_idx.items()}
        y_tr = y2[tr_idx].astype(float)
        days_used = sorted(set(dates[tr_idx]))
        print(f"\n[{cut}] 训练 {len(tr_idx):,} 条 / {len(days_used)} 个采样日，"
              f"评估 {eval_dates[0]} ~ {eval_dates[-1]}")

        for v in variants:
            f = OUT / f"{v}_c{cut.replace('-','')}.json"
            if f.exists():
                summary.append(json.loads(f.read_text())); print(f"  {v}: 已有，跳过"); continue
            t0 = time.time()
            Xtr = build_X(raw_tr, v)
            model = train_reg(Xtr, y_tr, cfg, search_optuna=False)
            sec = time.time() - t0
            ics = []
            for d in eval_dates:
                p = predict_reg(model, build_X(raw_ev[d], v)).flatten()
                ics.append(round(float(spearmanr(p, ev_y[d]).statistic), 4))
            a = np.array(ics)
            r = {"cut": cut, "variant": v, "n_feat": int(Xtr.shape[1]), "n_used": int(len(tr_idx)),
                 "n_days": len(days_used), "sec": round(sec, 1),
                 "ic_mean": round(float(a.mean()), 4),
                 "ic_std": round(float(a.std(ddof=1)), 4),
                 "icir": round(float(a.mean() / a.std(ddof=1)), 3),
                 "per_date": [{"date": d, "ic": ic} for d, ic in zip(eval_dates, ics)]}
            f.write_text(json.dumps(r, ensure_ascii=False, indent=2))
            del Xtr
            summary.append(r)
            print(f"  {v:16s} 维度={r['n_feat']:>6,}  {sec:>5.0f}s  "
                  f"均值IC={r['ic_mean']:+.4f}  ICIR={r['icir']:+.3f}")

    # ── 与基线配对比较 ──
    print("\n" + "=" * 96)
    print("【配对比较：各变体 vs 生产基线（同切点、同评估日、同 cap=5000）】")
    base = {}
    for cut in CUTS:
        f = D0_OUT / f"cap_5000_c{cut.replace('-','')}.json"
        if f.exists():
            base[cut] = {p["date"]: p["ic"] for p in json.loads(f.read_text())["per_date"]}
    rows = []
    for v in variants:
        diffs, mb, mv = [], [], []
        for r in summary:
            if r["variant"] != v or r["cut"] not in base:
                continue
            b = base[r["cut"]]
            for p in r["per_date"]:
                if p["date"] in b:
                    diffs.append(p["ic"] - b[p["date"]]); mb.append(b[p["date"]]); mv.append(p["ic"])
        if len(diffs) < 5:
            continue
        diffs = np.array(diffs)
        t, p = stats.ttest_rel(np.array(mv), np.array(mb))
        rows.append((v, len(diffs), np.mean(mb), np.mean(mv), diffs.mean(), t, p))
    print(f"{'变体':>16} {'n':>4} {'基线IC':>9} {'变体IC':>9} {'配对差':>9} {'t':>7} {'p':>8}  判定")
    for v, n, b, m, d, t, p in rows:
        flag = "✅ 显著更好" if (p < 0.05 and d > 0) else ("❌ 显著更差" if (p < 0.05 and d < 0) else "— 不显著")
        print(f"{v:>16} {n:>4} {b:>+9.4f} {m:>+9.4f} {d:>+9.4f} {t:>+7.2f} {p:>8.4f}  {flag}")
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n结论解读：配对差为正 = 该变体更好；不显著 = 与现状无实质差别。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
