#!/usr/bin/env python3
"""D0 对照实验：训练样本量（MAX_TRAIN_SAMPLES）对 T2 预测能力的影响。

背景
----
`scripts/build_prediction_cache.py:509` 把训练集截断为**按「日期→股票」排序的最后
5000 行**；每采样日约 2,825 样本 → 有效训练窗口仅 ~1.8 个采样日（实测 2026-09 那个
模型只用到了 ['2026-08-07','2026-08-21'] 两天）。本实验回答：放开这个上限，
模型会不会更好？

设计（**严格时间外**）
----------------------
· 训练：12 个月窗口，截止到 2026-06-22（该日标签已实现）
· 评估：4 个**更晚**的采样日（07-07 / 07-21 / 08-07 / 08-21）—— 模型完全没见过
· 三档训练量：5000（现状）/ 20000 / 全量（~71k）
· 指标：每个评估日的 Rank IC（Spearman）、均值、IC 标准差、ICIR=均值/标准差、
  训练耗时、预测值离散度
· 超参**固定**（search_optuna=False，与生产一致），否则 Optuna 会混淆结论

重要说明
--------
IC 需要**已实现**的未来 20 日收益，故评估日必须早于数据末日（2026-09-24）足够远。
本实验因此评估的是**两个月前**的选股能力，而不是当下这一期 —— 这是唯一诚实的做法。

用法
----
    python experiments/d0_train_size/run.py            # 跑全部档位（断点续跑）
    python experiments/d0_train_size/run.py --caps 5000,20000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CACHE = ROOT / "data/cache/v2_dataset/3ccb10244903"
OUT = Path(__file__).resolve().parent / "out"
TRAIN_END = "2026-06-22"
WINDOW_MONTHS = 12
EVAL_DATES = ["2026-07-07", "2026-07-21", "2026-08-07", "2026-08-21"]
CAPS = [5000, 20000, 0]        # 0 = 全量（不截断）


def _train_start(end: str, months: int) -> str:
    y, m = int(end[:4]), int(end[5:7])
    m -= months
    while m <= 0:
        m += 12
        y -= 1
    return f"{y}-{m:02d}-01"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", type=str, default="", help="逗号分隔，如 5000,20000（默认全部）")
    ap.add_argument("--train-end", type=str, default="", help="覆盖 TRAIN_END（做多切点扩充验证）")
    ap.add_argument("--eval-dates", type=str, default="", help="逗号分隔，覆盖默认 4 个评估日")
    ap.add_argument("--tag", type=str, default="", help="输出文件名后缀，区分不同切点")
    args = ap.parse_args()
    caps = [int(c) for c in args.caps.split(",")] if args.caps else CAPS
    global TRAIN_END, EVAL_DATES
    if args.train_end:
        TRAIN_END = args.train_end
    if args.eval_dates:
        EVAL_DATES = [d.strip() for d in args.eval_dates.split(",") if d.strip()]
    tag = args.tag or ""

    OUT.mkdir(parents=True, exist_ok=True)
    X = np.load(str(CACHE / "X.npy"), mmap_mode="r")
    y2 = np.load(str(CACHE / "y2.npy"))
    dates = np.array(json.load(open(CACHE / "dates.json")))
    n_feat = X.shape[2] * X.shape[1]
    print("=" * 92)
    print(f"缓存 {CACHE.name}  X={X.shape}  y2={y2.shape}  特征维度(展平)={n_feat:,}")
    print("=" * 92)

    start = _train_start(TRAIN_END, WINDOW_MONTHS)
    tr_mask = (dates >= start) & (dates <= TRAIN_END)
    tr_idx_all = np.where(tr_mask)[0]
    print(f"训练窗口 [{start} ~ {TRAIN_END}]  n_train={len(tr_idx_all):,}  "
          f"覆盖采样日 {len(set(dates[tr_idx_all]))} 天")
    # 评估日索引
    ev = {}
    for d in EVAL_DATES:
        i = np.where(dates == d)[0]
        if len(i):
            ev[d] = i
    print(f"评估日 {len(ev)} 个: {list(ev)}（均在训练窗口之后 → 严格时间外）\n")

    # 评估集特征/标签只读一次（复用）
    ev_X = {d: np.asarray(X[i]) for d, i in ev.items()}
    ev_y = {d: y2[i].astype(float) for d, i in ev.items()}

    from sequoia_x.model_selection_v2.config import get_config
    from sequoia_x.model_selection_v2.models.tree_reg import train_reg, predict_reg
    cfg = get_config()

    summary = []
    for cap in caps:
        out_f = OUT / f"cap_{cap or 'full'}{tag}.json"
        if out_f.exists():
            r = json.loads(out_f.read_text())
            print(f"[cap={cap or '全量'}] 已有结果，跳过（{r.get('ic_mean', float('nan')):+.4f}）")
            summary.append(r)
            continue

        idx = tr_idx_all if cap == 0 else tr_idx_all[-cap:]
        days_used = sorted(set(dates[idx]))
        print("-" * 92)
        print(f"[cap={cap or '全量'}] 实际样本 {len(idx):,}  覆盖采样日 {len(days_used)} 天: "
              f"{days_used[:3]}{' ... ' if len(days_used) > 6 else ''}{days_used[-3:] if len(days_used) > 6 else ''}")
        t0 = time.time()
        X_tr = np.asarray(X[idx]).reshape(len(idx), -1)
        y_tr = y2[idx].astype(float)
        print(f"  读取训练集完成 {time.time()-t0:.0f}s  内存 {X_tr.nbytes/1e9:.2f}GB")
        t1 = time.time()
        model = train_reg(X_tr, y_tr, cfg, search_optuna=False)
        train_s = time.time() - t1
        print(f"  ✅ 训练完成 {train_s:.0f}s")
        del X_tr, y_tr

        ics, rows = [], []
        for d, Xi in ev_X.items():
            pred = predict_reg(model, Xi.reshape(len(Xi), -1)).flatten()
            ic = float(spearmanr(pred, ev_y[d]).statistic)
            ics.append(ic)
            rows.append({"date": d, "ic": round(ic, 4),
                         "pred_std": round(float(pred.std()), 4),
                         "pred_mean": round(float(pred.mean()), 4)})
            print(f"    {d}  IC={ic:+.4f}  pred_std={pred.std():.4f}  pred_mean={pred.mean():+.4f}")
        ic_mean = float(np.mean(ics)); ic_std = float(np.std(ics, ddof=1)) if len(ics) > 1 else 0.0
        res = {"cap": cap or "full", "n_used": int(len(idx)), "n_days": len(days_used),
               "days": days_used, "train_sec": round(train_s, 1),
               "ic_mean": round(ic_mean, 4), "ic_std": round(ic_std, 4),
               "icir": round(ic_mean / ic_std, 3) if ic_std > 0 else None,
               "per_date": rows}
        out_f.write_text(json.dumps(res, ensure_ascii=False, indent=2))
        print(f"  ⇒ 均值 IC={ic_mean:+.4f}  std={ic_std:.4f}  ICIR={res['icir']}")
        summary.append(res)

    print("\n" + "=" * 92)
    print(f"{'训练量':>10} {'采样日':>6} {'训练耗时':>9} {'均值IC':>9} {'IC std':>8} {'ICIR':>7}  各日 IC")
    print("-" * 92)
    for r in summary:
        per = " ".join(f"{p['ic']:+.3f}" for p in r["per_date"])
        print(f"{r['n_used']:>10,} {r['n_days']:>6} {r['train_sec']:>8.0f}s "
              f"{r['ic_mean']:>+9.4f} {r['ic_std']:>8.4f} {str(r['icir']):>7}  {per}")
    print("=" * 92)
    (OUT / f"summary{tag}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
