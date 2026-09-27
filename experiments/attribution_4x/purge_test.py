#!/usr/bin/env python3
"""泄漏验证：训练窗口做 20 交易日 purge 后，IC 是否崩塌。

泄漏假设：训练样本的标签 y2 用未来 20 日收益，而 train_end 只卡在预测月之前，
于是"最后 1~2 个采样日"的标签直接落在**被预测的那个月**里 → 模型是用答案训练的。
验证：把训练窗口往前推 20+ 交易日（标签已完整）再训，IC 应显著下降。
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, ttest_rel
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
CACHE = ROOT / "data/cache/v2_dataset/3ccb10244903"
CAP = 5000; H = 20
CUTS = ["2025-08-07","2025-09-19","2025-10-29","2025-11-21","2025-12-19",
        "2026-01-23","2026-02-06","2026-02-27","2026-03-20","2026-04-22"]

def main():
    X = np.load(str(CACHE/"X.npy"), mmap_mode="r")
    y2 = np.load(str(CACHE/"y2.npy"))
    dates = np.array(json.load(open(CACHE/"dates.json")))
    cal = np.array(sorted(set(dates)))
    from sequoia_x.model_selection_v2.config import get_config
    from sequoia_x.model_selection_v2.models.tree_reg import train_reg, predict_reg
    cfg = get_config()
    res = []
    for cut in CUTS:
        if cut not in cal: continue
        ci = int(np.searchsorted(cal, cut))
        if ci - 12 < 0 or ci + 1 >= len(cal): continue
        ev = cal[ci+1]                                  # 下一个采样日做评估
        ev_idx = np.where(dates == ev)[0]
        if len(ev_idx) < 100: continue
        row = {"cut": cut, "eval": ev}
        # 2026-09-27 修（审计 C1）：原来 purged 组的**上界是 cal[ci-25]、下界却是 cal[ci-24]**
        #   ⇒ 下界 > 上界 ⇒ 掩码恒空 ⇒ `purged` 列全是 None/NaN，配对检验的守卫永不执行
        #   （out/purge_test.json 10 行全是 purged:null；purge.log 里 "均值 IC=+nan n=0"）。
        #   正确做法：**整个训练窗口一起前推** shift 个交易日（窗口长度不变）。
        for tag, shift in [("leaky", 0), ("purged", H + 5)]:
            end = cal[max(0, ci - shift)]
            start = cal[max(0, ci - 24 - shift)]
            tr = np.where((dates >= start) & (dates <= end))[0]
            if len(tr) < 100: row[tag] = None; continue
            tr = tr[-CAP:]
            Xtr = np.asarray(X[tr]).reshape(len(tr), -1)
            m = train_reg(Xtr, y2[tr].astype(float), cfg, search_optuna=False)
            del Xtr
            Xev = np.asarray(X[ev_idx]).reshape(len(ev_idx), -1)
            p = predict_reg(m, Xev).flatten()
            row[tag] = float(spearmanr(p, y2[ev_idx].astype(float)).statistic)
            print(f"  {cut} → {ev}  {tag:7s} IC={row[tag]:+.4f}", flush=True)
        res.append(row)
    L = np.array([r["leaky"] for r in res if r.get("leaky") is not None])
    P = np.array([r["purged"] for r in res if r.get("purged") is not None])
    print("\n" + "="*70)
    print(f"  leaky （现状：训练到最后采样日）  均值 IC={L.mean():+.4f}  n={len(L)}")
    print(f"  purged（训练窗口前推 25 交易日） 均值 IC={P.mean():+.4f}  n={len(P)}")
    if len(L)==len(P) and len(L)>2:
        t,p = ttest_rel(L,P)
        print(f"  配对差（leaky−purged）={np.mean(L-P):+.4f}  t={t:+.2f}  p={p:.4f}  "
              f"{'✅ 泄漏确认' if p<0.05 else '❌ 不显著'}")
    json.dump(res, open(Path(__file__).parent/"out"/"purge_test.json","w"), ensure_ascii=False, indent=2)
main()
