#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Kronos 3a 相关性检查（step4）
============================
目的：判定 Kronos 零样本预测是否值得投入 70 个月全量（§9.6 门槛第 3 条）：
      corr(Kronos, T2) < 0.3 且 corr(Kronos, T4) < 0.3 → 融合候选 → 跑 70 个月全量
      （自身 IC ≥ +0.01 已由 step3 单月验证：+0.4503）

口径：
- Kronos 预测: experiments/kronos/output/month_2026-06.jsonl（exp_ret）
- T2/T4 预测:  output/backtest_v2/prediction_cache.json['2026-06']（同股票池 2978 只）
- 相关性: Spearman（与 Rank IC 同秩口径）

用法: /home/zhulei/anaconda3/envs/zhulei_py312/bin/python step4_corr.py [月份]
"""
import json
import os
import sys
from scipy.stats import spearmanr

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根
MONTH = sys.argv[1] if len(sys.argv) > 1 else "2026-06"
KRONOS_F = os.path.join(BASE, "experiments", "kronos", "output", f"month_{MONTH}.jsonl")
CACHE_F = os.path.join(BASE, "output", "backtest_v2", "prediction_cache.json")

# 1. 加载 Kronos 预测
kronos = {}
with open(KRONOS_F) as f:
    for line in f:
        r = json.loads(line)
        kronos[r["code"]] = r["exp_ret"]
print(f"Kronos 预测: {len(kronos)} 只")

# 2. 加载 T2/T4 预测
cache = json.load(open(CACHE_F))
m = cache[MONTH]
sym_map = {code: i for i, code in enumerate(m["symbols"])}
print(f"Cache {MONTH}: {len(m['symbols'])} 只 (t2/t4 均有: {len(m.get('t4', []))})")

# 3. 交集对齐
rows = []
for code, kr in kronos.items():
    if code in sym_map:
        i = sym_map[code]
        rows.append((code, kr, m["t2"][i], m["t4"][i]))
print(f"交集: {len(rows)} 只")

codes, kr, t2, t4 = zip(*rows)
kr, t2, t4 = list(kr), list(t2), list(t4)

# 4. Spearman 相关性
print("\n=== Spearman 相关性（交集截面, 2026-06）===")
for name, v in [("T2", t2), ("T4", t4)]:
    r, p = spearmanr(kr, v)
    flag = "✅ <0.3" if abs(r) < 0.3 else "⚠️ >=0.3"
    print(f"corr(Kronos, {name}) = {r:+.4f} (p={p:.2e}) {flag}")

r_ref, _ = spearmanr(t2, t4)
print(f"corr(T2, T4)   = {r_ref:+.4f}（参考：现有融合成员相关性）")

# 5. 门槛判定
c2 = abs(spearmanr(kr, t2)[0])
c4 = abs(spearmanr(kr, t4)[0])
print("\n=== 判定（§9.6 门槛 3）===")
if c2 < 0.3 and c4 < 0.3:
    print("✅ 与 T2/T4 均低相关 → 融合候选成立 → 值得投入 70 个月全量")
else:
    print("⚠️ 相关度过高 → 融合价值有限，需谨慎评估是否跑全量")
