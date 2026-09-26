#!/usr/bin/env python3
"""便宜版三项验证之 ②③：换标签期 + 换时段（纯计算，用已有的 purge 预测）。

② 换标签期：同一个（去泄漏的）预测，分别对"未来 5 / 20 / 60 交易日"的超额收益算 IC
   → 看信号在哪个周期上更强。
③ 换时段：把 69 个月的逐月 IC 按时段切片 → 看真实能力是否只在某个时期存在。
"""
import json, sqlite3, numpy as np, pandas as pd, sys
from pathlib import Path
from scipy import stats
from scipy.stats import spearmanr
ROOT = Path(__file__).resolve().parents[2]
cx = sqlite3.connect(ROOT/"data/sequoia_v2.db")
px = pd.read_sql("SELECT symbol,date,close FROM stock_daily WHERE date>='2019-12-01' AND close>0 ORDER BY symbol,date", cx)
idx = pd.read_sql("SELECT date,close FROM index_daily WHERE symbol='sh.000300' ORDER BY date", cx)
cm={s:(g["date"].to_numpy(), g["close"].to_numpy(float)) for s,g in px.groupby("symbol",sort=False)}
cal=np.array(sorted(set(idx["date"]))); ic={d:c for d,c in zip(idx["date"],idx["close"].astype(float))}
CACHE = json.load(open(ROOT/"output/backtest_v2/.purged_70m.json"))
HS=[5,20,60]

rows=[]
for m in sorted(CACHE):
    pv=cal[cal<f"{m}-01"]
    if len(pv)<25: continue
    ref=pv[-1]; pos=int(np.searchsorted(cal,ref))
    rec={"month":m}
    for H in HS:
        if pos+H>=len(cal): rec[f"ic{H}"]=None; continue
        d0,d1=cal[pos+1],cal[pos+H]
        ir=ic[d1]/ic[d0]-1.0
        ys,ps=[],[]
        for s,p in zip(CACHE[m]["symbols"],CACHE[m]["t2"]):
            a=cm.get(s)
            if a is None: continue
            dt,cl=a
            j0=np.searchsorted(dt,d0); j1=np.searchsorted(dt,d1)
            if j0>=len(dt) or j1>=len(dt) or dt[j0]!=d0 or dt[j1]!=d1: continue
            ys.append(np.clip(cl[j1]/cl[j0]-1-ir,-0.5,0.5)); ps.append(p)
        if len(ys)<100: rec[f"ic{H}"]=None; continue
        rec[f"ic{H}"]=float(spearmanr(ps,ys).statistic); rec[f"n{H}"]=len(ys)
    rows.append(rec)
df=pd.DataFrame(rows)
out=Path(__file__).parent/"out"; out.mkdir(parents=True,exist_ok=True)
df.to_csv(out/"cheap_checks.csv",index=False,encoding="utf-8-sig")

print("="*80); print("② 换标签期：同一个（已去泄漏的）预测，对不同预测期的预测力"); print("="*80)
for H in HS:
    c=df[f"ic{H}"].dropna()
    if len(c)<5: continue
    t,p=stats.ttest_1samp(c,0)
    print(f"  未来 {H:2d} 交易日: 均值 IC={c.mean():+.4f}  std={c.std(ddof=1):.4f}  "
          f"ICIR={c.mean()/c.std(ddof=1):+.3f}  IC>0 {(c>0).mean()*100:.0f}%  "
          f"t={t:+.2f} p={p:.4f} {'✅显著' if p<0.05 else '❌不显著'}")

print()
print("="*80); print("③ 换时段：按时期切片（预测期=20 交易日）"); print("="*80)
c20=df.dropna(subset=["ic20"]).copy()
c20["year"]=c20["month"].str[:4].astype(int)
for tag, lo, hi in [("2020–2022",2020,2022),("2023–2024",2023,2024),("2025–2026",2025,2026)]:
    s=c20[(c20["year"]>=lo)&(c20["year"]<=hi)]["ic20"]
    if len(s)<3: continue
    t,p=stats.ttest_1samp(s,0)
    print(f"  {tag}: 均值 IC={s.mean():+.4f}  n={len(s)}  ICIR={s.mean()/s.std(ddof=1):+.3f}  "
          f"IC>0 {(s>0).mean()*100:.0f}%  t={t:+.2f} p={p:.4f} {'✅显著' if p<0.05 else '❌不显著'}")
print(f"\n  逐年: " + "  ".join(f"{y}:{g['ic20'].mean():+.4f}" for y,g in c20.groupby('year')))
