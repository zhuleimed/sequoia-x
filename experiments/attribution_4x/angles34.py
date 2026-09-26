#!/usr/bin/env python3
"""角度3：泄漏模型的 TOP10 是否正好是当月涨幅榜？
   角度4：同一个预测，评估在"下个月/下下个月"，衰减多快？"""
import json, sqlite3, numpy as np, pandas as pd, sys
from pathlib import Path
from scipy.stats import spearmanr
ROOT = Path(__file__).resolve().parents[2]
cx = sqlite3.connect(ROOT/"data/sequoia_v2.db")
px = pd.read_sql("SELECT symbol,date,close FROM stock_daily WHERE date>='2019-12-01' AND close>0 ORDER BY symbol,date", cx)
idx = pd.read_sql("SELECT date,close FROM index_daily WHERE symbol='sh.000300' ORDER BY date", cx)
cm={s:(g["date"].to_numpy(), g["close"].to_numpy(float)) for s,g in px.groupby("symbol",sort=False)}
cal=np.array(sorted(set(idx["date"]))); ic=dict(zip(idx["date"],idx["close"].astype(float)))
A=json.load(open(ROOT/"output/backtest_v2/.purge_off.json"))

def fwd(m, shift=0):
    """返回 (pred, y) —— m 月预测；shift=0 评估当月, 1=下月, 2=下下月"""
    pv=cal[cal<f"{m}-01"]
    if len(pv)<25: return None
    ref=pv[-1]; pos=int(np.searchsorted(cal,ref))+shift*20
    if pos+20>=len(cal): return None
    d0,d1=cal[pos+1],cal[pos+20]; ir=ic[d1]/ic[d0]-1.0
    ys,ps,ss=[],[],[]
    for s,p in zip(A[m]["symbols"],A[m]["t2"]):
        a=cm.get(s)
        if a is None: continue
        dt,cl=a
        j0=np.searchsorted(dt,d0); j1=np.searchsorted(dt,d1)
        if j0>=len(dt) or j1>=len(dt) or dt[j0]!=d0 or dt[j1]!=d1: continue
        ys.append(np.clip(cl[j1]/cl[j0]-1-ir,-0.5,0.5)); ps.append(p); ss.append(s)
    return np.array(ps),np.array(ys),ss

print("=== 角度4：时间外衰减（同一个泄漏模型，评估不同滞后期）===")
dec={0:[],1:[],2:[]}
for m in sorted(A):
    for sh in (0,1,2):
        r=fwd(m,sh)
        if r and len(r[0])>100: dec[sh].append(spearmanr(r[0],r[1]).statistic)
for sh,v in dec.items():
    v=np.array([x for x in v if np.isfinite(x)])
    tag={0:"当月(有泄漏)",1:"下月",2:"下下月"}[sh]
    print(f"  {tag:12s}: 均值 IC={v.mean():+.4f}  n={len(v)}")

print("\n=== 角度3：TOP10 命中涨幅榜吗？（2026-03，A 档 IC 最高月 0.82）===")
te="2026-03"
for tag,model in [("A 泄漏模型",A)]:
    pass
m="2026-03"
r=fwd(m,0)
ps,ys,ss=r
order=np.argsort(-ps)
top10=[(ss[i],ys[i]) for i in order[:10]]
print(f"  {m} 预测 TOP10 的实际 20 日超额收益:")
for k,(s,y) in enumerate(top10,1): print(f"    {k:2d}. {s}  {y*100:+.2f}%")
print(f"  TOP10 均值={np.mean([y for _,y in top10])*100:+.2f}%   全池均值={ys.mean()*100:+.2f}%")
# 全池按实际收益排前10
best=np.argsort(-ys)[:10]
hit=len(set(ss[i] for i in order[:10]) & set(ss[i] for i in best))
print(f"  全池实际收益 TOP10 与预测 TOP10 的交集: {hit}/10")
print(f"  预测 TOP10 在全池实际收益中的平均排名: {np.mean([np.where(np.argsort(-ys)==i)[0][0] for i in order[:10]]):.0f} / {len(ys)}")
