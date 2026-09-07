"""③ 去"前视"隔离：canonical 缓存 vs 去前视 V4 缓存，A/B 两臂对比。

用户已确认停牌/涨跌停/成本约束在回测里到位；问放大器(超 IC) 从哪来。本脚本用两个现成 prediction_cache
(canonical 73月 与 prediction_cache_v4_nolook.json 去前视 70月) 跑同一引擎、M4/TOP10/500k/同窗口，隔离：
  - lookahead 是否真实抬高头部(尤其 extreme 月)？把"canonical 拿到、去前视拿不到"的差距量化成 CAGR/月超额。
  - 去前视后保留的 alpha vs IC(0.011-0.016) 是否量级更贴近"真实弱IC"。
用途：给出"超 IC 的放大里，前视占多少、需靠流动性/容量压多少"的可辩护分层。
运行: py312 experiments/nolook_vs_canonical.py
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
_REPO=Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path: sys.path.insert(0,str(_REPO))
os.environ.pop("KMP_AFFINITY",None)
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
from sequoia_x.core.config import Settings
from sequoia_x.data.engine import DataEngine
from sequoia_x.model_selection_v2.backtest.monthly_engine import MonthlyBacktestEngine
from sequoia_x.model_selection_v2.config import get_config

OUT=Path("output/backtest_v2")
def run(cachefp, pol, s="2020-09", e="2026-06"):
    cache=json.loads((OUT/cachefp).read_text())
    bt=MonthlyBacktestEngine(cfg=get_config(),engine=DataEngine(Settings()),top_n=10,risk_mode="M4",
        initial_capital=500_000.0,use_real_t4=True,prediction_cache=cache,fusion_method="pred_std",
        keep_survivors=False,intra_exit_policy=pol)
    t0=time.time(); m=bt.run(s,e); dt=time.time()-t0
    sells=[t for t in bt.trades if t.trade_type=="sell"]
    # real monthly returns (daily path total) also monthly win
    return dict(cache=cachefp.split('/')[-1], policy=pol,
        total=m["total_return"],annual=m["annual_return"],sharpe=m["sharpe"],mdd=m["max_drawdown"],
        win_month=m["win_rate"], trades=m["n_trades"], rule_sell=sum(1 for t in sells if "清仓" not in (t.reason or "")),
        elapsed=round(dt,1), monthly=m["monthly_returns"], labels=m.get("_monthly_labels"))
def main():
    print("=== canonical vs 去前视 V4, A(现状)/B(纯持有) — 隔离前视放大 ===",flush=True)
    res=[]
    combos=[("prediction_cache.json","all"),("prediction_cache.json","none"),
            ("prediction_cache_v4_nolook.json","all"),("prediction_cache_v4_nolook.json","none")]
    for cf,pol in combos:
        r=run(cf,pol); res.append(r)
        print(f"[{r['cache']} {pol}] tot={r['total']*100:+8.1f}% ann={r['annual']*100:+6.1f}% "
              f"sharpe={r['sharpe']:.2f} mdd={r['mdd']*100:+.1f}% win月={r['win_month']*100:.0f}% "
              f"trades={r['trades']} ruleSell={r['rule_sell']} ({r['elapsed']}s)",flush=True)
    # deltas
    def g(c,p): return next(x for x in res if x['cache']==c and x['policy']==p)
    cA=g('prediction_cache.json','all'); cB=g('prediction_cache.json','none')
    nA=g('prediction_cache_v4_nolook.json','all'); nB=g('prediction_cache_v4_nolook.json','none')
    print("\n=== 前视隔离 (canonical − 去前视) ===")
    for lab,a,n in [("A现状",cA,nA),("B纯持有",cB,nB)]:
        dT=a['total']-n['total']; dC=(a['annual']-n['annual'])*100
        print(f"{lab}: 总收益差 {dT*100:+,.0f}pp | 年化差 {dC:+,.1f}pp | 夏普 {a['sharpe']-n['sharpe']:+.2f}")
    Path(OUT/"nolook_vs_canonical.json").write_text(json.dumps(res,indent=2,ensure_ascii=False,default=str))
    print("已存 nolook_vs_canonical.json")

if __name__=="__main__": main()
