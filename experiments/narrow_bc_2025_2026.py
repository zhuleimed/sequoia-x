"""2025→2026H1 细粒度 B vs C(带 A 参考)：判定近期该保持 C(护栏)还是切 B(纯持有封顶)。

延续 robust_window_cmp.py 的口径(authoritative daily-path total/sharpe/maxdd)。
窗：2025-01起、2025-07起、2026-01起(2026只跑上半年)。每窗 3 臂 all/none/hard_stop_only，cache 极快。
输出 output/backtest_v2/narrow_bc_2025_2026.json + 终端表。
运行：/home/zhulei/anaconda3/envs/zhulei_py312/bin/python experiments/narrow_bc_2025_2026.py
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path: sys.path.insert(0, str(_REPO))
os.environ.pop("KMP_AFFINITY", None)
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v]="1"
import numpy as np
from sequoia_x.core.config import Settings
from sequoia_x.data.engine import DataEngine
from sequoia_x.model_selection_v2.backtest.monthly_engine import MonthlyBacktestEngine
from sequoia_x.model_selection_v2.config import get_config
CACHE=json.loads(Path("output/backtest_v2/prediction_cache.json").read_text())
POLS=["all","none","hard_stop_only"]
LABEL={"all":"A现状:月内全规则","none":"B纯持有:月末清仓","hard_stop_only":"C只留硬止损-8%"}
WIN=[("近12月 2025-07~26-06","2025-07","2026-06"),
     ("近18月 2025-01~26-06","2025-01","2026-06"),
     ("2026 H1","2026-01","2026-06")]

def main():
    out=[]
    print("窗 | A现状    B纯持有   C硬止损   (相对B: C-B 收益/夏普)")
    for wname,ws,we in WIN:
        res={}
        for pol in POLS:
            bt=MonthlyBacktestEngine(cfg=get_config(),engine=DataEngine(Settings()),top_n=10,risk_mode="M4",
                initial_capital=500_000.0,use_real_t4=True,prediction_cache=CACHE,fusion_method="pred_std",
                keep_survivors=False,intra_exit_policy=pol)
            t0=time.time(); m=bt.run(ws,we); dt=time.time()-t0
            sells=[t for t in bt.trades if t.trade_type=="sell"]
            neom=sum(1 for t in sells if "清仓" in (t.reason or ""))
            dmb={}
            for rec in bt.daily_records: dmb.setdefault(rec["date"][:7],[]).append(rec)
            mep=round(float(np.mean([v[-1]["positions"] for v in dmb.values()])),2)
            res[pol]={"总收益":m["total_return"]*100,"年化":m["annual_return"]*100,
                      "夏普":m["sharpe"],"回撤":m["max_drawdown"]*100,"月胜率":m["win_rate"]*100,
                      "规则卖":len(sells)-neom,"月末持均":mep,"年":ws[:4]}
            print(f"[{wname} {pol}] tot={res[pol]['总收益']:+.1f}% 夏普={res[pol]['夏普']:.2f} "
                  f"回撤={res[pol]['回撤']:+.1f}% 规则卖={res[pol]['规则卖']} ({dt:.0f}s)",flush=True)
        A,B,C=res["all"],res["none"],res["hard_stop_only"]
        cb=(C["总收益"]-B["总收益"]); cs=C["夏普"]-B["夏普"]
        print(f">>> {wname}: C-B 总收益 {cb:+.1f}pp | 夏普 {cs:+.2f} | "
              f"(A% {A['总收益']:+.0f} B% {B['总收益']:+.0f} C% {C['总收益']:+.0f})")
        out.append({"window":wname,"arms":res,"C_minus_B_pp":round(cb,1),"C_minus_B_sharpe":round(cs,2)})
    Path("output/backtest_v2/narrow_bc_2025_2026.json").write_text(json.dumps(out,indent=2,ensure_ascii=False))
    print("已存 narrow_bc_2025_2026.json")

if __name__=="__main__": main()
