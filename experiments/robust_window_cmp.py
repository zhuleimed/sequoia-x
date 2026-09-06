"""分时段稳健性校验：C(只留硬止损)/B(纯持有) vs A(现状全规则) 是否在去掉早期牛段后仍占优。

背景(2026-09-07)：用户对 70 月 A/B 的 +9553%/+7865% 存疑。疑点之一是用例切片用 monthly_returns
乘积会低估(日常路径 total_return 才是真值+已被独立 runner 交叉验证 ~+2077%)。故此处**直接调用
MonthlyBacktestEngine.run(start,end)** 取各窗口的 daily-path 权威 total_return/sharpe/max_drawdown。

对每个窗口(全70月/去2020-21/近4年/近3年/2025+)，跑 A/B/C 三臂(同一 prediction_cache / M4 / TOP10 / 模式A)，
得每日路径下的 total、年化、日夏普、最大回撤、月胜率、规则卖 vs 月末清仓。以此说明：
  - 即使排除 2020-21 大牛段暴露红利，C(与B) 是否仍显著优于 A。
运行(铁律六 py312)：
  /home/zhulei/anaconda3/envs/zhulei_py312/bin/python experiments/robust_window_cmp.py
输出 output/backtest_v2/robust_window_cmp.json + 终端表。
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path: sys.path.insert(0, str(_REPO))
os.environ.pop("KMP_AFFINITY", None)
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import numpy as np
from sequoia_x.core.config import Settings
from sequoia_x.data.engine import DataEngine
from sequoia_x.model_selection_v2.backtest.monthly_engine import MonthlyBacktestEngine
from sequoia_x.model_selection_v2.config import get_config
OUT=Path("output/backtest_v2/prediction_cache.json")
CACHE=json.loads(OUT.read_text())
POLICIES=["all","none","hard_stop_only"]
LABEL={"all":"A现状:月内全规则","none":"B纯持有:月末清仓","hard_stop_only":"C只留硬止损-8%"}
WIN=[("全70月","2020-09","2026-06"),("去2020-21","2022-01","2026-06"),
     ("近4年","2023-01","2026-06"),("近3年","2024-01","2026-06"),("2025起","2025-01","2026-06")]

def run_one(policy,start,end):
    bt=MonthlyBacktestEngine(cfg=get_config(),engine=DataEngine(Settings()),top_n=10,
        risk_mode="M4",initial_capital=500_000.0,use_real_t4=True,prediction_cache=CACHE,
        fusion_method="pred_std",keep_survivors=False,intra_exit_policy=policy)
    t0=time.time(); m=bt.run(start,end); dt=time.time()-t0
    sells=[t for t in bt.trades if t.trade_type=="sell"]
    neom=sum(1 for t in sells if "清仓" in (t.reason or ""))
    # 月末持均
    dmb={}
    for rec in bt.daily_records: dmb.setdefault(rec["date"][:7],[]).append(rec)
    mep=np.mean([recs[-1]["positions"] for recs in dmb.values()])
    keep=["total_return","annual_return","sharpe","max_drawdown","win_rate","n_months","n_trades","n_buys","n_sells"]
    r={k:m[k] for k in keep if k in m}
    r.update(final_value=m.get("final_value"),policy=policy,label=LABEL[policy],
             n_rule_sells=len(sells)-neom,n_eom_sells=neom,month_end_pos_avg=round(float(mep),2),
             elapsed_min=round(dt/60,2))
    return r

def main():
    print("=== 分时段稳健性校验: B/C vs A (daily-path 权威指标) ===")
    allres=[]
    for wname,ws,we in WIN:
        for pol in POLICIES:
            r=run_one(pol,ws,we)
            allres.append({"window":wname,**r})
            print(f"[{wname} {ws}~{we}] {r['label']:<16} tot={r['total_return']*100:+8.1f}% "
                  f"年化={r['annual_return']*100:+6.1f}% 夏普={r['sharpe']:.2f} 回撤={r['max_drawdown']*100:+.1f}% "
                  f"月胜率={r['win_rate']*100:.0f}% 规则卖={r['n_rule_sells']} 月末持均={r['month_end_pos_avg']}")
    # per-window structured table
    print("\n=== 汇总: 各窗口 A 合计 = C-A / B-A ===")
    for wn in [w[0] for w in WIN]:
        winrats=[r for r in allres if r['window']==wn]
        A={x['policy']:x for x in winrats}['all']
        for p in ['none','hard_stop_only']:
            P={x['policy']:x for x in winrats}[p]
            print(f"{wn:<10} {p:<6} tot={P['total_return']*100:+7.1f}% vs A={A['total_return']*100:+7.1f}%  Δ={ (P['total_return']-A['total_return'])*100:+8.1f}pp | "
                  f"C/A夏普: A={A['sharpe']} {p.split(':')[0]}={P['sharpe']}")
    out=Path("output/backtest_v2/robust_window_cmp.json"); out.write_text(json.dumps(allres,indent=2,ensure_ascii=False,default=str))
    print(f"\n已存 {out}")

if __name__=="__main__": main()
