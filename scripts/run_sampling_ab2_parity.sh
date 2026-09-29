#!/bin/bash
# 抽样×T4 的 2×2 受控对照（2026-09-29，**同一套政策**版）
#
# 为什么要重跑（2026-09-29 发现的混淆项）
# --------------------------------------
# 9/27 那三格（ab_tail_t2 / ab_random_t2 / ab_random_t2t4）跑在 **15:33~18:33**，
# 而回测引擎的默认政策是当晚**才改的**：
#   20:11 commit 6dd7112  默认 → E1（intra_exit_policy="hard_stop_only" + hard_stop_pct=-0.12）
#   21:45 commit 66e551b  月末清仓价默认 open → close
# ⇒ 旧三格用的是「月内全规则 + 开盘价清仓 + −8%」，9/29 新跑的 tail+T2+T4 用的是
#   「只留硬止损 −12% + 收盘价清仓」——**不是同一套政策，不可比**。
#   （实证：同一格 tail+T2+T4 在新政策下 M4/TOP10 = +345.4%/夏普1.06；旧政策的
#     tail_t2 只有 +23.9%/0.13 —— 差距主要来自政策，不能归给 T4。）
#
# 本脚本把**全部 4 格**统一到当前（= 生产对齐的）政策下重跑，得到干净的 2×2：
#             T2-only          T2+T4
#   tail      ab2_tail_t2      ab_tail_t2t4（9/29 已跑，政策一致，复用）
#   random    ab2_random_t2    ab2_random_t2t4
# 唯一变量 = 抽样方式 × 是否含 T4。
#
# 口径（不得改动）：purge 25、lastday_agg、129 维、--all --period full、引擎全用当前默认
# ⚠️ 必须固定 V4_SAMPLE_END_FIX=2026-09-24：否则重建分支找不到数据集缓存而秒挂
#    （详见 run_tail_t4_backtest.sh 里的完整说明）
#
# 用法：nohup bash scripts/run_sampling_ab2_parity.sh > logs/ab2_parity_wrapper.log 2>&1 &
set -u
PROJ=/public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x
PY=/home/zhulei/anaconda3/envs/zhulei_py312/bin/python
cd "$PROJ" || exit 1
PROG=logs/ab2_parity_progress.log
SE=2026-09-24

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$PROG"; }

# ── 1) 排队：等 9/29 那次 tail_t2t4 跑完（避免与它抢资源；它政策一致，直接复用）──
# 用**产物**判断而非 pgrep（2026-09-29 教训：pgrep -f 会匹配到含同名字符串的其它进程，永久死等）
WO=0
while true; do
  if [ -f output/ab_tail_t2t4/summary_all.csv ]; then log "✅ tail_t2t4 已完成（政策一致，直接复用）"; break; fi
  if ! ps -eo args | grep -q "[r]un_shared_backtest.py"; then
    log "⚠️ tail_t2t4 未产出且无回测进程在跑"; break
  fi
  if [ "$WO" -ge 180 ]; then log "❌ 等待超时"; exit 1; fi
  log "⏳ 等待 tail_t2t4 完成中（已等 ${WO}min）…"
  sleep 120; WO=$((WO+2))
done

# ── 2) 并行跑三格（183GB 内存 / 36 核，实测可并行）──
run_bt(){
  local tag="$1" cache="$2"
  log "  ▶ $tag 启动（cache=$cache）"
  V4_SAMPLE_END_FIX=$SE "$PY" -u scripts/run_shared_backtest.py --all --period full \
    --cache "$cache" --output-dir "output/$tag" > "logs/$tag.log" 2>&1
  log "  ■ $tag 结束 exit=$?"
}
log "═══ 2×2 受控对照启动（统一到当前政策；固定 sample_end=$SE）═══"
run_bt ab2_tail_t2      output/backtest_v2/.purged_70m.json      &
run_bt ab2_random_t2    output/backtest_v2/.t2_70m_random.json   &
run_bt ab2_random_t2t4  output/backtest_v2/.t4_70m_random.json   &
wait
log "═══ 三格全部结束 ═══"

# ── 3) 2×2 汇总 ──
"$PY" - <<'PYEOF' 2>&1 | tee -a "$PROG"
import pandas as pd, pathlib
cells=[("tail","T2-only","ab2_tail_t2"),("tail","T2+T4","ab_tail_t2t4"),
       ("random","T2-only","ab2_random_t2"),("random","T2+T4","ab2_random_t2t4")]
rows=[]
for s,m,tag in cells:
    p=pathlib.Path(f"output/{tag}/summary_all.csv")
    if not p.exists(): rows.append({"抽样":s,"模型":m,"状态":"无输出"}); continue
    d=pd.read_csv(p); r=d[(d["风控模式"]=="M4")&(d["TOP_N"]==10)]
    if len(r)==0: rows.append({"抽样":s,"模型":m,"状态":"无M4/TOPN10"}); continue
    r=r.iloc[0]
    rows.append({"抽样":s,"模型":m,"月数":r["月数"],"总收益%":round(r["总收益率"]*100,1),
                 "夏普":round(r["夏普比率"],2),"回撤%":round(r["最大回撤"]*100,1),
                 "月胜率%":round(r["月胜率"]*100,1),"笔数":int(r["交易笔数"])})
print(pd.DataFrame(rows).to_string(index=False))
print("\n⚠️ 这只是 24 组里的 M4/TOP10 一行；判稳须看全部 24 组一致性 + 剔除最好 3 个月的稳健性")
PYEOF
log "═══ 完成 ═══"
