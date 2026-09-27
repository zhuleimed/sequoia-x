#!/bin/bash
# 抽样模式 A/B 回测（2026-09-27，待办第 2 项的"证据"部分）
#
# 为什么跑它
# ----------
# 今天测出"random 抽样让 T2 原始 IC 翻倍（0.0152→0.0292）"——但 **IC 不等于钱**：
# 换手、成本、回撤、集中度都可能把 IC 优势吃掉。所以把两种口径的缓存各跑一遍**完整回测**。
#
# 三个变体（都用 --all --period full = 2020-09~2026-06 全 69 个月）
#   A  tail   T2-only   .purged_70m.json       ← 现状口径（已有，不动）
#   B  random T2-only   .t2_70m_random.json    ← 本脚本新建；**与 A 只差抽样规则** ⇒ 干净 A/B
#   C  random T2+T4     .t4_70m_random.json    ← 生产信号形态；与 B 比可看 **T4 的增量**
#
# 口径（**不得改动**）：purge 25、lastday_agg、129 维、12 workers（与基线一致）
#   ⚠️ tail+T4 的组合今天没有（需另建 ~2h），已记为待办。
#
# 输出隔离：每个变体写自己的 --output-dir（避免覆盖 output/backtest_v2/summary_all.csv 既有基准）
#
# 用法：nohup bash scripts/run_sampling_ab_backtest.sh > logs/sampling_ab_wrapper.log 2>&1 &
set -u
PROJ=/public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x
PY=/home/zhulei/anaconda3/envs/zhulei_py312/bin/python
cd "$PROJ" || exit 1
PROG=logs/sampling_ab_progress.log
T2R=output/backtest_v2/.t2_70m_random.json

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$PROG"; }

# ── Step 1: 建 random + T2-only 缓存（断点续跑：重跑同命令即跳过已完成月）──
if [ -f "$T2R" ] && [ "$("$PY" -c "import json;print(len(json.load(open('$T2R'))))" 2>/dev/null || echo 0)" -ge 60 ]; then
  log "Step1 跳过：$T2R 已有 $("$PY" -c "import json;print(len(json.load(open('$T2R'))))") 个月"
else
  log "Step1 开始：建 random + T2-only 70 月缓存（~1h）"
  env -u KMP_AFFINITY V4_BT_WORKERS=12 V2_TRAIN_SAMPLE_MODE=random V2_TRAIN_PURGE_DAYS=25 \
    "$PY" -u scripts/build_prediction_cache.py \
    --start-month 2020-09 --end-month 2026-06 --skip-t4 \
    --feature-view lastday_agg --output "$T2R" > logs/t2_70m_random.log 2>&1
  RC=$?
  N=$("$PY" -c "import json;print(len(json.load(open('$T2R'))))" 2>/dev/null || echo 0)
  log "Step1 结束 exit=$RC（**新增的退出码修复**：有失败月应为 2）｜ $N/70 个月"
  if [ "$N" -lt 60 ]; then log "❌ 月数不足，中止"; exit 1; fi
fi

# ── Step 2: 三个变体各跑完整回测 ──
run_bt(){
  local tag="$1" cache="$2"
  log "回测 $tag（cache=$cache）..."
  "$PY" -u scripts/run_shared_backtest.py --all --period full \
    --cache "$cache" --output-dir "output/ab_$tag" > "logs/ab_$tag.log" 2>&1
  log "  $tag 结束 exit=$? ｜ 结果: output/ab_$tag/summary_all.csv"
}
run_bt tail_t2     output/backtest_v2/.purged_70m.json
run_bt random_t2   "$T2R"
run_bt random_t2t4 output/backtest_v2/.t4_70m_random.json

# ── Step 3: 汇总 M4 / TOP10 三行对照 ──
log "── 汇总（M4 风控 × TOP_N=10）──"
"$PY" - <<'PYEOF' 2>&1 | tee -a "$PROG"
import pandas as pd, pathlib
rows=[]
for tag in ["tail_t2","random_t2","random_t2t4"]:
    p=pathlib.Path(f"output/ab_{tag}/summary_all.csv")
    if not p.exists(): rows.append({"变体":tag,"状态":"无输出"}); continue
    d=pd.read_csv(p)
    m=d[(d["风控模式"]=="M4")&(d["TOP_N"]==10)]
    if len(m)==0: rows.append({"变体":tag,"状态":"无M4/TOP10行"}); continue
    r=m.iloc[0]
    rows.append({"变体":tag,"月数":r.get("月数"),"总收益%":round(r["总收益率"]*100,1),
                 "年化%":round(r["年化收益率"]*100,1),"夏普":round(r["夏普比率"],2),
                 "最大回撤%":round(r["最大回撤"]*100,1),"月胜率":round(r.get("月胜率",float("nan"))*100,1)})
print(pd.DataFrame(rows).to_string(index=False))
PYEOF
log "═══ 全部完成 ═══"
