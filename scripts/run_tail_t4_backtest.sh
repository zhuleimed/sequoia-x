#!/bin/bash
# 补最后一个格子：tail 抽样 + T2+T4 的完整回测（2026-09-28，待办 #2 重做）
#
# 为什么跑它
# ----------
# 9/27 的"抽样方式 A/B"只跑了三个变体（`run_sampling_ab_backtest.sh`）：
#   A  tail   T2-only   .purged_70m.json        → output/ab_tail_t2   ✅已有
#   B  random T2-only   .t2_70m_random.json     → output/ab_random_t2 ✅已有
#   C  random T2+T4     .t4_70m_random.json     → output/ab_random_t2t4 ✅已有
#   D  tail   T2+T4     .t4_70m_tail.json       → **本脚本补的就是它**
# 当时 tail+T4 的缓存不存在（注释里写"需另建 ~2h，已记为待办"= 待办 #10），
# 于是"tail vs random 该用哪个"这个决策**只在 T2-only 下做过**（结论：不切换）。
#
# 但 9/28 复算发现 **random 恰是 T4 发挥的前提**：
#   · random 口径 T4 IC20 +0.0423 (p=0.001)，强于 T2 +0.0292
#   · tail   口径 T4 被饿死（训练集只剩 ~1.8 个市场截面）IC20 +0.0078 (p=0.752)
#   · 端到端：random 下纯 T2 的 24 个配置全负，加 T4 后 24/24 全转正
# ⇒ 用 T2-only 选的"不切换"，**可能恰好选中了会饿死最强模型的那一档**。
# 补齐 D 格后即可做 2×2 受控对照（唯一变量 = 抽样方式 × 是否含 T4），把 #2 重做。
#
# 口径（与 A/B/C 三格**完全一致**，不得改动）：purge 25、lastday_agg、129 维、--period full
# 输出隔离：写 output/ab_tail_t2t4/，不覆盖既有基准
#
# 断点续跑：回测本身按月推进；重跑同命令会重头再来（单次 ~60min，可接受）
# 用法：nohup bash scripts/run_tail_t4_backtest.sh > logs/tail_t4_bt_wrapper.log 2>&1 &
set -u
PROJ=/public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x
PY=/home/zhulei/anaconda3/envs/zhulei_py312/bin/python
cd "$PROJ" || exit 1
PROG=logs/tail_t4_bt_progress.log
CACHE=output/backtest_v2/.t4_70m_tail.json
TAG=tail_t2t4

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$PROG"; }

# ── 1) 排队：等 tail+T4 缓存**产物**就绪（避免与构建抢 36 核）──
# ⚠️ 2026-09-29 教训（损失 20 小时，勿重蹈）：
#   本步骤原用 `while pgrep -f "run_t4_70m_tail.sh"`，**永久死等**。根因是 `pgrep -f` 匹配
#   **命令行任意位置**的子串 —— 任何命令行里含这 19 个字符的进程都会被当成"构建还在跑"，
#   包括**为它写的监听器**（`while pgrep -f "run_t4_70m_tail.sh"; do sleep; done`）
#   ⇒ 两个脚本互相匹配、互为对方的"存活证据"，双双死锁。构建 11:58 就结束了，接力却等到次日。
#   修法：**等产物，不等进程**（产物是唯一可靠信号）；另加最大等待上限防死循环。
MAX_WAIT_MIN=420   # 最多等 7 小时
WAITED=0
while true; do
  if [ -f "$CACHE" ]; then
    M=$("$PY" -c "import json;print(len(json.load(open('$CACHE'))))" 2>/dev/null || echo 0)
    if [ "$M" -ge 60 ]; then log "✅ 缓存产物已就绪（$M 个月），开始回测"; break; fi
  else
    M=0
  fi
  if [ "$WAITED" -ge "$MAX_WAIT_MIN" ]; then
    log "❌ 中止：等待产物超时 ${MAX_WAIT_MIN} 分钟（当前 ${M} 个月）"; exit 1
  fi
  log "⏳ 等待缓存产物就绪中（当前 ${M} 个月，已等 ${WAITED}min）…"
  sleep 120
  WAITED=$((WAITED + 2))
done

# ── 2) 前置校验：缓存必须存在且月份数足够 ──
if [ ! -f "$CACHE" ]; then log "❌ 中止：$CACHE 不存在"; exit 1; fi
N=$("$PY" -c "import json;print(len(json.load(open('$CACHE'))))" 2>/dev/null || echo 0)
log "缓存 $CACHE 有 $N 个月"
if [ "$N" -lt 60 ]; then log "❌ 中止：月数 $N < 60，构建可能未完成或大量失败"; exit 1; fi

# 关键校验：t4 字段不能全 0（否则跑出来的是"T2-only 却贴 T4 标签"）
NZ=$("$PY" -c "
import json;d=json.load(open('$CACHE'))
print(sum(1 for v in d.values() if v.get('t4') and any(x!=0 for x in v['t4'][:50])))")
log "t4 非全零的月数：$NZ / $N（应≈$N，否则是 T2-only 缓存被贴了 T4 标签）"
if [ "$NZ" -lt 60 ]; then log "❌ 中止：t4 有值的月数仅 $NZ，缓存不含真 T4"; exit 1; fi

# ── 3) 回测（配置与 A/B/C 三格逐字一致）──
# ⚠️ 必须固定 V4_SAMPLE_END_FIX=2026-09-24（2026-09-29 踩坑，勿删）
#   机理：run_shared_backtest 的缓存完整性判据是 `months_ok >= len(test_months)`，
#     而 test_months 是 2020-09~2026-06 共 **70** 个月，我们的缓存只有 **69** 个
#     （2020-09 因 purge 后 0 样本被跳过，是**固有**的，不是构建失败）
#     ⇒ 69 < 70 ⇒ 永远判定"缓存不完整"，落入**重建分支**。
#   重建分支第一步要 load_full_dataset，而数据集缓存键含 `sample_end`（动态数据截止日，
#     见 labels.py 的 key_data），`resolve_sample_end` 取 **DB 最后交易日** ⇒ **每天前移**
#     ⇒ 隔天就找不到数据集缓存（2026-09-29 实测：想要 7440fe9883a5，实际只有
#     3ccb10244903@s2026-09-24）→ 回退 88 维 → 仍缺 → FileNotFoundError 秒挂。
#   固定成构建本缓存时用的 2026-09-24 后，重建分支会**命中数据集缓存并秒级空转**
#     （9/27 那三格也是这样通过的：16:31:55 重建 → 16:31:56 加载完成 → 直接回测）
#     ⇒ 对回测结果**无影响**，只是绕过这个陷阱。
#   两个构建日志已核对：tail 与 random 都用了 sample_end=2026-09-24 / 数据集 3ccb10244903
#     ⇒ 本对照的数据源完全一致，可比。
log "回测 $TAG（cache=$CACHE，固定 sample_end=2026-09-24 以命中数据集缓存）…约 60 分钟"
V4_SAMPLE_END_FIX=2026-09-24 "$PY" -u scripts/run_shared_backtest.py --all --period full \
  --cache "$CACHE" --output-dir "output/ab_$TAG" > "logs/ab_$TAG.log" 2>&1
log "  $TAG 结束 exit=$? ｜ 结果: output/ab_$TAG/summary_all.csv"

# ── 4) 2×2 受控对照汇总 ──
log "── 2×2 对照（M4 风控 × TOP_N=10，全周期 69 月）──"
"$PY" - <<'PYEOF' 2>&1 | tee -a "$PROG"
import pandas as pd, pathlib
cells = [("tail","T2-only","tail_t2"), ("tail","T2+T4","tail_t2t4"),
         ("random","T2-only","random_t2"), ("random","T2+T4","random_t2t4")]
rows=[]
for samp, model, tag in cells:
    p=pathlib.Path(f"output/ab_{tag}/summary_all.csv")
    if not p.exists(): rows.append({"抽样":samp,"模型":model,"状态":"无输出"}); continue
    d=pd.read_csv(p)
    m=d[(d["风控模式"]=="M4")&(d["TOP_N"]==10)]
    if len(m)==0: rows.append({"抽样":samp,"模型":model,"状态":"无 M4/TOPN10 行"}); continue
    r=m.iloc[0]
    rows.append({"抽样":samp,"模型":model,"月数":r.get("月数"),
                 "总收益%":round(r["总收益率"]*100,1),"夏普":round(r["夏普比率"],2),
                 "回撤%":round(r["最大回撤"]*100,1)})
print(pd.DataFrame(rows).to_string(index=False))
print("\n读法：① 同一模型下对比 tail vs random（抽样效应）")
print("      ② 同一样本下对比 T2-only vs T2+T4（T4 增量）")
print("      ③ 若 random+T2+T4 明显优于 tail+T2+T4 ⇒ 待办 #2 应改判为「切 random」")
PYEOF
log "═══ 完成 ═══"
