#!/bin/bash
# 70 月（2020-09~2026-06）**tail 抽样 + 含 T4** 的 purge 缓存（2026-09-28）
#
# 为什么建它（待办 #10）
# --------------------
# 现有 70 月含 T4 的缓存**只有 random 版**（`.t4_70m_random.json`），而**生产用的是 tail**
# （`V2_TRAIN_SAMPLE_MODE` 默认值）。所以"**生产现状口径下 T4 到底行不行**"一直是未测项。
#
# 这条直接决定待办 #2（tail vs random 该用哪个）能否正确回答：
#   9/27 的决策「不切换抽样」是**用 T2-only 得出的**（tail+T2 +23.9% vs random+T2 −9.2%）。
#   但 9/28 复算发现 **random 恰是 T4 发挥的前提**：
#     · random 口径：T4 IC20 +0.0423 (p=0.001) 强于 T2 +0.0292
#     · tail  口径：T4 IC20 +0.0078 (p=0.752) —— tail 只取最后 5000 条 ≈ **1.8 个市场截面**，
#       LSTM 靠序列学习，被饿死（同一 bug 对 LGBM 是"可利用的泄漏"，对 LSTM 是"断粮"）
#   ⇒ 用 T2-only 选的"不切换"，很可能**恰好选中了会饿死最强模型的那一档**。
#   本脚本产出 tail+含T4 的 70 月缓存后，即可与 `.t4_70m_random.json` 做**同窗口/同样本量/
#   同 purge、唯一差别=抽样方式**的受控对照，把 #2 重做一遍。
#
# 口径：
#   2020-09 ~ 2026-06 ｜ purge 25 ｜ **tail 抽样（默认）** ｜ lastday_agg ｜ 含 T4
#
# ⚠️ checkpoint 陷阱（必读，待办 #11 的实例）
# ------------------------------------------
# `t4_checkpoint_cache_{月}.keras` **只用月份做 key**，`deep_lstm.py:574` 只要文件存在就
# `load_model` 并 return（断点续跑机制）⇒ **别的口径的模型会被静默复用**。
# 当前 `data/models/v2_selection/` 下 2020-10~2026-06 共 69 个 checkpoint **全部是
# 2026-09-27 那次 `random` 口径跑出来的** ⇒ 不清掉的话，本脚本会**静默产出 random 模型的
# 预测却贴上 tail 的标签**，整个对照实验失效且不报错。
# 因此本脚本**启动即自检**：若发现任何 2020-01~2026-06 区间的 checkpoint 就直接中止，
# 并提示先执行归档（归档命令见下）。**2026-09 那个是生产模型，不在检查范围内，勿动。**
#
#   归档（启动前手动执行一次，可回滚）。⚠️ **必须排除生产模型 2026-09**（用 grep -v 而不是
#   宽 glob —— `202[0-6]-*` 会把 2026-09 也选中）：
#     CK=data/models/v2_selection; BAK=$CK/_t4_ckpt_random70m_20260928
#     mkdir -p "$BAK"
#     ls "$CK"/t4_checkpoint_cache_*.keras | grep -vE '2026-09\.keras$' \
#       | xargs -r -I{} mv {} "$BAK"/
#
# 断点续跑：重跑同命令即跳过已完成月（month_*.json）
# 用法：
#   nohup bash scripts/run_t4_70m_tail.sh > logs/t4_70m_tail_wrapper.log 2>&1 &
set -u
PROJ=/public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x
PY=/home/zhulei/anaconda3/envs/zhulei_py312/bin/python
cd "$PROJ" || exit 1
LOG=logs/t4_70m_tail.log
OUT=output/backtest_v2/.t4_70m_tail.json
TMP=output/backtest_v2/.cache_tmp_.t4_70m_tail
PROG=logs/t4_70m_tail_progress.log
CKPT=data/models/v2_selection
TOTAL=70

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$PROG"; }

# 与 random 版同线程配置（4 workers × 8 线程 ≤ 36 核），保证两缓存唯一差别=抽样方式
export V4_BT_WORKERS=4
export TF_NUM_INTRAOP_THREADS=8
export TF_NUM_INTEROP_THREADS=4
export OMP_NUM_THREADS=8
export V2_TRAIN_SAMPLE_MODE=tail
export V2_TRAIN_PURGE_DAYS=25

# ── 前置自检（仅首次启动）：本次要建的 69 个月不得残留任何 checkpoint ──
#   本次建 2020-09~2026-06（2020-09 因 purge 后 0 样本被跳过，实需 2020-10~2026-06 共 69 个）。
#   判据用「全部 checkpoint **排除生产月 2026-09**」——
#   写成宽 glob `202[0-6]-*` 会误把 2026-09 生产模型算进来（实测数出 70 而非 69），
#   且会让归档命令连生产模型一起搬走，属危险指令。勿改成宽 glob。
if [ ! -f "$LOG" ]; then
  BAD=$(ls "$CKPT"/t4_checkpoint_cache_*.keras 2>/dev/null | grep -vE '2026-09\.keras$' | wc -l)
  if [ "$BAD" -gt 0 ]; then
    log "❌ 中止：$CKPT 下有 $BAD 个历史月份的 t4_checkpoint_cache_*.keras（已排除生产月 2026-09）"
    log "   它们若不是本次 tail 口径建的，会被 deep_lstm.py:574 静默复用 → 对照实验失效。"
    log "   请先归档（可回滚；注意排除生产模型 2026-09）："
    log "     CK=$CKPT; BAK=\$CK/_t4_ckpt_random70m_20260928; mkdir -p \"\$BAK\""
    log "     ls \"\$CK\"/t4_checkpoint_cache_*.keras | grep -vE '2026-09\\.keras\$' | xargs -r -I{} mv {} \"\$BAK\"/"
    exit 1
  fi
fi

# ── 独立监控：每 15 分钟一次进度快照 ──
(
  while true; do
    sleep 900
    D=$(ls "$TMP"/month_*.json 2>/dev/null | wc -l)
    E=$(ls "$TMP"/month_*.error 2>/dev/null | wc -l)
    echo "[$(date '+%F %T')] 进度 $D/$TOTAL 个月，失败 $E ｜ 日志尾: $(tail -1 "$LOG" 2>/dev/null | cut -c1-110)" >> "$PROG"
    if [ "$D" -ge $TOTAL ]; then break; fi
  done
) &
MON=$!

log "═══ 70 月 tail+T4 构建启动（2020-09~2026-06，purge 25，**tail**，lastday_agg，4 workers × 8 线程）═══"
log "  目的：补齐"生产现状口径（tail）下 T4 的表现"，用于重做待办 #2（tail vs random）"
# 核数用 getconf 而非 nproc —— nproc 会读 OMP_NUM_THREADS，这里 OMP=8 会误显示成"8 核"
log "  线程配置：TF_INTRAOP=$TF_NUM_INTRAOP_THREADS TF_INTEROP=$TF_NUM_INTEROP_THREADS OMP=$OMP_NUM_THREADS（机器 $(getconf _NPROCESSORS_ONLN) 核）"
T0=$(date +%s)
# 只清 KMP_AFFINITY（会锁核，见项目铁律），**保留**显式设置的线程数
env -u KMP_AFFINITY \
    TF_NUM_INTRAOP_THREADS="$TF_NUM_INTRAOP_THREADS" \
    TF_NUM_INTEROP_THREADS="$TF_NUM_INTEROP_THREADS" \
    OMP_NUM_THREADS="$OMP_NUM_THREADS" \
    V4_BT_WORKERS="$V4_BT_WORKERS" \
    V2_TRAIN_SAMPLE_MODE=tail V2_TRAIN_PURGE_DAYS=25 \
    "$PY" -u scripts/build_prediction_cache.py \
  --start-month 2020-09 --end-month 2026-06 \
  --feature-view lastday_agg \
  --output "$OUT" >> "$LOG" 2>&1
RC=$?
T1=$(date +%s)
kill $MON 2>/dev/null

# tmp 目录在合并后会被清空，故从**输出文件**数月份（原写法恒显示 0/70，会误读成"零完成"）
D=$("$PY" -c "import json;print(len(json.load(open('$OUT'))))" 2>/dev/null || echo 0)
E=$(ls "$TMP"/month_*.error 2>/dev/null | wc -l)
log "═══ 结束 exit=$RC ｜ 耗时 $(( (T1-T0)/60 ))min ｜ 输出缓存 $D/$TOTAL 个月 ｜ 失败月 $E ═══"
log "   注：2020-09 因 purge 后 0 训练样本必然跳过 ⇒ 上限为 69 个月；exit=0 不代表无失败月"

# ── 自动分析：T2 / T4 各一套（多周期 + 中性化）──
if [ -f "$OUT" ]; then
  log "── 自动分析 ──"
  "$PY" -u experiments/attribution_4x/ic_by_horizon.py --cache="$OUT" --tag=t2_70m_tail --field=t2 >> "$PROG" 2>&1
  "$PY" -u experiments/attribution_4x/ic_by_horizon.py --cache="$OUT" --tag=t4_70m_tail --field=t4 >> "$PROG" 2>&1
  "$PY" -u experiments/attribution_4x/t4_ic.py --cache="$OUT" --tag=70m_tail >> "$PROG" 2>&1
  log "  分析完成 → experiments/attribution_4x/out/{ic_by_horizon_t2_70m_tail,ic_by_horizon_t4_70m_tail,t4_ic_70m_tail}.csv"
fi
