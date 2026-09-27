#!/bin/bash
# 审计三个最小证伪实验（2026-09-27 夜，无人值守；项目铁律：nohup + 断点续跑 + 详尽日志）
#
#   (c) 风格匹配对照      experiments/attribution_4x/style_matched_control.py
#                         判定"+1.6%/月"是能力还是风格暴露（同行业×成交额×波动桶抽对照）
#   (a) 月末清仓价 A/B    experiments/compare_eom_price.py
#                         回测用末日**开盘**价 vs 生产用**收盘**价（差额 = +0.41%/月）
#   (b) 组合层中性化      experiments/neutralized_portfolio.py
#                         把预测分数残差化后选股，看"中性化 IC"能否变成钱（裁决待办 #3）
#
# 断点续跑：三个脚本各自"结果已存在即跳过"；崩溃后**重跑本命令**即可从断点继续。
# 用法：
#   nohup bash scripts/run_audit_experiments.sh > logs/audit_experiments_wrapper.log 2>&1 &
set -u
PROJ=/public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x
PY=/home/zhulei/anaconda3/envs/zhulei_py312/bin/python
cd "$PROJ" || exit 1
PROG=logs/audit_experiments_progress.log
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$PROG"; }

log "═══ 审计实验 (a)(b) 启动（(c) 已在别处运行，先等它）═══"

# (c) 用的是同一套引擎 + 全池特征，并发会抢核；先等它结束
while pgrep -f "style_matched_control.py" > /dev/null 2>&1; do
  log "⏳ 等待实验 (c) 结束…"; sleep 120
done
log "实验 (c) 已结束（结果见 logs/style_match.log 与 out/style_matched_control.json）"

# ── (a) 月末清仓价 ──
log "── (a) 月末清仓价 A/B ──"
env -u KMP_AFFINITY "$PY" -u experiments/compare_eom_price.py >> logs/eom_price_ab.log 2>&1
log "  (a) exit=$? ｜ 产物 experiments/attribution_4x/out/eom_price_ab.json"

# ── (b) 组合层中性化 ──
log "── (b) 组合层中性化 ──"
env -u KMP_AFFINITY "$PY" -u experiments/neutralized_portfolio.py >> logs/neutralized_portfolio.log 2>&1
log "  (b) exit=$? ｜ 产物 experiments/attribution_4x/out/neutralized_portfolio.json"

# ── 汇总（供明早一眼看结论）──
log "═══ 全部结束，汇总 ═══"
{
  echo "==================== 三项审计实验汇总 $(date '+%F %T') ===================="
  echo; echo "### (c) 风格匹配对照 —— 能力 vs 风格"
  grep -E "【基准】|【风格匹配对照】|对照组月均|模型 − 风格匹配|配对 p|候选回退|【判读】" logs/style_match.log 2>/dev/null | sed 's/^\s*//'
  echo; echo "### (a) 月末清仓价 open vs close"
  grep -E "^【|^  [A-Za-z0-9]" logs/eom_price_ab.log 2>/dev/null | tail -20
  echo; echo "### (b) 组合层中性化"
  grep -E "【结论】|^口径|^原始|^中性化" logs/neutralized_portfolio.log 2>/dev/null | tail -8
} | tee -a "$PROG"
log "═══ 汇总完成 ═══"
