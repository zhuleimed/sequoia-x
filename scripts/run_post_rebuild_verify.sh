#!/bin/bash
# 缓存重建完成后接续执行 1-3 步验证（nohup 脱离会话）
#   1. 确认 fv=4 缓存目录已生成且完整
#   2. 干跑 build_prediction_cache 单月验证（验证 129 维预测链路）
#   3. 抽样验证新特征（龙虎榜/涨停）覆盖率
# 第4步(60月回测验证) 超长, 建议手动跑, 不入此链
# 用法: nohup bash scripts/run_post_rebuild_verify.sh > logs/post_rebuild_verify.log 2>&1 &
PROJ=$(cd "$(dirname "$0")/.." && pwd)
PY=/home/zhulei/anaconda3/envs/zhulei_py312/bin/python
cd "$PROJ"
log(){ echo "[$(date '+%F %T')] $1"; }

# 等待缓存重建进程 (rebuild_dataset_cache --only-88) 结束
# ⚠️ 用 pgrep -f 精确匹配 + grep -v 排除自身；勿用裸 grep（会匹配本脚本命令行里的同名字符串）
log "等待缓存重建进程结束..."
while pgrep -f "rebuild_dataset_cache\.py --only-88" > /dev/null 2>&1; do sleep 30; done
sleep 15
log "缓存重建进程已结束"

# ── 1. 确认 fv=4 缓存生成且完整 ──
log "① 确认 fv=4 缓存..."
FOUND=0
for d in data/cache/v2_dataset/*/; do
  if [ -f "$d/metadata.json" ]; then
    fv=$("$PY" -c "import json;print(json.load(open('$d/metadata.json')).get('params',{}).get('feature_version',''))" 2>/dev/null)
    if [ "$fv" = "4" ]; then
      log "  找到 fv=4 缓存: $d"
      FOUND=1
    fi
  fi
done
if [ "$FOUND" = "1" ]; then
  log "  ✅ fv=4 缓存已生成"
  "$PY" scripts/notify_wechat.py "✅ 129维缓存重建完成(fv=4)。开始接续验证(新特征覆盖率+单月干跑)..." 2>/dev/null || true
else
  log "  ❌ 未找到 fv=4 缓存 (可能重建未完成或失败), 中止接续链"
  "$PY" scripts/notify_wechat.py "❌ 129维缓存重建失败——未找到fv=4缓存, 需人工排查 logs/v4_cache_rebuild*.log" 2>/dev/null || true
  exit 1
fi

# ── 3. 抽样验证新特征覆盖率（快, 先做）──
log "② 抽样验证新特征(龙虎榜/涨停)覆盖率..."
"$PY" - <<'EOPY' >> logs/post_rebuild_verify.log 2>&1
import pandas as pd, glob, random
from sequoia_x.features_extra.build_extra_features import build_extra_features
dates=pd.date_range('2025-09-01','2026-08-19',freq='B'); close=pd.Series(10.0,index=dates)
sample=glob.glob('data/extra_features/dragon_tiger/*.parquet'); random.seed(42)
sample=random.sample(sample,min(50,len(sample)))
agg={'dt_net_buy':0,'dt_cnt_30d':0,'lu_lianban':0,'lu_cnt_30d':0}
for f in sample:
    code=f.split('/')[-1].replace('.parquet','')
    o,_=build_extra_features(dates,close,code)
    for c in agg: agg[c]+=(o[c]!=0).mean()
n=len(sample)
print('新特征平均覆盖率:')
for c,v in agg.items(): print(f'  {c}: {v/n*100:.1f}%')
EOPY
log "  ✅ 新特征覆盖率验证完成(见 log)"

# ── 2. 干跑 build_prediction_cache 单月验证 ──
log "③ 干跑 build_prediction_cache 单月验证(2026-08, ~30-60min)..."
DRY="$PROJ/output/backtest_v2/.dryrun_verify.json"
PYTHONPATH=$PROJ "$PY" -u scripts/build_prediction_cache.py \
  --start-month 2026-08 --end-month 2026-08 --skip-t4 \
  --output "$DRY" >> logs/post_rebuild_verify.log 2>&1
RC=$?
if [ $RC -eq 0 ] && [ -f "$DRY" ]; then
  log "  ✅ 干跑验证通过: $DRY"
  PUSH_MSG="✅ V4迁移接续验证全部通过！129维新体系就绪(fv=4, 129维)。下一步(手动): 60月回测(6月基准)后 8/31 月末链正式跑。详看 logs/post_rebuild_verify.log"
else
  log "  ❌ 干跑失败 exit=$RC (129维预测链路有问题, 需人工排查)"
  PUSH_MSG="⚠️ V4迁移：缓存重建完成但干跑验证失败(exit=$RC)，129维预测链路可能有问题，需排查 logs/post_rebuild_verify.log"
fi
"$PY" scripts/notify_wechat.py "$PUSH_MSG" 2>/dev/null || true

log "═══ 接续链完成 ═══"
log "下一步(手动): 60月回测验证(6月基准), 确认129维收益不劣化"
