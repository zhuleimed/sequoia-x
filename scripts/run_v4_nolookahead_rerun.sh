#!/bin/bash
# 去前视后 V4 完整重跑: 重建129维缓存(feature_version hash变) → 预测缓存 → 回测
PROJ=$(cd "$(dirname "$0")/.." && pwd); PY=/home/zhulei/anaconda3/envs/zhulei_py312/bin/python
cd "$PROJ"
log(){ echo "[$(date '+%F %T')] $1"; }
notify(){ "$PY" scripts/notify_wechat.py "$1" 2>/dev/null || true; }
export V4_SAMPLE_END_FIX="2026-08-19"
V4_CACHE="data/cache/v2_dataset/v4_nolookahead_"
log "阶段0: 重建129维缓存(去前视龙虎榜/涨停, feature_version hash变)..."
"$PY" -u scripts/rebuild_dataset_cache.py --only-88 --workers 12 >> logs/v4_nolook_bt.log 2>&1
log "阶段0 缓存重建 exit=$? (新缓存目录=$(ls -dt data/cache/v2_dataset/*/ 2>/dev/null | head -1))"
log "阶段1: 重建70月预测缓存..."
"$PY" -u scripts/build_prediction_cache.py --start-month 2020-09 --end-month 2026-06 --skip-t4 --output output/backtest_v2/prediction_cache_v4_nolook.json >> logs/v4_nolook_bt.log 2>&1
log "阶段1 预测缓存 exit=$? (共用正式output上级/stock_pool)"
log "阶段2: 并行回测(72组)..."
"$PY" -u scripts/run_shared_backtest.py --all --cache output/backtest_v2/prediction_cache_v4_nolook.json >> logs/v4_nolook_bt.log 2>&1
log "阶段2 回测 exit=$? → 去前视V4真实结果"
notify "去前视V4重跑完成(见 logs/v4_nolook_bt.log) — 对比旧121维判断'提升'是真是假"
