#!/bin/bash
# V4 迁移链：估值重建 → 129维缓存全量重建（nohup 脱离会话，断点续跑）
# 用法: nohup bash scripts/run_v4_migration_chain.sh > logs/v4_chain.log 2>&1 &
# 依赖: 生产环境 py312 + HITHINK_FINANCE_API_KEY 已 exported
set -x
PROJ=$(cd "$(dirname "$0")/.." && pwd)
PY=/home/zhulei/anaconda3/envs/zhulei_py312/bin/python
cd "$PROJ"

# 阶段1: 历史 peTTM 重建（方案A，PE 摆脱 baostock，PB 保留）——断点续跑，幂等
echo "[chain $(date '+%F %T')] 阶段1: 历史 peTTM 重建..."
"$PY" scripts/rebuild_valuation_history.py >> logs/valuation_rebuild_$(date +%Y%m%d).log 2>&1
echo "[chain $(date '+%F %T')] 阶段1 完成 exit=$?"

# 阶段2: 129 维缓存全量重建（feature_version=4 → 自动全量；16 worker 双 job）
#   --only-88: 重建 121/88 维（树模型 T2/T1/T3），129维扩展随 feature_version=4 自动纳入
echo "[chain $(date '+%F %T')] 阶段2: 129 维缓存全量重建（预计 2-6h）..."
"$PY" scripts/rebuild_dataset_cache.py --only-88 --workers 16 >> logs/v4_cache_rebuild_$(date +%Y%m%d).log 2>&1
echo "[chain $(date '+%F %T')] 阶段2 完成 exit=$?"

echo "[chain $(date '+%F %T')] ✅ V4 迁移链完成"
