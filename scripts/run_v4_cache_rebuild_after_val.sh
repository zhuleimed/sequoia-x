#!/bin/bash
# 等待在跑的 历史peTTM重建(PID 1902059) 完成后，自动启动 129维缓存全量重建
# 用于: 估值重建已在跑(nohup)，此脚本在它结束后接续跑缓存重建，均脱离会话
# 用法: nohup bash scripts/run_v4_cache_rebuild_after_val.sh > logs/v4_cache_waiter.log 2>&1 &
PROJ=$(cd "$(dirname "$0")/.." && pwd)
PY=/home/zhulei/anaconda3/envs/zhulei_py312/bin/python
cd "$PROJ"

VAL_PID=1902059
echo "[$(date '+%F %T')] 等待估值重建 PID=$VAL_PID 完成..."
while kill -0 "$VAL_PID" 2>/dev/null; do
    sleep 30
done
echo "[$(date '+%F %T')] 估值重建已结束，启动 129 维缓存全量重建..."

# 检查估值重建是否成功产出（进度文件含完成标记）
if [ -f scripts/tmp/valuation_rebuild_progress.json ]; then
    done_count=$(python3 -c "import json;print(len(json.load(open('scripts/tmp/valuation_rebuild_progress.json'))['done']))" 2>/dev/null)
    echo "[$(date '+%F %T')] 估值重建进度: $done_count 只"
fi

PYTHONPATH=$PROJ "$PY" scripts/rebuild_dataset_cache.py --only-88 --workers 16 \
    >> logs/v4_cache_rebuild_$(date +%Y%m%d).log 2>&1
echo "[$(date '+%F %T')] 缓存重建退出 exit=$?"
