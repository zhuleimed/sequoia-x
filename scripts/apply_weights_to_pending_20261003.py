#!/usr/bin/env python
"""给 V4 模拟盘"10-01 那批待买信号"补上目标权重（2026-10-03，P0 方向一上线动作）。

背景
    10-01 凌晨的月度重训产出 10 条买入信号（pending），当时"按权重买入"还没接入，
    故它们没有 target_weight（NULL = 等权）。本脚本按当前配置
    `config.BUY_WEIGHT_METHOD`（= min_variance 最小方差）补算权重并写回，
    使 **10-09 开盘（国庆后首个交易日）的买入即为"按权重买入"**（首次上线）。

安全性
    · 只改 `sim_v2.db` 中 `status='pending'` 的信号；已执行/已取消的**一律不动**
    · 幂等：重复跑结果一致（同样的权重覆盖同样的行）
    · 回滚：`UPDATE sim_buy_signals SET target_weight=NULL` 即回到等权
    · 默认 dry-run（只打印不改）；加 `--apply` 才写库

用法（铁律六：必须 py312）
    .../python scripts/apply_weights_to_pending_20261003.py            # 预览
    .../python scripts/apply_weights_to_pending_20261003.py --apply    # 写库
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "scripts"))

from sequoia_x.core.logger import get_logger          # noqa: E402
from v2_monthly_retrain import _compute_buy_weights    # noqa: E402

logger = get_logger(__name__)

SIM_V2_DB = PROJ / "data" / "sim_v2.db"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="真正写库（默认只预览）")
    ap.add_argument("--db", default=str(SIM_V2_DB))
    args = ap.parse_args()

    with sqlite3.connect(args.db) as c:
        rows = c.execute(
            "SELECT id, symbol, buy_date FROM sim_buy_signals "
            "WHERE status='pending' ORDER BY id").fetchall()

    if not rows:
        logger.info("没有 pending 信号，无需处理")
        return 0

    symbols = [r[1] for r in rows]
    logger.info(f"pending 信号 {len(rows)} 条: {symbols}")

    weights = _compute_buy_weights(symbols)
    if weights is None:
        logger.error("权重计算失败（返回 None）—— 不写库，保持等权")
        return 1

    print(f"\n{'id':>5} {'代码':<8} {'买入日':<12} {'权重':>6}")
    for (sid, sym, bd), w in zip(rows, weights):
        print(f"{sid:>5} {sym:<8} {bd:<12} {w:>6.2f}")
    zero = [s for s, w in zip(symbols, weights) if w == 0]
    print(f"\n权重均值={sum(weights)/len(weights):.4f}（应=1）"
          + (f" | ⚠ 0 权重（不买入）: {zero}" if zero else ""))

    if not args.apply:
        print("\n（dry-run：未写库；加 --apply 执行）")
        return 0

    with sqlite3.connect(args.db) as c:
        for (sid, _sym, _bd), w in zip(rows, weights):
            c.execute("UPDATE sim_buy_signals SET target_weight=? "
                      "WHERE id=? AND status='pending'", (float(w), sid))
        c.commit()
        n = c.execute("SELECT COUNT(*) FROM sim_buy_signals "
                      "WHERE status='pending' AND target_weight IS NOT NULL").fetchone()[0]
    logger.info(f"✅ 已写库：{n} 条 pending 信号带权重（回滚: SET target_weight=NULL）")
    print(f"\n✅ 已写库：{n} 条 pending 信号带权重")
    return 0


if __name__ == "__main__":
    sys.exit(main())
