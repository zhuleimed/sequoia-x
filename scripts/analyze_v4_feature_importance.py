#!/usr/bin/env python3
"""V4(129维) 特征重要性分析：搞清楚哪个维度/特征对 T2 模型有贡献。

2026-08-21: 用 V4 缓存的 129 维特征训练 LightGBM T2, 提取 feature_importance('gain')。
目标: 定位是 finance16 新增 / 龙虎榜 / 涨停 / 还是基础量价 在起作用, 指导后续聚焦。

方法:
  - mmap 加载 V4 缓存 X(399775,120,129) + y2 (不载全内存)
  - 随机采样 N 样本（默认 50000，够出 importance 趋势，避免全量 399775×15480 特征太慢）
  - LightGBM 训练(Huber/rmse, 默认参数, 不 Optuna——只求 importance 相对趋势)
  - feature_importance('gain') → 129 值
  - 标注: 前88基础(量价/市场), 后41扩展(有列名, 来自 build_extra_features)
  - 按 88基础/41扩展 + 扩展各维度(资金流/财务/股东/一致/新闻/分红/龙虎榜/涨停) 聚合

遵守铁律: 用系统时间(date)做时长分析(多进程CPU用时不准); 断点/仅读缓存不写生产; 只读。

用法: py312 python scripts/analyze_v4_feature_importance.py [--n 50000] [--seed 42]
输出: 终端 + data/feature_importance_v4.json
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parent.parent
CACHE = PROJ / "data/cache/v2_dataset/17bf7d4f4111"  # 去前视后 feature_version=4 缓存(2026-08-21 13:31 重建, 龙虎榜/涨停次日生效)


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# 41 扩展列名 (与 build_extra_features 输出列序一致)
EXTRA_COLS = [
    "ff_main_ratio", "ff_main_ratio_5d", "ff_main_inflow_days_5", "ff_main_amt_20d",
    "ff_main_momentum", "ff_xbig_ratio_5d",          # fund_flow 0-5
    "fin_roe", "fin_gp_margin", "fin_np_margin", "fin_debt_ratio", "fin_rev_yoy",
    "fin_profit_yoy", "fin_profit_yoy_chg", "fin_cf_quality", "fin_eps", "fin_bps",
    "fin_roe_tb", "fin_rev_yoy_chg", "fin_current_ratio", "fin_quick_ratio",
    "fin_inv_turn_days", "fin_ar_turn_days",          # finance 6-21
    "hd_num_chg", "hd_avg_mcap",                      # holders 22-23
    "cs_buy_ratio", "cs_org_num", "cs_pred_pe", "cs_aim_dev", "cs_aim_spread",  # consensus 24-28
    "nw_cnt_5d", "nw_cnt_20d", "nw_src_div",          # news 29-31
    "xd_yield", "xd_div_cnt_3y", "xd_song_cnt_3y",    # xdxr 32-34
    "dt_net_buy", "dt_net_rate", "dt_cnt_30d",        # dragon_tiger 35-37
    "lu_lianban", "lu_seal", "lu_cnt_30d",            # limit_up 38-40
]
BASE_N = 88  # 基础维度(量价+市场状态)
# 扩展维度分组(列索引偏移 → 维度名)
EXTRA_GROUPS = {
    "fund_flow": (0, 6), "finance": (6, 22), "holders": (22, 24),
    "consensus": (24, 29), "news": (29, 32), "xdxr": (32, 35),
    "dragon_tiger": (35, 38), "limit_up": (38, 41),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50000, help="采样样本数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--recent", action="store_true",
                    help="只采样近一年(2025-08后, 含龙虎榜/涨停数据段)")
    a = ap.parse_args()
    t0 = time.time()
    log(f"耗时基准: 系统时间 {datetime.now():%H:%M:%S} | 采样 N={a.n}")

    # ── 1. mmap 加载 (不载全内存) ──
    X = np.load(str(CACHE / "X.npy"), mmap_mode="r")
    y2 = np.load(str(CACHE / "y2.npy"), mmap_mode="r")
    log(f"X={X.shape}, y2={y2.shape}")

    # ── 2. 采样（近一年段 or 全期等距）──
    # ⚠️ 2026-08-21 教训: 龙虎榜/涨停仅近一年(2025-08后)有数据, 全期样本里这些特征大多0。
    #    分析它们必须单独采样近一年段(索引>=RECENT_IDX)。全期采样只反映全期平均贡献。
    #    X 前 329573 行=2020-08~2025-07(无情绪数据), 后 70202 行=2025-08后(含龙虎榜/涨停)
    RECENT_IDX = 329573
    n_total = X.shape[0]
    if a.recent:
        start = RECENT_IDX
        n_avail = n_total - RECENT_IDX
    else:
        start = 0
        n_avail = n_total
    n = min(a.n, n_avail)
    step = max(1, n_avail // n)
    idx_sam = np.arange(start, start + n_avail, step)[:n]
    Xs = X[idx_sam]
    ys = y2[idx_sam]
    log(f"采样段: {'近一年(2025-08后)' if a.recent else '全期等距'} | 起始索引{start} 采样{n}")
    # 展平为 2D: (N, 120*129=15480)
    X2d = Xs.reshape(len(Xs), -1)
    log(f"展平 X2d={X2d.shape} (15480 特征列)")

    # 排除 nan/inf
    bad = ~np.isfinite(X2d).all(axis=1) | ~np.isfinite(ys)
    X2d = X2d[~bad]; ys = ys[~bad]
    log(f"有效样本 {len(ys)} (剔 {bad.sum()})")

    # ── 3. LightGBM 训练 (Huber/rmse, 默认参数, 不求调优) ──
    import lightgbm as lgb
    params = {
        "objective": "huber", "alpha": 0.1, "metric": "rmse",
        "num_leaves": 31, "learning_rate": 0.1, "min_child_samples": 20,
        "subsample": 0.8, "colsample_bytree": 0.5,   # 降特征采样加速(15480列)
        "reg_alpha": 0.1, "reg_lambda": 1.0,
        "n_jobs": 8, "verbosity": -1, "random_state": a.seed,
    }
    dtr = lgb.Dataset(X2d, label=ys)
    model = lgb.train(params, dtr, num_boost_round=300)
    importances = model.feature_importance(importance_type="gain")  # (15480,)
    log(f"训练完成 用时 {(time.time()-t0)/60:.1f}min")

    # ── 4. 特征列重要性聚合 (129 特征 × 120 时间步) ──
    # X2d = (N, 120*129) 展平, C连续: 第 i 样本 [t*129 + f], t=时间步0..119, f=特征0..128
    # 所以 特征 f 占的展平列 = { t*129 + f for t in range(120) }
    feat_imp = np.zeros(129)
    imp_arr = np.asarray(importances)
    for f in range(129):
        idx_cols = np.arange(f, 120 * 129, 129)   # 同特征所有时间步的展平列
        feat_imp[f] = imp_arr[idx_cols].sum()

    # ── 5. 分组分析 ──
    # 基础 88 维 (无名, 合计)
    base_imp = feat_imp[:BASE_N].sum()
    extra_imp = feat_imp[BASE_N:].sum()
    total_imp = feat_imp.sum()
    log(f"\n══ 维度贡献 (gain 占比) ══")
    log(f"  基础88维(量价+市场状态): {base_imp/total_imp*100:.1f}%")
    log(f"  扩展41维: {extra_imp/total_imp*100:.1f}%")
    log(f"\n══ 扩展41维 各维度贡献 ══")
    group_imp = {}
    for g, (s, e) in EXTRA_GROUPS.items():
        gi = feat_imp[BASE_N + s: BASE_N + e].sum()
        group_imp[g] = gi
        log(f"  {g:<14}: {gi/total_imp*100:.2f}% (gain sum={gi:.0f})")

    # ── 6. 扩展特征 top 排序 ──
    log(f"\n══ 扩展41维 特征重要性 Top15 ══")
    ext_ims = feat_imp[BASE_N:]
    order = np.argsort(-ext_ims)[:15]
    for i in order:
        log(f"  {i:>2} {EXTRA_COLS[i]:<25} {ext_ims[i]/total_imp*100:.2f}%")

    # 保存结果
    out = {
        "generated": json.dumps({"date": str(datetime.now())}),
        "n_samples": len(ys),
        "total_gain": float(total_imp),
        "base88_pct": float(base_imp / total_imp),
        "extra41_pct": float(extra_imp / total_imp),
        "extra_group_pct": {k: float(v / total_imp) for k, v in group_imp.items()},
        "feature_importance_129": [float(x) for x in feat_imp],
        "extra_cols": EXTRA_COLS,
    }
    outfile = PROJ / "data" / "feature_importance_v4.json"
    outfile.write_text(json.dumps(out, indent=2))
    log(f"\n✅ 结果已存 {outfile}")
    log(f"总耗时 {(time.time()-t0)/60:.1f}min (系统时间基准)")


if __name__ == "__main__":
    main()
