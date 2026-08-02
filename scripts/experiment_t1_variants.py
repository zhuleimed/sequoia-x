"""T1 方向预测目标变体实验（2026-08-02）

目的：T1（XGBoost 5日方向分类）70 个月真实 AUC=0.499 无预测能力。
验证：换预测目标是否能获得 AUC>0.55 的预测能力。

变体：
  v0: y1_5d  （5 日绝对涨跌方向）——现状基线（预期 ≈0.5）
  v1: y1_10d （10 日绝对涨跌方向）
  v2: y1_20d （20 日绝对涨跌方向）
  v3: sign(y2)（20 日跑赢沪深300方向，与 T2 同源目标）

方法：代表性月份（2025-01 温和 / 2025-08 弱月 / 2026-03 极端）各训练 12 月滚动窗口
XGBoost → 当月全市场 AUC。
"""
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data/cache/v2_dataset/13132147f8e8"
DB_PATH = PROJECT_ROOT / "data/sequoia_v2.db"

TEST_MONTHS = ["2025-01", "2025-08", "2026-03"]  # 温和 / 弱月 / 极端
WINDOW_MONTHS = 12
N_ESTIMATORS = 300  # 实验用快速配置


def load_shared():
    X = np.load(str(CACHE_DIR / "X.npy"), mmap_mode="r")
    with open(CACHE_DIR / "dates.json") as f:
        dates = json.load(f)
    dates_arr = np.array(dates)
    return X, dates_arr


def compute_labels(conn, symbols, sample_dates, horizon):
    """对每个采样日计算未来 horizon 交易日方向标签 {date: {sym: 0/1}}。"""
    result = {}
    for sd in sample_dates:
        # T 日所有股票收盘
        t_prices = pd.read_sql(
            "SELECT symbol, close FROM stock_daily WHERE date=?",
            conn, params=(sd,),
        )
        if t_prices.empty:
            continue
        # 未来第 horizon 个交易日（全市场该日有交易的股票）
        fut_rows = pd.read_sql(
            "SELECT symbol, close FROM stock_daily "
            "WHERE date=(SELECT DISTINCT date FROM stock_daily "
            "             WHERE date>? ORDER BY date LIMIT 1 OFFSET ?)",
            conn, params=(sd, horizon - 1),
        )
        if fut_rows.empty:
            continue
        t_map = dict(zip(t_prices["symbol"], t_prices["close"]))
        day_labels = {}
        for _, r in fut_rows.iterrows():
            if r["symbol"] in t_map and t_map[r["symbol"]] > 0:
                day_labels[r["symbol"]] = 1 if r["close"] > t_map[r["symbol"]] else 0
        result[sd] = day_labels
    return result


def compute_excess_labels(conn, symbols, sample_dates):
    """sign(y2)：20 日超额收益方向（股票 20 日收益 - 沪深300 同期）。"""
    result = {}
    idx_df = pd.read_sql(
        "SELECT date, close FROM index_daily WHERE symbol='sh.000300' ORDER BY date",
        conn,
    )
    idx_map = dict(zip(idx_df["date"], idx_df["close"]))
    for sd in sample_dates:
        t_prices = pd.read_sql(
            "SELECT symbol, close FROM stock_daily WHERE date=?",
            conn, params=(sd,),
        )
        if t_prices.empty or sd not in idx_map:
            continue
        fut_rows = pd.read_sql(
            "SELECT symbol, close FROM stock_daily "
            "WHERE date=(SELECT DISTINCT date FROM stock_daily "
            "             WHERE date>? ORDER BY date LIMIT 1 OFFSET 19)",
            conn, params=(sd,),
        )
        if fut_rows.empty:
            continue
        t_map = dict(zip(t_prices["symbol"], t_prices["close"]))
        idx_t, idx_f = idx_map[sd], fut_rows["close"].iloc[0]
        idx_ret = idx_f / idx_t - 1
        day_labels = {}
        for _, r in fut_rows.iterrows():
            if r["symbol"] in t_map and t_map[r["symbol"]] > 0:
                stock_ret = r["close"] / t_map[r["symbol"]] - 1
                day_labels[r["symbol"]] = 1 if stock_ret > idx_ret else 0
        result[sd] = day_labels
    return result


def train_eval(X, dates_arr, train_sds, test_sd, labels, sym_list):
    """训练 + 测试 AUC。labels: {date: {sym: 0/1}}。"""
    # 训练集
    X_tr, y_tr = [], []
    for sd in train_sds:
        if sd not in labels:
            continue
        # 找到该采样日在 dates_arr 的索引（缓存行）
        idx = np.where(dates_arr == sd)[0]
        if len(idx) == 0:
            continue
        row = idx[0]
        for j, sym in enumerate(sym_list):
            if sym in labels[sd]:
                X_tr.append(X[row, j])
                y_tr.append(labels[sd][sym])
    if len(X_tr) < 500 or len(set(y_tr)) < 2:
        return None
    X_tr = np.array(X_tr).reshape(len(X_tr), -1)
    y_tr = np.array(y_tr)

    # 测试集
    X_te, y_te = [], []
    if test_sd not in labels:
        return None
    idx = np.where(dates_arr == test_sd)[0]
    if len(idx) == 0:
        return None
    row = idx[0]
    for j, sym in enumerate(sym_list):
        if sym in labels[test_sd]:
            X_te.append(X[row, j])
            y_te.append(labels[test_sd][sym])
    if len(X_te) < 100 or len(set(y_te)) < 2:
        return None
    X_te = np.array(X_te).reshape(len(X_te), -1)
    y_te = np.array(y_te)

    clf = xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, n_jobs=8,
        eval_metric="auc", use_label_encoder=False,
    )
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    return roc_auc_score(y_te, proba), len(X_tr), len(X_te)


def main():
    print("=" * 70)
    print("  T1 方向预测目标变体实验")
    print("=" * 70)
    X, dates_arr = load_shared()
    print(f"X={X.shape}, 采样日={len(set(dates_arr))}")

    conn = sqlite3.connect(DB_PATH)
    # 股票池（缓存第 0 采样日的 symbols 顺序 = 缓存行结构）
    # 注：缓存每行是同一采样日的全部股票，行内顺序 = 股票顺序。用首个采样日数量推断
    sample_dates_sorted = sorted(set(dates_arr))
    sym_list = [f"TEMP{i}" for i in range(X.shape[1])]  # 占位，实际用下标

    variants = {
        "v0_5日方向": 5,
        "v1_10日方向": 10,
        "v2_20日方向": 20,
    }
    results = {k: {} for k in variants}

    for month in TEST_MONTHS:
        # 训练窗口：month 前 12 个月的所有采样日；测试：month 最后采样日
        ym = month
        last_date = conn.execute(
            "SELECT MAX(date) FROM stock_daily WHERE date LIKE ?", (ym + "%",)
        ).fetchone()[0]
        if last_date is None:
            continue
        test_sd = last_date
        # 训练窗口采样日（<= 测试采样日前一个月的月末）
        from datetime import datetime, timedelta
        test_dt = datetime.strptime(test_sd, "%Y-%m-%d")
        start_dt = test_dt - timedelta(days=WINDOW_MONTHS * 31)
        train_sds = [d for d in sample_dates_sorted
                     if start_dt.strftime("%Y-%m-%d") <= d < test_sd]

        # 各变体标签
        labels_map = {}
        for name, horizon in variants.items():
            labels_map[name] = compute_labels(conn, None, train_sds + [test_sd], horizon)
        labels_map["v3_超额方向"] = compute_excess_labels(conn, None, train_sds + [test_sd])

        print(f"\n{'─'*70}")
        print(f"测试月: {month} (测试日 {test_sd}, 训练采样日 {len(train_sds)})")
        for name in list(variants.keys()) + ["v3_超额方向"]:
            auc = train_eval(X, dates_arr, train_sds, test_sd, labels_map[name], None)
            if auc:
                print(f"  {name:12}: AUC={auc[0]:.4f} (训练 {auc[1]}, 测试 {auc[2]})")
            else:
                print(f"  {name:12}: 数据不足")

    conn.close()
    print("\n完成")


if __name__ == "__main__":
    main()
