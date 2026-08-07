"""V3 方向三实验：可微 RankIC 损失 vs Huber 损失，纯 LSTM 70 个月逐月 IC 对照（2026-08-06）

不换架构只换损失函数——检验"直接优化排序目标（可微 RankIC）是否比 Huber 回归
更能提升 LSTM 的排序能力"（方正证券实证：可微 RankIC 损失 Rank IC 12.48% vs MSE）。

三向对照（变量隔离：只差损失函数，其余全同）:
  - 实验组: 纯 LSTM（V2 定稿参数）+ 可微 RankIC 损失（soft-rank 软排序）
  - 对照组: 纯 LSTM（同架构）+ Huber 损失（生产同款）
  - 参考列: production prediction_cache 的 t4（Huber + 5000 抽样，已有零成本）

工程要点（方向一/二教训全部应用）:
  - 固定 epochs=150，不做 early stopping（方向一：早停失效，树数<10 占 48-61%）
  - 同截面 batch：RankIC 排序必须同一采样日内 → 训练样本按采样日分组，
    每组（采样日）固定 seed 抽 300 只（24 组 × 300 = 7200 条/月），两组共享抽样
  - soft-rank 数值稳定（clip ±20 防 sigmoid 溢出；损失监控 NaN 立即中止——方向二教训）
  - 8 workers × 特征 8 进程（ProcessPoolExecutor 嵌套），KMP_AFFINITY 清除
  - 内存预算：~8GB/worker（LSTM 双模型 + 训练数据 + OHLCV）→ 8 × 8 = 64GB 安全
  - 训练后 del + gc（方向一 v2 修复）；nohup + 断点续跑

用法:
  python experiments/rankic_lstm/experiment_rankic.py --month 2026-06   # 单月验证
  python experiments/rankic_lstm/experiment_rankic.py                    # 全量 70 个月
  python experiments/rankic_lstm/experiment_rankic.py --analyze          # IC 对照分析
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── 铁律一：线程控制，必须在 import numpy/TF 之前 ──
os.environ.pop("KMP_AFFINITY", None)
os.environ["OMP_NUM_THREADS"] = "2"           # TF 训练用 2 线程（t4 worker 模式）
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# ── 常量 ──
CACHE_DIR_80 = PROJECT_ROOT / "data/cache/v2_dataset/62cf234c5440"
OUT_DIR = PROJECT_ROOT / "output/backtest_v2/experiments/rankic_lstm"
TMP_DIR = PROJECT_ROOT / ".tmp/rankic_lstm"
PREDICTIONS_PATH = PROJECT_ROOT / "output/backtest_v2/experiments/rankic_lstm_predictions.json"
IC_REPORT_PATH = PROJECT_ROOT / "output/backtest_v2/experiments/rankic_lstm_ic_report.csv"
DB_PATH = str(PROJECT_ROOT / "data/sequoia_v2.db")
STOCK_POOL_PATH = PROJECT_ROOT / "output/backtest_v2/.stock_pool.json"

TRAIN_MONTHS = 12
FEAT_WORKERS = 8
# ⚠️ 2026-08-07 方案 B 实测失败后回退（V3 文档 §8.7 记录）：
#   方案 B（12 workers × TF intra 4）实测：每 worker 仅 1.32→1.47 核（+11%），
#   但 workers 16→12 使总吞吐反降（21核→18核）——LSTM 瓶颈在单线程小算子，
#   intra 线程提升无效。回退方案 A：16 workers × TF intra 2（总吞吐 21 核最优）
N_POOL_WORKERS = 16

# 训练超参（V2 定稿 + 方向三设计）
LSTM_UNITS = 128
LSTM_UNITS2 = 64
DROPOUT = 0.285
LR = 0.0096
# ⚠️ 2026-08-06 单月验证调整（V3 文档 §8.6 记录）：epochs 150 → 60
#   实测收敛证据：epoch 20 时 RankIC loss 已达 -0.572（软 Spearman +0.57）——
#   RankIC 目标收敛远快于 Huber（生产 200ep 是 Huber 需求）；60ep 充分且两组公平
EPOCHS = 60
SAMPLE_PER_DAY = 300          # 每采样日抽 300 只（26 组 × 300 = 7800 条/月；同截面 batch）
SOFT_RANK_TAU = 1.0           # 软排序温度
SEED = 42
HUBER_DELTA = 0.1
GRAD_CLIP = 1.0


# ══════════════════ 可微 RankIC 损失（TF 实现）══════════════════

def build_rankic_loss(tau: float = SOFT_RANK_TAU):
    """构建可微 RankIC 损失函数（最小化负软 Spearman）。

    软排序: soft_rank_i = Σ_j sigmoid((x_i - x_j)/τ) / n
    软 Spearman = Pearson(soft_rank(pred), soft_rank(y))
    数值稳定: 距离矩阵 clip ±20（sigmoid 饱和区梯度≈0 但避免 NaN）
    """
    import tensorflow as tf

    def soft_rank(x: tf.Tensor, temp: float) -> tf.Tensor:
        xf = tf.cast(x, tf.float32)
        n = tf.cast(tf.shape(xf)[0], tf.float32)
        d = (xf - tf.transpose(xf)) / temp
        d = tf.clip_by_value(d, -20.0, 20.0)
        return tf.reduce_sum(tf.sigmoid(d), axis=1) / n

    def rankic_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        rp = soft_rank(y_pred, tau)
        ry = soft_rank(y_true, tau)
        rp_c = rp - tf.reduce_mean(rp)
        ry_c = ry - tf.reduce_mean(ry)
        denom = tf.sqrt(tf.reduce_sum(rp_c ** 2) * tf.reduce_sum(ry_c ** 2)) + 1e-8
        return -tf.reduce_sum(rp_c * ry_c) / denom

    return rankic_loss


def build_lstm_model(cfg) -> tuple:
    """构建纯 LSTM 模型（V2 定稿参数；num_transformers=0, l2_reg=0）。

    只读引用生产 deep_lstm._create_lstm_model（隔离规则 §3.1）。
    返回 (model, optimizer)：手动训练循环用（不 model.compile——损失函数随组切换）。
    """
    import tensorflow as tf
    from sequoia_x.model_selection_v2.models.deep_lstm import _create_lstm_model

    model = _create_lstm_model(
        window=cfg.window,
        n_features=80,                     # 80 维（T4 同口径）
        lstm_units=LSTM_UNITS,
        lstm_units2=LSTM_UNITS2,
        num_transformers=0,                # V2 定稿：纯 LSTM
        dropout_rate=DROPOUT,
        l2_reg=0.0,                        # V2 定稿：L2 杀死 kernel
        learning_rate=LR,
        gradient_clip_norm=GRAD_CLIP,
    )
    opt = tf.keras.optimizers.Adam(learning_rate=LR, clipnorm=GRAD_CLIP)
    return model, opt


def train_lstm_manual(X: np.ndarray, y: np.ndarray, group_ids: np.ndarray,
                      cfg, loss_fn, model_id: str, logger_print) -> tuple:
    """手动训练循环（同截面 batch + 自定义损失）。

    Args:
        X: (n, 120, 80) 训练样本
        y: (n,) y2 标签
        group_ids: (n,) 每样本的采样日索引（0~23），同组构成一个 batch
        loss_fn: 损失函数（rankic 或 huber）
        logger_print: 日志函数

    Returns:
        (model, loss_history) 或 (None, []) 若 NaN 中止
    """
    import tensorflow as tf

    model, opt = build_lstm_model(cfg)
    n = len(X)
    uniq_groups = np.unique(group_ids)
    loss_history = []
    t_start = time.time()

    for epoch in range(EPOCHS):
        epoch_losses = []
        for g in uniq_groups:
            m = group_ids == g
            xb = X[m].astype(np.float32)
            yb = y[m].astype(np.float32).reshape(-1, 1)
            with tf.GradientTape() as tape:
                pred = model(xb, training=True)
                loss = loss_fn(yb, pred)
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
            epoch_losses.append(float(loss.numpy()))
        mean_loss = float(np.mean(epoch_losses))
        loss_history.append(mean_loss)
        # ── 铁律一/五：运行时自检——NaN 立即中止，不留坏结果 ──
        if np.isnan(mean_loss) or np.isinf(mean_loss):
            logger_print(f"  ❌ [{model_id}] epoch {epoch}: loss={mean_loss} NaN → 中止")
            return None, loss_history
        if (epoch + 1) % 20 == 0 or epoch == 0:
            logger_print(f"  [{model_id}] epoch {epoch+1}/{EPOCHS}: loss={mean_loss:.5f} "
                         f"({time.time()-t_start:.0f}s)")
    return model, loss_history


# ══════════════════ 单月 worker ══════════════════

_TF_CONFIGURED = False  # 模块级标志：fork 后每 worker 独立；TF 线程只能初始化前设置一次


def _configure_tf():
    """TF 线程配置（一次性）。

    ⚠️ 2026-08-07 修复（V3 文档 §8.6 记录）：ProcessPoolExecutor 的 worker 常驻复用，
    处理第 2+ 个任务时 TF 已初始化并执行过 op，重复 set_intra_op 抛
    "Intra op parallelism cannot be modified after initialization" → 全量崩溃。
    修复：模块级标志保证每 worker 只设置一次 + TF_NUM_* 环境变量双保险。
    """
    global _TF_CONFIGURED
    if _TF_CONFIGURED:
        return
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(2)   # 方案 A 定稿（方案 B 实测无效）
    tf.config.threading.set_inter_op_parallelism_threads(1)
    _TF_CONFIGURED = True


def _rankic_month_worker(args: tuple) -> tuple:
    """单月双组训练+预测（RankIC-LSTM + Huber-LSTM，共享抽样与特征构建）。"""
    month, cfg_dict, max_pool_size = args

    import sqlite3
    import pandas as pd

    _os = __import__("os")
    _os.environ.pop("KMP_AFFINITY", None)
    _os.environ["OMP_NUM_THREADS"] = "2"
    _os.environ["OPENBLAS_NUM_THREADS"] = "1"
    _os.environ["MKL_NUM_THREADS"] = "1"
    # TF 线程：环境变量（TF import 时读取）+ 一次性 API 设置（防常驻 worker 二次调用崩溃）
    _os.environ["TF_NUM_INTRAOP_THREADS"] = "2"   # 方案 A 定稿（方案 B 实测无效）
    _os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    _configure_tf()
    print(f"[Worker {month}] 启动诊断: CPU核={_os.cpu_count()} TF=2/1 OMP=2 KMP清除", flush=True)

    def log(msg):
        print(f"[Worker {month}] {msg}", flush=True)

    marker = TMP_DIR / f"{month}.json"
    if marker.exists():
        log(f"跳过：已完成 ({marker})")
        return month, None, 0, 0

    # ── mmap 加载 80 维缓存 ──
    X = np.load(str(CACHE_DIR_80 / "X.npy"), mmap_mode="r")
    y2 = np.load(str(CACHE_DIR_80 / "y2.npy"), mmap_mode="r")
    with open(CACHE_DIR_80 / "dates.json") as f:
        dates = json.load(f)
    dates_arr = np.array(dates)
    log(f"mmap加载: X={X.shape} (80维), {len(set(dates))}采样日期")

    # ── 训练截止日 ──
    ym_year, ym_month = int(month[:4]), int(month[5:7])
    prev_m = ym_month - 1
    prev_y = ym_year
    if prev_m <= 0:
        prev_m += 12
        prev_y -= 1
    conn = sqlite3.connect(DB_PATH)
    last_date_row = conn.execute(
        "SELECT MAX(date) FROM stock_daily WHERE date >= ? AND date < ?",
        (f"{prev_y}-{prev_m:02d}-01", month + "-01"),
    ).fetchone()
    train_end_date = last_date_row[0] if last_date_row and last_date_row[0] else (month + "-01")

    from build_prediction_cache import _filter_stock_pool
    if STOCK_POOL_PATH.exists():
        stock_pool = json.loads(STOCK_POOL_PATH.read_text())
    else:
        all_symbols = [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM stock_daily").fetchall()]
        stock_pool = _filter_stock_pool(all_symbols, DB_PATH)
    if max_pool_size > 0 and len(stock_pool) > max_pool_size:
        import random
        random.seed(42)
        stock_pool = random.sample(stock_pool, max_pool_size)
    conn.close()

    # ── 训练数据：12 月窗口全量 → 按采样日分组 → 每组固定抽 SAMPLE_PER_DAY ──
    end_ym = train_end_date[:7]
    ey, em = int(end_ym[:4]), int(end_ym[5:7])
    sm = em - TRAIN_MONTHS
    sy = ey
    if sm <= 0:
        sm += 12
        sy -= 1
    train_start = f"{sy}-{sm:02d}-01"
    mask = (dates_arr >= train_start) & (dates_arr <= train_end_date)
    n_train = mask.sum()
    if n_train < 100:
        log(f"⚠ 训练样本不足({n_train})，跳过")
        return month, None, 0, 0
    X_all = X[mask]
    y_all = y2[mask]
    d_all = dates_arr[mask]

    # 按采样日分组 + 固定 seed 抽样（两组共享同一批样本 → 变量只有损失函数）
    rng = np.random.default_rng(SEED)
    keep = []
    uniq_days = np.unique(d_all)
    for d in uniq_days:
        idx = np.where(d_all == d)[0]
        if len(idx) > SAMPLE_PER_DAY:
            idx = rng.choice(idx, SAMPLE_PER_DAY, replace=False)
        keep.extend(idx)
    keep = np.sort(np.array(keep))
    X_tr = X_all[keep].astype(np.float32)
    y_tr = y_all[keep]
    group_ids = np.searchsorted(uniq_days, d_all[keep])
    log(f"训练数据: {len(y_tr)} 条, {len(uniq_days)} 采样日 × 每批{SAMPLE_PER_DAY}只 "
        f"(窗口 {train_start} ~ {train_end_date})")

    # ── 重建 cfg ──
    from sequoia_x.model_selection_v2.config import V2Config
    cfg = V2Config()
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)

    # ── 训练两组（先 RankIC 后 Huber；共享数据）──
    import tensorflow as tf
    from tensorflow.keras.losses import Huber

    t0 = time.time()
    log(f"Step1: RankIC-LSTM 训练({EPOCHS}ep)...")
    rankic_loss_fn = build_rankic_loss()
    model_rk, hist_rk = train_lstm_manual(X_tr, y_tr, group_ids, cfg, rankic_loss_fn,
                                          "RankIC", log)
    if model_rk is None:
        log("❌ RankIC 训练 NaN 中止，本月份失败")
        return month, None, 0, 0
    log(f"Step1完成: RankIC loss 终值={hist_rk[-1]:.5f} ({time.time()-t0:.0f}s)")

    t1 = time.time()
    log("Step2: Huber-LSTM 训练(对照组)...")
    huber_fn = Huber(delta=HUBER_DELTA)
    model_hb, hist_hb = train_lstm_manual(X_tr, y_tr, group_ids, cfg, huber_fn,
                                          "Huber", log)
    if model_hb is None:
        log("❌ Huber 训练 NaN 中止，本月份失败")
        return month, None, 0, 0
    log(f"Step2完成: Huber loss 终值={hist_hb[-1]:.5f} ({time.time()-t1:.0f}s)")
    del X_tr, y_tr, X_all, y_all
    import gc
    gc.collect()

    # ── OHLCV 预加载 → 80 维特征构建（复用方向二验证过的路径）──
    log(f"Step3: OHLCV预加载({len(stock_pool)}只)...")
    conn = sqlite3.connect(DB_PATH)
    ph = ",".join("?" * len(stock_pool))
    ohlcv_df = pd.read_sql(
        f"SELECT * FROM stock_daily WHERE symbol IN ({ph}) AND date <= ? ORDER BY symbol, date",
        conn, params=stock_pool + [train_end_date])
    idx_df = pd.read_sql(
        "SELECT * FROM index_daily WHERE symbol='sh.000300' AND date <= ? ORDER BY date",
        conn, params=(train_end_date,))
    conn.close()
    ohlcv_cache = {sym: g.reset_index(drop=True) for sym, g in ohlcv_df.groupby("symbol")}
    log(f"OHLCV={len(ohlcv_df)}行 {len(ohlcv_cache)}只, 80维特征构建({FEAT_WORKERS}进程)...")

    sys.path.insert(0, str(PROJECT_ROOT / "experiments/dlinear"))
    from experiment_dlinear import _build_one_features_80
    from concurrent.futures import ProcessPoolExecutor
    tasks = [(sym, ohlcv_cache.get(sym), idx_df, cfg) for sym in stock_pool]
    with ProcessPoolExecutor(max_workers=FEAT_WORKERS) as ex:
        results = list(ex.map(_build_one_features_80, tasks, chunksize=100))
    X_list = [r[1] for r in results if r is not None]
    sym_list = [r[0] for r in results if r is not None]
    if not X_list:
        log("❌ 特征构建全部失败")
        return month, None, 0, 0
    X_pred = np.stack(X_list, axis=0).astype(np.float32)
    n_valid = len(X_pred)
    log(f"特征完成: {n_valid}/{len(stock_pool)} 有效")

    # ── 两次预测 + 自检 ──
    pred_rk = model_rk.predict(X_pred, verbose=0).flatten()
    pred_hb = model_hb.predict(X_pred, verbose=0).flatten()
    rk_std = float(np.std(pred_rk))
    hb_std = float(np.std(pred_hb))
    if rk_std < 1e-7 or hb_std < 1e-7:
        log(f"❌ 严重: 预测无方差! rankic_std={rk_std:.2e} huber_std={hb_std:.2e}")
    else:
        log(f"✅ RankIC pred: std={rk_std:.4f} | Huber pred: std={hb_std:.4f}")

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({
        "month": month, "train_end_date": train_end_date, "n": n_valid,
        "symbols": sym_list,
        "rankic": [float(v) for v in pred_rk],
        "huber": [float(v) for v in pred_hb],
        "rankic_loss_last": float(hist_rk[-1]),
        "huber_loss_last": float(hist_hb[-1]),
    }))
    log(f"✅ 完成并保存 ({marker})")
    return month, n_valid, rk_std, time.time() - t0


# ══════════════════ 主流程 ══════════════════

def get_experiment_months(start_month: str = "2020-09", end_month: str = "2026-06") -> list[str]:
    y0, m0 = int(start_month[:4]), int(start_month[5:7])
    y1, m1 = int(end_month[:4]), int(end_month[5:7])
    months = []
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        months.append(f"{y}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def run_experiment(months: list[str], max_pool_size: int = 0, n_workers: int = N_POOL_WORKERS):
    from concurrent.futures import ProcessPoolExecutor

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    from sequoia_x.model_selection_v2.config import get_config
    cfg = get_config()
    cfg_dict = {k: v for k, v in cfg.__dict__.items() if not k.startswith("_")}

    print(f"═══ RankIC-LSTM 对照实验（{len(months)} 个月 × 2 组）═══", flush=True)
    print(f"  并行: {n_workers} workers × 特征{FEAT_WORKERS}进程 | LSTM {LSTM_UNITS}u | "
          f"epochs={EPOCHS}(固定) | 每采样日抽{SAMPLE_PER_DAY}只 | τ={SOFT_RANK_TAU}", flush=True)
    print(f"  KMP_AFFINITY: 已清除 | TF=2/1 OMP=2 | 环境: {sys.executable}", flush=True)

    t_start = time.time()
    args_list = [(m, cfg_dict, max_pool_size) for m in months]
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(_rankic_month_worker, args_list))

    done = [r for r in results if r and r[1]]
    print(f"\n═══ 完成: {len(done)}/{len(months)} 个月 总耗时 {time.time()-t_start:.0f}s ═══", flush=True)
    for m, n, std, sec in done:
        print(f"  {m}: n={n} std={std:.4f} 训练+预测={sec:.0f}s", flush=True)

    merge_predictions(months)


def merge_predictions(months: list[str]):
    merged = {}
    for month in months:
        marker = TMP_DIR / f"{month}.json"
        if marker.exists():
            d = json.loads(marker.read_text())
            merged[month] = {"symbols": d["symbols"], "rankic": d["rankic"], "huber": d["huber"]}
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_PATH.write_text(json.dumps(merged))
    print(f"合并完成: {len(merged)} 个月 → {PREDICTIONS_PATH}")


# ══════════════════ IC 对照分析 ══════════════════

def _load_y2_for_month(month: str, symbols: list[str]):
    import sqlite3
    import pandas as pd

    conn = sqlite3.connect(DB_PATH)
    last_date = conn.execute(
        "SELECT MAX(date) FROM stock_daily WHERE date LIKE ?", (month + "%",)
    ).fetchone()[0]
    if last_date is None:
        conn.close()
        return None, None
    ph = ",".join("?" * len(symbols))
    prices = pd.read_sql(
        f"SELECT symbol, date, close FROM stock_daily "
        f"WHERE symbol IN ({ph}) AND date >= ? ORDER BY symbol, date",
        conn, params=symbols + [last_date])
    idx_df = pd.read_sql(
        "SELECT date, close FROM index_daily WHERE symbol='sh.000300' AND date >= ? ORDER BY date",
        conn, params=(last_date,))
    conn.close()

    t_close, t20_close = {}, {}
    for sym, g in prices.groupby("symbol"):
        g = g.sort_values("date")
        row_t = g[g["date"] == last_date]
        if row_t.empty:
            continue
        t_close[sym] = float(row_t["close"].iloc[0])
        future = g[g["date"] > last_date]
        if len(future) >= 20:
            t20_close[sym] = float(future["close"].iloc[19])
    idx_t = idx_df[idx_df["date"] == last_date]
    if idx_t.empty:
        return None, None
    idx_future = idx_df[idx_df["date"] > last_date]
    if len(idx_future) < 20:
        return None, None
    idx_ret = float(idx_future["close"].iloc[19]) / float(idx_t["close"].iloc[0]) - 1

    valid, y2s = [], []
    for sym in symbols:
        if sym in t_close and sym in t20_close:
            stock_ret = t20_close[sym] / t_close[sym] - 1
            valid.append(sym)
            y2s.append(np.clip(stock_ret - idx_ret, -0.5, 0.5))
    return valid, np.array(y2s)


def analyze_ic_report():
    """逐月 IC 三向对照：RankIC-LSTM vs Huber-LSTM（全量对照组）vs 生产 t4（参考）。"""
    from scipy.stats import spearmanr
    from build_prediction_cache import OUTPUT_PATH

    cache = json.loads(OUTPUT_PATH.read_text())
    exp = {}
    for mf in sorted(TMP_DIR.glob("*.json")):
        d = json.loads(mf.read_text())
        if "rankic" in d:
            exp[d["month"]] = {"symbols": d["symbols"], "rankic": d["rankic"], "huber": d["huber"]}

    months = sorted(set(cache.keys()) & set(exp.keys()))
    print(f"═══ 逐月 IC 三向对照（{len(months)} 个月，同股票交集）═══")

    rows = []
    for month in months:
        t4_syms = cache[month]["symbols"]
        ex_syms = exp[month]["symbols"]
        common = sorted(set(t4_syms) & set(ex_syms))
        if len(common) < 50:
            continue
        valid, y2 = _load_y2_for_month(month, common)
        if valid is None or len(valid) < 50:
            continue
        y2_aligned = np.array([y2[valid.index(s)] for s in valid])
        t4_pred = np.array([cache[month]["t4"][t4_syms.index(s)] for s in valid])
        rk_pred = np.array([exp[month]["rankic"][ex_syms.index(s)] for s in valid])
        hb_pred = np.array([exp[month]["huber"][ex_syms.index(s)] for s in valid])
        ic_t4, _ = spearmanr(t4_pred, y2_aligned)
        ic_rk, _ = spearmanr(rk_pred, y2_aligned)
        ic_hb, _ = spearmanr(hb_pred, y2_aligned)
        corr_rk_hb, _ = spearmanr(rk_pred, hb_pred)
        rows.append({
            "month": month, "n": len(valid),
            "t4_ic": round(ic_t4, 4), "rankic_ic": round(ic_rk, 4), "huber_ic": round(ic_hb, 4),
            "delta_rk_vs_hb": round(ic_rk - ic_hb, 4),
            "corr_rk_hb": round(corr_rk_hb, 4),
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    print(df[["month", "n", "t4_ic", "rankic_ic", "huber_ic", "delta_rk_vs_hb", "corr_rk_hb"]]
          .to_string(index=False))

    print("\n═══ 总体统计 ═══")
    print(f"  月份数: {len(df)}")
    print(f"  生产 t4 IC 均值(参考): {df['t4_ic'].mean():+.4f}")
    print(f"  Huber-LSTM IC 均值(全量对照): {df['huber_ic'].mean():+.4f}")
    print(f"  RankIC-LSTM IC 均值(实验): {df['rankic_ic'].mean():+.4f}")
    print(f"  RankIC vs Huber 差值均值: {df['delta_rk_vs_hb'].mean():+.4f}")
    print(f"  RankIC 赢过 Huber 的月份占比: {(df['delta_rk_vs_hb'] > 0).mean()*100:.0f}%")
    print(f"  RankIC 正 IC 月份占比: {(df['rankic_ic'] > 0).mean()*100:.0f}% (Huber: {(df['huber_ic'] > 0).mean()*100:.0f}%)")
    print(f"  corr(RankIC, Huber) 均值: {df['corr_rk_hb'].mean():+.3f}")

    print("\n═══ 按年度汇总 ═══")
    df["year"] = df["month"].str[:4]
    for y, g in df.groupby("year"):
        print(f"  {y}: 生产t4={g['t4_ic'].mean():+.3f} Huber={g['huber_ic'].mean():+.3f} "
              f"RankIC={g['rankic_ic'].mean():+.3f} rk胜率={(g['delta_rk_vs_hb']>0).mean()*100:.0f}%")

    ic_hb_m = df["huber_ic"].mean()
    ic_rk_m = df["rankic_ic"].mean()
    pos_hb = (df["huber_ic"] > 0).mean()
    pos_rk = (df["rankic_ic"] > 0).mean()
    print("\n═══ 门槛判定（V3 §8.5）═══")
    print(f"  ① RankIC IC 均值 > Huber 全量对照?  {ic_rk_m:+.4f} vs {ic_hb_m:+.4f} → {'✅ 达标' if ic_rk_m > ic_hb_m else '❌ 未达标'}")
    print(f"  ② 正 IC 月占比 ≥ Huber?  {pos_rk*100:.0f}% vs {pos_hb*100:.0f}% → {'✅ 达标' if pos_rk >= pos_hb else '❌ 未达标'}")
    print(f"  ③ corr(RankIC, Huber) < 0.5?  {df['corr_rk_hb'].mean():+.3f} → {'✅ 达标' if df['corr_rk_hb'].mean() < 0.5 else '❌ 未达标'}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(IC_REPORT_PATH, index=False)
    print(f"\n已保存: {IC_REPORT_PATH}")


def main():
    ap = argparse.ArgumentParser(description="V3 方向三：可微 RankIC 损失 vs Huber 损失 LSTM 对照")
    ap.add_argument("--month", help="仅跑单月（测试）")
    ap.add_argument("--start-month", default="2020-09")
    ap.add_argument("--end-month", default="2026-06")
    ap.add_argument("--max-stocks", type=int, default=0, help="限制股票池（测试）")
    ap.add_argument("--workers", type=int, default=N_POOL_WORKERS)
    ap.add_argument("--analyze", action="store_true", help="只做 IC 对照分析")
    args = ap.parse_args()

    if args.analyze:
        analyze_ic_report()
        return

    months = [args.month] if args.month else get_experiment_months(args.start_month, args.end_month)
    print(f"实验月份: {months[0]} ~ {months[-1]}（{len(months)} 个月）")
    run_experiment(months, max_pool_size=args.max_stocks, n_workers=args.workers)


if __name__ == "__main__":
    main()
