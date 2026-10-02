"""Phase 1: 构建月度预测缓存 —— 集中训练+预测，供 72 组回测共享。

对每个月（2025-08 ~ 2026-06，共 11 个月）：
  1. 从全量数据集提取 12 月滚动窗口训练数据
  2. 训练 T2 (LightGBM) + T1 (XGBoost) + T3 (CatBoost)
  3. 对全股票池（~2977只）批量预测
  4. 保存到 JSON 缓存文件

输出:
  output/backtest_v2/prediction_cache.json
  { "2025-08": { "symbols": [...], "t2": [...], "t1": [...], "t3": [...] }, ... }

用法:
  python scripts/build_prediction_cache.py                  # 全量
  python scripts/build_prediction_cache.py --months 3       # 仅3个月（测试）
  python scripts/build_prediction_cache.py --max-stocks 500 # 限制股票池（测试）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── 线程控制：必须在 import numpy 之前设置 ──
# 1) KMP_AFFINITY 必须清除：.bashrc 的 granularity=fine,compact,1,0 会把所有
#    worker 的主线程绑定到同一核心集 → 24 worker 抢 1 核 → CPU 1 核假象
#    （2026-08-01 实测：24 worker 全卡 Step1，nonvoluntary 切换 3376 次；
#     v1 的 8 worker × 4 线程恰好错开绑定所以正常）
# 2) OMP_NUM_THREADS 硬赋值 1（非 setdefault）：.bashrc 的 36 必须覆盖，
#    OpenBLAS 在 BLAS 首次调用时读此值；24 worker × 1 = 24 ≤ 36 核
os.environ.pop("KMP_AFFINITY", None)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sequoia_x.core.config import Settings
from sequoia_x.core.logger import get_logger
from sequoia_x.data.engine import DataEngine
from sequoia_x.model_selection_v2.config import V2Config, get_config

logger = get_logger("build_prediction_cache")

OUTPUT_PATH = Path("output/backtest_v2/prediction_cache.json")


def get_test_months(start_month: str, end_month: str) -> list[str]:
    """获取需要预测的月份列表。"""
    start_ym = (int(start_month[:4]), int(start_month[5:7]))
    end_ym = (int(end_month[:4]), int(end_month[5:7]))

    months = []
    y, m = start_ym
    while (y, m) <= end_ym:
        months.append(f"{y}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def load_full_dataset(cfg: V2Config, engine: DataEngine, cache_dir=None):
    """直接加载已知缓存（绕过 build_training_dataset 避免 baostock 和哈希不匹配）。

    Args:
        cache_dir: 缓存目录（build_cache 按 cfg.extra_features 计算, 2026-08-07 动态化）。
                   None=默认 88 维缓存 13132147f8e8。
    """
    import json as _json
    from pathlib import Path as _Path

    # 使用已知的缓存目录（由之前的 build_training_dataset 创建）
    # 2026-08-07: extra_features=True 时由调用方传入 121 维缓存目录（hash 不同）
    if cache_dir is None:
        cache_dir = Path("data/cache/v2_dataset/13132147f8e8")
    cache_path = _Path(cache_dir)

    # 缓存存在性检查（2026-08-07: 目录由 hash 动态计算, 股票池/特征开关变化可能失配）
    # 缺失时给出清晰指引而非 np.load 文件错误（8月重训前需先重建缓存）
    if not (cache_path / "metadata.json").exists():
        raise FileNotFoundError(
            f"训练数据集缓存不存在: {cache_path}\n"
            f"请先运行 python scripts/rebuild_dataset_cache.py --only-88 重建"
            f"（extra_features=True 时需在 config.py 开启后再重建）")

    logger.info(f"从缓存直接加载全量数据集: {cache_path}...")
    t0 = time.time()

    X = np.load(str(cache_path / "X.npy"), mmap_mode="r")
    y1 = np.load(str(cache_path / "y1.npy"), mmap_mode="r")
    y2 = np.load(str(cache_path / "y2.npy"), mmap_mode="r")
    y3 = np.load(str(cache_path / "y3.npy"), mmap_mode="r")
    with open(cache_path / "dates.json") as f:
        dates = _json.load(f)

    # ── 加载侧不变量 fail-fast（2026-10-02 新增）──
    #   与 labels.py::_save_dataset_cache 的保存侧断言对称。9/30 事故就是坏缓存被
    #   静默加载, 一路走到 _process_month_worker 的 `X[mask]` 才抛 IndexError
    #   (X=418434 vs dates=412491) —— 报错点离病根很远, 排障多花数小时。
    #   这里提前报错: 病根(cache_dir)直接写在错误信息里。
    _n = X.shape[0]
    _ys = (y1.shape[0], y2.shape[0], y3.shape[0])
    if len(dates) != _n or any(v != _n for v in _ys):
        raise AssertionError(
            f"❌ 训练缓存数据不变量破坏: {cache_path.name} "
            f"X={_n}行, y=({_ys[0]},{_ys[1]},{_ys[2]}), dates={len(dates)}条"
            f"（X-dates 差={_n - len(dates)}）。该缓存不可信, 请删除后重建, 不要复用。"
        )

    elapsed = time.time() - t0
    logger.info(f"数据集加载完成: X={X.shape}, {len(set(dates))} 采样日期, {elapsed:.0f}s")
    return X, y1, y2, y3, dates


def _assert_pred_aligned(month: str, symbols: list, preds: dict) -> None:
    """断言「符号 ↔ 预测数组」逐行对齐（2026-10-02 加，同属"改行数必须联动"防护）。

    为什么要这道保险丝: 这份 preds 决定**买哪 10 只股票** —— symbols[i] 必须就是
    t2/t1/t3/t4[i] 的那只。长度一旦不一致（例如日后有人把 build_batch_features 改成
    "只过滤 X 不过滤符号"），下游 `[symbols[i] for i in argsort(-t2)]` 会**静默错位**：
    买错股票、且不报错——这是本项目最危险的一类故障（比崩溃危险得多）。
    与 labels.py 的缓存不变量断言、T4 合并的"按 symbol 对齐"是同一族保护。
    """
    n = len(symbols)
    bad = {k: len(v) for k, v in preds.items() if len(v) != n}
    if bad:
        raise AssertionError(
            f"[{month}] ❌ 符号与预测数组长度不一致: symbols={n}, 异常={bad} "
            f"→ 拒绝写缓存（若继续, 下游按序号取股票会静默错位）"
        )


def extract_training_data(
    X: np.ndarray, y: np.ndarray, dates_arr: np.ndarray,
    train_end_date: str, train_months: int = 12,
) -> np.ndarray:
    """提取训练窗口数据（标签）。"""
    end_ym = train_end_date[:7]
    end_year, end_month = int(end_ym[:4]), int(end_ym[5:7])
    start_month = end_month - train_months
    start_year = end_year
    if start_month <= 0:
        start_month += 12
        start_year -= 1
    train_start = f"{start_year}-{start_month:02d}-01"
    mask = (dates_arr >= train_start) & (dates_arr <= train_end_date)
    return y[mask]


def extract_training_xy(
    X: np.ndarray, y2: np.ndarray, dates_arr: np.ndarray,
    train_end_date: str, train_months: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """提取训练窗口数据（特征+标签）。

    Returns:
        (X_tr_3d, y_tr, X_tr_2d): LSTM用3D, 树模型用2D, 标签向量。
    """
    end_ym = train_end_date[:7]
    end_year, end_month = int(end_ym[:4]), int(end_ym[5:7])
    start_month = end_month - train_months
    start_year = end_year
    if start_month <= 0:
        start_month += 12
        start_year -= 1
    train_start = f"{start_year}-{start_month:02d}-01"
    mask = (dates_arr >= train_start) & (dates_arr <= train_end_date)
    n_train = mask.sum()
    if n_train < 100:
        return np.array([]), np.array([]), np.array([])
    X_tr = X[mask]
    y_tr = y2[mask]
    X_tr_2d = X_tr.reshape(n_train, -1)
    return X_tr, y_tr, X_tr_2d


def predict_full_pool(
    stock_pool: list[str],
    ref_date: str,
    t2_model,
    t1_model=None,
    t3_model=None,
    t4_model=None,
    cfg: V2Config | None = None,
    db_path: str = "data/sequoia_v2.db",
    max_pool_size: int = 0,
    include_extra: bool = False,
) -> dict:
    """对全股票池批量预测。

    Args:
        include_extra: True 时拼接 33 维扩展特征(121维), 关键面缺失股票不产生信号。

    Returns:
        {"symbols": [...], "t2": [...], "t1": [...], "t3": [...], "t4": [...]}
    """
    from sequoia_x.model_selection_v2.features import build_batch_features
    from sequoia_x.model_selection_v2.models.tree_reg import predict_reg
    from sequoia_x.model_selection_v2.models.tree_cls import predict_cls
    from sequoia_x.model_selection_v2.models.tree_vol import predict_vol

    if cfg is None:
        cfg = get_config()

    # 限制股票池
    pool = stock_pool
    if max_pool_size > 0 and len(pool) > max_pool_size:
        import random
        random.seed(42)
        pool = random.sample(pool, max_pool_size)

    n_total = len(pool)
    n_workers = min(8, (os.cpu_count() or 4) - 2, n_total)
    use_parallel = n_workers >= 2 and n_total >= 200

    logger.info(f"  构建特征: {n_total} 只 (ref={ref_date})"
                f"{', ' + str(n_workers) + '进程并行' if use_parallel else ''}..."
                f"{' [88+33=121维扩展特征]' if include_extra else ''}")
    t_feat = time.time()

    if use_parallel:
        from multiprocessing import Pool
        from sequoia_x.model_selection_v2.backtest.monthly_engine import \
            _build_features_chunk

        chunks = np.array_split(list(pool), n_workers)
        task_args = [(list(c), ref_date, db_path, cfg, include_extra) for c in chunks]

        with Pool(n_workers) as p:
            chunk_results = p.map(_build_features_chunk, task_args)

        X_list, sym_list = [], []
        for X_chunk, syms_chunk in chunk_results:
            if len(X_chunk) > 0:
                X_list.append(X_chunk)
                sym_list.extend(syms_chunk)
        X_pred = np.concatenate(X_list) if X_list else np.array([])
        valid_symbols = sym_list
    else:
        X_pred, valid_symbols = build_batch_features(
            pool, ref_date, DataEngine(Settings()), cfg, include_extra=include_extra)

    logger.info(f"  特征完成: {len(valid_symbols)}/{n_total} 有效 ({time.time()-t_feat:.0f}s)")

    if len(X_pred) == 0:
        return {"symbols": [], "t2": [], "t1": [], "t3": []}

    n_valid = len(X_pred)
    X_pred_2d = X_pred.reshape(n_valid, -1)

    # T2 预测
    t_pred = time.time()
    pred_t2 = predict_reg(t2_model, X_pred_2d).flatten()
    logger.info(f"  T2预测完成 ({time.time()-t_pred:.0f}s)")

    # T1 预测
    pred_t1 = np.zeros(n_valid)
    if t1_model is not None:
        pred_t1 = predict_cls(t1_model, X_pred_2d).flatten()

    # T3 预测
    pred_t3 = np.zeros(n_valid)
    if t3_model is not None:
        pred_t3 = predict_vol(t3_model, X_pred_2d).flatten()

    # T4 预测
    pred_t4 = np.zeros(n_valid)
    if t4_model is not None:
        from sequoia_x.model_selection_v2.models.deep_lstm import predict_lstm
        pred_t4 = predict_lstm(t4_model, X_pred).flatten()
        logger.info(f"  T4预测完成")
        # 铁律一：验证预测方差
        t4_std = float(np.std(pred_t4))
        if t4_std < 1e-7:
            logger.error(f"  ❌ T4预测值无方差！std={t4_std:.2e}")
        else:
            logger.info(f"  ✅ T4 pred: mean={pred_t4.mean():.4f} std={t4_std:.4f}")

    _preds = {
        "symbols": valid_symbols,
        "t2": [float(v) for v in pred_t2],
        "t1": [float(v) for v in pred_t1],
        "t3": [float(v) for v in pred_t3],
        "t4": [float(v) for v in pred_t4],
    }
    _assert_pred_aligned("pred", valid_symbols, _preds)   # 行对齐保险丝（2026-10-02）
    return _preds


# 特征构建并行数（2026-08-02：单只股票特征计算无状态，可安全并行）
FEAT_WORKERS = 8

# ── 特征视图（2026-09-26 新增，实验开关；默认 full = 与既有行为**逐字节一致**）──
# full        : (120天 × 129维) 展平 = 15,480 维（现状）
# lastday_agg : 最后一天 129 维 + 窗口均值 129 + 窗口标准差 129 = 387 维
# 依据：实测 10 切点 × 13 样本外评估日（n=130），387 维的 T2 Rank IC 点估计更高
#       （+0.064 vs +0.051，p=0.15 不显著），但训练快 **13.7 倍**（4s vs 60s）。
#       故定位为"实验加速用"；**换生产需先跑 70 月回测看收益/夏普/回撤**。
FEATURE_VIEWS = ("full", "lastday_agg")


def _apply_feature_view(X3d: np.ndarray, view: str) -> np.ndarray:
    """把 (n, window, n_feat) 的时间序列块转成模型输入矩阵。

    full 分支为 X3d.reshape(n, -1)，与改动前的写法完全等价（默认路径零风险）。
    """
    if view == "full":
        return X3d.reshape(len(X3d), -1)
    if view == "lastday_agg":
        return np.concatenate([X3d[:, -1, :], X3d.mean(axis=1), X3d.std(axis=1)], axis=1)
    raise ValueError(f"未知 feature_view: {view}（可选 {FEATURE_VIEWS}）")


def _build_one_features(args: tuple):
    """单只股票特征构建（Step5 并行化用，模块级供 Pool 调用）。

    Args:
        args: (symbol, df, idx_df, cfg, include_extra)
    Returns:
        (symbol, X_i) 或 None（数据不足/扩展维度关键面缺失）。
    """
    from sequoia_x.model_selection_v2.features import _extract_per_day_features
    sym, df, idx_df, cfg, include_extra = args
    if df is None or len(df) < cfg.window + 10:
        return None
    # 扩展维度特征（可选, 2026-08-07）: 关键面缺失 → 不产生信号
    extra_matrix = None
    if include_extra:
        from sequoia_x.features_extra.build_extra_features import build_extra_with_flag
        import pandas as pd
        dates = pd.DatetimeIndex(pd.to_datetime(df["date"]))
        close = pd.Series(df["close"].values.astype(float), index=dates, name="close")
        extra, incomplete, _ = build_extra_with_flag(dates, close, sym)
        if incomplete:
            return None
        extra_matrix = extra.values.astype(np.float32)
    per_day = _extract_per_day_features(
        df, idx_df if idx_df is not None and len(idx_df) else None, cfg,
        extra_matrix=extra_matrix, symbol=sym,
    )
    if len(per_day) < cfg.window:
        return None
    return sym, per_day[-cfg.window:]


def _synth_series_samples(series_dir: str, db_path: str, cfg, include_extra: bool):
    """V3 修订二: 合成完整序列 → 特征 → 滑窗样本（真·数据增强, 2026-08-09）。

    与 _synth_samples（标签替换）的本质区别: 特征+标签全从合成序列计算（自洽）,
    扩充的是样本量本身。每只合成序列(300 天) → ~160 个样本 (X(120,88), y2 20 日)。
    2026-08-10: 121 维支持——合成序列前 120 天为真实种子段, 扩展特征 = 种子股票
    最新基本面快照广播（语义: 合成序列是种子股票的价格延续）; 种子数据缺失 → 全 0
    （与真实缺失 fillna(0) 同语义）。
    """
    import sqlite3 as _sqlite3
    import pandas as _pd
    from sequoia_x.model_selection_v2.features import _extract_per_day_features
    from pathlib import Path as _Path
    sdir = _Path(series_dir)
    if not sdir.exists():
        return np.array([]), np.array([])
    conn = _sqlite3.connect(db_path)
    idx_df = _pd.read_sql(
        "SELECT date, open, high, low, close, volume FROM index_daily "
        "WHERE symbol='sh.000300' ORDER BY date", conn)
    conn.close()
    Xs, ys = [], []
    for fp in sorted(sdir.glob("syn_*.csv")):
        try:
            series = _pd.read_csv(fp)
            if len(series) < cfg.window + 40:
                continue
            extra_matrix = None
            if include_extra:
                extra_matrix = _synth_extra_matrix(series, fp.stem.replace("syn_", ""))
            per_day = _extract_per_day_features(series, idx_df, cfg,
                                                extra_matrix=extra_matrix)
            arr = np.asarray(per_day, dtype=float)
            if len(arr) < cfg.window + 20 or np.isnan(arr).any():
                continue
            X = np.lib.stride_tricks.sliding_window_view(
                arr[:len(arr) - 20], cfg.window, axis=0).transpose(0, 2, 1)
            y = (series["close"].values[cfg.window + 20:] /
                 series["close"].values[cfg.window:-20] - 1)
            n = min(len(X), len(y))
            Xs.append(X[:n])
            ys.append(y[:n])
        except Exception:
            continue
    if not Xs:
        return np.array([]), np.array([])
    return np.concatenate(Xs).astype(np.float32), np.concatenate(ys).astype(np.float32)


def _synth_extra_matrix(series: pd.DataFrame, seed_code: str):
    """合成序列的扩展特征矩阵: 种子股票最新基本面快照广播到全序列。

    合成序列前 120 行为真实种子段（日期真实, DB 有扩展数据）→ 取种子段最后
    可用扩展行 → tile 到全序列长度。种子无数据 → 全 0（与真实缺失同语义）。
    """
    import pandas as _pd
    try:
        from sequoia_x.features_extra.build_extra_features import build_extra_with_flag
        seed_part = series.head(120)
        dates = _pd.DatetimeIndex(_pd.to_datetime(seed_part["date"]))
        close = _pd.Series(seed_part["close"].values.astype(float), index=dates, name="close")
        extra, incomplete, _ = build_extra_with_flag(dates, close, seed_code)
        if extra is not None and len(extra):
            last = extra.ffill().iloc[-1].fillna(0).values
            return np.tile(last, (len(series), 1)).astype(np.float32)
    except Exception:
        pass
    return np.zeros((len(series), 33), dtype=np.float32)   # 33 维扩展（88+33=121）


def _synth_samples(synth_file: str, db_path: str, cfg, include_extra: bool,
                   ratio: float = 1.0):
    """V3 修订二: 为合成标签 (symbol, ref) 重算特征 → (X_syn, y_syn)。

    合成样本: X = ref 前 120 天真实特征窗口（复用 _build_one_features）,
    y = Kronos 合成 20 日收益（校准后, 与 y2 同口径）。
    ratio: 注入比例（2026-08-09 占比试调: 1.0=全量24%, 0.25≈5%, 0.5≈10%,
    固定 seed 抽样子集, 可复现）。
    仅 88 维模式支持（include_extra 时扩展特征对合成样本不可用 → 返回空）。
    """
    import json as _json
    import sqlite3 as _sqlite3
    import pandas as _pd
    if include_extra:
        return np.array([]), np.array([])
    synth_map = _json.loads(Path(synth_file).read_text())
    conn = _sqlite3.connect(db_path)
    idx_df = _pd.read_sql(
        "SELECT date, open, high, low, close, volume FROM index_daily "
        "WHERE symbol='sh.000300' ORDER BY date", conn)
    Xs, ys = [], []
    for symbol, ref_map in synth_map.items():
        for ref, y_list in ref_map.items():
            df = _pd.read_sql(
                "SELECT date, open, high, low, close, volume, amount FROM stock_daily "
                "WHERE symbol=? AND date<=? ORDER BY date DESC LIMIT 200",
                conn, params=[symbol, ref])
            if len(df) < cfg.window + 10:
                continue
            df = df.iloc[::-1].reset_index(drop=True)
            r = _build_one_features((symbol, df, idx_df, cfg, False))
            if r is None:
                continue
            _, X_i = r
            # 路径级标签: 每条采样路径 = 一个训练样本（X 同, y 不同 → 保留采样多样性）
            if isinstance(y_list, list):
                for y_syn in y_list:
                    Xs.append(X_i)
                    ys.append(y_syn)
            else:
                Xs.append(X_i)
                ys.append(y_list)
    conn.close()
    if not Xs:
        return np.array([]), np.array([])
    # 占比抽样（2026-08-09）: ratio<1 时固定 seed 抽子集（可复现, 试调占比用）
    if ratio < 1.0 and len(ys) > 1:
        rng = __import__("random").Random(42)
        idx = sorted(rng.sample(range(len(ys)), max(1, int(len(ys) * ratio))))
        Xs = [Xs[i] for i in idx]
        ys = [ys[i] for i in idx]
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def _process_month_worker(args: tuple) -> tuple:
    """处理单个月份的训练+预测（模块级函数，供 multiprocessing 使用）。

    Args:
        args: (month, db_path, cfg_dict, max_pool_size, skip_t4, cache_dir,
               include_extra, synth_file)

    Returns:
        (month, predictions_dict) 或 (month, None) 若失败。
    """
    month, db_path, cfg_dict, max_pool_size, skip_t4, cache_dir, include_extra, synth_file, synth_ratio, synth_series_dir = args[:10]

    import json as _json
    import sqlite3
    import numpy as np
    from pathlib import Path as _Path

    # ── 铁律一：启动诊断 + 限制 OpenMP 线程数 + 清除 KMP_AFFINITY ──
    # KMP_AFFINITY=compact 会把各 worker 主线程绑到同一核心（24 worker 抢 1 核），
    # 必须在 OpenMP 库初始化（BLAS 首次调用）前清除；OMP=1：24 worker × 1 = 24 ≤ 36 核
    _os = __import__('os')
    _cpu_count = _os.cpu_count()
    _cwd = _os.getcwd()
    _os.environ.pop('KMP_AFFINITY', None)
    _os.environ['OMP_NUM_THREADS'] = '1'
    _os.environ['OPENBLAS_NUM_THREADS'] = '1'
    _os.environ['MKL_NUM_THREADS'] = '1'
    _os.environ['NUMEXPR_NUM_THREADS'] = '1'
    print(f"[Worker {month}] 启动诊断: CPU核={_cpu_count} OMP=1 KMP清除 CWD={_cwd}"
          f"{' extra=121维' if include_extra else ''}", flush=True)

    # ── 加载 mmap 数据（使用绝对路径，子进程 CWD 可能不同）──
    # 缓存目录由父进程按 cfg.extra_features 计算（2026-08-07: 121维走新 hash 目录）
    cache_path = _Path(cache_dir)
    X = np.load(str(cache_path / "X.npy"), mmap_mode="r")
    y1 = np.load(str(cache_path / "y1.npy"), mmap_mode="r")
    y2 = np.load(str(cache_path / "y2.npy"), mmap_mode="r")
    y3 = np.load(str(cache_path / "y3.npy"), mmap_mode="r")
    with open(cache_path / "dates.json") as f:
        dates = _json.load(f)
    dates_arr = np.array(dates)
    print(f"[Worker {month}] mmap加载: X={X.shape}, {len(set(dates))}采样日期", flush=True)

    # ── 确定训练截止日 ──
    ym_year = int(month[:4])
    ym_month = int(month[5:7])
    prev_m = ym_month - 1
    prev_y = ym_year
    if prev_m <= 0:
        prev_m += 12
        prev_y -= 1
    conn = sqlite3.connect(db_path)
    last_date_row = conn.execute(
        "SELECT MAX(date) FROM stock_daily WHERE date >= ? AND date < ?",
        (f"{prev_y}-{prev_m:02d}-01", month + "-01"),
    ).fetchone()
    train_end_date = last_date_row[0] if last_date_row and last_date_row[0] else (month + "-01")
    # ── 训练样本 purge（2026-09-26 起**默认开启**，不再是实验开关）──
    # 为什么必须 purge：采样日 D 的标签 y2 = D **之后** 20 个交易日的收益。若 D 距
    #   train_end 不足 20 个交易日，该标签就用到了 train_end 之后的数据 —— 而 train_end
    #   之后正是**被预测的那个月** → 模型等于用答案训练。
    #   原 `X_tr[-5000:]`（取尾部）使该泄漏覆盖 **100% 训练集**，实测把 IC 从
    #   +0.015（真实）抬到 +0.224（70 月）/ +0.428（18 月），且 99% 的月份为正。
    #   详见 docs/2026-09_回测IC异常排查清单.md
    # V2_TRAIN_PURGE_DAYS=N：只保留"标签已完整"的样本（D ≤ train_end 往前 N 个交易日）。
    #   默认 25（> 标签期 20，留安全边界）。**要复现泄漏必须显式设 V2_ALLOW_LEAK=1**。
    import os as _os
    _purge = int(_os.environ.get("V2_TRAIN_PURGE_DAYS", "25") or 0)
    _allow_leak = _os.environ.get("V2_ALLOW_LEAK", "0") == "1"
    if _purge < 20 and not _allow_leak:
        raise RuntimeError(
            f"[{month}] 训练集含前视：V2_TRAIN_PURGE_DAYS={_purge} < 20（标签=未来 20 交易日"
            f"收益，会跨入被预测月）。真要做泄漏对照实验请显式设 V2_ALLOW_LEAK=1。")
    _purge_cutoff = ""
    if _purge > 0:
        _cal = [r[0] for r in conn.execute(
            "SELECT DISTINCT date FROM stock_daily WHERE date <= ? ORDER BY date",
            (train_end_date,)).fetchall()]
        if len(_cal) > _purge:
            _purge_cutoff = _cal[-_purge - 1]
    # 从缓存文件读取标准股票池（由父进程 baostock 获取）
    stock_pool_path = _Path(db_path).parent.parent / "output/backtest_v2/.stock_pool.json"
    if stock_pool_path.exists():
        stock_pool = _json.loads(stock_pool_path.read_text())
    else:
        all_symbols = [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM stock_daily").fetchall()]
        stock_pool = _filter_stock_pool(all_symbols, db_path)
    conn.close()

    # ── 提取训练数据 ──
    end_ym = train_end_date[:7]
    ey, em = int(end_ym[:4]), int(end_ym[5:7])
    sm = em - 12; sy = ey
    if sm <= 0:
        sm += 12; sy -= 1
    train_start = f"{sy}-{sm:02d}-01"
    mask = (dates_arr >= train_start) & (dates_arr <= train_end_date)
    if _purge_cutoff:
        mask &= (dates_arr <= _purge_cutoff)
        print(f"[Worker {month}] 训练 purge={_purge} 交易日 → 上界 {_purge_cutoff}"
              f"（原 {train_end_date}），剩余样本 {int(mask.sum()):,}", flush=True)
    n_train = mask.sum()
    if n_train < 100:
        # 2026-09-27（审计 I2b）：原为静默 `return month, None` —— 月份**无声消失**且退出码为 0，
        #   于是文档写"70 月"而实际缓存只有 69 月（2020-09 purge 后样本为 0）。
        #   样本不足属**数据边界**而非故障（不计为失败），但必须**可见**：
        #   这里打印 + 父进程末尾汇总"跳过月"（见 build_cache 的 skipped 报告）。
        print(f"[Worker {month}] ⚠️ 跳过该月：purge={_purge} 后训练样本仅 {int(n_train)}（<100）",
              flush=True)
        return month, None
    # ── 防回归断言（2026-09-26）──
    # 判据**独立于**上面的 purge 逻辑：直接算"train_end 往前 20 个交易日"，要求所有
    # 训练样本 ≤ 它。这样即使日后有人改坏 purge，只要泄漏回来就会立刻炸，而不是静默
    # 把 IC 从 0.015 抬到 0.22。
    if not _allow_leak:
        _need = _cal[-21] if len(_cal) > 20 else train_end_date
        # 注意：dates_arr 是**字符串**数组，np 的 .max() 会走 np.maximum 并抛
        #   UFuncNoLoopError（dtype('<U10') 无 loop）——用 Python 内建 max（ISO 日期
        #   字典序 = 时序）✓
        _maxd = max(dates_arr[mask])
        if _maxd > _need:
            raise RuntimeError(
                f"[{month}] 防回归断言失败：训练集最晚采样日 {_maxd} > 允许上界 {_need}"
                f"（= train_end {train_end_date} 往前 20 交易日）→ 标签跨入预测期（前视）。"
                f"详见 docs/2026-09_回测IC异常排查清单.md")

    X_tr = X[mask]
    y_tr = y2[mask]
    X_tr_2d = X_tr.reshape(n_train, -1)
    y1_tr = y1[mask]
    y3_tr = y3[mask]

    # 抽样 5000
    # 2026-09-26 实验开关 V2_TRAIN_SAMPLE_MODE：
    #   tail(默认) = X_tr[-5000:] 取**时间上最后**的 5000 条 → 全部落在窗口末端，
    #                其标签（未来 20 交易日）跨过 train_end 进到被预测月 → 泄漏
    #   random     = 从同窗口**随机**抽 5000 条（固定种子）→ 时间跨度完整，泄漏比例按
    #                抽样比例稀释。样本量与窗口**完全相同**，只有"取哪 5000 条"不同。
    MAX_TRAIN_SAMPLES = 5000
    if n_train > MAX_TRAIN_SAMPLES:
        _mode = _os.environ.get("V2_TRAIN_SAMPLE_MODE", "tail")
        if _mode == "random":
            _sel = np.random.RandomState(42).choice(n_train, MAX_TRAIN_SAMPLES, replace=False)
            _sel.sort()                     # 保持时间序（下游有依赖时序的假设）
            print(f"[Worker {month}] 取样模式=random（跨度 {dates_arr[mask][_sel][0]} ~ "
                  f"{dates_arr[mask][_sel][-1]}）", flush=True)
        else:
            _sel = np.arange(n_train - MAX_TRAIN_SAMPLES, n_train)
        X_tr = X_tr[_sel]
        y_tr = y_tr[_sel]
        y1_tr = y1_tr[_sel]
        y3_tr = y3_tr[_sel]
        X_tr_2d = X_tr.reshape(len(X_tr), -1)

    # ── 重建 cfg ──
    from sequoia_x.model_selection_v2.config import V2Config
    cfg = V2Config()
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)
    cfg.extra_features = include_extra  # 特征拼接开关（父进程 cfg_dict 同值）

    # ── 合成增强（V3 修订二, 2026-08-09）: 追加 Kronos 合成样本, 只进 T2/T4 ──
    X_tr_enh, y_tr_enh, X_tr_2d_enh = X_tr, y_tr, X_tr_2d
    if synth_series_dir:
        X_syn, y_syn = _synth_series_samples(synth_series_dir, db_path, cfg, include_extra)
        tag = "完整序列"
    elif synth_file and Path(synth_file).exists():
        X_syn, y_syn = _synth_samples(synth_file, db_path, cfg, include_extra,
                                      synth_ratio)
        tag = "标签替换"
    else:
        X_syn, y_syn = np.array([]), np.array([])
        tag = ""
    if len(y_syn) > 0:
        X_tr_enh = np.concatenate([X_tr, X_syn])
        y_tr_enh = np.concatenate([y_tr, y_syn])
        X_tr_2d_enh = X_tr_enh.reshape(len(X_tr_enh), -1)
        print(f"[Worker {month}] 合成增强({tag}): +{len(y_syn)} 样本 "
              f"({len(X_tr)}→{len(X_tr_enh)}, 仅 T2/T4)", flush=True)
        if len(y_syn) > 0:
            X_tr_enh = np.concatenate([X_tr, X_syn])
            y_tr_enh = np.concatenate([y_tr, y_syn])
            X_tr_2d_enh = X_tr_enh.reshape(len(X_tr_enh), -1)
            print(f"[Worker {month}] 合成增强: +{len(y_syn)} 样本 "
                  f"({len(X_tr)}→{len(X_tr_enh)}, 仅 T2/T4)", flush=True)
        else:
            print(f"[Worker {month}] ⚠️ 合成样本为 0（88 维模式才支持）, 跳过", flush=True)

    # ── 特征视图（2026-09-26 实验开关）──
    # 默认 full 时，_apply_feature_view 就是 reshape(n,-1)，与改动前**逐字节一致**；
    # 放在合成增强之后，保证 X_tr_enh 已是最终训练块（含合成样本）。
    feature_view = getattr(cfg, "feature_view", "full")
    X_tr_2d = _apply_feature_view(X_tr, feature_view)
    X_tr_2d_enh = _apply_feature_view(X_tr_enh, feature_view)
    if feature_view != "full":
        n_full = X_tr_enh.shape[1] * X_tr_enh.shape[2]
        print(f"[Worker {month}] 特征视图={feature_view} → 树模型训练维度 "
              f"{X_tr_2d_enh.shape[1]:,}（full 模式为 {n_full:,}）；T4 仍用 3D 序列不受影响",
              flush=True)

    # ── 训练 T2 ──
    # 2026-09-26 实验开关 V2_T2_ALGO（默认 lightgbm = 原行为）：验证"IC≈0 是不是算法问题"。
    # 后两者用固定轮数（不做早停）——与 LightGBM 的 early_stopping 口径略有差异，
    # 仅用于"换算法会不会变好"的粗判，不作为精细调参结论。
    print(f"[Worker {month}] Step1: T2训练(samples={len(X_tr_enh)})...", flush=True)
    _t2_algo = _os.environ.get("V2_T2_ALGO", "lightgbm").lower()
    if _t2_algo == "xgboost":
        from xgboost import XGBRegressor
        t2_model = XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
            colsample_bytree=0.8, n_jobs=cfg.n_jobs, random_state=cfg.random_seed)
        t2_model.fit(X_tr_2d_enh, y_tr_enh)
        print(f"[Worker {month}] 算法=xgboost", flush=True)
    elif _t2_algo == "catboost":
        from catboost import CatBoostRegressor
        t2_model = CatBoostRegressor(
            iterations=800, learning_rate=0.05, depth=6, verbose=0,
            random_seed=cfg.random_seed, thread_count=cfg.n_jobs)
        t2_model.fit(X_tr_2d_enh, y_tr_enh)
        print(f"[Worker {month}] 算法=catboost", flush=True)
    else:
        from sequoia_x.model_selection_v2.models.tree_reg import train_reg
        t2_model = train_reg(X_tr_2d_enh, y_tr_enh, cfg, search_optuna=False)

    # ── 训练 T1 ──
    print(f"[Worker {month}] Step2: T1训练...", flush=True)
    from sequoia_x.model_selection_v2.models.tree_cls import train_cls
    t1_model = train_cls(X_tr_2d, y1_tr, cfg, search_optuna=False)

    # ── 训练 T3 ──
    print(f"[Worker {month}] Step3: T3训练...", flush=True)
    from sequoia_x.model_selection_v2.models.tree_vol import train_vol
    t3_model = train_vol(X_tr_2d, y3_tr, cfg, search_optuna=False)
    print(f"[Worker {month}] 训练全部完成", flush=True)

    # ── T4 ──
    t4_model = None
    if not skip_t4:
        try:
            from sequoia_x.model_selection_v2.models.deep_lstm import train_lstm
            t4_model = train_lstm(X_tr_enh, y_tr_enh, cfg, search_optuna=False,
                                  model_id=f"cache_{month}")
        except Exception as e:
            # 2026-09-26：原来是 `except Exception: pass` —— T4 训练失败会被**完全静默**，
            #   该月输出里 t4 全 0 却无任何提示（生产 T2+T4 融合选股会悄悄少一半信号，
            #   且事后无法区分"T4 没能力"与"T4 没跑成"）。失败必须可见。
            import traceback as _tb
            print(f"\n[Worker {month}] ❌ T4(LSTM) 训练失败，该月 t4 将为全 0："
                  f"{type(e).__name__}: {e}\n{_tb.format_exc()}", flush=True)

    # ── 预测：一次 SQL 预加载 OHLCV → 内存特征构建（消除 SQLite 锁竞争）──
    print(f"[Worker {month}] Step4: OHLCV预加载...", flush=True)
    import pandas as pd
    from sequoia_x.model_selection_v2.features import _extract_per_day_features
    from sequoia_x.model_selection_v2.models.tree_reg import predict_reg
    from sequoia_x.model_selection_v2.models.tree_cls import predict_cls
    from sequoia_x.model_selection_v2.models.tree_vol import predict_vol

    pool = stock_pool
    if max_pool_size > 0 and len(pool) > max_pool_size:
        import random
        random.seed(42)
        pool = random.sample(pool, max_pool_size)

    print(f"[Worker {month}] Step4a: SQL查询({len(pool)}只)...", flush=True)
    conn = sqlite3.connect(db_path)
    ph = ','.join('?' * len(pool))
    ohlcv_df = pd.read_sql(
        f"SELECT * FROM stock_daily WHERE symbol IN ({ph}) AND date <= ? ORDER BY symbol, date",
        conn, params=pool + [train_end_date])
    # 2026-09-26 修复：原打印的 "...沪深300..." 是**源码里写死的省略号**，把
    # 「指数到底加载了几行」彻底藏住了 —— 而 §6/§6b 那 14 维是否生效正取决于它
    # （2026-09-25 排查「指数取数走散」时，日志里查不到任何线索）。
    # 改为查询后打印真实行数 + 日期范围；加载为空时显式告警。
    idx_df = pd.read_sql(
        "SELECT * FROM index_daily WHERE symbol='sh.000300' AND date <= ? ORDER BY date",
        conn, params=(train_end_date,))
    conn.close()
    idx_desc = (f"{len(idx_df)}行 ({idx_df['date'].min()} ~ {idx_df['date'].max()})"
                if len(idx_df) else "❌ 空！§6/§6b 将走全零分支")
    print(f"[Worker {month}] Step4b: OHLCV={len(ohlcv_df)}行, 沪深300={idx_desc}", flush=True)
    print(f"[Worker {month}] Step4c: 分组...", flush=True)
    ohlcv_cache = {sym: g.reset_index(drop=True) for sym, g in ohlcv_df.groupby('symbol')}
    print(f"[Worker {month}] Step5: 特征构建({len(ohlcv_cache)}只有OHLCV, {FEAT_WORKERS}进程并行)...", flush=True)

    # 纯内存特征构建（零 SQLite，多进程并行分块）
    # 2026-08-02 并行化：单只股票特征计算无状态 → 分块并行，8 进程 × 1 线程
    # ⚠️ 必须用 ProcessPoolExecutor（其 worker 非 daemon，可嵌套）——
    #    multiprocessing.Pool 的 worker 是 daemon，禁止再创建子进程
    #    （build 的 Pool(1) worker 内开 Pool(8) 会报 "daemonic processes..."）
    from concurrent.futures import ProcessPoolExecutor
    tasks = [(sym, ohlcv_cache.get(sym), idx_df, cfg, include_extra) for sym in pool]
    with ProcessPoolExecutor(max_workers=FEAT_WORKERS) as _ex:
        _results = list(_ex.map(_build_one_features, tasks, chunksize=100))
    X_list = [r[1] for r in _results if r is not None]
    sym_list = [r[0] for r in _results if r is not None]

    print(f"[Worker {month}] Step6: 预测({len(sym_list)}只有效)...", flush=True)
    if not X_list:
        return month, None
    X_pred = np.stack(X_list, axis=0)
    valid_symbols = sym_list
    n_valid = len(X_pred)
    if n_valid == 0:
        return month, None
    # 预测侧必须用与训练侧**同一个特征视图**，否则维度/语义都不匹配
    X_pred_2d = _apply_feature_view(X_pred, getattr(cfg, "feature_view", "full"))
    pred_t2 = predict_reg(t2_model, X_pred_2d).flatten()
    pred_t1 = predict_cls(t1_model, X_pred_2d).flatten()
    pred_t3 = predict_vol(t3_model, X_pred_2d).flatten()
    pred_t4 = np.zeros(n_valid)
    if t4_model is not None:
        from sequoia_x.model_selection_v2.models.deep_lstm import predict_lstm
        pred_t4 = predict_lstm(t4_model, X_pred).flatten()

    # ── 铁律一：运行时自检 - 预测值必须有效 ──
    t1_std = float(np.std(pred_t1))
    t3_std = float(np.std(pred_t3))
    t2_std = float(np.std(pred_t2))
    if t2_std < 1e-7:
        print(f"[Worker {month}] ❌ 严重: T2预测无方差! std={t2_std:.2e}", flush=True)
    if t1_std < 1e-4:
        print(f"[Worker {month}] ⚠ T1预测方差极小: std={t1_std:.4f} range=[{pred_t1.min():.3f},{pred_t1.max():.3f}]", flush=True)
    if t3_std < 1e-5:
        print(f"[Worker {month}] ⚠ T3预测方差极小: std={t3_std:.4f} range=[{pred_t3.min():.3f},{pred_t3.max():.3f}]", flush=True)
    print(f"[Worker {month}] ✅ 预测自检: T2 std={t2_std:.4f} | T1 std={t1_std:.4f} range=[{pred_t1.min():.3f},{pred_t1.max():.3f}] | T3 std={t3_std:.4f}",
          flush=True)

    preds = {
        "symbols": valid_symbols,
        "t2": [float(v) for v in pred_t2],
        "t1": [float(v) for v in pred_t1],
        "t3": [float(v) for v in pred_t3],
        "t4": [float(v) for v in pred_t4],
    }
    _assert_pred_aligned(month, valid_symbols, preds)     # 行对齐保险丝（2026-10-02）
    return month, preds


def _process_and_save(month: str, db_path: str, cfg_dict: dict,
                       max_pool_size: int, skip_t4: bool,
                       cache_dir: str, include_extra: bool,
                       synth_file: str = "", synth_ratio: float = 1.0,
                       synth_series_dir: str = "", tmp_dir: str = "") -> None:
    """Worker 入口：调用 _process_month_worker 并写入临时文件（避开 IPC 传大数据）。

    ⚠️ 2026-08-09 修复: tmp_dir 由 build_cache 按 output 文件名隔离传入——多个
    build_prediction_cache 任务并行时共享 .cache_tmp 会被先完成任务的 rmtree 删除,
    导致其他任务写 temp 失败（FileNotFoundError）。
    """
    import json as _json, traceback
    from pathlib import Path as _Path
    # 绝对路径，避免子进程 CWD 不一致
    _proj_root = _Path(__file__).resolve().parent.parent
    if not tmp_dir:
        tmp_dir = str(_proj_root / "output/backtest_v2/.cache_tmp")
    tmp_dir = _Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_file = tmp_dir / f"month_{month}.json"

    try:
        args = (month, db_path, cfg_dict, max_pool_size, skip_t4, cache_dir,
                include_extra, synth_file, synth_ratio, synth_series_dir)
        _, preds = _process_month_worker(args)
        if preds is not None:
            with open(tmp_file, "w") as f:
                _json.dump(preds, f, indent=2, default=str)
    except Exception as e:
        # 写入错误文件供调试
        err_file = tmp_dir / f"month_{month}.error"
        with open(err_file, "w") as f:
            f.write(traceback.format_exc())
        # 2026-09-26：**同时打到 stdout** —— 原来只写文件、控制台一声不吭，
        #   导致 worker 因"训练集含前视"断言失败时，外层只看到"0/1 个月完成"、
        #   退出码 0，故障被静默吞掉。失败必须可见。
        print(f"\n[Worker {month}] ❌ 失败: {type(e).__name__}: {e}\n"
              f"  完整堆栈见 {err_file}\n{traceback.format_exc()}", flush=True)


def _filter_stock_pool(symbols: list[str], db_path: str) -> list[str]:
    """本地过滤股票池（等效 get_base_stock_pool，无 baostock 依赖）。

    过滤规则: 板块(ST/科创/创业/北交所) + 次新(>1年) + 低价(<2元)
    """
    import sqlite3
    # 板块剔除
    exclude_prefixes = ('688', '689', '300', '301', '4', '8')
    symbols = [s for s in symbols if not s.startswith(exclude_prefixes)]

    # ST/退市 + 次新 + 低价
    conn = sqlite3.connect(db_path)
    # ST/退市检查：stock_list 表中的名称
    rows = conn.execute(
        "SELECT symbol, name, listed_date FROM stock_list WHERE symbol IN ({})".format(
            ','.join('?' * len(symbols))),
        symbols
    ).fetchall()
    name_map = {r[0]: (r[1] or '', r[2] or '') for r in rows}

    # 最新收盘价
    ph = ','.join('?' * len(symbols))
    price_rows = conn.execute(
        f"SELECT symbol, close FROM stock_daily WHERE symbol IN ({ph}) "
        f"AND (symbol, date) IN (SELECT symbol, MAX(date) FROM stock_daily GROUP BY symbol)",
        symbols
    ).fetchall()
    price_map = {r[0]: r[1] for r in price_rows if r[1]}
    conn.close()

    from datetime import date, timedelta
    today = date.today()
    one_year_ago = (today - timedelta(days=365)).isoformat()

    result = []
    for s in symbols:
        name, listed = name_map.get(s, ('', ''))
        if 'ST' in name or '退' in name:
            continue
        if listed and listed > one_year_ago:
            continue
        close = price_map.get(s, 0)
        if close < 2.0:
            continue
        result.append(s)

    return result


def _make_engine(db_path: str):
    """在子进程中创建 DataEngine（不调用 baostock）。"""
    from sequoia_x.core.config import Settings
    from sequoia_x.data.engine import DataEngine
    s = Settings()
    s.db_path = db_path
    return DataEngine(s)


def _merge_temp_files(tmp_dir, cache, output_path, total_start, test_months):
    """合并 temp 文件到主缓存（由父进程轮询调用）。"""
    import json as _json
    for tmp_file in sorted(tmp_dir.glob("month_*.json")):
        m = tmp_file.stem.replace("month_", "")
        if m not in cache:
            try:
                with open(tmp_file) as f:
                    cache[m] = _json.load(f)
                with open(output_path, "w") as f:
                    _json.dump(cache, f, indent=2, default=str)
                done = len(cache)
                elapsed = __import__('time').time() - total_start
                logger.info(f"  [{done}/{len(test_months)}] {m} 完成 → 缓存已保存 ({elapsed:.0f}s)")
            except Exception:
                pass


def build_cache(
    cfg: V2Config,
    engine: DataEngine,
    test_months: list[str],
    max_pool_size: int = 0,
    output_path: Path | None = None,
    skip_t4: bool = False,
    synth_file: str = "",
    synth_ratio: float = 1.0,
    synth_series_dir: str = "",
    feature_view: str = "full",
) -> dict:
    """构建预测缓存（串行逐月，稳定可靠）。

    Returns:
        完整的预测缓存 dict。
    """
    if output_path is None:
        output_path = OUTPUT_PATH

    # 0. sample_end 动态化（2026-08-07 月末自动链）: 与缓存重建同口径（DB 最后交易日),
    #     否则 hash 含写死的 sample_end → 9/1 重训与 8/31 重建 hash 失配
    # 2026-08-20: 允许 V4_SAMPLE_END_FIX 固定 sample_end——70月回测到6月, DB已是8/20,
    #   resolve 到 08-20 会失配现有 08-19 缓存; 固定 08-19 复用缓存, 无需重建/重做已完成的月份。
    import os as _os
    _se_fix = _os.environ.get("V4_SAMPLE_END_FIX", "")
    if _se_fix:
        cfg.sample_end = _se_fix
        logger.info(f"采样截止日（固定 V4_SAMPLE_END_FIX）: {cfg.sample_end}")
    else:
        from sequoia_x.model_selection_v2.labels import resolve_sample_end
        cfg.sample_end = resolve_sample_end(cfg, engine.db_path)
        logger.info(f"采样截止日（动态）: {cfg.sample_end}")

    # 1. 一次 baostock 获取标准股票池（写入文件供 worker 读取）
    stock_pool_path = output_path.parent / ".stock_pool.json"
    if stock_pool_path.exists():
        stock_pool = json.loads(stock_pool_path.read_text())
        logger.info(f"股票池（缓存）: {len(stock_pool)} 只")
    else:
        try:
            stock_pool = engine.get_base_stock_pool()
            stock_pool_path.write_text(json.dumps(stock_pool))
            logger.info(f"股票池（baostock）: {len(stock_pool)} 只")
        except Exception:
            stock_pool = _filter_stock_pool(engine.get_local_symbols(), engine.db_path)
            logger.warning(f"baostock 失败，本地过滤: {len(stock_pool)} 只")

    # 1a. 特征拼接开关（2026-08-07）: cfg.extra_features=True → 88+扩展N 维
    #     （N 随 feature_version 变: V3=121, V4/V5=129 —— 实际值读缓存 metadata, 见下）
    #     自动降级（回退机制）: 配置要扩展维但缓存未就绪（月末自动链未完成/数据不全回退）
    #     → 自动回退 88 维兜底, 不中断月度流程（微信告知）
    from sequoia_x.model_selection_v2.labels import _dataset_cache_path
    include_extra = bool(getattr(cfg, "extra_features", False))
    if include_extra:
        d121, _ = _dataset_cache_path(cfg, stock_pool, include_market_state=True,
                                      include_extra=True)
        if not (d121 / "metadata.json").exists():
            include_extra = False
            logger.warning("⚠️ 扩展维度训练缓存未就绪 → 自动回退 88 维（预测特征同步 88 维）")
            try:
                from wxpusher import WxPusher
                from sequoia_x.core.config import get_settings
                _s = get_settings()
                WxPusher.send_message(
                    content=f"⚠️ V2 重训自动回退 88 维\n扩展维度缓存未就绪（{d121.name} 缺失）, "
                            f"本次按 88 维重训（扩展维度数据不全的保底机制）",
                    token=_s.wxpusher_token, topic_ids=_s.wxpusher_topic_ids, content_type=1)
            except Exception:
                pass
        else:
            # 2026-10-02 修正: 原文案写死"88+33=121 维"（V3 时代），V4/V5 实为 129 维——
            #   陈旧文案在 9/30 事故复盘时误导过一次（日志说 121、缓存实为 129）。
            #   改为读该缓存 metadata 的**实际**维度, 以后升版本也不必再改这行。
            try:
                _dim = json.loads((d121 / "metadata.json").read_text())["X_shape"][2]
            except Exception:
                _dim = "?"
            logger.info(f"🔧 扩展特征已启用: 88+扩展={_dim} 维（训练缓存+预测特征一致）")

    # 1b. 训练数据集缓存目录（按 include_extra 哈希; 121 维走新目录）
    cache_dir, _ = _dataset_cache_path(cfg, stock_pool, include_market_state=True,
                                       include_extra=include_extra)
    cache_dir = str(cache_dir)
    logger.info(f"训练缓存目录: {cache_dir}")

    # 2. 加载 mmap 数据
    X, y1, y2, y3, dates = load_full_dataset(cfg, engine, cache_dir)
    dates_arr = np.array(dates)

    # 2. 断点续跑
    cache = {}
    if output_path.exists():
        with open(output_path) as f:
            cache = json.load(f)
        logger.info(f"从已有缓存加载: {len(cache)} 个月")

    # 3. 准备 cfg_dict
    # n_jobs=1: 24 worker × 1 线程 = 24 ≤ 36 核，避免训练阶段线程争抢（16×4=64 线程曾导致 CatBoost 卡死）
    db_path = engine.db_path
    cfg_dict = {
        "window": cfg.window, "n_jobs": 1, "random_seed": cfg.random_seed,
        "extra_features": include_extra,
        "feature_view": feature_view,   # 2026-09-26 实验开关（默认 "full" = 原行为）
    }

    total_start = time.time()

    # 过滤已完成月份
    pending_months = [m for m in test_months if m not in cache]
    if not pending_months:
        logger.info("所有月份已完成")
        return cache

    # 并行：多进程 + 文件通信（特征提取单线程 → 并行核；n_jobs=1 避免训练线程争抢）
    # 2026-08-20: 默认 24 → 可经 V4_BT_WORKERS 覆盖并降为 12；每 worker 内部再开 8 特征worker
    #   （24×8=192 并发峰值易触发 worker 被终止/BrokenProcessPool），降并发更稳。
    import os as _os
    _bt_w = int(_os.environ.get("V4_BT_WORKERS", "12"))
    n_workers = min(_bt_w, len(pending_months))
    logger.info(f"并行构建: {len(pending_months)} 个月, {n_workers} 进程 (V4_BT_WORKERS={_bt_w})")

    # ⚠️ 2026-08-09: tmp_dir 按 output 文件名隔离（多任务并行互不干扰）
    tmp_dir = (output_path.parent / f".cache_tmp_{output_path.stem}").resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"临时目录: {tmp_dir}")

    task_args = [
        (month, db_path, cfg_dict, max_pool_size, skip_t4, cache_dir, include_extra,
         synth_file, synth_ratio, synth_series_dir, str(tmp_dir))
        for month in pending_months
    ]

    from concurrent.futures import ProcessPoolExecutor, as_completed
    if n_workers == 1:
        # ── 单月任务：主进程直接执行（2026-08-02）──
        # multiprocessing.Pool 的 worker 是 daemon 进程，禁止创建子进程——
        # 而 Step5 特征构建并行（ProcessPoolExecutor）需要开子进程。
        # 单月时在主进程执行 _process_and_save，特征并行正常工作。
        for args in task_args:
            _process_and_save(*args)
            _merge_temp_files(tmp_dir, cache, output_path, total_start, test_months)
    else:
        # ── 多月并行：用 ProcessPoolExecutor 而非 multiprocessing.Pool（2026-08-20 修复）──
        # multiprocessing.Pool 的 worker 是 daemon 进程，禁止创建子进程；
        # 而每月 worker 内部 Step5 特征构建(ProcessPoolExecutor)需开子进程 →
        # Pool 会报 "daemonic processes are not allowed to have children"，全部月份失败。
        # ProcessPoolExecutor 的 worker 非 daemon，可嵌套子进程。
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_process_and_save, *args) for args in task_args]
            # 渐进合并：as_completed + 每完成一个月份即 merge 该月 temp 文件
            for tot in as_completed(futures):
                tot.result()  # 抛出 worker 异常（若有）
                _merge_temp_files(tmp_dir, cache, output_path, total_start, test_months)
            # 收尾 merge（确保所有月份入库）
            _merge_temp_files(tmp_dir, cache, output_path, total_start, test_months)

    # 2026-09-26：失败必须可见 —— 原来失败只写 .error 文件、控制台不出声、退出码仍为 0，
    #   于是"训练集含前视"这类致命故障会被静默吞掉（实测：只显示 0/1 个月完成）。
    # 2026-09-27 修（独立审计发现，有**生产风险**）：成功月份**不删除**旧的 .error，而这里按
    #   目录里 .error **总数**计数 ⇒ 进程被 kill/崩溃留下的陈旧 .error 会在下次"同命令续跑"里
    #   被当成本次失败 → main() 退出码 2 → 生产链 v2_monthly_retrain 见非 0 即中止 + 微信告警
    #   （数据其实全对，却误报并中断生产重训）。改为：**只把"没有对应 month_X.json"的 .error
    #   算作失败**（有 json = 该月已完成，其 .error 必为陈旧残留），并显式报告忽略掉的陈旧数。
    import glob as _glob
    _all_errs = _glob.glob(str(tmp_dir / "month_*.error"))
    _errs, _stale = [], []
    for _p in _all_errs:
        (_stale if (tmp_dir / (Path(_p).stem + ".json")).exists() else _errs).append(_p)
    if _stale:
        logger.warning(f"（忽略 {len(_stale)} 个陈旧 .error：对应月份已有 month_*.json，非本次失败）")
    # 2026-09-27（审计 I2b）：显式报告"跳过月"（worker 因样本不足 return None 的月份）
    _skipped = [m for m in test_months if m not in cache]
    if _skipped:
        logger.warning(f"⚠️ {len(_skipped)} 个月份被跳过（训练样本不足，非故障）: {_skipped}"
                       f" ⇒ 实际产出 {len(cache)}/{len(test_months)} 个月")
    if _errs:
        logger.error(f"❌ {len(_errs)} 个月份构建失败: "
                     f"{[Path(p).stem for p in _errs]}")
        for _p in _errs[:3]:
            logger.error(f"--- {Path(_p).name} ---\n{Path(_p).read_text()[-1000:]}")
    # 2026-09-27：把失败月挂到函数属性上，供**调用方**判断 —— 原先失败月只写日志，
    #   退出码仍是 0，wrapper 打印"完成 0/N"、exit=0，会被误读成"零失败"（09-27 实测咬了两次）。
    #   不用 sys.exit：本文件的调用方有两类 ——
    #     · 子进程（v2_monthly_retrain.py）：它检查 returncode，main() 里转成退出码 2；
    #     · 进程内（run_shared_backtest.py:97）：它取返回值 build_cache(...)，不能被杀掉。
    build_cache.last_failed_months = [Path(p).stem for p in _errs]

    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    total_elapsed = time.time() - total_start
    logger.info(f"\n{'='*60}")
    logger.info(f"缓存构建完成: {len(cache)}/{len(test_months)} 个月, "
                f"总耗时={total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    logger.info(f"输出: {output_path}")
    return cache


def main():
    parser = argparse.ArgumentParser(description="构建月度预测缓存 (Phase 1)")
    parser.add_argument("--months", type=int, default=0,
                        help="限制月份数（0=全部11个月）")
    parser.add_argument("--max-stocks", type=int, default=0,
                        help="限制股票池大小（0=全量~2977只）")
    parser.add_argument("--skip-t4", action="store_true",
                        help="跳过 T4 LSTM 训练（加速，默认训练T4）")
    parser.add_argument("--start-month", type=str, default="2025-08",
                        help="起始月 (YYYY-MM)")
    parser.add_argument("--end-month", type=str, default="2026-06",
                        help="结束月 (YYYY-MM)")
    parser.add_argument("--output", type=str,
                        default=str(OUTPUT_PATH),
                        help="输出路径")
    parser.add_argument("--synth-file", type=str, default="",
                        help="V3 修订二: 合成标签 JSON（Kronos 生成, 仅 88 维模式生效）")
    parser.add_argument("--no-extra", action="store_true",
                        help="实验用: 强制 88 维（覆盖 config 的 extra_features, 不动生产配置）")
    parser.add_argument("--synth-ratio", type=float, default=1.0,
                        # 2026-09-26: 原文案含裸 `%` → argparse 的 help 格式化会抛
                        #   ValueError: unsupported format character ',' → **--help 一直崩**。
                        #   argparse 对 help 做 %-格式化，字面百分号须写成 `%%`。
                        help="合成注入比例（1.0=全量24%%，0.25≈5%%，0.5≈10%%；占比试调）")
    parser.add_argument("--synth-series", type=str, default="",
                        help="V3 修订二: 合成完整序列目录（真·数据增强, 优先于 --synth-file）")
    parser.add_argument("--feature-view", type=str, default="full", choices=list(FEATURE_VIEWS),
                        help="特征视图（实验开关, 默认 full 保持原行为）。"
                             "full=(120天×129维)展平=15,480维; "
                             "lastday_agg=最后一天+窗口均值+窗口标准差=387维（训练快 ~13.7 倍）")
    args = parser.parse_args()

    cfg = get_config()
    engine = DataEngine(Settings())

    test_months = get_test_months(args.start_month, args.end_month)
    if args.months > 0:
        test_months = test_months[:args.months]

    logger.info(f"预测缓存构建:")
    logger.info(f"  月份: {len(test_months)} 个月 ({test_months[0]}~{test_months[-1]})")
    logger.info(f"  股票池: {'全量(~2977)' if args.max_stocks <= 0 else str(args.max_stocks)}")
    logger.info(f"  输出: {args.output}")

    if args.no_extra:
        cfg.extra_features = False   # 实验用 88 维（合成样本仅 88 维支持）
    build_cache(cfg, engine, test_months, args.max_stocks,
                output_path=Path(args.output), skip_t4=args.skip_t4,
                synth_file=args.synth_file, synth_ratio=args.synth_ratio,
                synth_series_dir=args.synth_series,
                feature_view=args.feature_view)

    # 2026-09-27：失败月必须反映到**退出码**（原为 0，会被 wrapper/人工误读成"零失败"）。
    #   调用方 v2_monthly_retrain.py 见非 0 即中止 + 微信告警（正是我们要的行为）。
    _failed = getattr(build_cache, "last_failed_months", None)
    if _failed:
        logger.error(f"❌ 退出码=2：{len(_failed)} 个月份构建失败 {_failed}"
                     f"（调用方应据此终止或重试；注意 exit=0 不代表无失败月的旧行为已废止）")
        sys.exit(2)


if __name__ == "__main__":
    main()
