"""V2 月度重训 + 选股 + 信号入库（cron 每月 1 日 03:00 触发（2026-08-10: 与月末链错开——19:00 月末链约 8h 至 02:00, 03:00 重训零重叠））

时机依据（用户设计）：
  - 月末最后交易日 18:10 同步完成 → 上月数据 100% 完整
  - 下月 1 日 03:00 重训（月末链 02:00 完成后 1h 缓冲, 凌晨训练不占盘面时间）
  - 重训完成 → 选股信号入库 sim_v2.db（strategy_from="V2"）
  - 信号日期标记为"上月最后交易日"→ 下一个交易日晚上 --sim-update
    执行买入（用当日开盘价）——若 1 日不是交易日则自动顺延（T+1 模型天然支持）

流程：
  1. 增量重建预测缓存：T2/T1/T3（build_prediction_cache，--skip-t4）
  2. T4 训练（t4_monthly_worker）
  3. 从 prediction_cache 读最新月预测 → Rank 融合选股 → TOP_N
  4. 写入 sim_v2.db（submit_buy_signals，strategy_name="V2"）
  5. V2 荐股推送（wxpusher）

用法（cron, 2026-08-10: 00:00 → 03:00 与月末链错开）：
  0 3 1 * * cd <project> && python scripts/v2_monthly_retrain.py >> logs/v2_retrain_$(date +%Y%m).log 2>&1
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from sequoia_x.core.config import get_settings
from sequoia_x.core.logger import get_logger
from sequoia_x.simulation.signals import submit_buy_signals

logger = get_logger(__name__)

SIM_V2_DB = str(PROJECT_DIR / "data" / "sim_v2.db")
CACHE_PATH = PROJECT_DIR / "output/backtest_v2/prediction_cache.json"
TOP_N_BUY = 10           # 每月买入数（与回测 M4+TOP_N=10 一致）
PYTHON = sys.executable


def get_target_month() -> str:
    """重训目标月 = 当前月（下月 1 日运行时，重训本月信号用）。"""
    return datetime.now().strftime("%Y-%m")


def _check_extra_coverage() -> None:
    """扩展维度覆盖率检查: 某类 <90% 时日志告警 + wxpusher 推送（不阻断重训）。

    2026-09-01 fix: 改用**实际数据文件数**统计覆盖率，而非 manifest.subsets.success
    ——后者是"本次采集新增数"，数据新鲜时跳过采集 → success=0 → 误报 0% 覆盖率。
    实际数据（data/extra_features/{subset}/*.parquet）才是真实覆盖。
    """
    import os
    import glob
    extra_dir = PROJECT_DIR / "data/extra_features"
    # 2026-09-01: 股票池总数动态取（不再硬编码 5206），从 .stock_pool.json 读；缺省 5206
    total = 5206
    try:
        import json as _json
        _pool = _json.loads((PROJECT_DIR / "output/backtest_v2/.stock_pool.json").read_text())
        if isinstance(_pool, list) and _pool:
            total = len(_pool)
    except Exception:
        pass
    low = []
    subsets = ["fund_flow", "finance", "holders", "consensus", "news", "xdxr"]
    for subset in subsets:
        sub_dir = extra_dir / subset
        if not sub_dir.is_dir():
            low.append(f"{subset}=0%")
            logger.warning(f"Step0: {subset} 目录缺失")
            continue
        ok = len(glob.glob(str(sub_dir / "*.parquet")))
        cov = ok / total * 100
        if cov < 90 and subset != "consensus":  # consensus 允许 64%（数据源限制，报警但差异）
            low.append(f"{subset}={cov:.0f}%")
            logger.warning(f"Step0: {subset} 覆盖率仅 {cov:.0f}% ({ok}/{total})")
    if low:
        try:
            from wxpusher import WxPusher
            settings = get_settings()
            WxPusher.send_message(
                content=f"⚠️ 扩展维度覆盖率不足: {', '.join(low)}（V2重训 Step0）",
                token=settings.wxpusher_token,
                topic_ids=settings.wxpusher_topic_ids,
                content_type=1,
            )
            logger.info(f"Step0: 覆盖率告警已推送: {low}")
        except Exception as e:
            logger.warning(f"Step0: 告警推送失败: {e}")
    else:
        logger.info("Step0: 扩展维度覆盖率全部达标(≥90%)")


def _notify(title: str, body: str) -> None:
    """wxpusher 微信推送（失败告警, 不阻断流程）。"""
    try:
        from wxpusher import WxPusher
        settings = get_settings()
        WxPusher.send_message(content=f"{title}\n{body}", token=settings.wxpusher_token,
                              topic_ids=settings.wxpusher_topic_ids, content_type=1)
        logger.info(f"告警已推送: {title}")
    except Exception as e:
        logger.warning(f"告警推送失败: {e}")


def wait_for_cache_ready(target_month: str, max_wait_h: float = 12.0) -> bool:
    """等待月末自动链的训练缓存就绪（2026-08-07: 8/31 19:00 拉取+重建 2-6h,
    9/1 03:00 重训启动时可能尚未完成 → 每 5min 轮询, 最长 12h 后失败告警）。

    判定: 树模型缓存(121/88 按 config) metadata 存在 + 维度正确 + 采样日覆盖到上月最后交易日。
    """
    import json as _json
    import time

    from sequoia_x.model_selection_v2.config import get_config
    from sequoia_x.model_selection_v2.labels import (
        _dataset_cache_path, _get_sample_dates, resolve_sample_end,
    )
    from sequoia_x.data.engine import DataEngine
    from sequoia_x.core.config import Settings as _Settings

    cfg = get_config()
    engine = DataEngine(_Settings())
    cfg.sample_end = resolve_sample_end(cfg, engine.db_path)  # DB 最后交易日（与月末重建同口径，动态非硬编码）
    # 2026-09-01 fix: 覆盖率基准 = "最后一个有完整未来标签窗口(20交易日)的采样日"。
    # 不能要求覆盖到 DB 最后交易日(如8/31)或采样日最后一天(8/21)——它们未来20日窗口不完整，
    # 该采样日本身在当前数据下 0 样本（如 8/21）。正确 = 在 sample_end 往前推 predict_horizon_t2
    # 个交易日的日期之前、最接近的采样日（当前=8/07；下月 9/30 时自动变 9/07 左右，动态不硬编码）。
    sample_dates = _get_sample_dates(engine, cfg)
    # 2026-09-01 fix: 覆盖率基准 = "最后一个有完整未来标签窗口(20交易日)的采样日"。
    # 直接用 sqlite 查 DB 交易日（稳健，不依赖单只股票 get_ohlcv——部分股票可能无数据导致 fallback 到 8/21）。
    _all_days = []
    try:
        import sqlite3
        _conn = sqlite3.connect(engine.db_path)
        raw_days = _conn.execute("SELECT DISTINCT date FROM stock_daily ORDER BY date").fetchall()
        _conn.close()
        _all_days = [str(r[0]) for r in raw_days if str(r[0]) <= cfg.sample_end]
    except Exception:
        _all_days = []
    if len(_all_days) > cfg.predict_horizon_t2:
        _cutoff = _all_days[-cfg.predict_horizon_t2]      # 未来窗口起始: 最后第20个交易日前
        _cands = [d for d in sample_dates if d <= _cutoff]
        last_date = _cands[-1] if _cands else sample_dates[0]  # 动态最后一个完整 label 采样日
    else:
        last_date = sample_dates[-1]
    pool_path = PROJECT_DIR / "output/backtest_v2/.stock_pool.json"
    if not pool_path.exists():
        _notify("❌ V2 重训: .stock_pool.json 缺失",
                "月末自动链未正常运行（应写入股票池）。请检查 logs/month_end_pull_*.log")
        return False
    symbols = _json.loads(pool_path.read_text())
    want_extra = bool(getattr(cfg, "extra_features", False))

    def _check_ready(include_extra: bool) -> tuple[bool, str]:
        """单缓存就绪判定: metadata 存在 + 维度正确 + 采样日覆盖到上月最后交易日。

        维度基准动态化：base 特征数(88) + 扩展特征数（若 cache_dir 存在则从其 metadata 读
        实际 X_shape[2]，否则按 base + 估算）。2026-09-01 起不再硬编码 129——
        feature_version 演变（129/更多）时自动适配。
        """
        cache_dir, _ = _dataset_cache_path(cfg, symbols, include_market_state=True,
                                           include_extra=include_extra)
        meta_path = cache_dir / "metadata.json"
        if not meta_path.exists():
            return False, f"缓存缺失 {cache_dir.name}"
        try:
            m = _json.loads(meta_path.read_text())
            # 目标维度：直接用该缓存自身应达到的维度（include_extra 决定是否含扩展特征）。
            # 以 base+extra 的实际列数推算，避免硬编码 129。若无法推算则读 metadata 已有维度。
            extra_feat = getattr(cfg, "extra_features", False)
            _base_dim = 88  # base 行情特征（含 market_state）
            want_dim = _base_dim + (0 if not include_extra else m["X_shape"][2] - _base_dim) \
                if include_extra else _base_dim
            # 动态：用 metadata 中记录的真实维度作为目标（最贴近实际，feature_version 无关）
            if include_extra and "feature_version" not in m.get("params", {}):
                want_dim = m["X_shape"][2]
            if m["X_shape"][2] != want_dim:
                return False, f"维度错误 {m['X_shape'][2]}≠{want_dim}"
            dates = _json.loads((cache_dir / "dates.json").read_text())
            if not dates or dates[-1] < last_date:
                return False, f"采样日未覆盖 {last_date}（止于 {dates[-1] if dates else '空'}）"
            return True, f"{cache_dir.name} {m['X_shape']} 止于 {dates[-1]}"
        except Exception as e:
            return False, f"缓存元数据异常: {e}"

    deadline = time.time() + max_wait_h * 3600
    waited = 0
    while True:
        # 就绪判定: 优先扩展特征维（V4 配置目标）; 缺失但 88 就绪（自动链数据不全回退）→ 降级接受
        ok, msg = _check_ready(want_extra)
        if not ok and want_extra:
            ok88, msg88 = _check_ready(False)
            if ok88:
                _notify("⚠️ V2 重训自动回退 88 维",
                        "扩展特征维缓存未就绪, 但 88 维缓存已就绪 → 本次按 88 维重训（保底机制）")
                logger.warning(f"扩展特征维未就绪, 接受 88 维降级: {msg88}")
                return True
            reason = f"{msg}; 88 维: {msg88}"
        elif not ok:
            reason = msg
        else:
            logger.info(f"缓存就绪: {msg}")
            return True

        if time.time() > deadline:
            _notify("❌ V2 重训等待缓存超时",
                    f"等待 {max_wait_h:.0f}h 未就绪: {reason}\n请检查 logs/month_end_pull_*.log（月末自动链）")
            logger.error(f"缓存等待超时: {reason}")
            return False
        if waited % 30 == 0 or waited == 0:
            logger.info(f"缓存未就绪（{waited}min）, 每 5min 重试, 最长 {max_wait_h:.0f}h: {reason}")
        time.sleep(300)
        waited += 5


def build_prediction_cache(target_month: str) -> bool:
    """T2/T1/T3 预测缓存构建（增量，断点续跑）。"""
    logger.info(f"Step1: T2/T1/T3 预测缓存构建（{target_month}）...")
    # 注: V3 合成序列增强（--synth-series）不纳入 V2 生产重训（2026-08-10 用户明确:
    #   V3 体系功能不在 V2 生产 cron 中实现; 将来 V3 替代 V2 时再集成, 见 V3 §20.3）
    r = subprocess.run(
        [PYTHON, "-u", "scripts/build_prediction_cache.py",
         "--start-month", target_month, "--end-month", target_month, "--skip-t4"],
        cwd=str(PROJECT_DIR),
    )
    return r.returncode == 0


def train_t4(target_month: str) -> bool:
    """T4 LSTM 单月训练+预测（追加到缓存）。"""
    logger.info(f"Step2: T4 LSTM 训练（{target_month}）...")
    r = subprocess.run(
        [PYTHON, "-u", "scripts/t4_monthly_worker.py", "--month", target_month],
        cwd=str(PROJECT_DIR),
    )
    return r.returncode == 0


def select_stocks(target_month: str) -> tuple[list[str], dict]:
    """从预测缓存读最新月预测 → Rank 融合 → TOP_N 选股。

    Returns:
        (buy_list, pred_info): 选股列表 + 预测摘要（用于荐股推送）。
    """
    cache = json.loads(CACHE_PATH.read_text())
    entry = cache.get(target_month)
    if entry is None:
        logger.error(f"缓存无 {target_month} 预测")
        return [], {}

    symbols = entry["symbols"]
    t2 = np.array(entry["t2"])
    t4 = np.array(entry["t4"])

    # ── 风控 1：T2 分布预警（与回测一致）──
    # 最优 100 只的 T2 预测均值 < -5% → 系统性看空 → 本月空仓
    top100_idx = np.argsort(-t2)[:min(100, len(symbols))]
    top100_t2_mean = float(np.mean(t2[top100_idx]))
    if top100_t2_mean < -0.05:
        logger.warning(
            f"T2 分布预警: top100均值={top100_t2_mean:.4f} < -0.05, "
            f"系统性看空 → 本月空仓"
        )
        return [], {
            "n_pool": len(symbols), "t2_mean_top": top100_t2_mean,
            "t4_mean_top": 0.0, "buy_list": [], "signal": "空仓",
        }

    # ── Rank 融合选股 ──
    if t4.std() < 1e-9:  # T4 未完成（占位 0）
        logger.warning("T4 预测为占位（std≈0），本次仅用 T2")
        rank_t2 = rankdata(-t2, method="average")
        rank_scores = rank_t2
    else:
        rank_t2 = rankdata(-t2, method="average")
        rank_t4 = rankdata(-t4, method="average")
        rank_scores = (rank_t2 + rank_t4) / 2.0

    order = np.argsort(rank_scores)[:TOP_N_BUY]
    buy_list = [symbols[i] for i in order]

    pred_info = {
        "n_pool": len(symbols),
        "t2_mean_top": float(np.mean(t2[order])),
        "t4_mean_top": float(np.mean(t4[order])),
        "buy_list": buy_list,
    }
    return buy_list, pred_info


def push_recommendation(target_month: str, pred_info: dict) -> None:
    """V2 荐股报告推送（wxpusher，直接 send_message，不依赖已废弃的选股播报格式）。"""
    try:
        from wxpusher import WxPusher

        from sequoia_x.data.engine import DataEngine

        settings = get_settings()
        lines = [f"【V2 模型月度荐股 {target_month}】"]
        lines.append(f"股票池: {pred_info['n_pool']} 只 | "
                     f"T2 头部均值: {pred_info['t2_mean_top']:+.2%} | "
                     f"T4 头部均值: {pred_info['t4_mean_top']:+.2%}")
        lines.append("买入候选（T+1 开盘执行）:")
        eng = DataEngine(settings)
        for i, sym in enumerate(pred_info["buy_list"], 1):
            name = eng.get_stock_name(sym)
            lines.append(f"  {i}. {sym} {name}")
        result = WxPusher.send_message(
            content="\n".join(lines),
            token=settings.wxpusher_token,
            topic_ids=settings.wxpusher_topic_ids,
            content_type=1,
        )
        if result.get("code") == 1000:
            logger.info("V2 荐股已推送")
        else:
            logger.warning(f"V2 荐股推送失败: {result}")
    except Exception as e:
        logger.warning(f"V2 荐股推送失败: {e}")


def main() -> None:
    import time as _time
    t_main = _time.time()
    target_month = get_target_month()
    logger.info("=" * 60)
    logger.info(f"V2 月度重训启动 | 目标月={target_month} | {datetime.now()}")
    logger.info("=" * 60)

    # ── Step0: 辅助维度增量刷新（资金流向/财务/股东/研报/新闻/分红, 见 collect_extra_features.py）──
    #    月末 19:00 已全量刷新（month_end_pull.py, 24-48h 缓冲）; 1 号只补缺失+failed 清单
    #    完成后做覆盖率检查: <90% 告警推送（当月特征质量可见, 不阻断重训）
    try:
        subprocess.run(
            [PYTHON, str(PROJECT_DIR / "scripts/collect_extra_features.py"),
             "--codes", str(PROJECT_DIR / "scripts/all_a_codes.txt"),
             "--refresh-days", "40"],
            check=False, timeout=3600,
        )
        logger.info(f"Step0: 辅助维度刷新完成（{(_time.time()-t_main)/60:.0f}min）")
        _check_extra_coverage()
    except Exception as e:
        logger.warning(f"Step0: 辅助维度刷新异常(不阻断重训): {e}")

    # ── Step0.5: 等待月末自动链训练缓存就绪（2026-08-07）──
    #    8/31 19:00 拉取 + 重建 2-6h, 9/1 03:00 启动时可能未完成 → 轮询等待（最长 12h）
    if not wait_for_cache_ready(target_month):
        _notify("❌ V2 月度重训中止（缓存未就绪）", "9 月信号未产生, 请人工介入排查月末自动链")
        sys.exit(1)
    logger.info(f"训练缓存就绪，继续重训（累计 {(_time.time()-t_main)/60:.0f}min）")

    # ── Step1: T2/T1/T3 缓存构建（增量，断点续跑）──
    if not build_prediction_cache(target_month):
        _notify("❌ V2 重训 Step1 失败", "T2/T1/T3 预测缓存构建失败, 请查 logs/v2_retrain_*.log")
        logger.error("T2/T1/T3 构建失败，重训终止")
        sys.exit(1)
    logger.info(f"Step1 完成（累计 {(_time.time()-t_main)/60:.0f}min）")

    # ── Step2: T4 训练（追加 T4 预测到缓存）──
    if not train_t4(target_month):
        _notify("❌ V2 重训 Step2 失败", "T4 LSTM 训练失败, 请查 logs/v2_retrain_*.log")
        logger.error("T4 训练失败，重训终止（可重跑续跑）")
        sys.exit(1)
    logger.info(f"Step2 T4 完成（累计 {(_time.time()-t_main)/60:.0f}min）")

    # ── Step3: 选股（Rank 融合）──
    buy_list, pred_info = select_stocks(target_month)
    if not buy_list:
        logger.error("选股为空，重训终止")
        sys.exit(1)
    logger.info(f"选股完成: {len(buy_list)} 只（累计 {(_time.time()-t_main)/60:.0f}min）")
    logger.info(f"  预测摘要: 池子={pred_info.get('n_pool')} 只 | "
                f"T2 头部均值={pred_info.get('t2_mean_top', 0):+.2%} | "
                f"T4 头部均值={pred_info.get('t4_mean_top', 0):+.2%}")

    # ── Step4: 信号入库 sim_v2.db（strategy_from="V2"）──
    n = submit_buy_signals(
        db_path=SIM_V2_DB,
        symbols=buy_list,
        strategy_name="V2",
        top_n=TOP_N_BUY,
    )
    logger.info(f"V2 买入信号已写入 sim_v2.db: {n} 条")

    # ── Step5: 荐股推送 ──
    push_recommendation(target_month, pred_info)

    # ── Step6: V2 + LLM 模拟盘月度汇总报告推送（对应 V1 月末月度报告）──
    try:
        from sequoia_x.simulation.reporter import build_monthly_report_text, push_daily_summary
        settings = get_settings()
        # 上个月（重训目标月的前一月）的模拟盘月度汇总
        y, m = int(target_month[:4]), int(target_month[5:7])
        prev_m = m - 1
        if prev_m <= 0:
            prev_m, y = 12, y - 1

        # 合并两个模拟盘的月度报告：
        #   V2  模拟盘 → sim_v2.db（独立库）
        #   LLM 模拟盘 → settings.db_path（sequoia_v2.db 的 sim_* 表）
        v2_text = build_monthly_report_text(y, prev_m, SIM_V2_DB)
        llm_text = build_monthly_report_text(y, prev_m, settings.db_path)

        if not v2_text and not llm_text:
            logger.info("V2/LLM 月度报告均为空，跳过")
        else:
            report = [f"【V2 + LLM 模拟盘月度报告 {y}-{prev_m:02d}】", ""]
            report.append("═══ V2 模拟盘 ═══")
            report.append(v2_text or "（本月无模拟盘交易数据）")
            report.append("")
            report.append("═══ LLM 模拟盘 ═══")
            report.append(llm_text or "（本月无模拟盘交易数据）")
            push_daily_summary(settings, "\n".join(report))
            logger.info(f"V2+LLM 月度报告已推送（{y}-{prev_m:02d}）")
    except Exception as e:
        logger.warning(f"V2+LLM 月度报告推送失败: {e}")

    logger.info(f"V2 月度重训完成 | 耗时见日志 | 信号待下个交易日执行")


if __name__ == "__main__":
    main()
