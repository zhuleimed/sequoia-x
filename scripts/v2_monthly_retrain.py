"""V2 月度重训 + 选股 + 信号入库（cron 每月 1 日 00:00 触发）

时机依据（用户设计）：
  - 月末最后交易日 18:10 同步完成 → 上月数据 100% 完整
  - 下月 1 日 00:00 重训（此时数据完整，凌晨训练不占盘面时间）
  - 重训完成 → 选股信号入库 sim_v2.db（strategy_from="V2"）
  - 信号日期标记为"上月最后交易日"→ 下一个交易日晚上 --sim-update
    执行买入（用当日开盘价）——若 1 日不是交易日则自动顺延（T+1 模型天然支持）

流程：
  1. 增量重建预测缓存：T2/T1/T3（build_prediction_cache，--skip-t4）
  2. T4 训练（t4_monthly_worker）
  3. 从 prediction_cache 读最新月预测 → Rank 融合选股 → TOP_N
  4. 写入 sim_v2.db（submit_buy_signals，strategy_name="V2"）
  5. V2 荐股推送（wxpusher）

用法（cron）：
  0 0 1 * * cd <project> && python scripts/v2_monthly_retrain.py >> logs/v2_retrain_$(date +%Y%m).log 2>&1
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


def build_prediction_cache(target_month: str) -> bool:
    """T2/T1/T3 预测缓存构建（增量，断点续跑）。"""
    logger.info(f"Step1: T2/T1/T3 预测缓存构建（{target_month}）...")
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
    target_month = get_target_month()
    logger.info("=" * 60)
    logger.info(f"V2 月度重训启动 | 目标月={target_month} | {datetime.now()}")
    logger.info("=" * 60)

    # ── Step1: T2/T1/T3 缓存构建（增量，断点续跑）──
    if not build_prediction_cache(target_month):
        logger.error("T2/T1/T3 构建失败，重训终止")
        sys.exit(1)

    # ── Step2: T4 训练（追加 T4 预测到缓存）──
    if not train_t4(target_month):
        logger.error("T4 训练失败，重训终止（可重跑续跑）")
        sys.exit(1)

    # ── Step3: 选股（Rank 融合）──
    buy_list, pred_info = select_stocks(target_month)
    if not buy_list:
        logger.error("选股为空，重训终止")
        sys.exit(1)
    logger.info(f"选股完成: {buy_list}")

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
