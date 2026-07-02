#!/usr/bin/env python
"""Sequoia-X V2 全自动管线编排器。

由 cron 在 18:10 启动，依次执行 sync → strategy → 018 → 未来项目。
上一步完成立即启动下一步，不依赖固定时间。

使用方法：
    python pipeline/pipeline.py

新增项目：在下方 STEPS 配置列表中加一项即可，无需改其他代码。
状态文件：/public/home/hpc/zhulei/superman/quant/code/pipeline_status.json
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path

# ── 让 Python 能找到 sequoia_x 包 ──
PROJECT_DIR: Path = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_DIR))

from pipeline.status import PipelineStatus, push_pipeline_summary
from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)

# ════════════════════════════════════════════════════════════
#  路径常量（请根据实际环境调整）
# ════════════════════════════════════════════════════════════

PY312: str = "/home/zhulei/anaconda3/envs/zhulei_py312/bin/python"  # 004 项目 Python 3.12
PY39: str = "/home/zhulei/anaconda3/envs/zhulei/bin/python"         # 018 项目 Python 3.9
P018_DIR: str = "/public/home/hpc/zhulei/superman/quant/code/018_unified_trading"

# ════════════════════════════════════════════════════════════
#  步骤配置（按顺序执行）
# ════════════════════════════════════════════════════════════
#
# 新增项目在这里加一项，格式：
#   {
#       "id":       唯一标识（snake_case，用于 status.json key）
#       "name":     中文名称（日志和推送显示）
#       "cmd":      命令参数列表（不含 python 自身）
#       "cwd":      工作目录
#       "python":   Python 可执行文件路径
#       "required": True=进程崩溃退出时终止管线；False=失败告警继续
#       "timeout":  超时秒数，到达后强制 kill 该进程（0=不限制）
#   }
#

STEPS: list[dict] = [
    # ── 1. 数据同步（必需） ──
    {
        "id": "sync",
        "name": "数据同步",
        "cmd": ["main.py", "--sync-only"],
        "cwd": str(PROJECT_DIR),
        "python": PY312,
        "required": True,
        "timeout": 14400,  # 4h（baostock 慢时 3h 不够用）
    },
    # ── 2. 策略选股 + LLM（必需） ──
    {
        "id": "strategy",
        "name": "策略选股+LLM",
        "cmd": ["main.py"],
        "cwd": str(PROJECT_DIR),
        "python": PY312,
        "required": False,   # 可选：失败不阻断管线
        "timeout": 1800,  # 30min
    },
    # ── 3. 模拟盘更新（策略选股后执行，T+1 模式） ──
    {
        "id": "simulation",
        "name": "模拟盘更新",
        "cmd": ["main.py", "--sim-update"],
        "cwd": str(PROJECT_DIR),
        "python": PY312,
        "required": False,
        "timeout": 600,  # 10min（主要是数据库操作）
    },
    # ── 4. 018 LSTM 策略（可选，已暂停） ──
    # {
    #     "id": "p018_lstm",
    #     "name": "018 LSTM 策略",
    #     "cmd": ["run_daily.py", "--strategy", "lstm"],
    #     "cwd": P018_DIR,
    #     "python": PY39,
    #     "required": False,
    #     "timeout": 1800,  # 30min
    # },
    # ── 4. 018 指标策略（可选，已暂停） ──
    # {
    #     "id": "p018_indicator",
    #     "name": "018 指标策略",
    #     "cmd": ["run_daily.py", "--strategy", "indicator"],
    #     "cwd": P018_DIR,
    #     "python": PY39,
    #     "required": False,
    #     "timeout": 1800,  # 30min
    # },
    # ── 未来项目加入示例 ──
    # {
    #     "id": "p019",
    #     "name": "019 项目",
    #     "cmd": ["run.py"],
    #     "cwd": "/public/home/hpc/zhulei/superman/quant/code/019_xxx",
    #     "python": PY312,
    #     "required": False,
    #     "timeout": 1800,
    # },
]


# ════════════════════════════════════════════════════════════
#  交易日判断（轻量级，无 baostock 依赖）
# ════════════════════════════════════════════════════════════


def is_trade_day(check_date: date | None = None) -> bool:
    """判断当日是否为 A 股交易日。

    两层策略：
    1. 周末过滤（最快）
    2. chinese_calendar 节假日判断

    Args:
        check_date: 检查日期，默认当天。

    Returns:
        True 表示交易日或无法确定（fail-open）。
    """
    if check_date is None:
        check_date = date.today()
    # 第1层：周末
    if check_date.weekday() >= 5:
        return False
    # 第2层：chinese_calendar 离线库
    try:
        from chinese_calendar import is_workday

        return is_workday(check_date)
    except ImportError:
        logger.warning("chinese_calendar 未安装，默认视为交易日")
        return True


# ════════════════════════════════════════════════════════════
#  步骤执行器
# ════════════════════════════════════════════════════════════


def run_step(step: dict, status: PipelineStatus) -> bool:
    """执行一个管线步骤。

    Args:
        step:    步骤配置字典
        status:  PipelineStatus 实例

    Returns:
        True=可继续管线（成功 或 非必需步骤失败），
        False=必须终止。
    """
    sid: str = step["id"]
    name: str = step["name"]
    cmd: list[str] = [step["python"]] + step["cmd"]
    cwd: str = step["cwd"]
    required: bool = step["required"]
    timeout: int | None = step["timeout"] or None  # None = 不限制

    logger.info(f"═══ 步骤 [{sid}] {name} 开始 ═══")
    status.start_step(sid)
    t0: float = time.time()

    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            timeout=timeout,
        )
        elapsed: float = time.time() - t0
        success: bool = result.returncode == 0

        if success:
            logger.info(f"✅ [{sid}] {name} 完成（耗时 {elapsed:.0f}s）")
            status.complete_step(sid, success=True)
            return True
        else:
            logger.error(
                f"❌ [{sid}] {name} 失败"
                f"（退出码 {result.returncode}, 耗时 {elapsed:.0f}s）"
            )
            status.complete_step(
                sid, success=False, error=f"exit code {result.returncode}"
            )
            if required:
                logger.error(f"管线终止：必需步骤 [{sid}] 失败")
                return False
            logger.warning(f"可选步骤 [{sid}] 失败，继续管线")
            return True

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        err = f"超时（{timeout}s）"
        logger.error(f"❌ [{sid}] {name} {err}（耗时 {elapsed:.0f}s）")
        status.complete_step(sid, success=False, error=err)
        if required:
            return False
        return True

    except Exception as e:
        elapsed = time.time() - t0
        err = str(e)
        logger.error(f"❌ [{sid}] {name} 异常: {err}（耗时 {elapsed:.0f}s）")
        status.complete_step(sid, success=False, error=err)
        if required:
            return False
        return True


# ════════════════════════════════════════════════════════════
#  管线入口
# ════════════════════════════════════════════════════════════


def main() -> None:
    t0: float = time.time()

    logger.info("=" * 60)
    logger.info("Sequoia-X 全自动管线启动")
    logger.info(f"日期: {date.today().isoformat()}")
    logger.info(f"时间: {datetime.now().strftime('%H:%M:%S')}")
    logger.info("=" * 60)

    # ────────────────────────────────────────────────
    #  交易日判断（管线级前置条件）
    # ────────────────────────────────────────────────
    if not is_trade_day():
        logger.info("⏭️ 非交易日，管线跳过")
        status = PipelineStatus()
        status.reset()
        status.data["pipeline_status"] = "skipped"
        status.data["started_at"] = datetime.now().strftime("%H:%M:%S")
        status.finish("skipped")

        # 推送跳过通知
        _push_skip_notice()
        return

    # ────────────────────────────────────────────────
    #  初始化状态文件
    # ────────────────────────────────────────────────
    status = PipelineStatus()
    if status.data is not None:
        prev = status.data.get("pipeline_status")
        if prev == "running":
            logger.warning("检测到上一轮管线状态为 running，可能异常中断，重置")
    status.reset()
    status.data["started_at"] = datetime.now().strftime("%H:%M:%S")
    status.save()

    # 注册所有步骤
    for step_cfg in STEPS:
        status.add_step(step_cfg["id"], step_cfg["name"])

    # ────────────────────────────────────────────────
    #  逐步骤执行
    # ────────────────────────────────────────────────
    pipeline_ok: bool = True
    completed: int = 0
    total: int = len(STEPS)

    for step_cfg in STEPS:
        ok = run_step(step_cfg, status)
        if ok:
            completed += 1
        else:
            pipeline_ok = False
            break  # 必需步骤失败，终止管线

    # ────────────────────────────────────────────────
    #  结束
    # ────────────────────────────────────────────────
    elapsed: float = time.time() - t0
    final_status: str = "completed" if pipeline_ok else "failed"
    status.finish(final_status)

    logger.info("=" * 60)
    logger.info(
        f"管线运行完成: {final_status}"
        f"（{completed}/{total} 步骤通过, 总耗时 {elapsed:.0f}s）"
    )
    logger.info("=" * 60)

    # 推送全管线汇总
    push_pipeline_summary(status.data)


def _push_skip_notice() -> None:
    """推送非交易日跳过通知到微信。"""
    try:
        from wxpusher import WxPusher

        from sequoia_x.core.config import get_settings

        settings = get_settings()
        today_str: str = date.today().strftime("%m-%d")
        now_str: str = datetime.now().strftime("%H:%M")
        WxPusher.send_message(
            content=(
                f"Sequoia-X 管线 | {today_str}\n\n"
                f"⏭️ 非交易日，今日管线跳过\n"
                f"{now_str} 检测到周末/节假日"
            ),
            token=settings.wxpusher_token,
            topic_ids=settings.wxpusher_topic_ids,
            content_type=1,
        )
        logger.info("跳过通知推送成功")
    except Exception as e:
        logger.warning(f"推送跳过通知异常: {e}")


if __name__ == "__main__":
    main()
