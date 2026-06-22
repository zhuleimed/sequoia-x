"""管线状态管理：status.json 读写 + 微信推送。

status.json 存放位置：/public/home/hpc/zhulei/superman/quant/code/pipeline_status.json
（在 code/ 根目录而非项目内，方便跨项目访问）
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)

# ── 状态文件路径（code 根目录，非项目内） ──
STATUS_FILE = Path(
    "/public/home/hpc/zhulei/superman/quant/code/pipeline_status.json"
)


class PipelineStatus:
    """管理 pipeline_status.json 的读写。

    用法：
        status = PipelineStatus()
        status.reset()                  # 创建当日新记录
        status.add_step("sync", "数据同步")
        status.start_step("sync")
        ... 执行步骤 ...
        status.complete_step("sync", success=True, detail={...})
        status.finish("completed")
    """

    def __init__(self) -> None:
        self.data: dict | None = self._load_or_create()

    # ── 内部读写 ──

    def _load_or_create(self) -> dict | None:
        """加载当日已有状态文件，若不存在或无当日记录则返回 None。"""
        if not STATUS_FILE.exists():
            return None
        try:
            data: dict = json.loads(STATUS_FILE.read_text())
            today: str = datetime.now().strftime("%Y-%m-%d")
            if data.get("date") == today:
                return data
        except Exception:
            pass
        return None

    def _save(self) -> None:
        STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATUS_FILE.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2)
        )

    # ── 生命周期 ──

    def reset(self) -> None:
        """创建新的当日状态记录（覆盖已有）。"""
        self.data = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "pipeline_status": "running",
            "started_at": None,
            "finished_at": None,
            "current_step": None,
            "steps": {},
        }
        self._save()

    def finish(self, status: str = "completed") -> None:
        """标记管线结束。"""
        if self.data is None:
            return
        self.data["pipeline_status"] = status
        self.data["finished_at"] = datetime.now().strftime("%H:%M:%S")
        self.data["current_step"] = None
        self._save()

    # ── 步骤生命周期 ──

    def add_step(self, step_id: str, step_name: str) -> None:
        """向状态文件注册一个步骤（pipeline 启动时调用）。"""
        if self.data is None:
            return
        self.data["steps"][step_id] = {
            "name": step_name,
            "status": "pending",
            "started_at": None,
            "finished_at": None,
            "duration": None,
            "error": None,
            "detail": {},
        }
        self._save()

    def start_step(self, step_id: str) -> None:
        """标记步骤开始运行。"""
        if self.data is None or step_id not in (self.data.get("steps") or {}):
            return
        self.data["current_step"] = step_id
        self.data["steps"][step_id]["status"] = "running"
        self.data["steps"][step_id]["started_at"] = datetime.now().strftime(
            "%H:%M:%S"
        )
        self._save()

    def complete_step(
        self,
        step_id: str,
        success: bool = True,
        detail: dict | None = None,
        error: str | None = None,
    ) -> None:
        """标记步骤完成。自动计算耗时。"""
        if self.data is None or step_id not in self.data.get("steps", {}):
            return
        step = self.data["steps"][step_id]
        step["status"] = "completed" if success else "failed"
        step["finished_at"] = datetime.now().strftime("%H:%M:%S")
        # 计算耗时（秒）
        if step["started_at"]:
            try:
                fmt = "%H:%M:%S"
                s = datetime.strptime(step["started_at"], fmt)
                e = datetime.strptime(step["finished_at"], fmt)
                step["duration"] = int((e - s).total_seconds())
            except ValueError:
                step["duration"] = None
        step["error"] = error
        if detail:
            step["detail"] = detail
        self._save()


# ════════════════════════════════════════════════════════════
#  推送
# ════════════════════════════════════════════════════════════


def push_pipeline_summary(status_data: dict) -> None:
    """通过 WxPusher 推送全管线执行摘要。

    Args:
        status_data: PipelineStatus.data（字典结构）。
    """
    from wxpusher import WxPusher

    from sequoia_x.core.config import get_settings

    try:
        settings = get_settings()
    except Exception as e:
        logger.warning(f"推送管线摘要: 获取配置失败 {e}")
        return

    today = status_data.get("date", "??")
    overall = status_data.get("pipeline_status", "unknown")
    started = status_data.get("started_at", "--:--")
    finished = status_data.get("finished_at", "--:--")
    steps: dict = status_data.get("steps", {})

    # ── 组装步骤行 ──
    step_lines: list[str] = []
    all_passed: bool = True
    total_duration: int = 0
    emoji: dict[str, str] = {
        "completed": "✅",
        "failed": "❌",
        "skipped": "⏭️",
        "running": "⏳",
        "pending": "⬜",
    }

    for sid, s in steps.items():
        d: int = s.get("duration") or 0
        total_duration += d
        e = emoji.get(s.get("status", ""), "❓")
        dur_str: str = f"{d // 60}min" if d >= 60 else f"{d}s"
        line: str = f"{e} {s.get('name', sid)}    {dur_str}"
        # 追加明细（如有）
        detail: dict = s.get("detail", {}) or {}
        if detail.get("stock_count"):
            line += f" 股票{detail['stock_count']}只"
        if s.get("error"):
            line += f" ⚠️ {s['error']}"
            all_passed = False
        step_lines.append(line)

    # ── 总计 ──
    total_min: int = total_duration // 60
    total_sec: int = total_duration % 60
    total_str: str = (
        f"{total_min}min{total_sec}s" if total_min > 0 else f"{total_sec}s"
    )

    status_emoji = (
        "✅"
        if overall == "completed" and all_passed
        else "⚠️" if overall == "completed"
        else "❌"
    )
    status_text = (
        "全部完成 ✓"
        if overall == "completed" and all_passed
        else "部分异常"
        if overall == "completed"
        else "失败"
    )

    message: str = (
        f"Sequoia-X 全管线报告 | {today}\n\n"
        f"{status_emoji} {status_text}\n"
        f"⏱ {started} → {finished}（共 {total_str}）\n\n"
        + "\n".join(step_lines)
    )

    try:
        r = WxPusher.send_message(
            content=message,
            token=settings.wxpusher_token,
            topic_ids=settings.wxpusher_topic_ids,
            content_type=1,
        )
        if r.get("code") == 1000:
            logger.info("管线完成摘要推送成功")
        else:
            logger.warning(f"管线摘要推送失败: {r}")
    except Exception as e:
        logger.warning(f"管线摘要推送异常: {e}")
