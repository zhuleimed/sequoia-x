#!/usr/bin/env python
"""验证：SimEngine 启动时打印"卖出档位自检"（2026-10-04，用户要求）。

背景：V4 盘 = 只留硬止损（用户 09-05 拍板）、LLM 盘 = 全规则（有意保留）——
两个盘档位不同，必须一眼能从日志分清，防止"以为被改回去了"这类误会。

覆盖三种档位各打印正确的含义说明 + 库名（账户=库文件）。

运行（铁律六：py312）：
    .../python -m pytest tests/test_sell_policy_selfcheck.py -q
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

PROJ = Path(__file__).resolve().parents[1]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))


class _Capture(logging.Handler):
    """抓取日志原文（项目 logger 设了 propagate=False，pytest caplog 收不到）。"""

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.mark.parametrize("mode,expect", [
    ("hard_stop_only", "只留硬止损"),
    ("all", "全规则"),
    ("none", "完全不止损"),
])
def test_sell_policy_selfcheck(tmp_path, mode, expect):
    from sequoia_x.core.config import get_settings
    from sequoia_x.core.logger import get_logger
    from sequoia_x.simulation.engine import SimEngine

    # "all" 是默认值：不传参的实例（= LLM 盘口径）也应打印自检
    kwargs = {} if mode == "all" else {"sell_rules_mode": mode}

    cap = _Capture()
    lg = get_logger("sequoia_x.simulation.engine")   # 与 engine.py 模块内是同一实例
    lg.addHandler(cap)
    try:
        SimEngine(get_settings(), db_path=str(tmp_path / "probe.db"), **kwargs)
    finally:
        lg.removeHandler(cap)

    hits = [m for m in cap.messages if "卖出档位自检" in m]
    assert hits, "未打印卖出档位自检"
    line = hits[0]
    assert expect in line, f"档位说明不符: {line}"
    assert "probe.db" in line, "自检未带库名（账户=库文件，用于区分 V4/LLM）"
    assert "硬止损=" in line, "自检未带硬止损阈值"
