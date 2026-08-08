"""研究状态快照生成器（铁律七：研究状态管理，2026-08-07 建立）

从文件系统自动提取研究状态 → 生成 RESEARCH_STATE.md（单一事实源）。
会话启动（SessionStart hook）与实验完成后自动运行，无需人工提醒。

提取内容:
  1. 各方向实验进度（.tmp/*/ 完成标记计数）
  2. 监督链/实验进程运行状态（ps 探测）
  3. 最新结果摘要（output/backtest_v2/experiments/*_ic_report.csv）
  4. 当前时间 + 各实验日志尾部
  5. 待办提示（V3 文档 §14 锚点）

用法:
  python scripts/save_research_state.py            # 生成 RESEARCH_STATE.md
  python scripts/save_research_state.py --print     # 生成并打印简短摘要（hook 用）
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = ROOT / "RESEARCH_STATE.md"

# 各实验的 tmp 目录与名称（按 V3 文档方向编号）
EXPERIMENTS = [
    ("方向一 LGBMRanker", ".tmp/t2_ranker", "experiments/t2_ranker/experiment_t2_ranker.py"),
    ("方向二 DLinear", ".tmp/dlinear", "experiments/dlinear/experiment_dlinear.py"),
    ("方向三 RankIC-LSTM", ".tmp/rankic_lstm", "experiments/rankic_lstm/experiment_rankic.py"),
    ("方向四 Kronos", None, None),          # 无 tmp，状态从 V3 文档标题提取
    ("方向五 PatchTST", None, None),        # 前置门槛，未启动
]

# V3 文档章节标题 → 方向名（状态从【状态: ...】标记提取，单一事实源）
DOC_SECTIONS = [
    ("## 6. 方向一", "方向一"),
    ("## 7. 方向二", "方向二"),
    ("## 8. 方向三", "方向三"),
    ("## 9. 方向四", "方向四"),
    ("## 10. 方向五", "方向五"),
]

# 各方向 IC 报告（存在则提取最新总体统计）
IC_REPORTS = {
    "方向一": "output/backtest_v2/experiments/t2_ranker_ic_report.csv",
    "方向二": "output/backtest_v2/experiments/dlinear_ic_report.csv",
    "方向三": "output/backtest_v2/experiments/rankic_lstm_ic_report.csv",
}


def _count_json(d: Path) -> int:
    return len(list(d.glob("*.json"))) if d.exists() else 0


def _doc_status(name: str) -> str:
    """从 V3 文档章节标题提取【状态: ...】标记（找不到回退空串）。"""
    for sec_marker, dir_name in DOC_SECTIONS:
        if name.startswith(dir_name):
            for line in (ROOT / "V3研究方向与实验研究记录.md").read_text(
                    errors="ignore").splitlines():
                if line.startswith(sec_marker):
                    i = line.find("【状态:")
                    return line[i + 4:line.find("】", i)] if i >= 0 else ""
    return ""


def _running_procs() -> list[str]:
    """探测正在运行的相关进程（实验/监督/管线）。"""
    out = subprocess.run(
        ["ps", "-eo", "args", "--no-headers"],
        capture_output=True, text=True, timeout=10,
    ).stdout
    found = []
    for kw in ["experiment_t2_ranker", "experiment_dlinear", "experiment_rankic",
               "supervisor", "pipeline/pipeline.py"]:
        if any(kw in line for line in out.splitlines()):
            found.append(kw)
    return found


def _last_lines(path: Path, n: int = 3) -> str:
    if not path.exists():
        return "(无日志)"
    try:
        return " | ".join(line.strip()[:100] for line in path.read_text(errors="ignore")
                          .splitlines()[-n:] if line.strip())
    except Exception:
        return "(日志读取失败)"


def _ic_summary(csv_path: str) -> str:
    p = ROOT / csv_path
    if not p.exists():
        return "无报告"
    try:
        import csv
        rows = list(csv.DictReader(p.open()))
        if not rows:
            return "空报告"
        # 取总体统计由脚本生成时写在 stdout——这里只报行数
        return f"{len(rows)} 个月明细"
    except Exception:
        return "报告解析失败"


def generate(print_summary: bool = False) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# 研究状态快照（RESEARCH_STATE）",
        "",
        f"> 自动生成: {now} | 生成器: scripts/save_research_state.py | 铁律七",
        "> 本文件是研究状态的**单一事实源**——会话启动/恢复时优先读取，1 分钟重建全部状态。",
        "> 详细过程记录见 `V3研究方向与实验研究记录.md`；教训/规则见 memory/。",
        "",
        "## 一、各方向实验状态",
        "",
        "| 方向 | 状态 | 进度 | 关键结果 | 日志尾部 |",
        "|------|------|------|---------|---------|",
    ]

    for name, tmp_rel, script in EXPERIMENTS:
        doc_st = _doc_status(name)
        if tmp_rel is None:
            lines.append(f"| {name} | {doc_st or '未启动'} | — | — | — |")
            continue
        tmp = ROOT / tmp_rel
        n = _count_json(tmp)
        # 判断预期总数（脚本中 70 个月）; 文档标记优先（证伪/完成是终态）
        state = doc_st if doc_st else ("运行中" if (script and _probe_script(script)) else ("已完成" if n >= 70 else "部分完成"))
        lines.append(f"| {name} | {state} | {n}/70 个月 | {_ic_summary(IC_REPORTS[name[:3]])} | {_last_lines(ROOT / 'logs' / _log_for(script))} |")

    lines += [
        "",
        "## 二、运行进程",
        "",
    ]
    procs = _running_procs()
    lines.append("、".join(procs) if procs else "无相关进程（实验/监督/管线均未运行）")
    lines += [
        "",
        "## 三、监督链状态",
        "",
        f"监督日志尾部: {_last_lines(ROOT / 'logs' / 'exp_rankic_supervisor_20260806.log', 4)}",
        "",
        "## 四、待办（详见 V3 文档 §14）",
        "",
        "- 方向一/二/三/四均已完结（前三完成 70 个月, 方向四 3a+3b 证伪）",
        "- 后续: 融合矩阵实验 / 72 组回测验证 / 2026-07 月补测",
        "",
    ]
    text = "\n".join(lines)
    STATE_PATH.write_text(text)

    if print_summary:
        # hook 用：只打印一行摘要注入会话
        procs_s = "、".join(procs) if procs else "无"
        statuses = " ".join(f"{name.split()[0]}:{_doc_status(name) or ('运行中' if _probe_script(script) else '完成')}"
                            for name, _, script in EXPERIMENTS)
        print(f"[状态] {now} | {statuses} | 进程: {procs_s}")
    return text


def _probe_script(script: str) -> bool:
    """探测该实验脚本是否在运行。"""
    out = subprocess.run(["ps", "-eo", "args", "--no-headers"],
                         capture_output=True, text=True, timeout=10).stdout
    return any(script.split("/")[-1] in line for line in out.splitlines())


def _log_for(script: str) -> str:
    """实验脚本 → 日志名映射（约定）。"""
    base = script.split("/")[-1].replace(".py", "")
    return f"{base}.log"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true", help="生成并打印简短摘要（hook 用）")
    args = ap.parse_args()
    generate(print_summary=args.print)


if __name__ == "__main__":
    main()
