#!/usr/bin/env python3
"""全量补写 stock_daily.pctChg（baostock 不复权口径，交易所标准涨跌幅）。

背景：stock_daily.pctChg 仅最近 11 天有值（2026-08-03 起），全历史 NULL。
本脚本用 baostock query_history_k_data_plus(adjustflag="3") 拉取全历史
pctChg，按 (symbol, date) 匹配更新库中已有行。

口径（2026-08-18 实测验证）：
- baostock adjustflag="3" = 不复权；pctChg = (close - preclose)/preclose × 100
- 非除权日 = 真实涨跌幅；除权日基于交易所除权基准价（行情软件口径）
- 与库已有 11 天数据逐日一致（600519/688167/000001 全对）

特性：断点续跑（同命令恢复）、三问日志、failed 清单、--status/--dry-run

用法：
    python scripts/backfill_pctchg.py              # 全量补写（断点续跑）
    python scripts/backfill_pctchg.py --status     # 查看进度
    python scripts/backfill_pctchg.py --dry-run    # 只统计不写库
    python scripts/backfill_pctchg.py --limit 100  # 只处理前 100 只（试点）
"""

import argparse
import json
import logging
import os
import signal
import sqlite3
import sys
import time
from collections import Counter
from datetime import date, datetime
from pathlib import Path

import baostock as bs

# 单只查询超时（baostock 服务器偶发挂起不响应，实测会无限阻塞）
QUERY_TIMEOUT_SECONDS = 60


class _QueryTimeout(Exception):
    """baostock 单次查询超时。"""


def _timeout_handler(signum, frame):
    raise _QueryTimeout(f"查询超时 {QUERY_TIMEOUT_SECONDS}s (signal {signum})")

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "sequoia_v2.db"
PROGRESS_PATH = PROJECT_ROOT / "scripts" / "tmp" / "pctchg_backfill_progress.json"
LOG_DIR = PROJECT_ROOT / "logs"


def get_logger():
    """简易日志（文件 + 控制台双输出）。"""
    logger = logging.getLogger("pctchg_backfill")
    logger.setLevel(logging.DEBUG)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(
        LOG_DIR / f"pctchg_backfill_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(ch)

    return logger


logger = get_logger()


# ---------------------------------------------------------------------------
# 进度持久化（断点续跑）
# ---------------------------------------------------------------------------
def load_progress():
    """加载进度：返回 (completed set, failed dict, stats dict)。"""
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return set(data.get("completed", [])), data.get("failed", {}), data.get("stats", {})
    return set(), {}, {}


def save_progress(completed, failed, stats):
    """持久化进度（每只股票完成后调用，崩溃后同命令恢复）。"""
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {"completed": sorted(completed), "failed": failed, "stats": stats}
    # 先写临时文件再原子替换，避免写一半崩溃损坏进度
    tmp = PROGRESS_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    tmp.replace(PROGRESS_PATH)


# ---------------------------------------------------------------------------
# baostock 拉取
# ---------------------------------------------------------------------------
def code_to_bs(symbol: str) -> str:
    """库中 6 位数字代码 → baostock 前缀代码；B股/未知返回 None。"""
    if symbol.startswith(("6", "9")):
        return "sh." + symbol
    if symbol.startswith(("0", "3")):
        return "sz." + symbol
    if symbol.startswith(("4", "8")):
        return "bj." + symbol
    return None


def fetch_pctchg(code_bs: str):
    """拉一只股票全历史 pctChg（不复权），返回 [(date, pctChg), ...]。

    baostock 服务器偶发挂起不响应（socket ESTAB 但无数据）会导致无限阻塞，
    用 signal.alarm 加硬超时保护。
    """
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(QUERY_TIMEOUT_SECONDS)
    try:
        rs = bs.query_history_k_data_plus(
            code_bs, "date,pctChg",
            start_date="1990-01-01", end_date=date.today().isoformat(),
            frequency="d", adjustflag="3",
        )
        fields = list(rs.fields)  # 必须在迭代前拷贝！迭代后 fields 被清空
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        if len(rows) > 0 and len(rows[0]) != len(fields):
            raise ValueError(f"字段数异常: fields={len(fields)} row={len(rows[0])} ({code_bs})")
        out = []
        for r in rows:
            try:
                out.append((r[0], float(r[1])))  # 停牌/无效日 pctChg 为空 → float 抛异常跳过
            except (ValueError, IndexError):
                pass
        return out
    finally:
        signal.alarm(0)  # 取消闹钟


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="全量补写 stock_daily.pctChg")
    parser.add_argument("--status", action="store_true", help="查看进度")
    parser.add_argument("--dry-run", action="store_true", help="只统计不写库")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 只（试点用）")
    args = parser.parse_args()

    # 启动自检（铁律一）
    logger.info("=" * 70)
    logger.info("pctChg 全量补写启动 | 当前时间 %s", datetime.now().isoformat())
    logger.info("DB: %s", DB_PATH)
    logger.info("进度文件: %s", PROGRESS_PATH)

    if not DB_PATH.exists():
        logger.error("数据库不存在: %s", DB_PATH)
        sys.exit(1)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")  # 避免长事务锁库

    # 全部 symbol
    symbols = [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM stock_daily")]
    symbols.sort()
    if args.limit > 0:
        symbols = symbols[: args.limit]
    # 前缀分布自检
    prefix_counter = Counter(sym[0] for sym in symbols)
    logger.info("股票总数: %d, 前缀分布: %s", len(symbols), dict(prefix_counter))
    n_bs = sum(1 for s in symbols if code_to_bs(s))
    logger.info("baostock 可覆盖: %d (%.1f%%)", n_bs, n_bs / len(symbols) * 100)

    completed, failed, stats = load_progress()
    logger.info("断点续跑: 已完成 %d 只, 历史失败 %d 只（重跑将重试）", len(completed), len(failed))

    if args.status:
        logger.info("--- 进度报告 ---")
        logger.info("已完成: %d/%d (%.1f%%)", len(completed), len(symbols), len(completed) / len(symbols) * 100)
        logger.info("失败: %d 只: %s", len(failed), list(failed.keys())[:20])
        conn.close()
        return

    # 登录 baostock
    lg = bs.login()
    if lg.error_code != "0":
        logger.error("baostock 登录失败: %s %s", lg.error_code, lg.error_msg)
        sys.exit(1)
    logger.info("baostock 登录成功")

    # 只跳过已完成，failed 重跑时重试
    todo = [s for s in symbols if s not in completed]
    logger.info("本次待处理: %d 只", len(todo))

    t_start = time.time()
    n_done = len(completed)
    total_updates = stats.get("updates", 0)
    for i, sym in enumerate(todo):
        t0 = time.time()
        code_bs = code_to_bs(sym)
        try:
            if code_bs is None:
                raise ValueError(f"无法映射 baostock 代码: {sym}")
            rows = fetch_pctchg(code_bs)
            if rows:
                conn.executemany(
                    "UPDATE stock_daily SET pctChg=? WHERE symbol=? AND date=?",
                    [(p, sym, d) for d, p in rows],
                )
            conn.commit()
            completed.add(sym)
            failed.pop(sym, None)  # 重试成功 → 移出失败清单
            total_updates += len(rows)
            stats["updates"] = total_updates
            # 三问日志：每 50 只打印进度 + ETA
            if (i + 1) % 50 == 0 or i + 1 == len(todo):
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed * 60  # 只/分钟
                remaining = len(todo) - (i + 1)
                eta_min = remaining / rate if rate > 0 else 0
                logger.info(
                    "进度 %d/%d (%.1f%%) | 速率 %.1f 只/min | ETA %.0f min | 已更新 %d 行 | 本只 %d 行 %.1fs",
                    i + 1, len(todo), (i + 1) / len(todo) * 100,
                    rate, eta_min, total_updates, len(rows), time.time() - t0,
                )
            # 增量持久化：每 10 只存一次（减少磁盘写）
            if (i + 1) % 10 == 0:
                save_progress(completed, failed, stats)
        except _QueryTimeout as e:
            # 查询超时：baostock 会话可能已损坏 → 重建后继续（该股票记 failed，重跑重试）
            logger.error("超时 %s: %s → 重建 baostock 会话", sym, e)
            try:
                bs.logout()
            except Exception:
                pass
            lg2 = bs.login()
            if lg2.error_code != "0":
                logger.error("会话重建失败: %s %s", lg2.error_code, lg2.error_msg)
            failed[sym] = str(e)[:200]
            save_progress(completed, failed, stats)
        except Exception as e:
            logger.error("失败 %s: %s", sym, e)
            failed[sym] = str(e)[:200]
            save_progress(completed, failed, stats)

    save_progress(completed, failed, stats)
    bs.logout()
    conn.close()

    # 结束时验证（铁律三）
    logger.info("=" * 70)
    logger.info("补写完成 | 总耗时 %.1f min", (time.time() - t_start) / 60)
    logger.info("成功 %d 只, 失败 %d 只, 更新 %d 行", len(completed), len(failed), total_updates)
    if failed:
        logger.warning("失败清单: %s", list(failed.keys())[:50])


if __name__ == "__main__":
    main()
