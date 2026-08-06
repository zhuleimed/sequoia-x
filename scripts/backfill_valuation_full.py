#!/usr/bin/env python
"""全量估值修复：用 baostock 拉取全周期 4 个估值字段（真实历史值）。

背景（2026-08-03 调查结论）:
    2026-07-31 backfill 用 TDX 当前财务快照推算全史 PE/PB/PS/PCF，导致:
    1. 当前亏损股（1394 只）全史 peTTM 被抹 0（197 万条，26.8%）
    2. 当前盈利股全史 PE 用 2026 年 EPS 计算，严重失真（茅台 2020 PE 库 9.17 vs 真实 56.30）
    3. psTTM/pcfNcfTTM 同样被当前财务推算污染（pcfNcf 50.4% 被抹 0）
    本脚本用 baostock 拉取 2020-01-01 ~ 2026-07-31 全部真实估值，覆盖失真值。
    baostock 对亏损股返回负值（真实语义，如万科 -0.45），保留不转 0。

铁律:
    六（py312）、四（断点续跑）、五（详尽日志+自检）、二（nohup 解绑由启动命令保证）

用法:
    nohup /home/zhulei/anaconda3/envs/zhulei_py312/bin/python -u \
        scripts/backfill_valuation_full.py > logs/valuation_fix_$(date +%Y%m%d).log 2>&1 &
"""

import json
import logging
import os
import sqlite3
import sys
import time
from datetime import date, timedelta
from pathlib import Path

# ── KMP_AFFINITY 清除（铁律一：import 任何数值库之前）──
os.environ.pop("KMP_AFFINITY", None)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import baostock as bs  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("valuation_fix")

DB_PATH = PROJECT_ROOT / "data" / "sequoia_v2.db"
START_DATE = "2020-01-01"      # 全周期起点（数据库最早数据）
END_DATE = "2026-07-31"         # 修复目标截止（7/31 最后交易日；之后由日常同步负责）
PROGRESS_FILE = PROJECT_ROOT / "data" / "cache" / "valuation_fix_progress.json"
MAX_REQ_PER_LOGIN = 1200        # baostock 连接上限（sync.py 用 1400，保守取 1200）
RETRY_TIMES = 2                 # 单只失败重试次数


def _to_bs_code(symbol: str) -> str:
    """纯数字代码 → baostock 格式（与 engine._to_baostock_code 一致）。"""
    prefix = "sh" if symbol.startswith(("5", "6", "9")) else "sz"
    return f"{prefix}.{symbol}"


def _safe_float(v) -> float | None:
    """baostock 字符串转 float；空/异常 → None（None 表示该日 baostock 无值，不覆盖原值）。"""
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def load_progress() -> set[str]:
    """断点续跑：加载已完成 symbol 集合。"""
    if PROGRESS_FILE.exists():
        try:
            return set(json.loads(PROGRESS_FILE.read_text()))
        except Exception as e:
            logger.warning(f"进度文件损坏，从头开始: {e}")
    return set()


def save_progress(done: set[str]):
    """增量保存：每完成一只立即写盘（铁律四）。"""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = PROGRESS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(sorted(done)))
    tmp.rename(PROGRESS_FILE)


def fetch_valuation(bs_code: str) -> list[tuple]:
    """baostock 拉单股全周期估值，返回 [(date, pe, pb, ps, pcf), ...]。"""
    rs = bs.query_history_k_data_plus(
        bs_code,
        "date,peTTM,pbMRQ,psTTM,pcfNcfTTM",
        start_date=START_DATE,
        end_date=END_DATE,
        frequency="d",
        adjustflag="2",
    )
    if rs.error_code != "0":
        raise RuntimeError(f"baostock 错误 {rs.error_code}: {rs.error_msg}")
    rows = []
    while rs.next():
        row = rs.get_row_data()  # [date, peTTM, pbMRQ, psTTM, pcfNcfTTM]
        if len(row) < 5:
            continue
        vals = tuple(_safe_float(x) for x in row[1:5])
        if all(v is None for v in vals):
            continue  # 全空行跳过（该日无任何估值）
        rows.append((row[0], *vals))
    return rows


def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info(f"全量估值修复 | baostock 真实值 | {START_DATE} ~ {END_DATE}")
    logger.info(f"PID={os.getpid()} | Python={sys.executable}")
    logger.info("=" * 60)

    # ── 启动诊断（铁律五）──
    logger.info(f"KMP_AFFINITY 已清除: {os.environ.get('KMP_AFFINITY') is None}")

    done = load_progress()
    logger.info(f"断点续跑: 已完成 {len(done)} 只")

    # ── 登录 baostock ──
    lg = bs.login()
    if lg.error_code != "0":
        logger.error(f"baostock 登录失败: {lg.error_msg}")
        sys.exit(1)
    logger.info("baostock 登录成功")
    requests_since_login = 0

    # ── 获取全部股票 ──
    conn = sqlite3.connect(DB_PATH)
    symbols = [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM stock_daily")]
    conn.close()
    logger.info(f"待处理股票: {len(symbols)} 只（已完成 {len(done)} 只，剩余 {len(symbols)-len(done)} 只）")

    total_updated = 0
    total_failed: list[str] = []
    t_batch = time.time()

    for i, sym in enumerate(symbols):
        if sym in done:
            continue  # 断点续跑：跳过已完成

        bs_code = _to_bs_code(sym)
        updated = 0
        try:
            # baostock 连接上限：每 MAX_REQ_PER_LOGIN 次请求重新登录
            if requests_since_login >= MAX_REQ_PER_LOGIN and i > 0:
                bs.logout()
                lg = bs.login()
                if lg.error_code != "0":
                    raise RuntimeError(f"重新登录失败: {lg.error_msg}")
                requests_since_login = 0
                logger.info(f"baostock 已重新登录（{i}/{len(symbols)}）")

            # 带重试的拉取
            rows = None
            relogin_count = 0  # 连接失效连续重登保护
            for attempt in range(RETRY_TIMES + 1):
                try:
                    rows = fetch_valuation(bs_code)
                    requests_since_login += 1
                    break
                except Exception as e:
                    err = str(e)
                    if "未登录" in err or "10001001" in err:
                        # 连接失效（TCP 踢线）→ 强制重新登录后重试（不消耗重试次数）
                        relogin_count += 1
                        if relogin_count > 3:
                            raise RuntimeError(f"连续重新登录 {relogin_count} 次仍失败: {err}")
                        try:
                            bs.logout()
                        except Exception:
                            pass
                        lg = bs.login()
                        if lg.error_code != "0":
                            raise RuntimeError(f"重新登录失败: {lg.error_msg}")
                        requests_since_login = 0
                        logger.warning(f"  {sym} baostock 连接失效已重新登录（第{relogin_count}次），重试")
                        time.sleep(0.5)
                    elif attempt < RETRY_TIMES:
                        logger.warning(f"  {sym} 第{attempt+1}次失败({e})，重试")
                        time.sleep(1.0)
                    else:
                        raise

            # 批量 UPDATE（只覆盖 baostock 有值的行；None 行不动）
            if rows:
                conn = sqlite3.connect(DB_PATH)
                conn.executemany(
                    "UPDATE stock_daily SET peTTM=?, pbMRQ=?, psTTM=?, pcfNcfTTM=? "
                    "WHERE symbol=? AND date=?",
                    [(*vals, sym, dt) for dt, *vals in rows],
                )
                conn.commit()
                conn.close()
                updated = len(rows)
                total_updated += updated
        except Exception as e:
            total_failed.append(sym)
            logger.error(f"  {sym} 最终失败: {e}")

        # 增量保存（铁律四）：只保存成功的——失败的股票重启后自动重试
        done.add(sym)
        if updated > 0:
            save_progress(done)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_batch
            rate = elapsed / (i + 1)
            remaining = rate * (len(symbols) - i - 1)
            logger.info(
                f"进度: {i+1}/{len(symbols)} ({100*(i+1)//len(symbols)}%), "
                f"更新 {total_updated:,} 条, 失败 {len(total_failed)} 只, "
                f"速率 {rate:.2f}s/只, ETA {remaining/60:.0f}min"
            )

    bs.logout()

    # ── 运行时自检（铁律五：验证输出有效性）──
    logger.info("=" * 60)
    logger.info("修复完成，运行时自检:")
    conn = sqlite3.connect(DB_PATH)
    for f in ["peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"]:
        n = conn.execute(f"SELECT COUNT(*), SUM(CASE WHEN {f} > 0 THEN 1 ELSE 0 END) FROM stock_daily").fetchone()
        nz = conn.execute(f"SELECT SUM(CASE WHEN {f} != 0 AND {f} IS NOT NULL THEN 1 ELSE 0 END) FROM stock_daily").fetchone()[0]
        logger.info(f"  {f}: 总计 {n[0]:,} 条, >0 {n[1]:,} ({100*n[1]/n[0]:.1f}%), 非零(含负) {nz:,}")
    neg = conn.execute("SELECT COUNT(*) FROM stock_daily WHERE peTTM < 0").fetchone()[0]
    logger.info(f"  peTTM<0（亏损股真实负值）: {neg:,} 条")
    conn.close()

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(f"完成 | 更新 {total_updated:,} 条 | 失败 {len(total_failed)} 只 | 总耗时 {elapsed/60:.1f}min")
    if total_failed:
        logger.warning(f"失败股票: {','.join(total_failed[:50])}")
    logger.info(f"断点续跑进度: {PROGRESS_FILE} | 重跑同命令可跳过已完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
