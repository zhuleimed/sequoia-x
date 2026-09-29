#!/usr/bin/env python
"""月末链就绪检查（preflight）—— 2026-09-29 建立

用途
----
月末自动链（`scripts/month_end_pull.py`，每工作日 19:00）在**本月最后交易日**会跑：
覆盖率检查 → 股票池刷新 → 入库数据体检 → **重建训练数据集缓存(2-6h)** → 自检 →
单月干跑验证 → 微信推送；随后 1 号 03:00 `v2_monthly_retrain.py` 轮询等缓存就绪并重训选股。

**这条链是全自动的，但"自动"≠"一定成功"。** 本脚本在月末当天**白天**跑一次，
把"会导致链条静默出错/空转/失败"的前提逐条查出来 —— 全部只读，不写任何东西。

为什么需要它（真实教训）
------------------------
· 2026-08-31 实测：`.rebuild_done` 标记已写但 `_verify_caches` 崩溃，**验证缺失未被发现**。
· 2026-09-29 发现：`output/backtest_v2/.dryrun_cache.json` 是 **09-25 的陈旧产物**，
  而链条干跑判据是 `if r.returncode != 0 or not dry.exists()` —— **陈旧文件会让失败的干跑"假通过"**。
  （已改名留档 `.stale_20260925.bak`；本脚本会把这条列为检查项，防止再次发生。）
· 数据集缓存键含**动态 `sample_end`（= DB 最后交易日）** ⇒ **每天前移**、缓存隔天失效；
  月末当天该键必然不存在（**这是正常的**，链条 ② 会建）—— 本脚本负责说清"缺失是正常的、还是异常的"。

用法
----
    python scripts/month_end_preflight.py            # 检查"今天"
    python scripts/month_end_preflight.py --date 2026-09-30

退出码：0 = 就绪（无阻断项）；1 = 有阻断项（**不要放任自动链空跑**，按提示处置）。
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

OK, WARN, BAD = "✅", "⚠️ ", "❌"
issues: list[str] = []


def say(flag: str, title: str, detail: str = "") -> None:
    print(f"{flag} {title}" + (f"\n     {detail}" if detail else ""))
    if flag is BAD:
        issues.append(title)


def check_trade_day(today: dt.date) -> set[str]:
    print("── 1) 交易日 ──")
    try:
        sys.path.insert(0, str(PROJ / "scripts"))
        import month_end_pull as mep  # 与链条**同一实现**，保证判定一致
        td = mep.get_trade_dates()
        is_last = mep.is_last_trade_day(today, td)
        month_left = sorted(d for d in td if d.startswith(today.strftime("%Y-%m")) and d >= today.isoformat())
        say(OK if today.isoformat() in td else WARN,
            f"{today} {'是' if today.isoformat() in td else '不是'}交易日",
            f"本月剩余交易日: {month_left[:8]}{' …' if len(month_left) > 8 else ''}")
        if is_last:
            say(OK, "★ 今天是本月最后交易日 ⇒ 19:00 月末链**会触发**")
        else:
            say(OK, f"今天不是本月最后交易日 ⇒ 19:00 链条只做日更（正常）",
                f"（本月最后交易日 = {month_left[-1] if month_left else '未知'}）")
        return td
    except Exception as e:  # akshare 需要网络
        say(WARN, f"交易日历不可用（{type(e).__name__}: {e}）", "日更链会自行处理；如需精确判定请联网重跑")
        return set()


def check_code() -> None:
    print("── 2) 代码与版本 ──")
    try:
        from sequoia_x.model_selection_v2.labels import FEATURE_VERSION
        say(OK, f"FEATURE_VERSION = {FEATURE_VERSION}",
            "（v4→v5 = 深市复权口径/amount 补缺/指数取数/死列 等真数据修复）")
    except Exception as e:
        say(BAD, f"无法读取 FEATURE_VERSION: {e}")
    r = subprocess.run(["git", "status", "--porcelain"], cwd=PROJ, capture_output=True, text=True)
    # 排除**自动生成**的文件：RESEARCH_STATE.md 由 SessionStart hook 每次会话重写时间戳，
    # 与链条行为无关。不排除会让这条检查天天报 ⚠️ ⇒ 警告失去意义、掩盖真正的未提交改动。
    AUTO_GEN = {"RESEARCH_STATE.md"}
    changed = [ln for ln in r.stdout.splitlines() if ln[3:].strip() not in AUTO_GEN and ln[3:].strip()]
    if changed:
        say(WARN, "工作区有未提交改动", " ｜ ".join(changed)[:300] +
            "\n     ⇒ 链条会跑**当前工作区**的代码；确认这些改动是有意的")
    else:
        say(OK, "工作区干净（链条跑的代码 = 已提交版本；已排除自动生成的 RESEARCH_STATE.md）")


def check_stale_artifacts(today: dt.date) -> None:
    print("── 3) 陈旧产物（最容易被忽略的一类，会导致空转/假通过）──")
    dry = PROJ / "output/backtest_v2/.dryrun_cache.json"
    if dry.exists():
        m = dt.datetime.fromtimestamp(dry.stat().st_mtime)
        say(BAD, f"存在陈旧干跑产物 {dry.name}（{m:%Y-%m-%d %H:%M}）",
            "链条干跑判据是 `not dry.exists()` ⇒ **陈旧文件会让失败的干跑假通过**。\n"
            "     处置：mv 改名留档（勿直接删，留审计链）")
    else:
        say(OK, "无陈旧干跑产物（`.dryrun_cache.json` 不存在）")

    markers = sorted((PROJ / "output/backtest_v2").glob(".rebuild_done_*"))
    today_marker = [m for m in markers if today.strftime("%Y%m%d") in m.name]
    if today_marker:
        say(BAD, f"存在**本日**重建标记 {[m.name for m in today_marker]}",
            "⇒ 链条 ② 会**跳过重建**！若今天的重建并未真正完成，必须清掉该标记")
    else:
        say(OK, f"无本日重建标记（历史标记 {len(markers)} 个不触发跳过，文件名含日期）")

    tmps = sorted((PROJ / "output/backtest_v2").glob(".cache_tmp_*"))
    errs = [p for p in tmps if list(p.glob("*.error"))]
    if errs:
        say(WARN, f"{len(errs)} 个 tmp 目录内有 .error 残留",
            " ｜ ".join(f"{p.name}:{len(list(p.glob('*.error')))}" for p in errs[:4]))
    else:
        say(OK, f"无 .error 残留（tmp 目录 {len(tmps)} 个）")


def check_dataset_cache(today: dt.date) -> None:
    print("── 4) 训练数据集缓存键（含动态 sample_end，每天前移）──")
    try:
        from sequoia_x.model_selection_v2.config import V2Config
        from sequoia_x.model_selection_v2.labels import FEATURE_VERSION
        cfg = V2Config()
        db = PROJ / "data/sequoia_v2.db"
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        dblast = con.execute("SELECT MAX(date) FROM stock_daily").fetchone()[0]
        con.close()
        pool = PROJ / "output/backtest_v2/.stock_pool.json"
        n = len(json.loads(pool.read_text())) if pool.exists() else 2978

        def key(se: str, market_state: bool = True) -> str:
            d = {"n_stocks": n, "sample_start": cfg.sample_start, "sample_end": se,
                 "window": cfg.window, "feature_version": FEATURE_VERSION,
                 "market_state": market_state}
            d["extra_features"] = True
            return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]

        sample_end = max(cfg.sample_end, dblast or "")
        k = key(sample_end)
        p = PROJ / f"data/cache/v2_dataset/{k}"
        exists = p.exists()
        say(OK, f"sample_end={sample_end}（= max(config {cfg.sample_end}, DB 末日 {dblast})）→ 键 {k}",
            f"缓存{'已存在' if exists else '**不存在**'}"
            + ("" if exists else " ⇒ 正常：链条 ② 会重建（2-6h），这正是月末链的主要耗时"))
        # 顺带报告所有现存缓存，便于人眼判断
        all_c = sorted((PROJ / "data/cache/v2_dataset").glob("*/"))
        total_gb = sum((p.stat().st_size for p in (PROJ / "data/cache/v2_dataset").rglob("*") if p.is_file()), 0) / 1e9
        say(OK, f"现有数据集缓存 {len(all_c)} 个，合计 {total_gb:.0f} GB")
    except Exception as e:
        say(BAD, f"数据集缓存键计算失败: {type(e).__name__}: {e}")


def check_db_freshness(today: dt.date) -> None:
    print("── 5) 数据库时效（月末链 19:00 跑，数据由 18:10 日更链同步）──")
    try:
        con = sqlite3.connect(f"file:{PROJ / 'data/sequoia_v2.db'}?mode=ro", uri=True)
        d = con.execute("SELECT MAX(date) FROM stock_daily").fetchone()[0]
        n = con.execute("SELECT COUNT(*) FROM stock_daily WHERE date = ?", (d,)).fetchone()[0]
        con.close()
        gap = (today - dt.date.fromisoformat(d)).days
        if d == today.isoformat():
            say(OK, f"DB 最后交易日 = {today}（{n} 行）⇒ 当天数据已入库")
        elif gap <= 3:
            say(OK, f"DB 最后交易日 = {d}（距今 {gap} 天；周末/节假日属正常）")
        else:
            say(WARN, f"DB 最后交易日 = {d}，距今 {gap} 天", "确认 18:10 日更链是否正常")
    except Exception as e:
        say(BAD, f"数据库检查失败: {e}")


def check_disk() -> None:
    print("── 6) 磁盘（月末链会新建 2 个数据集缓存 ≈ 24GB×2）──")
    u = shutil.disk_usage(PROJ / "data")
    free_gb = u.free / 1e9
    say(OK if free_gb > 100 else BAD, f"可用 {free_gb:.0f} GB（总 {u.total/1e9:.0f} GB）",
        "" if free_gb > 100 else "空间不足，链条 ② 会失败")


def check_cron_and_scripts() -> None:
    print("── 7) crontab 与关键脚本 ──")
    r = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    cron = r.stdout
    for label, needle in [("月末拉取+重建 (19:00 工作日)", "month_end_pull.py"),
                          ("月度重训 (1 号 03:00)", "v2_monthly_retrain.py"),
                          ("日更管线 (18:10 工作日)", "pipeline/pipeline.py")]:
        say(OK if needle in cron else BAD, f"cron: {label}", "条目缺失" if needle not in cron else "")
    for s in ["scripts/month_end_pull.py", "scripts/rebuild_dataset_cache.py",
              "scripts/build_prediction_cache.py", "scripts/v2_monthly_retrain.py"]:
        p = PROJ / s
        say(OK if p.exists() else BAD, f"脚本存在: {s}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="", help="检查哪一天（YYYY-MM-DD），默认今天")
    ap.add_argument("--notify", action="store_true",
                    help="发现阻断项时用 wxpusher 推送（**只在有阻断项时推**，避免刷屏）。"
                         "供无人值守的后台/定时运行使用。")
    a = ap.parse_args()
    today = dt.date.fromisoformat(a.date) if a.date else dt.date.today()

    print("=" * 78)
    print(f"  月末链就绪检查 (preflight) | 目标日 {today} | 生成 {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 78)
    check_trade_day(today)
    check_code()
    check_stale_artifacts(today)
    check_dataset_cache(today)
    check_db_freshness(today)
    check_disk()
    check_cron_and_scripts()

    print("=" * 78)
    if issues:
        print(f"❌ 有 {len(issues)} 项阻断，**处置后再放任自动链**：")
        for i in issues:
            print(f"   · {i}")
        if a.notify:
            # 复用 month_end_pull 的推送路径（wxpusher），失败不阻断
            try:
                from wxpusher import WxPusher
                from sequoia_x.core.config import get_settings
                s = get_settings()
                body = "\n".join(f"· {i}" for i in issues)
                WxPusher.send_message(
                    content=f"❌ 月末链就绪检查未通过（{today}）\n{body}\n"
                            f"处置后再放任 19:00 自动链；详见 logs/preflight_{today:%Y%m%d}.log",
                    token=s.wxpusher_token, topic_ids=s.wxpusher_topic_ids, content_type=1)
                print("[notify] ✅ 已推送阻断告警")
            except Exception as e:
                print(f"[notify] ⚠️ 推送失败: {e}")
        return 1
    print("✅ 全部通过 —— 可安全交给 19:00 月末链与人肉不介入")
    print("   注：链条跑完会微信推送成功/失败；19:00–次日 03:00 请不要跑重 CPU 任务（链占 32 进程）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
