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


# 日更链 18:10 启动（cron），实测整链约 37 分钟 ⇒ 数据入库约在 18:50 前后。
# 本检查**必须区分"还没到同步时间"与"该入库却没有"** —— 否则会给出误导性的"正常"：
# 例如 9/30 早上跑时 DB 只有 9/29 的数据，若笼统报"距今 1 天，正常"，
# 会让人误以为当天数据已验证过（实际上那时它根本还没产生）。
SYNC_HHMM = (18, 50)


def check_db_freshness(today: dt.date) -> None:
    print("── 5) 数据库时效（月末链 19:00 跑，当日数据由 18:10 日更链同步）──")
    try:
        con = sqlite3.connect(f"file:{PROJ / 'data/sequoia_v2.db'}?mode=ro", uri=True)
        d = con.execute("SELECT MAX(date) FROM stock_daily").fetchone()[0]
        n = con.execute("SELECT COUNT(*) FROM stock_daily WHERE date = ?", (d,)).fetchone()[0]
        con.close()
        gap = (today - dt.date.fromisoformat(d)).days
        running_today = (today == dt.date.today())
        if d == today.isoformat():
            say(OK, f"DB 最后交易日 = {today}（{n} 行）⇒ 当天数据已入库 ✓ 这一项真验证过了")
        elif running_today and dt.datetime.now().time() < dt.time(*SYNC_HHMM):
            say(OK, f"DB 最后交易日 = {d}（**当天数据尚未同步，属正常**）",
                f"日更链 18:10 启动、约 {SYNC_HHMM[0]}:{SYNC_HHMM[1]:02d} 前后入库；"
                f"链条 19:00 跑时应已就绪。\n"
                f"     ⚠️ 注意：**本次并未验证到当天数据**（那时它还没产生）—— 这项要 18:50 后跑才有意义")
        elif running_today:
            say(BAD, f"已过 {SYNC_HHMM[0]}:{SYNC_HHMM[1]:02d}，但 DB 最后交易日仍是 {d}（无当天数据）",
                "⇒ 日更链可能异常；19:00 月末链会缺当日数据（覆盖率检查会降级 88 维）。"
                "请查 logs/pipeline_$(date +%Y%m%d).log")
        elif gap <= 3:
            say(OK, f"DB 最后交易日 = {d}（距今 {gap} 天；非目标日运行，周末/节假日属正常）")
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
    print("── 7) 调度是否在跑（**看日志，不看 crontab**）──")
    # ★★ 2026-10-10 修正：原实现用 `crontab -l` 找三条 cron 条目 —— 但**调度在 HPC 侧**
    #   （项目路径是 `/public/home/hpc/...` 共享盘），**本机 crontab 为空**
    #   ⇒ 该检查会**天天报 ❌ 假警**；假警多了真警就没人看 ✗
    #   ⇒ 改成看**"近期有没有在跑"的证据**（日志文件的新鲜度）—— 比看 crontab 可靠：
    #     调度在哪台机器上跑都无所谓，只要有"跑过"的痕迹。
    import glob as _g
    specs = [
        ("月末拉取+重建（19:00 工作日）", "month_end_pull_*.log", 4,
         "每个工作日 19:00 都记一行（非月末写「跳过」）⇒ 4 天内应有更新"),
        ("日更管线（18:10 工作日）", "pipeline_*.log", 4, "每个工作日 18:10 跑"),
        ("月度重训（1 号 03:00）", "v2_retrain_*.log", 40, "每月 1 号跑 ⇒ 40 天内应有更新"),
    ]
    for label, pat, max_age, why in specs:
        fs = sorted(_g.glob(str(PROJ / "logs" / pat)),
                    key=lambda x: os.path.getmtime(x), reverse=True)
        if not fs:
            say(BAD, f"调度：{label}", f"找不到日志 {pat} ⇒ 可能从未跑过（{why}）")
            issues.append(f"{label} 无日志证据（{pat}）")
            continue
        age = (dt.date.today() - dt.date.fromtimestamp(os.path.getmtime(fs[0]))).days
        ok = age <= max_age
        say(OK if ok else BAD,
            f"调度：{label} {'✓ 在跑' if ok else f'❌ 已 {age} 天无更新（> {max_age} 天）'}",
            f"最近日志 {Path(fs[0]).name}（{age} 天前）；{why}")
        if not ok:
            issues.append(f"{label} 已 {age} 天无日志更新（>{max_age} 天）")
    for s in ["scripts/month_end_pull.py", "scripts/rebuild_dataset_cache.py",
              "scripts/build_prediction_cache.py", "scripts/v2_monthly_retrain.py"]:
        p = PROJ / s
        say(OK if p.exists() else BAD, f"脚本存在: {s}")


def check_exp_feature_freshness(today: dt.date) -> None:
    print("── 8) 实验类维度新鲜度（V6-D 同花顺财务 / V6-E mootdx 股东机构）──")
    #  ★ 这两批是**季度数据**（财报按季披露）⇒ 不需要日更，但**必须自动更新**：
    #    否则一旦采纳进生产，数据停更 ⇒ **制造新的"半截列"**
    #    （历史是全的、回测看不出问题，但实盘会慢慢失效 —— 最危险的一类）。
    #
    #  ★★ 2026-10-10 修正（用户问"150 天够吗"→ 查证后重做了判据）：
    #    （a）"报告期年龄"这个判据**必须放宽到 220 天**：健康的季报节奏下，
    #        "最新完整期"最老能到 **212 天** —— 三季报(9/30)要一直撑到**次年 4/30** 年报披露完
    #        ⇒ 阈值 150 天会**每年春天误报** ✗。（222 天 = 212 + 余量）
    #    （b）但"报告期年龄"**发现停更太慢**（要 7 个月）⇒ 必须配一个**更灵敏的判据**：
    #        **"数据文件最后被写入"距今天数** —— 月末链每月跑一次、且
    #        `--refresh`(THS) / `--force-recent N`(mootdx) 会**重写最近 N 期**
    #        ⇒ 健康时**每月都应有写入**。>40 天 ⇒ 漏了一期（⚠️）；>70 天 ⇒ 漏两期（❌）。
    #    （c）顺带治了一个**真 Bug**：mootdx 采集器原逻辑"文件已存在就跳过" ⇒
    #        **季报"先残缺后补全"时永远不会补上**（实测 gpcw20260630 卡在 261 家）
    #        ⇒ 已加 `--force-recent`（实测 261 → 5573 家 ✓）。
    cutoff = today - dt.timedelta(days=222)
    ACT_WARN, ACT_BAD = 40, 70          # 采集活性阈值（数据文件多久没被写）
    deep_on = (os.environ.get("FEATURE_V6_DEEP_FINANCE") == "1"
               or os.environ.get("FEATURE_V6_MOOTDX_HOLDER") == "1")
    moot_on = os.environ.get("FEATURE_V6_MOOTDX_HOLDER") == "1"

    def _activity(dirp):
        """该数据目录里**最新文件的写入时间** → (距今天数, 文件名)。判"采集有没有真的在跑"。"""
        try:
            fs = [p for p in dirp.glob("*.parquet")]
            if not fs:
                return None, None
            newest = max(fs, key=lambda p: p.stat().st_mtime)
            return (today - dt.date.fromtimestamp(newest.stat().st_mtime)).days, newest.name
        except Exception:
            return None, None

    def _say_activity(tag: str, dirp, gate_on: bool, missing_msg: str) -> None:
        age, fn = _activity(dirp)
        if age is None:
            say(BAD if gate_on else WARN, f"{tag}：**目录为空**（采集器没跑过）", missing_msg)
            if gate_on:
                issues.append(f"{tag} 数据缺失（从未采集）")
            return
        if age > ACT_BAD:
            say(BAD if gate_on else WARN,
                f"{tag}：数据文件已 **{age} 天**没被写入（>{ACT_BAD} 天 = 漏了约两期月度链）"
                f"　最近写入 {fn}")
            if gate_on:
                issues.append(f"{tag} 采集似已停摆（{age} 天无写入）")
        elif age > ACT_WARN:
            say(WARN, f"{tag}：数据文件已 **{age} 天**没被写入（>{ACT_WARN} 天，可能漏了一期）",
                f"最近写入 {fn}；月末链每月跑一次、且会重写最近几期 ⇒ 健康时每月都应有写入")
        else:
            say(OK, f"{tag}：采集活跃（数据文件 {age} 天前刚写过：{fn}）")

    # ① 同花顺财务（report 形如 "2026-2" ⇒ 2026-06-30）
    Q_END = {"1": (3, 31), "2": (6, 30), "3": (9, 30), "4": (12, 31)}
    try:
        import random as _rnd
        import pandas as _pd
        _say_activity("同花顺财务", PROJ / "data/extra_features/finance_ths_v2/indicators",
                      deep_on, "采集器没跑过或目录为空")
        files = sorted((PROJ / "data/extra_features/finance_ths_v2/indicators").glob("*.parquet"))
        latest = None
        if files:
            for f in _rnd.Random(0).sample(files, min(20, len(files))):
                try:
                    r = _pd.read_parquet(f, columns=["report"])
                except Exception:
                    continue
                if len(r):
                    v = str(r["report"].iloc[-1])
                    if latest is None or v > latest:
                        latest = v
        if not latest:
            say(BAD if deep_on else WARN, "同花顺财务：**读不到 report 字段**",
                "采集器没跑过或目录为空；开关打开时这会阻断。")
            if deep_on:
                issues.append("同花顺财务（V6-D）数据缺失/不可读")
        else:
            y, q = latest.split("-")
            mm, dd = Q_END.get(q, (12, 31))
            pe = dt.date(int(y), mm, dd)
            ok = pe >= cutoff
            say(OK if ok else (BAD if deep_on else WARN),
                f"同花顺财务：最新报告期 {latest}（≈ {pe}）"
                f"{'✓' if ok else f'  ❌ 早于 {cutoff}（超一个季度未更新）'}",
                f"季度数据、允许滞后；开关 " +
                ("**已打开** ⇒ 该数据是生产特征，过期将阻断" if deep_on else "未打开 ⇒ 仅提示"))
            if not ok and deep_on:
                issues.append(f"同花顺财务（V6-D）最新报告期 {latest} 过旧（< {cutoff}）")
    except Exception as e:
        say(WARN, f"同花顺财务新鲜度检查异常: {type(e).__name__}: {e}")

    # ② mootdx（文件名 gpcwYYYYMMDD = 报告期末；最新期常常是"未发布完整"的，要跳过它看最新完整期）
    try:
        import pandas as _pd
        _say_activity("mootdx 财务", PROJ / "data/extra_features/mootdx_finance",
                      moot_on, "采集器没跑过或目录为空")
        fs = sorted((PROJ / "data/extra_features/mootdx_finance").glob("gpcw*.parquet"))
        if not fs:
            say(BAD if moot_on else WARN, "mootdx 财务：**目录为空**（采集器没跑过）")
            if moot_on:
                issues.append("mootdx 财务（V6-E）数据缺失")
        else:
            # 从最新往回找第一个"完整期"（股票数 ≥ 上一期的 30%）
            latest_complete, partial_note = None, ""
            counts = []
            for f in reversed(fs[-4:]):
                try:
                    counts.append((f, _pd.read_parquet(f, columns=["report_date"]).shape[0]))
                except Exception:
                    continue
            for i, (f, n) in enumerate(counts):
                nxt = counts[i + 1][1] if i + 1 < len(counts) else 0
                if nxt == 0 or n >= 0.3 * nxt:
                    latest_complete = (f, n)
                    break
            if counts and counts[0][0] != (latest_complete[0] if latest_complete else None):
                partial_note = f"（最新一期 {counts[0][1]} 只 = 尚未发布完整，已跳过）"
            if latest_complete is None:
                say(WARN, "mootdx：找不到完整期（近 4 期股票数都在骤降？）", partial_note)
            else:
                f, n = latest_complete
                per = f.stem.replace("gpcw", "")
                pe = dt.date(int(per[:4]), int(per[4:6]), int(per[6:8]))
                ok = pe >= cutoff
                say(OK if ok else (BAD if moot_on else WARN),
                    f"mootdx：最新完整期 {per}（{n} 只）"
                    f"{'✓' if ok else f'  ❌ 早于 {cutoff}'} {partial_note}",
                    f"季度数据、允许滞后；开关 " +
                    ("**已打开** ⇒ 该数据是生产特征，过期将阻断" if moot_on else "未打开 ⇒ 仅提示"))
                if not ok and moot_on:
                    issues.append(f"mootdx 财务（V6-E）最新完整期 {per} 过旧（< {cutoff}）")
    except Exception as e:
        say(WARN, f"mootdx 新鲜度检查异常: {type(e).__name__}: {e}")


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
    check_exp_feature_freshness(today)

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
