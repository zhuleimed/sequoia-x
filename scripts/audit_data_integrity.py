#!/usr/bin/env python3
"""入库数据体检：价格跳变自检 + 多源抽样对拍（2026-09-25 新增）。

为什么需要它
------------
2026-09-25 归因诊断时发现 `stock_daily` 有两类既有检查**完全抓不到**的事故：

  ① **口径错误**：深市 2024-01-02 ~ 2026-06-08 约 111 万行存的是**后复权价**而非
     不复权价。段内收益率本来就对（只差一个每股恒定因子），行内也自洽
     （amount/volume ≈ close），事发当日 sync_log 还是 `status=ok`、`coverage=1.0`。
  ② **单日整行错乱**：2026-07-06 / 07-07 的 off-by-one 写错股票。

两类事故的共同点是：**coverage / 完整性类检查对它们完全无效**。
唯一有效的办法是拿**独立数据源对拍**——这也是当时唯一逼出真相的手段
（修复脚本里的交叉验证先挡下了错误修复，才顺藤摸瓜查到 ①）。

本脚本做两件事
--------------
  A. **本地跳变自检**（秒级，无需联网）：复用 `DataSync.check_price_jumps`，
     抓单日 |涨跌| > 25% 的行（A股涨跌停 ±10%/±20%，>25% 必为异常）。
  B. **多源抽样对拍**（分钟级，联网）：随机抽 N 只股票 × K 个交易日，
     与 **baostock 不复权（adjustflag=3）** 逐日比对收盘价。
     比值偏离 1 即告警 —— 这是唯一能发现"整段口径错误"的检查。

判读阈值
--------
  跳变：单日异常行数 ≥ 30 告警（合法极端行情每日 ≤14 行，真实事故日为 631~2,053 行）。
  对拍：异常比例 ≥ 5% 告警（正常应接近 0%；实测口径错误期约 55%）。

用法
----
    python scripts/audit_data_integrity.py                 # 完整体检（默认近 15 日 + 抽样）
    python scripts/audit_data_integrity.py --quick         # 只做本地跳变检查（秒级）
    python scripts/audit_data_integrity.py --no-push       # 不推微信
    python scripts/audit_data_integrity.py --json out.json # 结果落盘

退出码：0 = 通过；1 = 有告警（供调用方决定是否中止后续步骤）
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

JUMP_ALERT_ROWS = 30      # 单日跳变行数告警阈值
CROSS_ALERT_RATIO = 0.05  # 对拍【总体】异常比例告警阈值（辅助判据）
DATE_ALERT_RATIO = 0.30   # 对拍【单日】异常比例告警阈值（主判据）
DATE_MIN_SAMPLE = 5       # 单日判据要求的最小样本数
TOL = 0.005               # 对拍相对误差容差（0.5%）


def check_jumps(days: int) -> dict:
    """A. 本地价格跳变自检（复用生产实现，保证与日常告警同口径）。"""
    from sequoia_x.core.config import get_settings
    from sequoia_x.data.sync import DataSync
    return DataSync(get_settings()).check_price_jumps(days=days)


def sample_dates(con: sqlite3.Connection, n_recent: int, n_old: int) -> list[str]:
    """取最近 n_recent 个交易日，外加 n_old 个较早的参照日（用于发现"追溯性口径变更"）。"""
    days = [r[0] for r in con.execute(
        "SELECT DISTINCT date FROM stock_daily ORDER BY date DESC LIMIT 2000")]
    days = days[::-1]                       # 升序
    if len(days) <= n_recent:
        return days
    recent = days[-n_recent:]
    # 较早参照日：在更早的历史里等距取 n_old 个（口径变更会同时污染历史，故必须往回看）
    earlier = days[:-n_recent]
    old: list[str] = []
    if earlier and n_old > 0:
        step = max(1, len(earlier) // n_old)
        old = earlier[::step][-n_old:]
    return sorted(set(old + recent))


def check_cross_source(con: sqlite3.Connection, n_stocks: int, n_recent: int,
                       n_old: int, seed: int = 20260925) -> dict:
    """B. 多源抽样对拍：库内收盘价 vs baostock 不复权（adjustflag=3）。"""
    syms = [r[0] for r in con.execute(
        "SELECT DISTINCT symbol FROM stock_daily WHERE date >= '2024-06-01'")]
    deep = sorted(s for s in syms if s[0] in "03")
    sh = sorted(s for s in syms if s[0] == "6")
    rng = random.Random(seed)
    n_each = max(1, n_stocks // 2)
    sample = (rng.sample(deep, min(n_each, len(deep)))
              + rng.sample(sh, min(n_each, len(sh))))
    dates = sample_dates(con, n_recent, n_old)
    if not dates:
        return {"ok": False, "reason": "无可用交易日"}

    import baostock as bs
    if bs.login().error_code != "0":
        return {"ok": False, "reason": "baostock 登录失败"}

    rows: list[pd.DataFrame] = []
    try:
        for i, sym in enumerate(sample, 1):
            # 与 sync.py 同款：baostock 长连接约 300 次查询后静默失效，需定期重连
            if i > 1 and (i - 1) % 250 == 0:
                try:
                    bs.logout()
                except Exception:
                    pass
                if bs.login().error_code != "0":
                    break
            code = f"sz.{sym}" if sym[0] in "03" else f"sh.{sym}"
            try:
                rs = bs.query_history_k_data_plus(
                    code, "date,close", start_date=dates[0], end_date=dates[-1],
                    frequency="d", adjustflag="3")
                if rs.error_code != "0":
                    continue
                data = []
                while rs.next():
                    data.append(rs.get_row_data())
                if data:
                    rows.append(pd.DataFrame(data, columns=rs.fields).assign(symbol=sym))
            except Exception:
                continue
    finally:
        try:
            bs.logout()
        except Exception:
            pass

    if not rows:
        return {"ok": False, "reason": "baostock 未返回任何数据"}

    bsdf = pd.concat(rows, ignore_index=True)
    bsdf["close"] = pd.to_numeric(bsdf["close"], errors="coerce")
    local = pd.read_sql(
        "SELECT symbol, date, close FROM stock_daily WHERE date BETWEEN ? AND ?",
        con, params=(dates[0], dates[-1]))
    local["close"] = pd.to_numeric(local["close"], errors="coerce")
    local = local[local["date"].isin(dates)]

    m = bsdf.merge(local, on=["symbol", "date"], suffixes=("_bs", "_db"))
    m = m[(m["close_bs"] > 0) & (m["close_db"] > 0)]
    if m.empty:
        return {"ok": False, "reason": "无可比对样本"}
    m["ratio"] = m["close_db"] / m["close_bs"]
    m["bad"] = (m["ratio"] - 1).abs() > TOL
    m["exch"] = m["symbol"].str[0].map(lambda c: "深市" if c in "03" else "沪市")

    by_date = (m.groupby("date")["bad"].agg(["sum", "size"])
               .assign(ratio=lambda d: d["sum"] / d["size"]))
    return {
        "ok": True, "n_stocks": len(sample), "n_pairs": int(len(m)),
        "bad_pairs": int(m["bad"].sum()),
        "bad_ratio": float(m["bad"].mean()),
        "worst_dates": (by_date.sort_values("ratio", ascending=False)
                        .head(5).reset_index().to_dict("records")),
        "by_exch": (m.groupby("exch")["bad"].agg(["sum", "size"]).to_dict("index")),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="入库数据体检（跳变 + 多源对拍）")
    ap.add_argument("--days", type=int, default=15, help="跳变自检与对拍的近期天数")
    ap.add_argument("--stocks", type=int, default=60, help="对拍抽样股票数（沪深各半）")
    ap.add_argument("--old-dates", type=int, default=3, help="回看的较早参照日数量")
    ap.add_argument("--quick", action="store_true", help="只做本地跳变检查")
    ap.add_argument("--no-push", action="store_true", help="不推送微信")
    ap.add_argument("--json", type=str, default="", help="结果另存 JSON")
    args = ap.parse_args()

    t0 = time.time()
    result: dict = {"ts": datetime.now().isoformat(timespec="seconds")}
    problems: list[str] = []

    # ── A. 本地跳变自检 ──
    print("=" * 84)
    print("A. 价格跳变自检（本地，单日 |涨跌| > 25%）")
    print("=" * 84)
    jump = check_jumps(max(args.days, 5))
    result["jump_check"] = jump
    print(f"  状态: {jump.get('status')}  近 {len(jump.get('checked_days', []))} 日"
          f"  异常 {jump.get('total_anomalies', 0)} 行"
          f"  最严重 {jump.get('worst_date', '-')}: {jump.get('worst_count', 0)} 行")
    if jump.get("status") == "warning":
        problems.append(
            f"价格跳变：{jump.get('worst_date')} 有 {jump.get('worst_count')} 行超阈值"
            f"（阈值 {JUMP_ALERT_ROWS}）")

    # ── B. 多源对拍 ──
    cross = {"skipped": True}
    if not args.quick:
        print("\n" + "=" * 84)
        print("B. 多源抽样对拍（库内 vs baostock 不复权 adjustflag=3）")
        print("=" * 84)
        con = sqlite3.connect(ROOT / "data" / "sequoia_v2.db")
        cross = check_cross_source(con, args.stocks, args.days, args.old_dates)
        con.close()
        result["cross_check"] = cross
        if not cross.get("ok"):
            print(f"  ⚠️ 对拍未能完成: {cross.get('reason')}")
            problems.append(f"多源对拍未完成：{cross.get('reason')}")
        else:
            print(f"  抽样 {cross['n_stocks']} 只 × {cross['n_pairs']} 个可比对样本")
            print(f"  比值偏离 1（>{TOL:.1%}）: {cross['bad_pairs']} 个"
                  f"（{cross['bad_ratio']:.2%}）")
            for e, v in cross.get("by_exch", {}).items():
                print(f"    {e}: {v['sum']}/{v['size']} 异常")
            if cross["worst_dates"]:
                print("  异常比例最高的日期：")
                for r in cross["worst_dates"]:
                    print(f"    {r['date']}: {int(r['sum'])}/{int(r['size'])}")
            # 判据一（主）：**单日系统性错**——某一天大多数抽样股都对不上。
            #   历史事故正是这种形态（整段口径错误 → 该段内每天都大面积错），
            #   而按总体比例判会被大量正常日期稀释掉（实测单日全错时总体才 3%）。
            worst = next((r for r in cross["worst_dates"]
                          if r["size"] >= DATE_MIN_SAMPLE
                          and r["sum"] / r["size"] >= DATE_ALERT_RATIO), None)
            if worst:
                problems.append(
                    f"多源对拍：{worst['date']} 有 {int(worst['sum'])}/{int(worst['size'])} "
                    f"抽样股与独立源对不上（≥{DATE_ALERT_RATIO:.0%}）→ 疑似该日数据整体错误")
            # 判据二（辅）：总体异常比例过高
            elif cross["bad_ratio"] >= CROSS_ALERT_RATIO:
                problems.append(
                    f"多源对拍总体异常比例 {cross['bad_ratio']:.1%} ≥ {CROSS_ALERT_RATIO:.0%}"
                    f"（各日均不突出，疑似分散性错误）")

    # ── 汇总 ──
    elapsed = time.time() - t0
    ok = not problems
    print("\n" + "=" * 84)
    print(f"体检结论: {'✅ 通过' if ok else '⚠️ 有告警'}   耗时 {elapsed:.0f}s")
    for p in problems:
        print(f"  ⚠️ {p}")
    print("=" * 84)
    result["ok"] = ok
    result["problems"] = problems

    if args.json:
        Path(args.json).write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"结果已保存: {args.json}")

    if problems and not args.no_push:
        try:
            from wxpusher import WxPusher
            from sequoia_x.core.config import get_settings
            s = get_settings()
            body = "入库数据体检未通过：\n\n" + "\n".join(f"• {p}" for p in problems)
            body += ("\n\n含多源对拍异常 → 疑似口径错误（历史事故：深市 2024-01-02~2026-06-08 "
                     "曾整段写成后复权）。\n处置：python scripts/audit_data_integrity.py --json /tmp/audit.json")
            WxPusher.send_message(content=f"⚠️ Sequoia-X 数据体检告警\n\n{body}",
                                  token=s.wxpusher_token,
                                  topic_ids=s.wxpusher_topic_ids, content_type=1)
            print("[notify] 已推送告警")
        except Exception as e:
            print(f"[notify] 推送失败（不影响结论）: {e}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
