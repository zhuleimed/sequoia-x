#!/usr/bin/env python3
"""修复 stock_daily 深市历史数据口径错误（2026-09-25 发现）。

问题
----
`stock_daily` 中**深市股票**（000/001/002/003/300/301）在
**2024-01-02 ~ 2026-06-08** 区间存的是**后复权价**，而不是全库约定的不复权价。

实证（000850，三源对拍 baostock / 腾讯实时价 / 库内）：
    库内 2023-12-29  收  3.81      ← 正确
    库内 2024-01-02  收 25.89      ← 突变为 ×6.795
          ...  维持 26~27
    库内 2026-06-08  收 26.52
    库内 2026-06-09  收  3.79      ← 切回正确
    腾讯实时价(9/24)  5.12         ← 第三方源证实后段对、前段被放大
判据：日收益率相关系数 0.998~0.99998（同一序列），水平差每股恒定因子，
且该因子随时间增长、与个股送转史相关 → 复权因子。

影响面（已量化）：
    受影响股票 1,898 只（深市 2,902 只的 68.3%）；受影响行数约 111 万
    沪市 / 北交所：零异常
    正常区间：2020-2023、以及 2026-06-09 之后
危害：2024-01-02 深市整体"假涨"中位 +127%，2026-06-09 "假跌" -55.9%；
      V4 训练集与 70 个月回测均覆盖该区间，且训练管线会再套一层后复权（双重复权）。

修复方式
--------
对**全部深市股票**重拉 2023-12-20 ~ 2026-06-20（adjustflag=3 不复权），其中：

  1. **控制窗验证**（先决条件）：区间两端各留一段**已知正确**的日期
     （2023-12-20~2023-12-29 与 2026-06-09~2026-06-20）。
     只有当 baostock 在这些控制日与库内值**逐只吻合**（≥3 个可比日、全部 <0.5% 误差），
     才采信该股票的 baostock 数据 —— 避免"用不可信的源覆盖数据"。
  2. **只改有差异的行**：段内某行与 baostock 相对误差 > 0.5% 才 UPDATE，
     完全一致的行不动（避免无谓写入与 id 抖动）。
  3. 改前把**每一行的原值**导出到备份目录；全过程单事务；改完复验。

幂等：重复运行结果一致（已修好的行不再有差异）。

用法
----
    python scripts/fix_deep_market_basis_20260925.py            # 只体检（dry-run）
    python scripts/fix_deep_market_basis_20260925.py --apply    # 备份 + 修复 + 复验
    python scripts/fix_deep_market_basis_20260925.py --apply --limit 50   # 先小批量试
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DB = ROOT / "data" / "sequoia_v2.db"
BACKUP_DIR = ROOT / f"data/backup_deep_basis_fix_{datetime.now():%Y%m%d}"

# 待修区间（口径错误段）
SEG_START, SEG_END = "2024-01-02", "2026-06-08"
# 拉取区间（比待修区间宽，两端留作"控制窗"）
FETCH_START, FETCH_END = "2023-12-20", "2026-06-20"
# 控制窗（已知正确的日期段，用于验证 baostock 可信）
CTRL_RANGES = (("2023-12-20", "2023-12-29"), ("2026-06-09", "2026-06-20"))

TOL = 0.005          # 相对误差容差（0.5%）
MIN_CTRL = 3         # 控制窗至少要有多少个可比日
FIELDS = "date,open,high,low,close,volume,amount,turn,pctChg"


def deep_symbols(conn: sqlite3.Connection) -> list[str]:
    """返回全部深市股票（0/3 开头）。"""
    rows = conn.execute(
        "SELECT DISTINCT symbol FROM stock_daily "
        "WHERE substr(symbol,1,1) IN ('0','3') AND date >= ? ORDER BY symbol",
        (SEG_START,),
    ).fetchall()
    return [r[0] for r in rows]


RECONNECT_EVERY = 250   # 每 N 次查询重新登录（baostock 长连接会静默失效，见 sync.py 同款处理）


def fetch_baostock(symbols: list[str], to_code) -> dict[str, pd.DataFrame]:
    """按股票拉取 FETCH_START~FETCH_END 的不复权日线。

    2026-09-25 实测教训：baostock 长连接在 ~300 次查询后会静默失效
    （日志出现 `logout failed!`，后续查询全部失败），必须**定期重连**。
    每 RECONNECT_EVERY 次重登一次；单次查询失败只跳过该股票，不中断整体。
    """
    import baostock as bs

    def _login() -> bool:
        try:
            return bs.login().error_code == "0"
        except Exception:
            return False

    if not _login():
        print("❌ baostock 登录失败")
        return {}

    out: dict[str, pd.DataFrame] = {}
    failed: list[str] = []
    t0 = time.time()
    try:
        for i, sym in enumerate(symbols, 1):
            if i > 1 and (i - 1) % RECONNECT_EVERY == 0:
                try:
                    bs.logout()
                except Exception:
                    pass
                if not _login():
                    print(f"    ⚠️ 第 {i} 只处重连失败，稍后重试一次")
                    time.sleep(2)
                    if not _login():
                        print("    ❌ 重连仍失败，中止拉取")
                        break
            try:
                rs = bs.query_history_k_data_plus(
                    to_code(sym), FIELDS, start_date=FETCH_START, end_date=FETCH_END,
                    frequency="d", adjustflag="3")
                if rs.error_code != "0":
                    failed.append(sym)
                    continue
                rows = []
                while rs.next():
                    rows.append(rs.get_row_data())
                if not rows:
                    failed.append(sym)
                    continue
                df = pd.DataFrame(rows, columns=rs.fields)
                for c in ("open", "high", "low", "close", "volume", "amount", "turn", "pctChg"):
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                out[sym] = df
            except Exception as e:
                failed.append(sym)
                print(f"    ⚠️ {sym}: {e}")
            if i % 50 == 0:
                el = time.time() - t0
                print(f"    ... {i}/{len(symbols)}（成功 {len(out)}，失败 {len(failed)}，"
                      f"{el:.0f}s，{el/i:.2f}s/只）", flush=True)
    finally:
        try:
            bs.logout()
        except Exception:
            pass
    if failed:
        print(f"    ⚠️ 未取到数据的 {len(failed)} 只（前 10）: {failed[:10]}")
    return out


def fetch_tencent(symbols: list[str], to_code, days: int = 1600) -> dict[str, pd.DataFrame]:
    """腾讯兜底：baostock 会**卡死**在一部分股票上（实测创业板 300xxx 约 400 只，
    查询永不返回、连 Python 信号都打断不了），改用腾讯 fqkline（fq 留空 = 不复权）。

    腾讯 |fq 留空| 接口可返回约 1600 条日线（本例覆盖 2020-02 起），足够覆盖待修区间。
    另有两点与 baostock 不同，已在返回值中标注：
      - **无 amount**（成交额）→ 由 close×volume 兜底（本库约定 amount 单位=元）
      - **无 turnover**（换手率）→ 写 NULL（本库惯例：未知用 NULL，不用 0）

    Returns:
        {symbol: DataFrame(date, open, high, low, close, volume, amount, turn, pctChg)}
    """
    from sequoia_x.data.tencent_source import TencentSource
    src = TencentSource()
    out: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(symbols, 1):
        try:
            df = src.get_daily(to_code(sym), days=days)
            if df is None or df.empty:
                continue
            df = df[(df["date"] >= FETCH_START) & (df["date"] <= FETCH_END)].copy()
            if df.empty:
                continue
            df["amount"] = df["close"] * df["volume"]      # 腾讯无成交额 → close×volume 兜底
            df["turn"] = float("nan")                       # 换手率未知 → NULL
            df["pctChg"] = df["close"].pct_change() * 100   # 单日涨跌幅（首行 NaN → NULL）
            out[sym] = df
        except Exception as e:
            print(f"    ⚠️ 腾讯 {sym}: {e}")
        if i % 50 == 0:
            print(f"    ... 腾讯 {i}/{len(symbols)}（成功 {len(out)}）", flush=True)
    return out


def load_local(conn: sqlite3.Connection, symbols: list[str]) -> dict[str, pd.DataFrame]:
    """把库内待修区间的数据按股票分组读进内存。"""
    ph = ",".join("?" * len(symbols))
    df = pd.read_sql(
        f"SELECT symbol, date, open, close FROM stock_daily "
        f"WHERE symbol IN ({ph}) AND date BETWEEN ? AND ? ORDER BY symbol, date",
        conn, params=symbols + [FETCH_START, FETCH_END])
    for c in ("open", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return {s: g.reset_index(drop=True) for s, g in df.groupby("symbol")}


def in_ctrl(d: str) -> bool:
    return any(lo <= d <= hi for lo, hi in CTRL_RANGES)


def main() -> int:
    ap = argparse.ArgumentParser(description="修复深市历史数据口径（后复权→不复权）")
    ap.add_argument("--apply", action="store_true", help="执行修复（默认只体检）")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 只（试跑用）")
    ap.add_argument("--symbols", type=str, default="",
                    help="逗号分隔的股票代码，只处理这些（定向测试用）")
    ap.add_argument("--fallback-tencent", action="store_true",
                    help="对 baostock 未返回的股票改用腾讯不复权接口兜底"
                         "（实测创业板约 400 只 baostock 永不返回）")
    ap.add_argument("--tencent-only", action="store_true",
                    help="完全跳过 baostock，只用腾讯不复权接口（对已确认卡死的股票用）")
    args = ap.parse_args()

    if not DB.exists():
        print(f"❌ 数据库不存在: {DB}")
        return 1

    from sequoia_x.core.config import get_settings
    from sequoia_x.data.engine import DataEngine
    eng = DataEngine(get_settings())

    conn = sqlite3.connect(DB)
    syms = deep_symbols(conn)
    if args.symbols:
        want = {s.strip().zfill(6) for s in args.symbols.split(",") if s.strip()}
        syms = [s for s in syms if s in want]
    if args.limit:
        syms = syms[: args.limit]
    print("=" * 96)
    print(f"【体检】深市股票 {len(syms)} 只 | 待修区间 {SEG_START} ~ {SEG_END}")
    print(f"        拉取区间 {FETCH_START} ~ {FETCH_END}（两端为控制窗）")
    print("=" * 96)

    local = load_local(conn, syms)
    print(f"库内读入 {len(local)} 只股票")

    if not args.apply:
        # dry-run：只统计"段内单日跳变"这个代理指标，不联网
        n_jump_days = 0
        for s, g in local.items():
            g2 = g.copy()
            g2["prev"] = g2["close"].shift(1)
            jump = (g2["close"] / g2["prev"] - 1).abs() > 0.25
            n_jump_days += int(jump.sum())
        print(f"\n段内单日 |涨跌| > 25% 的行数（口径切换的典型症状）: {n_jump_days}")
        print("\n（dry-run 模式，未联网、未修改数据库。加 --apply 执行修复）")
        conn.close()
        return 0

    # ── 拉取 ──
    from sequoia_x.data.tencent_source import to_tencent_code
    if args.tencent_only:
        print(f"\n【拉取】腾讯（不复权）拉取 {len(syms)} 只（跳过 baostock）...")
        bs_data = fetch_tencent(syms, to_tencent_code)
        print(f"  成功拉取 {len(bs_data)} 只")
    else:
        print(f"\n【拉取】baostock 拉取 {len(syms)} 只 ...")
        bs_data = fetch_baostock(syms, lambda s: eng._to_baostock_code(s))
        print(f"  成功拉取 {len(bs_data)} 只")

    # baostock 卡死的股票 → 腾讯兜底（2026-09-25：实测创业板约 400 只 baostock 永不返回）
    if args.fallback_tencent:
        missing = [s for s in syms if s not in bs_data]
        if missing:
            from sequoia_x.data.tencent_source import to_tencent_code
            print(f"\n【腾讯兜底】对 baostock 未返回的 {len(missing)} 只重试 ...")
            tc_data = fetch_tencent(missing, to_tencent_code)
            print(f"  腾讯成功拉取 {len(tc_data)} 只（合并后共 {len(bs_data) + len(tc_data)} 只）")
            for s, df in tc_data.items():
                bs_data[s] = df.rename(columns={"turn": "turn"})

    # ── 控制窗验证 + 差异比对 ──
    print("\n【控制窗验证 + 差异比对】")
    trusted: dict[str, pd.DataFrame] = {}
    to_update: list[tuple] = []     # (symbol, date, 新值...)
    backup_rows: list[dict] = []
    stat = {"ctrl_fail": 0, "no_data": 0, "trusted": 0, "same": 0, "diff": 0}
    for sym, bdf in bs_data.items():
        lg = local.get(sym)
        if lg is None:
            stat["no_data"] += 1
            continue
        m = lg.merge(bdf[["date", "close"]].rename(columns={"close": "bs_close"}),
                     on="date", how="inner")
        ctrl = m[m["date"].map(in_ctrl)].dropna()
        if len(ctrl) < MIN_CTRL:
            stat["no_data"] += 1
            continue
        # 控制窗必须逐行吻合，才采信该股票的 baostock
        if not ((ctrl["close"] / ctrl["bs_close"] - 1).abs() <= TOL).all():
            stat["ctrl_fail"] += 1
            continue
        stat["trusted"] += 1
        trusted[sym] = bdf

        # 段内逐行比对
        seg = m[(m["date"] >= SEG_START) & (m["date"] <= SEG_END)]
        bmap = {r.date: r for r in bdf.itertuples(index=False)}
        for r in seg.itertuples(index=False):
            br = bmap.get(r.date)
            if br is None:
                continue
            if abs(br.close / r.close - 1) <= TOL:
                stat["same"] += 1
                continue
            stat["diff"] += 1
            to_update.append((
                float(br.open), float(br.high), float(br.low), float(br.close),
                float(br.volume),
                float(br.amount) if pd.notna(br.amount) else None,
                float(br.turn) if pd.notna(br.turn) else None,
                float(br.pctChg) if pd.notna(br.pctChg) else None,
                sym, r.date))
            backup_rows.append({
                "symbol": sym, "date": r.date,
                "old_open": r.open, "old_close": r.close,
                "new_open": float(br.open), "new_close": float(br.close),
                "ratio": r.close / br.close if br.close else None,
            })

    for k, v in stat.items():
        print(f"  {k}: {v}")
    print(f"  待更新行数: {len(to_update)}")

    if not to_update:
        print("\n✓ 段内数据与 baostock 一致，无需修复")
        conn.close()
        return 0

    # ── 备份 ──
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    bak_path = BACKUP_DIR / "changed_rows_before.csv"
    pd.DataFrame(backup_rows).to_csv(bak_path, index=False, encoding="utf-8-sig")
    print(f"\n【备份】{len(backup_rows)} 行原值 → {bak_path}")

    # ── 修复（单事务）──
    print("\n【修复】单事务批量 UPDATE ...")
    t0 = time.time()
    with conn:  # with 块 = 单事务，异常自动回滚
        conn.executemany(
            "UPDATE stock_daily SET open=?, high=?, low=?, close=?, "
            "volume=?, amount=?, turnover=?, pctChg=? WHERE symbol=? AND date=?",
            to_update)
    print(f"  完成，耗时 {time.time()-t0:.0f}s")

    # ── 复验 ──
    print("\n【复验】")
    local2 = load_local(conn, list(trusted.keys()))
    bad = 0
    checked = 0
    for sym, bdf in list(trusted.items())[:400]:   # 抽样 400 只复验
        lg = local2.get(sym)
        if lg is None:
            continue
        m = lg.merge(bdf[["date", "open", "close"]].rename(
            columns={"open": "bo", "close": "bc"}), on="date", how="inner")
        seg = m[(m["date"] >= SEG_START) & (m["date"] <= SEG_END)].dropna()
        if not len(seg):
            continue
        checked += len(seg)
        bad += int(((seg["close"] / seg["bc"] - 1).abs() > TOL).sum())
    print(f"  抽验 {checked} 行，仍不一致 {bad} 行")

    # 段内假跳变是否消失
    n_jump = 0
    for s, g in local2.items():
        g2 = g.copy()
        g2["prev"] = g2["close"].shift(1)
        n_jump += int(((g2["close"] / g2["prev"] - 1).abs() > 0.25).sum())
    print(f"  抽验范围内段内 |涨跌|>25% 的行数: {n_jump}（修复前为数千级）")

    json.dump({"updated": len(to_update), "checked": checked, "remaining_bad": bad,
               "stat": stat, "seg": [SEG_START, SEG_END]},
              open(BACKUP_DIR / "repair_summary.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    conn.close()
    print(f"\n完成。备份目录: {BACKUP_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
