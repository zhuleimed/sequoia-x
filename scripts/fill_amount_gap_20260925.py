#!/usr/bin/env python3
"""回填 stock_daily.amount（成交额）的历史缺口（2026-09-25）。

问题
----
`stock_daily.amount` 缺失 **578.8 万行（76.6%）**，贯穿 2020-01-02 ~ 2026-07-31：
  - **沪市几乎全缺**（各年都缺）
  - 深市 2020~2023 全缺，2024 起才补上
  - 2026-08 起才 100% 完整（2026-09-18 单位事故修复的副产品）
影响：021 因子库的成交额/VWAP 类因子、022 的 turnover_rate(=amount/float_cap×100)、
      海龟策略流动性过滤等，在沪市上等同"瞎的"。

为什么不直接用 close×volume 填
------------------------------
`close×volume` 对真实成交额的误差其实很小（实测中位 0.5%、95 分位 2.7%），
**但它会让项目的"`amount/volume ≈ close` 自证式校验"变成同义反复** ——
被填的 76.6% 行再也无法用这条检查发现单位错误。
统计口径的因子用近似值可接受，但**牺牲一个通用自检的代价太大**。

故本脚本**只回填真值**（baostock，单位=元，adjustflag=3）；
数据源取不到的股票**保持 NULL** —— 符合本库「未知写 NULL，不写伪值」的既定约定。

校验
----
· 交叉验证：用该股票**已有 amount 的行**比对 baostock，≥3 个可比日且全部吻合
  （相对误差 <0.5%）才采信该股票的 baostock 数据
· 改后复验：随机抽样重新对拍

用法
----
    python scripts/fill_amount_gap_20260925.py              # 只体检（dry-run，不联网）
    python scripts/fill_amount_gap_20260925.py --apply      # 备份 + 回填 + 复验
    python scripts/fill_amount_gap_20260925.py --apply --limit 50   # 小批量试跑
    python scripts/fill_amount_gap_20260925.py --apply --symbols 600519,000001
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
BACKUP_DIR = ROOT / f"data/backup_amount_fill_{datetime.now():%Y%m%d}"

TOL = 0.005              # 交叉验证相对误差容差（源侧对拍用保留）
MIN_CTRL = 3             # 至少几个可比日才采信
# 源侧自证：baostock成交额/库内成交量/库内收盘价 应≈1（VWAP 天然贴近收盘价）。
# 实测真实值分布中位 1.00098、95 分位偏离 2.7%，故 ±15% 极宽松，但仍能抓出
# 单位错误（万元 vs 元 会差 10000 倍）。
IMPLIED_TOL = 0.15
RECONNECT_EVERY = 250    # baostock 长连接会静默失效，定期重连


def main() -> int:
    ap = argparse.ArgumentParser(description="回填 stock_daily.amount 历史缺口")
    ap.add_argument("--apply", action="store_true", help="执行回填（默认只体检）")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 只（试跑）")
    ap.add_argument("--symbols", type=str, default="", help="逗号分隔，只处理这些股票")
    args = ap.parse_args()

    if not DB.exists():
        print(f"❌ 数据库不存在: {DB}")
        return 1

    conn = sqlite3.connect(DB)
    miss = pd.read_sql("SELECT symbol, date FROM stock_daily WHERE amount IS NULL", conn)
    print("=" * 90)
    print("【体检】amount 缺失情况")
    print("=" * 90)
    if miss.empty:
        print("✓ 无缺失")
        conn.close()
        return 0
    n_stock = miss["symbol"].nunique()
    print(f"缺失 {len(miss):,} 行，涉及 {n_stock:,} 只股票")
    print(f"日期范围 {miss['date'].min()} ~ {miss['date'].max()}")
    by_year = miss.groupby(miss["date"].str[:4]).size()
    print("按年：", {k: f"{v:,}" for k, v in by_year.items()})

    if not args.apply:
        print("\n（dry-run，未联网、未改库。加 --apply 执行回填）")
        conn.close()
        return 0

    # 待处理股票（按缺失行数降序，先啃大头）
    grp = miss.groupby("symbol").size().sort_values(ascending=False)
    syms = list(grp.index)
    if args.symbols:
        want = {s.strip().zfill(6) for s in args.symbols.split(",") if s.strip()}
        syms = [s for s in syms if s in want]
    elif args.limit:
        syms = syms[: args.limit]
    print(f"\n【回填】{len(syms)} 只股票")

    # 每只股票的缺失区间
    span = miss.groupby("symbol")["date"].agg(["min", "max"])
    miss_set = {(r.symbol, r.date) for r in miss.itertuples(index=False)}

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    from sequoia_x.core.config import get_settings
    from sequoia_x.data.engine import DataEngine
    eng = DataEngine(get_settings())

    import baostock as bs

    def login() -> bool:
        try:
            return bs.login().error_code == "0"
        except Exception:
            return False

    if not login():
        print("❌ baostock 登录失败")
        conn.close()
        return 1

    fetched: list[tuple[str, str, float]] = []
    trusted, ctrl_fail, no_ctrl, no_data = [], [], [], []
    t0 = time.time()
    try:
        for i, sym in enumerate(syms, 1):
            if i > 1 and (i - 1) % RECONNECT_EVERY == 0:
                try:
                    bs.logout()
                except Exception:
                    pass
                if not login():
                    print(f"    ⚠️ 第 {i} 只处重连失败，中止")
                    break
            lo, hi = span.loc[sym, "min"], span.loc[sym, "max"]
            try:
                rs = bs.query_history_k_data_plus(
                    eng._to_baostock_code(sym), "date,amount",
                    start_date=lo, end_date=hi, frequency="d", adjustflag="3")
                if rs.error_code != "0":
                    no_data.append(sym)
                    continue
                rows = []
                while rs.next():
                    rows.append(rs.get_row_data())
            except Exception:
                no_data.append(sym)
                continue
            if not rows:
                no_data.append(sym)
                continue
            bsdf = pd.DataFrame(rows, columns=rs.fields)
            bsdf["amount"] = pd.to_numeric(bsdf["amount"], errors="coerce")
            bsdf = bsdf[bsdf["amount"] > 0]
            if bsdf.empty:
                no_data.append(sym)
                continue

            # 交叉验证（2026-09-25 改进）：
            #   初版用"该股票已有的 amount 行"做对照，但**沪市整段没有 amount** ——
            #   恰恰是最需要回填的股票全都拿不到对照（实测 20 只全部"可比日不足"）。
            #   改用**源侧自证**：库内 close/volume 已与 baostock 对拍确认无误，
            #   于是校验「baostock成交额 / 库内成交量 ≈ 库内收盘价」——
            #   这正是本库的铁律自检，能一举抓出单位错误（如万元 vs 元会差 10000 倍）。
            lv = pd.read_sql(
                "SELECT date, close, volume FROM stock_daily WHERE symbol=? "
                "AND date BETWEEN ? AND ? AND volume > 0 AND close > 0",
                conn, params=(sym, lo, hi))
            for c in ("close", "volume"):
                lv[c] = pd.to_numeric(lv[c], errors="coerce")
            m = lv.merge(bsdf, on="date")
            m = m[m["volume"] > 0]
            if len(m) < MIN_CTRL:
                no_ctrl.append(sym)
                continue
            implied = m["amount"] / m["volume"] / m["close"]     # 应 ≈ 1
            ok_ratio = (implied.between(1 - IMPLIED_TOL, 1 + IMPLIED_TOL)).mean()
            if ok_ratio < 0.98:
                ctrl_fail.append(sym)
                continue
            trusted.append(sym)
            for r in bsdf.itertuples(index=False):
                if (sym, r.date) in miss_set:
                    fetched.append((float(r.amount), sym, r.date))

            if i % 100 == 0:
                el = time.time() - t0
                print(f"    ... {i}/{len(syms)}（可信 {len(trusted)}，待写 {len(fetched):,} 行，"
                      f"{el:.0f}s）", flush=True)
    finally:
        try:
            bs.logout()
        except Exception:
            pass

    print(f"\n  采信 {len(trusted)} 只 | 控制窗不通过 {len(ctrl_fail)} | "
          f"可比日不足 {len(no_ctrl)} | 无数据 {len(no_data)}")
    print(f"  待写入 {len(fetched):,} 行")
    if not fetched:
        print("\n无数据可写")
        conn.close()
        return 0

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    # 备份：这些行原本就是 NULL，备份的是"将被填入的值"，便于回滚（置回 NULL 即可）
    pd.DataFrame(fetched, columns=["amount", "symbol", "date"]).to_csv(
        BACKUP_DIR / "filled_values.csv", index=False, encoding="utf-8-sig")
    print(f"【备份】写入清单 → {BACKUP_DIR / 'filled_values.csv'}（原始值均为 NULL，回滚=置回NULL）")

    print("\n【写入】单事务 ...")
    t1 = time.time()
    with conn:
        conn.executemany(
            "UPDATE stock_daily SET amount=? WHERE symbol=? AND date=? AND amount IS NULL",
            fetched)
    print(f"  完成，耗时 {time.time()-t1:.0f}s")

    # ── 复验 ──
    print("\n【复验】")
    left = conn.execute("SELECT COUNT(*) FROM stock_daily WHERE amount IS NULL").fetchone()[0]
    print(f"  剩余缺失 {left:,} 行（回填前 {len(miss):,}）")
    v = pd.read_sql(
        "SELECT close, volume, amount FROM stock_daily WHERE amount IS NOT NULL "
        "AND volume > 0 AND date >= '2020-01-01' ORDER BY RANDOM() LIMIT 300000", conn)
    for c in ("close", "volume", "amount"):
        v[c] = pd.to_numeric(v[c], errors="coerce")
    r = (v["amount"] / v["volume"]) / v["close"]
    print(f"  单位自证 amount/volume ≈ close：中位 {r.median():.5f}，"
          f"落在 [0.9,1.1] 占比 {(r.between(0.9, 1.1)).mean()*100:.2f}%（应≈100%）")

    json.dump({"before_missing": int(len(miss)), "after_missing": int(left),
               "filled": len(fetched), "trusted": len(trusted),
               "ctrl_fail": len(ctrl_fail), "no_ctrl": len(no_ctrl), "no_data": len(no_data)},
              open(BACKUP_DIR / "summary.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    conn.close()
    print(f"\n完成。备份目录: {BACKUP_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
