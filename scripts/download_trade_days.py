#!/usr/bin/env python3
"""交易日历本地快照（2026-08-21）。

背景: 同花顺交易日历接口只有 1 年窗口(243 天, 参数 start/end 均无效), 撑不起
2020-09 起的历史回测/缓存重建。baostock `query_trade_dates` 能一次拉全历史。

方案:
  - baostock 拉 2020-01-01~今天+1年 全部交易日 → 存本地 data/trade_days.json
  - is_trade_day 改为: 本地快照优先(历史回测离线够用), 未覆盖的新日期用同花顺补,
    同花顺也没有才 fail-open.
  - 快照定期刷新(月末链每次跑): 重新拉取覆盖到最新的窗口.

遵守铁律: 用系统时间做时长分析; py312 运行; 断点续跑(文件已存在且覆盖足够则跳过)。

用法: py312 python scripts/download_trade_days.py [--start 2020-01-01] [--end auto]
输出: data/trade_days.json ({"generated":..., "start":..., "end":..., "days":[...]})
"""
import argparse
import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
OUT = PROJ / "data" / "trade_days.json"


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default=None,
                    help="结束日期(默认今天+1年, 覆盖未来)")
    a = ap.parse_args()
    t0 = time.time()
    end = a.end or (date.today() + timedelta(days=365)).strftime("%Y-%m-%d")
    log(f"系统时间 {datetime.now():%H:%M:%S} | 拉取交易日 {a.start} ~ {end}")

    try:
        import baostock as bs
    except ImportError:
        log("❌ 缺 baostock (须用 py312 环境)")
        sys.exit(1)
    lg = bs.login()
    if lg.error_code != "0":
        log(f"❌ baostock 登录失败: {lg.error_msg}")
        sys.exit(1)
    try:
        rs = bs.query_trade_dates(start_date=a.start, end_date=end)
        if rs.error_code != "0":
            log(f"❌ query_trade_dates 失败: {rs.error_msg}")
            sys.exit(1)
        days: list[str] = []
        while rs.error_code == "0" and rs.next():
            row = rs.get_row_data()
            if row[1] == "1":  # is_trading_day
                # row[0] = 'YYYY-MM-DD' → 存 yyyyMMdd (与同花顺 _hithink_trade_days 同格式)
                days.append(row[0].replace("-", ""))
        days.sort()
        if not days:
            log("❌ 未拉到任何交易日")
            sys.exit(1)
        data = {
            "generated": datetime.now().isoformat(),
            "source": "baostock query_trade_dates",
            "start": a.start,
            "end": end,
            "n_days": len(days),
            "days": days,
        }
        OUT.write_text(json.dumps(data, ensure_ascii=False))
        log(f"✅ 写入 {OUT}: {len(days)} 个交易日 ({days[0]}~{days[-1]})")
        log(f"   总耗时 {(time.time()-t0):.1f}s (系统时间)")
    finally:
        bs.logout()


if __name__ == "__main__":
    main()
