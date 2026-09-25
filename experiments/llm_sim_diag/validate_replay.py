"""校验：策略回放是否忠实（用存档的当日真实选股结果对照）。

`strategy_backtest.py` 把"时间卡住"来历史重放策略。这只有在**回放逻辑与生产一致**时才有意义，
否则 70 个月的结论就是错的。本脚本给出直接证据：

  取 `data/results/results_YYYYMMDD.json`（当日真实运行 main.py 的产物）中的
  RPS动量突破 / 多周期RPS突破 选股，与"把 as_of 设成同一天再跑一遍"的结果逐日比对。

比对两种股票池，以区分"策略逻辑差异"与"股票池差异"：
  - 逐时点池（回测实际用的）：板块前缀 + 上市满 250 交易日 + 收盘价 ≥ 2 元
  - 当前实盘池（get_base_stock_pool）：与存档当日生产所用更接近

判读：若"当前实盘池"那列重合度高（≥4/5），说明策略逻辑回放忠实。
"""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from dotenv import load_dotenv                                       # noqa: E402
load_dotenv(PROJECT / ".env")                                        # 读取项目根 .env（wxpusher_token 等必填项）

from sequoia_x.core.config import get_settings                       # noqa: E402
from strategy_backtest import ReplayDb, ReplayEngine, point_in_time_pool  # noqa: E402

DB = PROJECT / "data" / "sequoia_v2.db"
CHECK_DATES = ["2026-06-30", "2026-07-31", "2026-08-31", "2026-09-22"]
STRATS = ["RPS动量突破", "多周期RPS突破"]

pd.set_option("display.width", 220)


def main() -> int:
    con = sqlite3.connect(DB)
    settings = get_settings()

    print("载入价格面板 ...")
    px = pd.read_sql(
        "SELECT symbol, date, open, high, low, close, volume, amount, turnover "
        "FROM stock_daily WHERE date >= '2019-06-01'", con)
    for c in ("open", "high", "low", "close"):
        px[c] = pd.to_numeric(px[c], errors="coerce")
    panel = {s: g.reset_index(drop=True) for s, g in px.groupby("symbol")}

    from sequoia_x.data.engine import DataEngine
    live_pool = DataEngine(settings).get_base_stock_pool()
    print(f"当前实盘池 {len(live_pool)} 只")

    from sequoia_x.strategy.rps_breakout import RpsBreakoutStrategy
    from sequoia_x.strategy.rps_multi_period import RpsMultiPeriodStrategy
    CLS = {"RPS动量突破": RpsBreakoutStrategy, "多周期RPS突破": RpsMultiPeriodStrategy}

    tmpdir = tempfile.mkdtemp(prefix="val_")
    rdb = ReplayDb(con, str(Path(tmpdir) / "v.db"))

    rows = []
    for d in CHECK_DATES:
        fp = PROJECT / "data" / "results" / f"results_{d.replace('-', '')}.json"
        if not fp.exists():
            print(f"  {d}: 无存档，跳过")
            continue
        arch = json.load(open(fp, encoding="utf-8"))["strategies"]
        rdb.advance_to(d)
        p_pit = point_in_time_pool(panel, d)
        for label, pool in (("逐时点池", p_pit), ("当前实盘池", live_pool)):
            eng = ReplayEngine(panel, d)
            eng.db_path = rdb.path
            for name, cls in CLS.items():
                want = arch.get(name, [])
                if not want:
                    continue
                got = cls(engine=eng, settings=settings, stock_pool=pool).run() or []
                inter = len(set(want) & set(got))
                rows.append({
                    "日期": d, "策略": name, "池": label,
                    "存档": len(want), "回放": len(got), "重合": inter,
                    "重合率": inter / max(len(want), 1),
                    "存档选股": ",".join(want), "回放选股": ",".join(got),
                })

    t = pd.DataFrame(rows)
    print("\n" + "=" * 110)
    print("回放忠实性校验（回放 vs 存档真实选股）")
    print("=" * 110)
    print(t[["日期", "策略", "池", "存档", "回放", "重合", "重合率"]].to_string(index=False))

    print("\n按股票池汇总：")
    print(t.groupby("池")["重合率"].agg(["mean", "min", "count"]).to_string())

    print("\n逐条明细（便于人工核对差异来源）：")
    for r in t.itertuples(index=False):
        mark = "✓" if r.重合率 >= 0.8 else "✗"
        print(f"  {mark} {r.日期} {r.池:<8} {r.策略:<12} 重合 {r.重合}/{r.存档}")
        if r.重合率 < 1.0:
            print(f"      存档: {r.存档选股}")
            print(f"      回放: {r.回放选股}")
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
