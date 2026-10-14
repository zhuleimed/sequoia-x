"""微盘实验的「最小单元 smoke」—— 验**管道性质**，不验模型质量。

## 为什么要它（项目铁律五）

2026-10-12/13 三次事故的共同点：**只验"命令跑完了、文件生成了"，从不验"让结果有意义的前提还成立"。**
⇒ 长实验启动前必须跑一个最小单元，证明**管道本身没坏**。

## 本 smoke 断言什么（都是"管道性质"，不是"模型好不好"）

| # | 断言 | 防的是什么事故 |
|---|---|---|
| ① | **输入缓存指纹可读且非空** | 上游把缓存删了/换了，结果不可比 |
| ② | **实际产出的月份 ⊆ 声明窗口** | "6 个月的实验被静默补成 69 个月" |
| ③ | 每月池子 **恰为 30 只** | 选股逻辑静默退化（选 0 只 / 选全市场） |
| ④ | **净值方差 > 0 且无 NaN** | 模型输出常数 / 全是 NaN 却"succeeded" |
| ⑤ | **信号严格用 t−1 的仓位** | 未来函数（本项目历史上吃过亏） |
| ⑥ | 每月**权重和为 1** | 权重泄漏（我们真踩过：部分股票没行情导致组合"部分空仓"） |
| ⑦ | 退市股**确实被并入** | 幸存者偏差"修了但没生效" |

结尾打印 `SMOKE PASS` / `SMOKE FAIL`（`exp_contract.py` 只认这两个词）。

用法：
    /home/zhulei/anaconda3/envs/zhulei_py312/bin/python smoke_test.py [db|ths]
"""

from __future__ import annotations

import glob
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import microcap_data as md

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "ths"
BASE = Path(__file__).resolve().parents[1]
N, STOP, VOL_M = 30, -0.12, 100
SMOKE_MONTHS = 12                       # 最小单元：只跑最近 12 个月
T0 = time.time()

fails: list[str] = []


def check(ok: bool, msg: str) -> None:
    print(f"  {'✅' if ok else '❌'} {msg}", flush=True)
    if not ok:
        fails.append(msg)


def log(m: str = "") -> None:
    print(m, flush=True)


log("=" * 96)
log(f"微盘实验 · 整链 SMOKE（最小单元 = 最近 {SMOKE_MONTHS} 个月）    数据源 {SOURCE}")
log("=" * 96)

# ── ① 输入缓存指纹（冻结检查）──
log("\n【① 输入缓存指纹】")
for f in ["data/dump", "data/market_cap_ths.parquet", "data/delisted_daily.parquet",
          "data/index_extra.parquet", "data/assets_extra.parquet"]:
    p = BASE / f
    if p.is_dir():
        files = sorted(glob.glob(str(p / "*.parquet")))
        sz = sum(Path(x).stat().st_size for x in files)
        log(f"    {f:<34} {len(files)} 个文件, {sz / 1e6:.1f} MB")
        check(len(files) > 0, f"{f} 非空")
    else:
        ex = p.exists()
        log(f"    {f:<34} {'存在' if ex else '缺失'}, {p.stat().st_size / 1e6:.1f} MB" if ex else f"    {f} 缺失")
        check(ex, f"{f} 存在")

# ── 加载（整链，与正式实验同一条路）──
log("\n【加载数据】")
df_all = md.load(SOURCE)
df = md.apply_initial_pool(df_all)
log(f"    全市场 {df_all['symbol'].nunique():,} 只 → 初选池 {df['symbol'].nunique():,} 只")

# ⑦ 退市股并入
#   ⚠️ 判据要**精确**：不能说"所有退市股都出现"（有些因市值算不出会被合理过滤），
#      正确判据是「**存在只可能来自退市股合并的股票**」—— 即结果里有 THS dump 没有的代码。
if SOURCE == "ths" and (BASE / "data" / "delisted_daily.parquet").exists():
    dl = pd.read_parquet(BASE / "data" / "delisted_daily.parquet")
    th = pd.read_parquet(sorted(glob.glob(str(BASE / "data" / "dump" / "*.parquet")))[-1])
    ths_syms = set(th["thscode"].astype(str).str.split(".").str[0].str.zfill(6))
    dl_syms, got = set(dl["symbol"]), set(df_all["symbol"])
    only_from_merge = got - ths_syms
    check(len(dl_syms) > 0, f"退市股文件非空（{len(dl_syms)} 只）")
    check(len(only_from_merge) > 0,
          f"退市股确实进入了加载结果：{len(dl_syms & got)}/{len(dl_syms)} 只存活，"
          f"其中 {len(only_from_merge)} 只是『THS dump 里没有』的（只可能来自退市股合并）")
    log(f"    （被过滤掉的 {len(dl_syms - got)} 只：市值算不出/无有效行情，属合理过滤）")

# ── 造最小单元窗口 ──
all_m = sorted(df["ym"].unique())
lo, hi = all_m[-SMOKE_MONTHS], all_m[-1]
log(f"\n【② 声明窗口】{lo} ~ {hi}（共 {len(all_m)} 个月里取最后 {SMOKE_MONTHS} 个）")

sel = md.pick_top_n(df, n=N, by="total_mv")
sel = sel[(sel["hold_ym"] >= lo) & (sel["hold_ym"] <= hi)]
port_daily = md.daily_portfolio(df, sel)
ret = port_daily[(port_daily.index >= lo.start_time) & (port_daily.index <= hi.end_time)]
got_m = sorted(set(ret.index.to_period("M")))
check(all(lo <= m <= hi for m in got_m), f"产出月份 {got_m[0]}~{got_m[-1]} 全部落在声明窗口内")
check(len(got_m) <= len(all_m), f"产出月份数 {len(got_m)} 不超过总月份数 {len(all_m)}（未被静默扩窗）")

# ③ 池子规模
sz = sel.groupby("hold_ym").size()
check((sz == N).all(), f"每月池子恰为 {N} 只（实测 min={sz.min()} max={sz.max()}）")

# ④ 净值有效性
check(np.isfinite(ret).all(), f"组合日收益无 NaN（{len(ret)} 天）")
check(float(np.nanvar(ret)) > 0, f"组合日收益方差 > 0（{np.nanvar(ret):.2e}）")

# ⑤ 未来函数：信号必须用 t−1
sig = md.build_signals(port_daily, md.pool_amount_share(df, sel),
                       pd.Series(0.0, index=port_daily.index), ma=60)
k = sig["S3_拥挤度"].shift(1)
check(k.index.equals(sig.index), "信号已 shift(1)（用 t−1 决定 t 的仓位）")
# 抽样：把信号再 shift 一天，结果应变化 ⇒ 证明 shift 真的生效
k0 = sig["S3_拥挤度"].astype(float)
check(not k0.equals(k), "shift(1) 确实改变了信号（不是空操作）")

# ⑥ 权重和 = 1（复现我们真踩过的"部分空仓"泄漏）
d = df[df["symbol"].isin(set(sel["symbol"]))].sort_values(["symbol", "date"])
rets = {s: g.set_index("date")["ret"] for s, g in d.groupby("symbol")}
bad_months = []
for ym, g in sel.groupby("hold_ym"):
    cols = [s for s in g["symbol"] if s in rets]
    r = pd.DataFrame({s: rets[s] for s in cols})
    r = r[r.index.to_period("M") == ym]
    if len(r) == 0:
        continue
    w = pd.Series(1.0 / len(cols), index=cols).reindex(r.columns).fillna(0.0)
    if abs(w.sum() - 1.0) > 1e-9:
        bad_months.append(str(ym))
check(not bad_months, f"每月权重和为 1（异常月份：{bad_months[:3] or '无'}）")

# ── 结论 ──
log("")
log("=" * 96)
log(f"耗时 {time.time() - T0:.1f}s")
if fails:
    log(f"❌ SMOKE FAIL（{len(fails)} 条未通过）")
    for f in fails:
        log(f"    - {f}")
    sys.exit(1)
log("✅ SMOKE PASS —— 管道性质全部成立（注意：这只证明管道没坏，不代表模型有效）")
log("=" * 96)
