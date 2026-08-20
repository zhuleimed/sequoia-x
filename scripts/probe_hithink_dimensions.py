#!/usr/bin/env python3
"""方案1+2 一次性探测：同花顺历史取数能力 + 新维度落地可行性（只读）。

覆盖：
  [A] 方案1：财务指标 report 各历史报告期回查 ↳ 判断是否可重建历史财务/估值段
  [B] 方案1：财务报表 income/balance 历史区间取数（ms 戳）
  [C] 方案2：龙虎榜 / 涨停池 / 集合竞价 / 热榜 逐项实测
  [D] mootdx 已拉维度 + mootdx_finance 字段抽样（本地，不联网）
  输出统一 JSON-ish 到 stdout，供决策。

用法：HITHINK_FINANCE_API_KEY=<key> py312 python scripts/probe_hithink_dimensions.py
"""
import json
import os
import sys
import sqlite3
import requests
from pathlib import Path

B = "https://fuyao.aicubes.cn"
KEY = os.environ.get("HITHINK_FINANCE_API_KEY", "")
H = {"X-api-key": KEY}
PROJ = Path(__file__).resolve().parent.parent


def q(path, params, tb=20):
    try:
        r = requests.get(B + path, params=params, headers=H, timeout=tb)
        return r.json(), r.status_code
    except Exception as e:
        return {"_err": str(e)}, None


def show(name, j, c, pick):
    code = j.get("code") if isinstance(j, dict) else "?"
    print(f"\n[{name}] HTTP={c} code={code} message={isinstance(j, dict) and j.get('message','')}")
    if code != 0:
        print(f"    {json.dumps(j, ensure_ascii=False)[:250]}")
        return None
    d = j.get("data", {})
    items = d.get("item") or d.get("abilities") or d.get("stock_items") or []
    return items


def main():
    if len(KEY) < 8:
        print("缺 KEY"); sys.exit(1)
    print(f"Key: {KEY[:6]}...\n")

    # ── [A] 财务指标 多报告期回查（历史取数能力）──
    print("=" * 60)
    print("[A] 方案1: 财务指标 历史报告期回查（茅 台, 多个 report）")
    ab_names = {}
    for rp in ["2025-1", "2024-3", "2023-1", "2022-4"]:
        j, c = q("/api/a-share/financials/indicators", {"thscode": "600519.SH", "report": rp})
        if (isinstance(j, dict) and j.get("code") == 0):
            ab = j.get("data", {}).get("abilities", [])
            print(f"  report={rp}: code=0 OK, {len(ab)} 能力")
            for a in ab:
                ab_names.setdefault(a.get("ability"), len(a.get("indicators", [])))
    print(f"  → 财务指标 历史报告期可回查 ✅; 5类指标数: {json.dumps(ab_names, ensure_ascii=False)}")
    # 完整指标名清单（取最近一期）
    j, c = q("/api/a-share/financials/indicators", {"thscode": "600519.SH", "report": "2025-1"})
    if j.get("code") == 0:
        print("  [完整指标名]")
        for a in j["data"]["abilities"]:
            ids = [i["index_id"] for i in a["indicators"]]
            print(f"    {a['ability']}({len(ids)}): {', '.join(ids[:6])}{'...' if len(ids)>6 else ''}")

    # ── [B] 财务报表 历史区间取数 ──
    import datetime
    def ms(s): return int(datetime.datetime.strptime(s, "%Y%m%d").timestamp() * 1000)
    print("\n[B] 方案1: 财务报表 历史区间（2024Q1~2024Q4）")
    j, c = q("/api/a-share/financials/income-statements",
             {"thscode": "600519.SH", "period": "quarterly",
              "start": ms("20240331"), "end": ms("20241231")})
    items = show("income 历史区间", j, c, None)
    if items:
        f0 = items[0]
        print(f"    返回 {len(items)} 期; 字段: {sorted(f0.keys())}")
        print(f"    首期 fiscal_period={f0.get('fiscal_period')} 净利={f0.get('parent_holder_net_profit')}")

    # 资产负债表（含股本 → 重建 PB 所需）
    j, c = q("/api/a-share/financials/balance-sheets",
             {"thscode": "600519.SH", "period": "quarterly",
              "start": ms("20240331"), "end": ms("20241231")})
    items = show("balance 历史区间", j, c, None)
    if items:
        f0 = items[0]
        keys = sorted(f0.keys())
        print(f"    返回 {len(items)} 期; 字段({len(keys)}): {keys}")
        # 找股本/净资产相关
        eq = [k for k in keys if "equity" in k or "share" in k or "asset" in k or "liab" in k]
        print(f"    权益/股本/资产字段: {eq}")

    # ── [C] 方案2: 新维度端点实测 ──
    print("\n[C] 方案2: 同花顺短线/情绪维度落地")
    # C1 龙虎榜
    j, c = q("/api/a-share/special-data/dragon-tiger-list", {"board_type": "all", "date": "2026-08-19"})
    it = show("龙虎榜", j, c, None)
    if it:
        # 结构：可能含 stock_items
        print(f"    样例: {json.dumps(it, ensure_ascii=False)[:300]}")
    # C2 涨停池
    j, c = q("/api/a-share/special-data/limit-up-pool", {"date": "2026-08-19"})
    it = show("涨停池", j, c, None)
    if it:
        print(f"    返回 {len(it)} 条; 样例: {json.dumps(it[0], ensure_ascii=False)[:300] if isinstance(it,list) else json.dumps(it,ensure_ascii=False)[:200]}")
    # C3 集合竞价
    j, c = q("/api/a-share/auction/snapshot", {"thscodes": "600519.SH"})
    show("集合竞价(试路径)", j, c, None)
    # C4 热榜
    j, c = q("/api/a-share/special-data/hot-stock-list", {"count": "50"})
    it = show("热榜", j, c, None)
    if it:
        print(f"    返回 {len(it)} 条; 样例: {json.dumps(it[0] if isinstance(it,list) else it, ensure_ascii=False)[:250]}")
    # C5 异动
    j, c = q("/api/a-share/special-data/anomaly-analysis-list", {"date": "2026-08-19"})
    it = show("异动", j, c, None)
    if it:
        print(f"    返回 {len(it)} 条; 样例: {json.dumps(it[0] if isinstance(it,list) else it, ensure_ascii=False)[:250]}")

    # ── [D] mootdx 已拉维度 + mootdx_finance 字段（本地）──
    print("\n[D] mootdx 维度资产（本地核实，不联网）")
    ef = PROJ / "data/extra_features"
    for sub in ["finance", "forecast", "fund_flow", "holders", "xdxr", "consensus", "news"]:
        d = ef / sub
        n = len(list(d.glob("*.parquet"))) if d.exists() else 0
        print(f"  {sub}: {n} 只 parquet")
    mf = ef / "mootdx_finance"
    files = sorted(mf.glob("*.parquet"))
    print(f"  mootdx_finance: {len(files)} 期")
    if files:
        import pandas as pd
        try:
            df = pd.read_parquet(files[-1])
            print(f"    最新期 {files[-1].name}: {df.shape[0]} 行 × {df.shape[1]} 列")
            print(f"    列样例: {list(df.columns)[:30]}")
        except Exception as e:
            print(f"    读取失败: {e}")

    print("\n✅ 方案1+2 探测完成（只读）")


if __name__ == "__main__":
    main()
